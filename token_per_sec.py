import time
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"
SEQ_LEN = 512
BATCH_SIZE = 16
STEPS = int(os.environ.get("STEPS", "100"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
).to(device)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

texts = dataset["text"][:10000]
texts = [t for t in texts if len(t) > 0]

encodings = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=SEQ_LEN,
    return_tensors="pt"
)

input_ids = encodings["input_ids"]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Disable KV cache during training for efficiency
model.config.use_cache = False

# Mixed precision scaler
scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else torch.amp.GradScaler()

model = torch.compile(model)

print("Starting benchmark...")

total_tokens = 0
start = time.time()

for step in range(STEPS):
    idx = (step * BATCH_SIZE) % input_ids.shape[0]
    batch = input_ids[idx:idx + BATCH_SIZE].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(batch, labels=batch, use_cache=False)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_tokens += batch.numel()

end = time.time()

elapsed = end - start
tokens_per_sec = total_tokens / elapsed

print("========== RESULT ==========")
print(f"Elapsed time: {elapsed:.2f} sec")
print(f"Tokens processed: {total_tokens}")
print(f"Tokens/sec: {tokens_per_sec:.2f}")