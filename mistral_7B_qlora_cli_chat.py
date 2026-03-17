import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "mistral-test/checkpoint-100"

# ---- Load model ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("\nModel ready. Type 'exit' or 'quit' to stop.\n")

# ---- Chat loop ----
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    prompt = f"""### Instruction:
{user_input}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    print(f"\nModel: {response}\n")