# Notes during development

## For Python env:

```bash
# once
sudo apt install -y python3-venv python3-pip
mkdir ~/python3-venvs

# for new envs
python3 -m venv ~/python3-venvs/test_model_training
source ~/python3-venvs/test_model_training/bin/activate
pip install --upgrade pip
```

## login from CLI intop HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
# or
export HF_TOKEN=your_token_here
```

## NVIDIA

- `watch -n 1 nvidia-smi` to monitor usage

## For token/sec testing

```bash
# python packages
pip install torch transformers datasets accelerate

# verify gpu
python -c "import torch; print(torch.cuda.get_device_name())"

# run
python token_per_sec.py
```

## For mistral 7B qlora

```bash
pip install torch transformers datasets peft accelerate bitsandbytes trl

# optional
pip install flash-attn --no-build-isolation

# run
python mistral_7B_qlora.py
```