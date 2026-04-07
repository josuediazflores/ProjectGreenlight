#!/bin/bash
# Setup script for TPU v4 VM
# Run on the TPU VM after `git clone https://github.com/josuediazflores/ProjectGreenlight.git`
#
# Usage: bash scripts/setup_tpu.sh

set -e

echo "==> Verifying Python version"
python3 --version

echo "==> Ensuring python3-venv is installed"
if ! python3 -c "import ensurepip" 2>/dev/null; then
    sudo apt update
    sudo apt install -y python3.10-venv
fi

echo "==> Creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing PyTorch/XLA for TPU v4"
pip install --upgrade pip
pip install "torch~=2.5.0" "torch_xla[tpu]~=2.5.0" \
    -f https://storage.googleapis.com/libtpu-releases/index.html

echo "==> Installing training stack"
pip install \
    "transformers>=4.51.0" \
    "peft>=0.13.0" \
    "datasets>=3.0.0" \
    "accelerate>=1.0.0" \
    "huggingface_hub>=0.26.0" \
    sentencepiece \
    protobuf

echo "==> Setting up TPU log directory"
sudo mkdir -p /tmp/tpu_logs && sudo chmod 777 /tmp/tpu_logs

echo "==> Verifying TPU access"
python3 -c "
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f'XLA device: {device}')
devices = xm.get_xla_supported_devices()
print(f'Device count: {len(devices)}')
print(f'Devices: {devices}')
" 2>&1 | grep -v "Could not open" | grep -v "log file"

echo "==> Logging into Hugging Face"
echo "If prompted, paste your HF token (read access to gemma-4-E4B-it required)"
huggingface-cli login

echo "==> Downloading dataset from Hugging Face"
mkdir -p data/extracted
huggingface-cli download josuediazflores/aao-eb1a-decisions \
    --repo-type=dataset \
    --local-dir=data/extracted

echo "==> Generating LoRA training data"
python3 scripts/format_training_data.py --min-score 7.0 --tasks all

echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  - Run baseline benchmark: python3 scripts/benchmark_eval.py --run-name baseline"
echo "  - Run LoRA fine-tuning:   python3 scripts/train_lora.py --rank 16"
