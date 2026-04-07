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

echo "==> Clearing any stale TPU locks"
sudo rm -f /tmp/libtpu_lockfile

# Force single-host TPU mode (use only worker 0's local 4 chips)
# Without these env vars, PJRT tries to coordinate across all pod workers and hangs
export PJRT_DEVICE=TPU
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0,1,2,3

# Persist these env vars for future shell sessions
ENV_FILE="$HOME/.tpu_env"
cat > "$ENV_FILE" <<'EOF'
export PJRT_DEVICE=TPU
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0,1,2,3
EOF
if ! grep -q "tpu_env" "$HOME/.bashrc"; then
    echo "source $ENV_FILE" >> "$HOME/.bashrc"
fi

echo "==> Verifying TPU access (non-fatal)"
set +e
timeout 60 python3 -c "import torch_xla.core.xla_model as xm; print('Device:', xm.xla_device()); print('Count:', len(xm.get_xla_supported_devices()))" || echo "WARNING: TPU verification failed."
set -e

echo "==> Logging into Hugging Face"
echo "If prompted, paste your HF token (read access to gemma-4-E4B-it required)"
# Newer huggingface_hub uses 'hf', older uses 'huggingface-cli'
if command -v hf >/dev/null 2>&1; then
    HF_CLI=hf
    hf auth login
else
    HF_CLI=huggingface-cli
    huggingface-cli login
fi

echo "==> Downloading dataset from Hugging Face"
mkdir -p data/extracted
$HF_CLI download josuediazflores/aao-eb1a-decisions \
    --repo-type=dataset \
    --local-dir=data/extracted

echo "==> Generating LoRA training data"
python3 scripts/format_training_data.py --min-score 7.0 --tasks all

echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  - Run baseline benchmark: python3 scripts/benchmark_eval.py --run-name baseline"
echo "  - Run LoRA fine-tuning:   python3 scripts/train_lora.py --rank 16"
