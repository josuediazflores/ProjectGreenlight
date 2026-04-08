#!/usr/bin/env bash
# queue_next_training.sh
#
# Watches for a running training process to exit, verifies its adapter saved
# successfully, then launches the next training run with the same hyperparams.
#
# Used to chain back-to-back LoRA training runs on the TPU without manual
# handoff, e.g., kicking off Llama 3.1 8B as soon as Qwen 2.5 7B finishes.
#
# Usage (run on the TPU VM, in nohup):
#
#   nohup bash scripts/queue_next_training.sh \
#       --watch-pid 2509114 \
#       --verify-file checkpoints/qwen-7b-r16/adapter_model.bin \
#       --next-model meta-llama/Llama-3.1-8B-Instruct \
#       --next-output-dir checkpoints/llama-3.1-8b-r16 \
#       --next-log train_llama.log \
#       > watcher.log 2>&1 &
#
# The watcher polls every 60 seconds. It only launches the next run if the
# verify-file exists and is non-empty (i.e., the previous run actually saved
# its adapter — guards against silent crashes).
#
# Hyperparameters for the next run match what we used for the rest of the
# study: rank 16, grad_accum 16, batch_size 1, max_length 1024, 3 epochs.

set -uo pipefail

# Defaults
WATCH_PID=""
VERIFY_FILE=""
NEXT_MODEL=""
NEXT_OUTPUT_DIR=""
NEXT_LOG="train_next.log"
POLL_INTERVAL=60

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch-pid)
            WATCH_PID="$2"
            shift 2
            ;;
        --verify-file)
            VERIFY_FILE="$2"
            shift 2
            ;;
        --next-model)
            NEXT_MODEL="$2"
            shift 2
            ;;
        --next-output-dir)
            NEXT_OUTPUT_DIR="$2"
            shift 2
            ;;
        --next-log)
            NEXT_LOG="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$WATCH_PID" || -z "$VERIFY_FILE" || -z "$NEXT_MODEL" || -z "$NEXT_OUTPUT_DIR" ]]; then
    echo "Missing required arguments." >&2
    echo "Usage: $0 --watch-pid PID --verify-file PATH --next-model MODEL --next-output-dir DIR [--next-log FILE] [--poll-interval SEC]" >&2
    exit 1
fi

cd "$(dirname "$0")/.."  # Move to repo root

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "==> Watcher starting"
log "  Watching PID: $WATCH_PID"
log "  Verify file:  $VERIFY_FILE"
log "  Next model:   $NEXT_MODEL"
log "  Next output:  $NEXT_OUTPUT_DIR"
log "  Next log:     $NEXT_LOG"
log "  Poll every:   ${POLL_INTERVAL}s"

# Phase 1: wait for the training PID to exit
log "==> Phase 1: polling PID $WATCH_PID until it exits"
while ps -p "$WATCH_PID" > /dev/null 2>&1; do
    sleep "$POLL_INTERVAL"
done
log "  PID $WATCH_PID has exited"

# Brief pause for the TPU to release any remaining locks
log "==> Phase 2: waiting 30s for TPU to clear"
sleep 30

# Phase 3: verify the previous run actually saved its adapter
log "==> Phase 3: verifying previous run saved successfully"
if [[ ! -f "$VERIFY_FILE" ]]; then
    log "  ERROR: $VERIFY_FILE does not exist — previous training did not save"
    log "  Refusing to launch next run. Investigate manually."
    exit 1
fi
SIZE=$(stat -c%s "$VERIFY_FILE" 2>/dev/null || stat -f%z "$VERIFY_FILE" 2>/dev/null)
if [[ -z "$SIZE" || "$SIZE" -lt 1000000 ]]; then
    log "  ERROR: $VERIFY_FILE exists but is too small (${SIZE:-unknown} bytes)"
    log "  Expected at least 1MB. Refusing to launch next run."
    exit 1
fi
log "  Verified: $VERIFY_FILE exists and is ${SIZE} bytes"

# Phase 4: clean up TPU locks before next run
log "==> Phase 4: cleaning up stale TPU locks"
sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true
LOCK_HOLDERS=$(sudo lsof /dev/accel0 /dev/accel1 /dev/accel2 /dev/accel3 2>/dev/null | tail -n +2 | awk '{print $2}' | sort -u)
if [[ -n "$LOCK_HOLDERS" ]]; then
    log "  TPU still held by: $LOCK_HOLDERS"
    log "  Waiting another 60s..."
    sleep 60
fi

# Phase 5: launch next training run
log "==> Phase 5: launching next training run"
log "  Activating venv and starting nohup python..."

source .venv/bin/activate || {
    log "  ERROR: Could not activate .venv"
    exit 1
}

PYTHONUNBUFFERED=1 nohup python3 -u scripts/train_lora_xla.py \
    --model "$NEXT_MODEL" \
    --rank 16 \
    --grad-accum 16 \
    --batch-size 1 \
    --max-length 1024 \
    --log-interval 5 \
    --epochs 3 \
    --output-dir "$NEXT_OUTPUT_DIR" \
    > "$NEXT_LOG" 2>&1 &

NEXT_PID=$!
log "  Launched PID $NEXT_PID, output → $NEXT_LOG"
log "==> Watcher done"
