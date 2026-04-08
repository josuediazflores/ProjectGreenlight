"""
Parse a training log from train_lora_xla.py into structured JSONL metrics.

The training script prints lines like:
    step 510/1962 | loss 1.1376 | 0.06 steps/s | elapsed 8961s

This script extracts those into a JSONL file with one object per logged step,
plus summary metadata. Used by the marimo notebook for plotting and by the
auto-update loop to track progress.

Usage:
    # Parse a local log file
    python scripts/parse_train_log.py --log train_qwen.log --out data/metrics/qwen_r16.jsonl

    # With summary printout
    python scripts/parse_train_log.py --log train_qwen.log --out data/metrics/qwen_r16.jsonl --summary
"""

import argparse
import json
import re
from pathlib import Path

# Regex to capture: step N/M | loss X | Y steps/s | elapsed Zs
STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*([\d.]+)\s*steps/s\s*\|\s*elapsed\s+(\d+)s"
)
EPOCH_RE = re.compile(r"==>\s*Epoch\s+(\d+)/(\d+)")
VAL_RE = re.compile(r"val_loss[:\s]+([\d.]+)")
SAVE_RE = re.compile(r"Saving adapter to (.+)")


def parse_log(log_path: Path) -> dict:
    """Parse a training log into structured metrics."""
    if not log_path.exists():
        return {"error": f"Log not found: {log_path}", "steps": []}

    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Convert tqdm carriage returns to newlines so regex works on each update
    text = text.replace("\r", "\n")

    steps = []
    epochs = []
    val_losses = []
    saves = []
    current_epoch = None

    for line in text.splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append({
                "epoch": current_epoch,
                "total_epochs": int(epoch_match.group(2)),
            })
            continue

        step_match = STEP_RE.search(line)
        if step_match:
            steps.append({
                "step": int(step_match.group(1)),
                "total_steps": int(step_match.group(2)),
                "loss": float(step_match.group(3)),
                "steps_per_sec": float(step_match.group(4)),
                "elapsed_sec": int(step_match.group(5)),
                "epoch": current_epoch,
            })
            continue

        val_match = VAL_RE.search(line)
        if val_match:
            val_losses.append({
                "epoch": current_epoch,
                "val_loss": float(val_match.group(1)),
                "after_step": steps[-1]["step"] if steps else None,
            })
            continue

        save_match = SAVE_RE.search(line)
        if save_match:
            saves.append({
                "epoch": current_epoch,
                "path": save_match.group(1).strip(),
                "after_step": steps[-1]["step"] if steps else None,
            })

    if not steps:
        return {
            "error": "No training steps found in log",
            "steps": [],
        }

    last = steps[-1]
    first_loss = steps[0]["loss"]
    last_loss = last["loss"]

    # Compute average step rate over last 20 steps
    recent = steps[-20:] if len(steps) >= 20 else steps
    if len(recent) >= 2:
        recent_elapsed = recent[-1]["elapsed_sec"] - recent[0]["elapsed_sec"]
        recent_steps = recent[-1]["step"] - recent[0]["step"]
        recent_rate = recent_steps / recent_elapsed if recent_elapsed > 0 else 0
    else:
        recent_rate = 0

    # Estimate time remaining
    steps_remaining = last["total_steps"] - last["step"]
    eta_sec = steps_remaining / recent_rate if recent_rate > 0 else None

    return {
        "log_path": str(log_path),
        "current_step": last["step"],
        "total_steps": last["total_steps"],
        "progress_pct": round(100 * last["step"] / last["total_steps"], 2),
        "current_epoch": current_epoch,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_reduction": round(first_loss - last_loss, 4),
        "loss_reduction_pct": round(100 * (first_loss - last_loss) / first_loss, 1),
        "elapsed_sec": last["elapsed_sec"],
        "elapsed_hours": round(last["elapsed_sec"] / 3600, 2),
        "recent_steps_per_sec": round(recent_rate, 4),
        "eta_sec": int(eta_sec) if eta_sec else None,
        "eta_hours": round(eta_sec / 3600, 2) if eta_sec else None,
        "epochs_seen": epochs,
        "saves": saves,
        "val_losses": val_losses,
        "steps": steps,
    }


def write_jsonl(metrics: dict, out_path: Path):
    """Write each step as a JSONL line, plus a summary header."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # Header: summary fields (without the per-step list)
        summary = {k: v for k, v in metrics.items() if k != "steps"}
        summary["_type"] = "summary"
        f.write(json.dumps(summary) + "\n")
        # One line per step
        for step in metrics.get("steps", []):
            step["_type"] = "step"
            f.write(json.dumps(step) + "\n")


def print_summary(metrics: dict):
    if "error" in metrics and not metrics.get("steps"):
        print(f"ERROR: {metrics['error']}")
        return

    print(f"=== Training Progress ===")
    print(f"  Step:       {metrics['current_step']}/{metrics['total_steps']} ({metrics['progress_pct']}%)")
    print(f"  Epoch:      {metrics['current_epoch']}")
    print(f"  Loss:       {metrics['first_loss']} → {metrics['last_loss']} (Δ -{metrics['loss_reduction']}, -{metrics['loss_reduction_pct']}%)")
    print(f"  Elapsed:    {metrics['elapsed_hours']}h ({metrics['elapsed_sec']}s)")
    print(f"  Step rate:  {metrics['recent_steps_per_sec']} steps/s")
    if metrics.get("eta_hours"):
        print(f"  ETA:        {metrics['eta_hours']}h remaining")
    if metrics.get("saves"):
        print(f"  Saves:      {len(metrics['saves'])} checkpoints")
    if metrics.get("val_losses"):
        print(f"  Val losses: {[v['val_loss'] for v in metrics['val_losses']]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to train log file")
    parser.add_argument("--out", type=str, default="data/metrics/train_metrics.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--summary", action="store_true", help="Print summary to stdout")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    metrics = parse_log(log_path)
    write_jsonl(metrics, out_path)

    if args.summary:
        print_summary(metrics)
        print(f"\n  Wrote metrics to: {out_path}")


if __name__ == "__main__":
    main()
