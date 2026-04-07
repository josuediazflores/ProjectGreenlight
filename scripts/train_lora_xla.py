"""
Manual LoRA fine-tuning loop for Gemma 4 E4B-it on TPU.

Bypasses HuggingFace Trainer entirely (which spawns hung Inductor workers).
Uses peft + torch_xla directly with a custom training loop.

Usage:
    python scripts/train_lora_xla.py --rank 16
"""

import argparse
import json
import os
import time
from pathlib import Path

# Disable Inductor/Dynamo before any torch imports — these spawn 32 hung workers
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_VISIBLE_CHIPS", "0,1,2,3")
# Tell XLA to use bf16 throughout
os.environ.setdefault("XLA_USE_BF16", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "training"
OUTPUT_BASE = Path(__file__).resolve().parent.parent / "checkpoints"
DEFAULT_MODEL = "google/gemma-4-E4B-it"


class JsonlConversationDataset(Dataset):
    """Loads ChatML conversations from a JSONL file and tokenizes on demand."""

    def __init__(self, path: Path, tokenizer, max_length: int = 1024):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = self.tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # static shape required for XLA performance
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        # Mask out padding in labels so loss isn't computed on pad tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.alpha is None:
        args.alpha = 2 * args.rank
    if args.output_dir is None:
        args.output_dir = str(OUTPUT_BASE / f"lora-r{args.rank}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"==> Acquiring XLA device")
    device = xm.xla_device()
    print(f"  device: {device}")

    print(f"==> Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"==> Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    # Enable gradient flow through frozen base model so LoRA adapters can train
    model.enable_input_require_grads()
    # Note: gradient_checkpointing not used — torch.utils.checkpoint tries to
    # access torch.xla as a module which doesn't exist in torch 2.5

    print(f"==> Configuring LoRA (rank={args.rank}, alpha={args.alpha})")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=[
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"==> Moving model to {device}", flush=True)
    move_start = time.time()
    model = model.to(device)
    xm.mark_step()  # Force the device transfer to complete
    print(f"  model on device in {time.time() - move_start:.1f}s", flush=True)
    model.train()

    print(f"==> Loading datasets from {DATA_DIR}")
    train_ds = JsonlConversationDataset(DATA_DIR / "train.jsonl", tokenizer, args.max_length)
    val_ds = JsonlConversationDataset(DATA_DIR / "val.jsonl", tokenizer, args.max_length)
    print(f"  train: {len(train_ds)} examples")
    print(f"  val:   {len(val_ds)} examples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    # Wrap loaders with XLA's parallel loader for device prefetching
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_device_loader = pl.MpDeviceLoader(val_loader, device)

    print(f"==> Setting up optimizer (lr={args.lr})")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    print(f"==> Starting training: {args.epochs} epochs, {total_steps} optimizer steps")
    print(f"  effective batch size: {args.batch_size * args.grad_accum}")

    global_step = 0
    optimizer.zero_grad()
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\n==> Epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0
        running_count = 0

        for step, batch in enumerate(train_device_loader):
            if step == 0:
                print(f"  first batch loaded, starting forward...", flush=True)
                first_step_start = time.time()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            if step == 0:
                print(f"  forward done at {time.time() - first_step_start:.1f}s, starting backward...", flush=True)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            if step == 0:
                print(f"  backward done at {time.time() - first_step_start:.1f}s", flush=True)
            running_loss += loss.item() * args.grad_accum
            running_count += 1

            if (step + 1) % args.grad_accum == 0:
                if global_step == 0:
                    print(f"  calling optimizer step...", flush=True)
                    opt_start = time.time()
                xm.optimizer_step(optimizer, barrier=True)
                if global_step == 0:
                    print(f"  optimizer step done in {time.time() - opt_start:.1f}s", flush=True)
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_interval == 0 or global_step == 1:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / running_count
                    steps_per_sec = global_step / elapsed
                    print(f"  step {global_step}/{total_steps} | loss {avg_loss:.4f} | "
                          f"{steps_per_sec:.2f} steps/s | elapsed {elapsed:.0f}s")
                    running_loss = 0.0
                    running_count = 0

        # End-of-epoch validation
        print(f"==> Validating after epoch {epoch + 1}")
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_device_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
                val_count += 1
        avg_val_loss = val_loss / max(val_count, 1)
        print(f"  val_loss: {avg_val_loss:.4f}")
        model.train()

        # Save checkpoint after each epoch
        ckpt_dir = Path(args.output_dir) / f"epoch-{epoch + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"==> Saving adapter to {ckpt_dir}")
        # Move to CPU before saving to avoid XLA tensor issues
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

    # Final save
    print(f"\n==> Final save to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("==> Done!")


if __name__ == "__main__":
    main()
