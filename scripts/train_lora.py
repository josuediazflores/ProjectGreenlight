"""
LoRA fine-tuning for Gemma 4 E4B-it on EB-1A/O-1A case law.

Trains on the structured AAO decisions dataset to specialize the model for
immigration legal reasoning tasks: criteria analysis, single-criterion
deep-dive, gap identification, and outcome prediction.

Usage:
    # Single TPU chip (1 device)
    python scripts/train_lora.py --rank 16

    # 8 TPU chips on a single host (recommended for v4-8 or v4-32 worker 0)
    python -m torch_xla.distributed.xla_multiprocessing \
        --nprocs 8 scripts/train_lora.py --rank 16
"""

import argparse
import json
import os
from pathlib import Path

# Disable torch.compile / Inductor — these are CUDA/GPU compilers that hang on TPU
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.disable = True
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "training"
OUTPUT_BASE = Path(__file__).resolve().parent.parent / "checkpoints"
DEFAULT_MODEL = "google/gemma-4-E4B-it"


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_for_training(example: dict, tokenizer, max_length: int = 2048) -> dict:
    """Apply the model's chat template to a conversation example.

    Returns input_ids and attention_mask only — DataCollatorForLanguageModeling
    will handle labels automatically (copying from input_ids after padding).
    """
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (16, 32, 64)")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha (defaults to 2*rank)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    args = parser.parse_args()

    if args.alpha is None:
        args.alpha = 2 * args.rank
    if args.output_dir is None:
        args.output_dir = str(OUTPUT_BASE / f"lora-r{args.rank}")

    print(f"==> Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"==> Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    print(f"==> Configuring LoRA (rank={args.rank}, alpha={args.alpha})")
    # Gemma 4 wraps Linear layers in Gemma4ClippableLinear; target the inner .linear
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

    print(f"==> Loading training data from {DATA_DIR}")
    train_examples = load_jsonl(DATA_DIR / "train.jsonl")
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    print(f"  train: {len(train_examples)} examples")
    print(f"  val:   {len(val_examples)} examples")

    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)

    print("==> Tokenizing")
    train_ds = train_ds.map(
        lambda ex: format_for_training(ex, tokenizer, args.max_length),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    val_ds = val_ds.map(
        lambda ex: format_for_training(ex, tokenizer, args.max_length),
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    print(f"==> Configuring trainer (output: {args.output_dir})")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("==> Starting training")
    trainer.train()

    print(f"==> Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"==> Pushing adapter to Hugging Face Hub: {args.hub_model_id}")
        trainer.push_to_hub()

    print("==> Done!")


if __name__ == "__main__":
    main()
