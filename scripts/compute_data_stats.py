"""
Compute statistics on the training/val/test datasets.

Generates a structured JSON describing:
  - Overall counts per split
  - Per-task-type counts
  - Token length distributions (input + output)
  - Criteria coverage
  - Outcome distribution (for outcome_prediction tasks)

Saved to data/metrics/dataset_stats.json so the marimo notebook can plot
distributions, validate the dataset is balanced, and compute summary tables.

Usage:
    python scripts/compute_data_stats.py
    python scripts/compute_data_stats.py --tokenizer Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
METRICS_DIR = DATA_DIR / "metrics"

VALID_CRITERIA = [
    "awards", "membership", "published material", "judging",
    "original contributions", "scholarly articles", "exhibition",
    "leading role", "high salary", "commercial success",
]


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    def pct(p):
        idx = max(0, min(n - 1, int(p * n / 100)))
        return s[idx]
    return {
        "min": s[0],
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": s[-1],
        "mean": round(sum(s) / n, 1),
    }


def analyze_split(examples: list[dict], tokenizer) -> dict:
    """Compute stats for one split (train/val/test)."""
    n = len(examples)
    if n == 0:
        return {"n": 0}

    task_counts = Counter()
    quality_scores = []
    criteria_counts = Counter()  # for single_criterion tasks
    outcome_counts = Counter()  # for outcome_prediction tasks

    user_token_lens = []
    assistant_token_lens = []
    total_token_lens = []

    user_token_lens_by_task = defaultdict(list)
    assistant_token_lens_by_task = defaultdict(list)

    for ex in examples:
        task = ex.get("task_type", "unknown")
        task_counts[task] += 1
        if "quality_score" in ex:
            quality_scores.append(ex["quality_score"])
        if task == "single_criterion" and "criterion" in ex:
            criteria_counts[ex["criterion"]] += 1

        # Find user / assistant messages
        user_msg = ""
        assistant_msg = ""
        for msg in ex.get("conversations", []):
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]

        if task == "outcome_prediction":
            # The format is "## Predicted Outcome: SUSTAIN/DISMISS/REMAND"
            import re as _re
            m = _re.search(r"Predicted Outcome:\s*(SUSTAIN|DISMISS|REMAND)", assistant_msg)
            if m:
                outcome_counts[m.group(1).lower()] += 1

        if tokenizer is not None:
            u_tok = len(tokenizer.encode(user_msg, add_special_tokens=False))
            a_tok = len(tokenizer.encode(assistant_msg, add_special_tokens=False))
            user_token_lens.append(u_tok)
            assistant_token_lens.append(a_tok)
            total_token_lens.append(u_tok + a_tok)
            user_token_lens_by_task[task].append(u_tok)
            assistant_token_lens_by_task[task].append(a_tok)

    result = {
        "n": n,
        "task_counts": dict(task_counts),
        "outcome_counts": dict(outcome_counts),
        "criteria_counts": dict(criteria_counts),
    }

    if quality_scores:
        result["quality_score"] = {
            "min": min(quality_scores),
            "max": max(quality_scores),
            "mean": round(sum(quality_scores) / len(quality_scores), 2),
        }

    if tokenizer is not None:
        result["token_lengths"] = {
            "user": percentiles(user_token_lens),
            "assistant": percentiles(assistant_token_lens),
            "total": percentiles(total_token_lens),
        }
        result["token_lengths_by_task"] = {
            task: {
                "user": percentiles(user_token_lens_by_task[task]),
                "assistant": percentiles(assistant_token_lens_by_task[task]),
            }
            for task in user_token_lens_by_task
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Tokenizer to use for length stats (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="Skip token length stats (faster, no model download needed)")
    parser.add_argument("--out", type=str, default="data/metrics/dataset_stats.json")
    args = parser.parse_args()

    tokenizer = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"==> Loading tokenizer: {args.tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"  WARNING: Could not load tokenizer ({e}). Skipping token length stats.")
            tokenizer = None

    stats = {
        "tokenizer": args.tokenizer if tokenizer else None,
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        path = TRAINING_DIR / f"{split_name}.jsonl"
        if not path.exists():
            print(f"==> Skipping {split_name}.jsonl (not found)")
            continue
        print(f"==> Analyzing {split_name}.jsonl")
        examples = load_jsonl(path)
        stats["splits"][split_name] = analyze_split(examples, tokenizer)

    # Overall totals
    if stats["splits"]:
        all_examples = sum(s["n"] for s in stats["splits"].values())
        stats["total_examples"] = all_examples

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n==> Wrote stats to {out_path}")
    print(f"\n=== Summary ===")
    for split, s in stats["splits"].items():
        print(f"\n{split}: {s['n']} examples")
        print(f"  Task counts: {s['task_counts']}")
        if "outcome_counts" in s and s["outcome_counts"]:
            print(f"  Outcomes:    {s['outcome_counts']}")
        if "token_lengths" in s:
            tl = s["token_lengths"]["total"]
            print(f"  Total tokens (median/p95/max): {tl['median']}/{tl['p95']}/{tl['max']}")


if __name__ == "__main__":
    main()
