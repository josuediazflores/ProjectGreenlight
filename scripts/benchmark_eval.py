"""
Benchmark evaluation for O-1A/EB-1A analysis model.

Evaluates model performance on test examples by comparing outputs against
ground truth from AAO decisions. Run before and after LoRA fine-tuning
to measure improvement.

Requires a vLLM endpoint serving Gemma 4 (or any OpenAI-compatible API).

Usage:
    python scripts/benchmark_eval.py --run-name baseline --endpoint http://localhost:8000/v1
    python scripts/benchmark_eval.py --run-name post-lora --endpoint http://localhost:8000/v1
    python scripts/benchmark_eval.py --run-name baseline --llm-judge
    python scripts/benchmark_eval.py --compare baseline post-lora
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_PATH = DATA_DIR / "training" / "test.jsonl"
BENCHMARK_DIR = DATA_DIR / "benchmark"

VALID_CRITERIA = {
    "awards", "membership", "published material", "judging",
    "original contributions", "scholarly articles", "exhibition",
    "leading role", "high salary", "commercial success",
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_model(conversations: list[dict], endpoint: str, max_tokens: int = 2048) -> str:
    """Send a chat completion request to a vLLM / OpenAI-compatible endpoint."""
    resp = requests.post(
        f"{endpoint}/chat/completions",
        json={
            "model": "default",
            "messages": conversations,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_criteria_from_response(text: str) -> dict[str, bool]:
    """Parse model output to find criteria and met/not-met status.

    Returns dict of {criterion_name: met_bool}.
    """
    results = {}
    text_lower = text.lower()

    for criterion in VALID_CRITERIA:
        # Look for the criterion name in the response
        pattern = re.compile(
            rf"(#{1,3}\s*)?{re.escape(criterion)}.*?(met|not met|not satisfied|satisfied|fails?|does not meet)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            verdict = match.group(2).lower()
            met = verdict in ("met", "satisfied")
            results[criterion] = met

    return results


def extract_outcome_from_response(text: str) -> str | None:
    """Parse model output to find predicted outcome."""
    text_lower = text.lower()

    # Look for explicit outcome statements
    for pattern in [
        r"predicted?\s+outcome[:\s]*(sustain|dismiss|remand)",
        r"outcome[:\s]*(sustain|dismiss|remand)",
        r"(sustain|dismiss|remand)(?:ed|al)?",
    ]:
        match = re.search(pattern, text_lower)
        if match:
            raw = match.group(1)
            if "sustain" in raw:
                return "sustain"
            elif "dismiss" in raw:
                return "dismiss"
            elif "remand" in raw:
                return "remand"

    return None


def extract_gaps_from_response(text: str) -> set[str]:
    """Parse model output to find which criteria were flagged as weak."""
    gaps = set()
    text_lower = text.lower()

    for criterion in VALID_CRITERIA:
        # Check if criterion appears near weakness/gap language
        pattern = re.compile(
            rf"{re.escape(criterion)}.*?(weakness|gap|rfe|insufficient|lacking|deficien|fail|not meet|not met|unlikely)",
            re.IGNORECASE,
        )
        if pattern.search(text):
            gaps.add(criterion)

    return gaps


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def get_ground_truth(example: dict) -> dict:
    """Extract ground truth from the assistant message in a test example."""
    task = example["task_type"]
    assistant_msg = ""
    for msg in example["conversations"]:
        if msg["role"] == "assistant":
            assistant_msg = msg["content"]

    truth = {"task_type": task, "raw": assistant_msg}

    if task == "criteria_analysis":
        truth["criteria"] = extract_criteria_from_response(assistant_msg)

    elif task == "single_criterion":
        criterion = example.get("criterion", "")
        truth["criterion"] = criterion
        # Check if met or not from the conclusion
        met = "not met" not in assistant_msg.lower().split("conclusion")[-1] if "conclusion" in assistant_msg.lower() else None
        truth["met"] = met

    elif task == "gap_identification":
        truth["gaps"] = extract_gaps_from_response(assistant_msg)

    elif task == "outcome_prediction":
        truth["outcome"] = extract_outcome_from_response(assistant_msg)

    return truth


# ---------------------------------------------------------------------------
# Evaluation per task type
# ---------------------------------------------------------------------------

def eval_criteria_analysis(pred_text: str, truth: dict) -> dict:
    """Evaluate a criteria analysis prediction."""
    pred_criteria = extract_criteria_from_response(pred_text)
    true_criteria = truth.get("criteria", {})

    if not true_criteria:
        return {"skipped": True}

    # Criteria identification: did the model find the same criteria?
    pred_set = set(pred_criteria.keys())
    true_set = set(true_criteria.keys())

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Met/not-met accuracy on overlapping criteria
    overlap = pred_set & true_set
    correct_assessments = sum(
        1 for c in overlap if pred_criteria[c] == true_criteria[c]
    )
    met_accuracy = correct_assessments / len(overlap) if overlap else 0

    return {
        "criteria_precision": precision,
        "criteria_recall": recall,
        "criteria_f1": f1,
        "met_accuracy": met_accuracy,
        "pred_criteria": list(pred_set),
        "true_criteria": list(true_set),
    }


def eval_single_criterion(pred_text: str, truth: dict) -> dict:
    """Evaluate a single criterion prediction."""
    true_met = truth.get("met")
    if true_met is None:
        return {"skipped": True}

    # Check if model predicted met or not
    criterion = truth.get("criterion", "")
    pred_criteria = extract_criteria_from_response(pred_text)

    if criterion in pred_criteria:
        pred_met = pred_criteria[criterion]
    else:
        # Fallback: look for general met/not met language
        lower = pred_text.lower()
        if "does not meet" in lower or "not met" in lower or "fails" in lower:
            pred_met = False
        elif "meets" in lower or "is met" in lower or "satisfied" in lower:
            pred_met = True
        else:
            return {"skipped": True}

    return {
        "correct": pred_met == true_met,
        "pred_met": pred_met,
        "true_met": true_met,
    }


def eval_gap_identification(pred_text: str, truth: dict) -> dict:
    """Evaluate gap identification prediction."""
    true_gaps = truth.get("gaps", set())
    if not true_gaps:
        return {"skipped": True}

    pred_gaps = extract_gaps_from_response(pred_text)

    tp = len(pred_gaps & true_gaps)
    fp = len(pred_gaps - true_gaps)
    fn = len(true_gaps - pred_gaps)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "gap_precision": precision,
        "gap_recall": recall,
        "gap_f1": f1,
        "pred_gaps": list(pred_gaps),
        "true_gaps": list(true_gaps),
    }


def eval_outcome_prediction(pred_text: str, truth: dict) -> dict:
    """Evaluate outcome prediction."""
    true_outcome = truth.get("outcome")
    if not true_outcome:
        return {"skipped": True}

    pred_outcome = extract_outcome_from_response(pred_text)

    return {
        "correct": pred_outcome == true_outcome,
        "pred_outcome": pred_outcome,
        "true_outcome": true_outcome,
    }


EVAL_FUNCS = {
    "criteria_analysis": eval_criteria_analysis,
    "single_criterion": eval_single_criterion,
    "gap_identification": eval_gap_identification,
    "outcome_prediction": eval_outcome_prediction,
}


# ---------------------------------------------------------------------------
# LLM Judge (optional)
# ---------------------------------------------------------------------------

def llm_judge_reasoning(pred_text: str, truth_text: str) -> int | None:
    """Use Claude Sonnet to score reasoning quality 1-5."""
    prompt = f"""You are evaluating an AI model's legal reasoning on an O-1A/EB-1A immigration case.

## Ground Truth (AAO Decision Analysis)
{truth_text[:3000]}

## Model Output
{pred_text[:3000]}

Rate the model's reasoning quality on a 1-5 scale:
1 = Irrelevant or completely wrong reasoning
2 = Partially relevant but major errors or omissions
3 = Adequate reasoning, gets the gist but misses important points
4 = Good reasoning, covers key points with minor gaps
5 = Excellent reasoning, thorough and well-grounded

Return ONLY a single integer (1-5), nothing else."""

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "sonnet", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None

        score = int(result.stdout.strip()[0])
        if 1 <= score <= 5:
            return score
        return None
    except (ValueError, IndexError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    run_name: str,
    endpoint: str,
    max_examples: int = 0,
    llm_judge: bool = False,
    batch_size: int = 5,
):
    """Run evaluation on test examples."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_PATH.exists():
        print(f"Test file not found: {TEST_PATH}")
        print("Run format_training_data.py first.")
        return

    # Load test examples
    examples = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    if max_examples > 0:
        examples = examples[:max_examples]

    print(f"Evaluating {len(examples)} test examples")
    print(f"Endpoint: {endpoint}")
    print(f"Run name: {run_name}")
    print(f"LLM judge: {'yes' if llm_judge else 'no'}")

    results = []

    def process_example(example):
        task_type = example["task_type"]

        # Build input: system + user messages only
        input_msgs = [
            msg for msg in example["conversations"]
            if msg["role"] in ("system", "user")
        ]

        # Get model prediction
        try:
            pred_text = call_model(input_msgs, endpoint)
        except Exception as e:
            return {"task_type": task_type, "error": str(e), "skipped": True}

        # Get ground truth
        truth = get_ground_truth(example)

        # Evaluate
        eval_func = EVAL_FUNCS.get(task_type)
        if not eval_func:
            return {"task_type": task_type, "skipped": True}

        result = eval_func(pred_text, truth)
        result["task_type"] = task_type
        result["source_file"] = example.get("source_file", "")

        # Optional LLM judge
        if llm_judge and not result.get("skipped"):
            score = llm_judge_reasoning(pred_text, truth["raw"])
            result["reasoning_score"] = score

        return result

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(process_example, ex): ex for ex in examples
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(future.result())

    # Aggregate metrics
    metrics = aggregate_metrics(results)
    metrics["run_name"] = run_name
    metrics["total_examples"] = len(examples)
    metrics["llm_judge"] = llm_judge

    # Save
    run_dir = BENCHMARK_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print_metrics(metrics)


def aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics per task type."""
    by_task = defaultdict(list)
    for r in results:
        if not r.get("skipped"):
            by_task[r["task_type"]].append(r)

    metrics = {}

    # Criteria analysis
    ca = by_task.get("criteria_analysis", [])
    if ca:
        metrics["criteria_analysis"] = {
            "n": len(ca),
            "criteria_f1": avg(ca, "criteria_f1"),
            "criteria_precision": avg(ca, "criteria_precision"),
            "criteria_recall": avg(ca, "criteria_recall"),
            "met_accuracy": avg(ca, "met_accuracy"),
        }

    # Single criterion
    sc = by_task.get("single_criterion", [])
    if sc:
        correct = sum(1 for r in sc if r.get("correct"))
        metrics["single_criterion"] = {
            "n": len(sc),
            "binary_accuracy": correct / len(sc) if sc else 0,
        }

    # Gap identification
    gi = by_task.get("gap_identification", [])
    if gi:
        metrics["gap_identification"] = {
            "n": len(gi),
            "gap_f1": avg(gi, "gap_f1"),
            "gap_precision": avg(gi, "gap_precision"),
            "gap_recall": avg(gi, "gap_recall"),
        }

    # Outcome prediction
    op = by_task.get("outcome_prediction", [])
    if op:
        correct = sum(1 for r in op if r.get("correct"))
        metrics["outcome_prediction"] = {
            "n": len(op),
            "accuracy": correct / len(op) if op else 0,
        }

    # LLM judge scores (across all tasks)
    all_scores = [r["reasoning_score"] for r in results if r.get("reasoning_score") is not None]
    if all_scores:
        metrics["reasoning_quality"] = {
            "n": len(all_scores),
            "mean": sum(all_scores) / len(all_scores),
            "distribution": {i: all_scores.count(i) for i in range(1, 6)},
        }

    return metrics


def avg(items: list[dict], key: str) -> float:
    """Average a numeric field across results."""
    vals = [r[key] for r in items if key in r]
    return sum(vals) / len(vals) if vals else 0


def print_metrics(metrics: dict):
    """Print a readable metrics summary."""
    print(f"\n{'='*50}")
    print(f"  Benchmark: {metrics['run_name']}")
    print(f"  Examples: {metrics['total_examples']}")
    print(f"{'='*50}")

    ca = metrics.get("criteria_analysis")
    if ca:
        print(f"\n  Criteria Analysis (n={ca['n']}):")
        print(f"    Criteria ID F1:    {ca['criteria_f1']:.2f}")
        print(f"    Met/Not-Met Acc:   {ca['met_accuracy']:.2f}")

    sc = metrics.get("single_criterion")
    if sc:
        print(f"\n  Single Criterion (n={sc['n']}):")
        print(f"    Binary Accuracy:   {sc['binary_accuracy']:.2f}")

    gi = metrics.get("gap_identification")
    if gi:
        print(f"\n  Gap Identification (n={gi['n']}):")
        print(f"    Gap Recall:        {gi['gap_recall']:.2f}")
        print(f"    Gap Precision:     {gi['gap_precision']:.2f}")

    op = metrics.get("outcome_prediction")
    if op:
        print(f"\n  Outcome Prediction (n={op['n']}):")
        print(f"    Accuracy:          {op['accuracy']:.2f}")

    rq = metrics.get("reasoning_quality")
    if rq:
        print(f"\n  Reasoning Quality (n={rq['n']}):")
        print(f"    Mean Score (1-5):  {rq['mean']:.2f}")

    print(f"\n{'='*50}")


# ---------------------------------------------------------------------------
# Compare two runs
# ---------------------------------------------------------------------------

def compare_runs(run_a: str, run_b: str):
    """Compare metrics between two benchmark runs."""
    path_a = BENCHMARK_DIR / run_a / "metrics.json"
    path_b = BENCHMARK_DIR / run_b / "metrics.json"

    if not path_a.exists():
        print(f"Run '{run_a}' not found at {path_a}")
        return
    if not path_b.exists():
        print(f"Run '{run_b}' not found at {path_b}")
        return

    ma = json.loads(path_a.read_text())
    mb = json.loads(path_b.read_text())

    print(f"\n{'='*60}")
    print(f"  Comparison: {run_a} vs {run_b}")
    print(f"{'='*60}")

    task_metrics = {
        "criteria_analysis": [("criteria_f1", "Criteria ID F1"), ("met_accuracy", "Met/Not-Met Acc")],
        "single_criterion": [("binary_accuracy", "Binary Accuracy")],
        "gap_identification": [("gap_recall", "Gap Recall"), ("gap_precision", "Gap Precision")],
        "outcome_prediction": [("accuracy", "Accuracy")],
    }

    for task, metric_list in task_metrics.items():
        ta = ma.get(task)
        tb = mb.get(task)
        if not ta or not tb:
            continue

        print(f"\n  {task.replace('_', ' ').title()}:")
        for key, label in metric_list:
            va = ta.get(key, 0)
            vb = tb.get(key, 0)
            diff = vb - va
            arrow = "+" if diff > 0 else ""
            print(f"    {label:20s}  {va:.2f} -> {vb:.2f}  ({arrow}{diff:.2f})")

    # Reasoning quality
    rqa = ma.get("reasoning_quality")
    rqb = mb.get("reasoning_quality")
    if rqa and rqb:
        diff = rqb["mean"] - rqa["mean"]
        arrow = "+" if diff > 0 else ""
        print(f"\n  Reasoning Quality:")
        print(f"    {'Mean (1-5)':20s}  {rqa['mean']:.2f} -> {rqb['mean']:.2f}  ({arrow}{diff:.2f})")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for O-1A model")
    parser.add_argument("--run-name", type=str, help="Name for this eval run (e.g., 'baseline', 'post-lora')")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1",
                        help="vLLM endpoint URL (default: http://localhost:8000/v1)")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Limit number of test examples (0 = all)")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Use Claude Sonnet to score reasoning quality (adds cost)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Concurrent inference requests (default: 5)")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Compare two benchmark runs")

    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
    elif args.run_name:
        run_eval(
            run_name=args.run_name,
            endpoint=args.endpoint,
            max_examples=args.max_examples,
            llm_judge=args.llm_judge,
            batch_size=args.batch_size,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
