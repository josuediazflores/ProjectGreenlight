"""
Local benchmark evaluation for fine-tuned LoRA adapters.

Loads a base model (optionally with a LoRA adapter) directly via transformers,
runs inference on the held-out benchmark cases (cases the model has never seen),
parses outputs, and computes metrics for criteria identification, met/not-met,
gap identification, and outcome prediction.

Use it to compare base vs fine-tuned models on the same out-of-distribution cases.

Usage:
    # Baseline: vanilla Qwen 2.5 7B
    python scripts/benchmark_eval_local.py --run-name baseline-qwen7b

    # Post-LoRA: Qwen 2.5 7B + adapter
    python scripts/benchmark_eval_local.py --run-name lora-r16-epoch1 \\
        --adapter checkpoints/qwen-7b-r16/epoch-1

    # Compare two runs
    python scripts/benchmark_eval_local.py --compare baseline-qwen7b lora-r16-epoch1
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCHMARK_EXTRACTED_DIR = DATA_DIR / "benchmark" / "extracted"
BENCHMARK_RESULTS_DIR = DATA_DIR / "benchmark" / "results"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

VALID_CRITERIA = {
    "awards", "membership", "published material", "judging",
    "original contributions", "scholarly articles", "exhibition",
    "leading role", "high salary", "commercial success",
}

SYSTEM_PROMPT = (
    "You are an expert immigration attorney specializing in O-1A and EB-1A "
    "extraordinary ability petitions. You analyze evidence against the USCIS "
    "evidentiary criteria, identify strengths and weaknesses, predict likely "
    "Requests for Evidence (RFEs), and provide strategic guidance grounded in "
    "AAO case law and the USCIS Policy Manual."
)


# ---------------------------------------------------------------------------
# Prompt builders — same task types as format_training_data.py
# ---------------------------------------------------------------------------

def build_criteria_prompt(case: dict) -> list[dict]:
    criteria = case.get("evidence_per_criterion", [])
    evidence_lines = [
        f"**{c['criterion'].title()}:** {c['evidence_submitted']}"
        for c in criteria if c.get("criterion") and c.get("evidence_submitted")
    ]
    user_msg = (
        f"Analyze the following O-1A/EB-1A petition evidence against the applicable "
        f"evidentiary criteria.\n\n"
        f"## Petitioner Background\n{case['petitioner_background']}\n\n"
        f"## Evidence Submitted\n" + "\n\n".join(evidence_lines)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_outcome_prompt(case: dict) -> list[dict]:
    criteria = case.get("evidence_per_criterion", [])
    summary = []
    for c in criteria:
        if c.get("criterion"):
            status = "met" if c.get("met") else "not met"
            summary.append(f"- {c['criterion'].title()}: {status}")

    user_msg = (
        f"Based on the following petition profile, predict the likely outcome "
        f"of this O-1A/EB-1A petition and explain your reasoning.\n\n"
        f"## Petitioner\n{case['petitioner_background']}\n\n"
        f"## Criteria Results\n" + "\n".join(summary)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_criteria_from_response(text: str) -> dict[str, bool]:
    results = {}
    for criterion in VALID_CRITERIA:
        pattern = re.compile(
            rf"(#{1,3}\s*)?{re.escape(criterion)}.*?(met|not met|not satisfied|satisfied|fails?|does not meet)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            verdict = match.group(2).lower()
            results[criterion] = verdict in ("met", "satisfied")
    return results


def extract_outcome_from_response(text: str) -> str | None:
    text_lower = text.lower()
    for pattern in [
        r"predicted?\s+outcome[:\s]*(sustain|dismiss|remand)",
        r"outcome[:\s]*(sustain|dismiss|remand)",
        r"\b(sustain|dismiss|remand)(?:ed|al)?\b",
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(base_model: str, adapter_path: str | None):
    """Load model and tokenizer. If adapter_path is given, load it on top."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"==> Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        print(f"==> Loading LoRA adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, conversation: list[dict], max_new_tokens: int = 1024) -> str:
    """Run inference on a single conversation."""
    import torch

    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_criteria(pred_text: str, case: dict) -> dict:
    """Compare predicted criteria against the case's true criteria."""
    pred = extract_criteria_from_response(pred_text)
    true = {c["criterion"]: bool(c.get("met")) for c in case.get("evidence_per_criterion", []) if c.get("criterion")}

    pred_set = set(pred.keys())
    true_set = set(true.keys())
    if not true_set:
        return {"skipped": True}

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    overlap = pred_set & true_set
    correct_met = sum(1 for c in overlap if pred[c] == true[c])
    met_acc = correct_met / len(overlap) if overlap else 0.0

    return {
        "criteria_precision": precision,
        "criteria_recall": recall,
        "criteria_f1": f1,
        "met_accuracy": met_acc,
        "n_criteria_true": len(true_set),
        "n_criteria_pred": len(pred_set),
    }


def eval_outcome(pred_text: str, case: dict) -> dict:
    pred = extract_outcome_from_response(pred_text)
    true = case.get("outcome", "").lower()
    if true not in {"sustain", "dismiss", "remand"}:
        return {"skipped": True}
    return {
        "outcome_correct": pred == true,
        "pred_outcome": pred,
        "true_outcome": true,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """Average metrics across cases, ignoring skipped ones."""
    out = {}
    keys = ["criteria_precision", "criteria_recall", "criteria_f1", "met_accuracy", "outcome_correct"]
    for key in keys:
        vals = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        if vals:
            out[f"avg_{key}"] = sum(vals) / len(vals)
            out[f"n_{key}"] = len(vals)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(run_name: str, base_model: str, adapter_path: str | None, n_cases: int | None):
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = sorted(BENCHMARK_EXTRACTED_DIR.glob("*.json"))
    if n_cases:
        cases = cases[:n_cases]
    print(f"==> Evaluating {len(cases)} benchmark cases")

    model, tokenizer = load_model(base_model, adapter_path)

    per_case_results = []
    for i, case_path in enumerate(cases):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        if not case.get("petitioner_background") or not case.get("evidence_per_criterion"):
            continue

        try:
            criteria_resp = generate_response(model, tokenizer, build_criteria_prompt(case))
            outcome_resp = generate_response(model, tokenizer, build_outcome_prompt(case))
        except Exception as e:
            print(f"  [{i+1}/{len(cases)}] {case_path.stem}: ERROR {e}")
            continue

        result = {
            "filename": case_path.stem,
            "criteria_response": criteria_resp,
            "outcome_response": outcome_resp,
        }
        result.update(eval_criteria(criteria_resp, case))
        result.update(eval_outcome(outcome_resp, case))
        per_case_results.append(result)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(cases)}] processed")

    aggregate = aggregate_metrics(per_case_results)

    output = {
        "run_name": run_name,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "n_cases": len(per_case_results),
        "aggregate": aggregate,
        "per_case": per_case_results,
    }

    out_path = BENCHMARK_RESULTS_DIR / f"{run_name}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n==> Results saved to {out_path}")
    print(f"==> Aggregate metrics:")
    for k, v in aggregate.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def compare_runs(run_a: str, run_b: str):
    a_path = BENCHMARK_RESULTS_DIR / f"{run_a}.json"
    b_path = BENCHMARK_RESULTS_DIR / f"{run_b}.json"
    if not a_path.exists() or not b_path.exists():
        print(f"Missing run files: {a_path} or {b_path}")
        return

    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())

    print(f"\n=== {run_a} vs {run_b} ===")
    print(f"{'Metric':<30} {run_a:<20} {run_b:<20} {'Δ':<10}")
    print("-" * 80)
    for key in sorted(set(a["aggregate"]) | set(b["aggregate"])):
        va = a["aggregate"].get(key)
        vb = b["aggregate"].get(key)
        if isinstance(va, float) and isinstance(vb, float):
            delta = vb - va
            print(f"{key:<30} {va:<20.4f} {vb:<20.4f} {delta:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, help="Name for this evaluation run")
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--n-cases", type=int, default=None, help="Limit number of cases (for testing)")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two runs")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
    elif args.run_name:
        run_eval(args.run_name, args.base_model, args.adapter, args.n_cases)
    else:
        parser.error("Either --run-name or --compare is required")


if __name__ == "__main__":
    main()
