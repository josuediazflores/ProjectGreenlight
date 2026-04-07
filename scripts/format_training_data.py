"""
Format extracted case structures into LoRA fine-tuning examples.

Takes the structured extractions from extract_structure.py and generates
multi-task ChatML/ShareGPT training examples in .jsonl format.

Task types generated per case:
1. Full criteria analysis — given facts, analyze all criteria
2. Single criterion deep-dive — given evidence for one criterion, assess it
3. Gap identification — given evidence summary, identify weaknesses and likely RFEs
4. Outcome prediction — given case facts + criteria results, predict outcome

Usage:
    python scripts/format_training_data.py
    python scripts/format_training_data.py --min-score 7.5
    python scripts/format_training_data.py --tasks all
    python scripts/format_training_data.py --tasks criteria,gaps
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
OUTPUT_DIR = DATA_DIR / "training"

SYSTEM_PROMPT = (
    "You are an expert immigration attorney specializing in O-1A and EB-1A "
    "extraordinary ability petitions. You analyze evidence against the USCIS "
    "evidentiary criteria, identify strengths and weaknesses, predict likely "
    "Requests for Evidence (RFEs), and provide strategic guidance grounded in "
    "AAO case law and the USCIS Policy Manual. Be thorough, specific, and cite "
    "relevant legal standards when applicable."
)

VALID_TASKS = {"criteria", "single", "gaps", "outcome"}


def _valid_criterion(c: dict) -> bool:
    """Check that a criterion entry has all the fields we need."""
    return (
        isinstance(c, dict)
        and c.get("criterion")
        and c.get("evidence_submitted")
        and c.get("aao_analysis")
    )


def make_criteria_analysis(case: dict) -> dict | None:
    """Task 1: Full criteria analysis.

    User provides petitioner background + evidence summary.
    Assistant provides per-criterion analysis.
    """
    criteria = [c for c in case.get("evidence_per_criterion", []) if _valid_criterion(c)]
    if len(criteria) < 2 or not case.get("petitioner_background"):
        return None

    # Build user message: petitioner facts + evidence listing
    evidence_lines = []
    for c in criteria:
        evidence_lines.append(f"**{c['criterion'].title()}:** {c['evidence_submitted']}")

    user_msg = (
        f"Analyze the following O-1A/EB-1A petition evidence against the applicable "
        f"evidentiary criteria.\n\n"
        f"## Petitioner Background\n{case['petitioner_background']}\n\n"
        f"## Evidence Submitted\n" + "\n\n".join(evidence_lines)
    )

    # Build assistant response: per-criterion analysis
    analysis_parts = []
    met_count = 0
    for c in criteria:
        status = "MET" if c.get("met") else "NOT MET"
        if c.get("met"):
            met_count += 1
        analysis_parts.append(
            f"### {c['criterion'].title()} — {status}\n{c['aao_analysis']}"
        )

    summary_line = (
        f"\n\n## Summary\nOf the {len(criteria)} criteria analyzed, "
        f"{met_count} were met and {len(criteria) - met_count} were not met. "
        f"{'The petitioner meets the threshold of at least 3 criteria.' if met_count >= 3 else 'The petitioner does not meet the minimum threshold of 3 criteria.'}"
    )

    assistant_msg = "\n\n".join(analysis_parts) + summary_line

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "task_type": "criteria_analysis",
        "source_file": case["filename"],
        "quality_score": case.get("score", 0),
    }


def make_single_criterion(case: dict) -> list[dict]:
    """Task 2: Single criterion deep-dive.

    One example per criterion in the case.
    User provides evidence for one criterion.
    Assistant provides detailed assessment.
    """
    examples = []
    if not case.get("petitioner_background"):
        return examples
    criteria = [c for c in case.get("evidence_per_criterion", []) if _valid_criterion(c)]

    for c in criteria:
        if len(c["aao_analysis"]) < 50:
            continue

        user_msg = (
            f"Assess the following evidence for the **{c['criterion'].title()}** "
            f"criterion in an O-1A/EB-1A extraordinary ability petition.\n\n"
            f"## Petitioner Background\n{case['petitioner_background']}\n\n"
            f"## Evidence for {c['criterion'].title()}\n{c['evidence_submitted']}"
        )

        status = "meets" if c.get("met") else "does not meet"
        assistant_msg = (
            f"## Assessment: {c['criterion'].title()}\n\n"
            f"{c['aao_analysis']}\n\n"
            f"**Conclusion:** This evidence {status} the {c['criterion']} criterion "
            f"under 8 C.F.R. § 204.5(h)(3)."
        )

        examples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "task_type": "single_criterion",
            "criterion": c["criterion"],
            "source_file": case["filename"],
            "quality_score": case.get("score", 0),
        })

    return examples


def make_gap_identification(case: dict) -> dict | None:
    """Task 3: Gap identification / RFE prediction.

    User provides evidence summary.
    Assistant identifies weaknesses and likely RFEs.
    """
    criteria = [c for c in case.get("evidence_per_criterion", []) if _valid_criterion(c)]
    if not case.get("petitioner_background"):
        return None
    failed = [c for c in criteria if not c.get("met")]
    if not failed:
        return None

    # User message: present the evidence as if from a prospective petitioner
    evidence_lines = []
    for c in criteria:
        evidence_lines.append(f"- **{c['criterion'].title()}:** {c['evidence_submitted']}")

    user_msg = (
        f"I'm preparing an O-1A/EB-1A petition. Review my evidence and identify "
        f"weaknesses, gaps, and likely Requests for Evidence (RFEs).\n\n"
        f"## Background\n{case['petitioner_background']}\n\n"
        f"## My Evidence\n" + "\n".join(evidence_lines)
    )

    # Assistant: identify gaps based on what the AAO actually found lacking
    gap_parts = []
    for c in failed:
        gap_parts.append(
            f"### {c['criterion'].title()} — Weakness Identified\n"
            f"{c['aao_analysis']}"
        )

    met = [c for c in criteria if c.get("met")]
    strength_note = ""
    if met:
        met_names = ", ".join(c["criterion"].title() for c in met)
        strength_note = f"\n\n## Strengths\nYour evidence for **{met_names}** appears solid."

    assistant_msg = (
        f"## Identified Weaknesses and Likely RFEs\n\n"
        + "\n\n".join(gap_parts)
        + strength_note
        + f"\n\n## Recommendation\n"
        f"Based on this analysis, {len(failed)} of {len(criteria)} criteria have "
        f"significant weaknesses that would likely trigger an RFE or denial. "
        f"{'You currently meet fewer than 3 criteria, which is below the regulatory threshold.' if len(met) < 3 else 'While you may meet 3+ criteria at initial count, the weaknesses identified could reduce that number on closer adjudication.'}"
    )

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "task_type": "gap_identification",
        "source_file": case["filename"],
        "quality_score": case.get("score", 0),
    }


def make_outcome_prediction(case: dict) -> dict | None:
    """Task 4: Outcome prediction with reasoning.

    User provides case summary.
    Assistant predicts outcome with legal reasoning.
    """
    outcome_reasoning = case.get("outcome_reasoning", "")
    if not outcome_reasoning or len(outcome_reasoning) < 50:
        return None
    if not case.get("petitioner_background"):
        return None

    criteria = [c for c in case.get("evidence_per_criterion", []) if isinstance(c, dict) and c.get("criterion")]
    criteria_summary = []
    for c in criteria:
        status = "met" if c.get("met") else "not met"
        criteria_summary.append(f"- {c['criterion'].title()}: {status}")

    citations = case.get("legal_citations", [])
    citation_str = ""
    if citations:
        citation_str = "\n\n**Key legal citations:** " + "; ".join(citations[:5])

    user_msg = (
        f"Based on the following petition profile, predict the likely outcome "
        f"of this O-1A/EB-1A petition and explain your reasoning.\n\n"
        f"## Petitioner\n{case['petitioner_background']}\n\n"
        f"## Criteria Results\n" + "\n".join(criteria_summary)
    )

    outcome = case.get("outcome", case.get("original_outcome", "unknown"))
    met_count = sum(1 for c in criteria if c.get("met"))

    assistant_msg = (
        f"## Predicted Outcome: {outcome.upper()}\n\n"
        f"**Criteria met:** {met_count} of {len(criteria)} analyzed\n\n"
        f"### Reasoning\n{outcome_reasoning}"
        f"{citation_str}"
    )

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "task_type": "outcome_prediction",
        "source_file": case["filename"],
        "quality_score": case.get("score", 0),
    }


def format_all(min_score: float = 7.0, tasks: set[str] = VALID_TASKS, seed: int = 42):
    """Generate training examples from all extracted cases."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extracted_files = sorted(EXTRACTED_DIR.glob("*.json"))
    if not extracted_files:
        print("No extracted files found. Run extract_structure.py first.")
        return

    all_examples = []
    task_counts = {t: 0 for t in VALID_TASKS}
    skipped = 0

    for path in extracted_files:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue

        if not isinstance(case, dict) or case.get("score", 0) < min_score:
            skipped += 1
            continue

        if "criteria" in tasks:
            ex = make_criteria_analysis(case)
            if ex:
                all_examples.append(ex)
                task_counts["criteria"] += 1

        if "single" in tasks:
            for ex in make_single_criterion(case):
                all_examples.append(ex)
                task_counts["single"] += 1

        if "gaps" in tasks:
            ex = make_gap_identification(case)
            if ex:
                all_examples.append(ex)
                task_counts["gaps"] += 1

        if "outcome" in tasks:
            ex = make_outcome_prediction(case)
            if ex:
                all_examples.append(ex)
                task_counts["outcome"] += 1

    # Shuffle for training
    random.seed(seed)
    random.shuffle(all_examples)

    # Split: 80/10/10
    n = len(all_examples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": all_examples[:train_end],
        "val": all_examples[train_end:val_end],
        "test": all_examples[val_end:],
    }

    for split_name, examples in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"--- Training Data Summary ---")
    print(f"Total examples: {n}")
    print(f"Skipped (below {min_score} score): {skipped}")
    print(f"Task breakdown:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Output: {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Format extracted cases into LoRA training data")
    parser.add_argument("--min-score", type=float, default=7.0, help="Minimum quality score (default: 7.0)")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task types: criteria,single,gaps,outcome,all (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    if args.tasks == "all":
        tasks = VALID_TASKS
    else:
        tasks = set(args.tasks.split(","))
        invalid = tasks - VALID_TASKS
        if invalid:
            print(f"Invalid task types: {invalid}. Valid: {VALID_TASKS}")
            return

    format_all(min_score=args.min_score, tasks=tasks, seed=args.seed)


if __name__ == "__main__":
    main()
