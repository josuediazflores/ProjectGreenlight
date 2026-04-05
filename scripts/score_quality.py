"""
Quality scoring orchestrator for EB-1A AAO decisions.

Spawns one Claude subagent per case to score against the rubric.
Processes cases in parallel batches for efficiency.

Usage:
    python scripts/score_quality.py
    python scripts/score_quality.py --batch-size 10 --threshold 6.0
    python scripts/score_quality.py --report  # generate summary report only
"""

import argparse
import json
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_DIR = DATA_DIR / "extracted_text"
SCORED_DIR = DATA_DIR / "scored"
RUBRIC_PATH = Path(__file__).resolve().parent.parent / "rubric" / "scoring_rubric.md"
REPORT_PATH = DATA_DIR / "scoring_report.json"


def load_rubric() -> str:
    """Load the scoring rubric."""
    return RUBRIC_PATH.read_text(encoding="utf-8")


def get_unscored_cases() -> list[Path]:
    """Find text files that haven't been scored yet."""
    scored = set()
    if SCORED_DIR.exists():
        scored = {p.stem for p in SCORED_DIR.glob("*.json")}

    all_cases = sorted(TEXT_DIR.glob("*.txt"))
    return [p for p in all_cases if p.stem not in scored]


def score_single_case(text_path: Path, rubric: str) -> dict | None:
    """Score a single case by calling Claude CLI as a subagent."""
    case_text = text_path.read_text(encoding="utf-8")
    filename = text_path.stem

    # Truncate very long cases to avoid token limits
    max_chars = 50000
    if len(case_text) > max_chars:
        case_text = case_text[:max_chars] + "\n\n[TRUNCATED]"

    prompt = f"""You are scoring an immigration case decision for training data quality.

## Scoring Rubric
{rubric}

## Case Decision Text
Filename: {filename}

{case_text}

---

Score this case according to the rubric above. Return ONLY a valid JSON object with the exact fields specified in the rubric's Output Format section. No other text, no markdown code fences — just the raw JSON object."""

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "sonnet", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"  Claude CLI error for {filename}: {result.stderr[:200]}")
            return None

        output = result.stdout.strip()

        # Try to extract JSON from the response
        # Handle cases where Claude wraps in markdown code fences
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()

        parsed = json.loads(output)
        parsed["filename"] = filename  # ensure correct filename
        return parsed

    except subprocess.TimeoutExpired:
        print(f"  Timeout scoring {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Invalid JSON from Claude for {filename}: {e}")
        print(f"  Raw output: {output[:300]}")
        return None
    except Exception as e:
        print(f"  Error scoring {filename}: {e}")
        return None


def score_all(batch_size: int = 10, threshold: float = 6.0, fail_rate_limit: float = 0.2):
    """Score all unscored cases. Halts if failure rate exceeds fail_rate_limit."""
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    rubric = load_rubric()

    unscored = get_unscored_cases()
    if not unscored:
        print("No unscored cases found. Run parse_pdfs.py first or all cases already scored.")
        return

    print(f"Found {len(unscored)} cases to score (batch size: {batch_size})")
    print(f"Will halt if failure rate exceeds {fail_rate_limit:.0%} (checked every 50 results)")

    scored_count = 0
    failed_count = 0
    halted = False

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(score_single_case, path, rubric): path
            for path in unscored
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            path = futures[future]
            result = future.result()

            if result is not None:
                out_path = SCORED_DIR / f"{path.stem}.json"
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                scored_count += 1
            else:
                failed_count += 1

            # Check failure rate every 50 results after at least 50 processed
            total_processed = scored_count + failed_count
            if total_processed >= 50 and total_processed % 50 == 0:
                current_fail_rate = failed_count / total_processed
                if current_fail_rate > fail_rate_limit:
                    print(f"\n⚠ HALTING: Failure rate {current_fail_rate:.0%} "
                          f"({failed_count}/{total_processed}) exceeds {fail_rate_limit:.0%} limit.")
                    print("Cancelling remaining tasks...")
                    for f in futures:
                        f.cancel()
                    halted = True
                    break

    print(f"\nScored: {scored_count}, Failed: {failed_count}")
    if halted:
        print("Run was halted early. Diagnose failures before retrying.")
    else:
        generate_report(threshold)


def generate_report(threshold: float = 6.0):
    """Generate a summary report of scoring results."""
    scored_files = sorted(SCORED_DIR.glob("*.json"))
    if not scored_files:
        print("No scored files found.")
        return

    scores = []
    outcomes = {"sustain": 0, "dismiss": 0, "remand": 0, "approve": 0, "other": 0}
    auto_rejects = 0
    above_threshold = 0
    criteria_counts = {}

    for path in scored_files:
        data = json.loads(path.read_text())
        score = data.get("score", 0)
        scores.append(score)

        if data.get("auto_reject"):
            auto_rejects += 1

        outcome = data.get("outcome", "other").lower()
        if outcome in outcomes:
            outcomes[outcome] += 1
        else:
            outcomes["other"] += 1

        if score >= threshold:
            above_threshold += 1

        for criterion in data.get("criteria_discussed", []):
            criteria_counts[criterion] = criteria_counts.get(criterion, 0) + 1

    report = {
        "total_scored": len(scores),
        "auto_rejects": auto_rejects,
        "above_threshold": above_threshold,
        "below_threshold": len(scores) - above_threshold,
        "threshold": threshold,
        "score_distribution": {
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "mean": sum(scores) / len(scores) if scores else 0,
            "median": sorted(scores)[len(scores) // 2] if scores else 0,
        },
        "outcomes": outcomes,
        "criteria_frequency": dict(sorted(criteria_counts.items(), key=lambda x: -x[1])),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n--- Scoring Report ---")
    print(f"Total scored: {report['total_scored']}")
    print(f"Auto-rejects: {report['auto_rejects']}")
    print(f"Above threshold ({threshold}): {report['above_threshold']}")
    print(f"Below threshold: {report['below_threshold']}")
    print(f"Score range: {report['score_distribution']['min']:.1f} - {report['score_distribution']['max']:.1f}")
    print(f"Mean score: {report['score_distribution']['mean']:.1f}")
    print(f"Outcomes: {outcomes}")
    print(f"Report saved to {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Score AAO case quality using Claude")
    parser.add_argument("--batch-size", type=int, default=10, help="Concurrent scoring agents (default: 10)")
    parser.add_argument("--threshold", type=float, default=6.0, help="Quality threshold score (default: 6.0)")
    parser.add_argument("--fail-rate-limit", type=float, default=0.2, help="Max failure rate before halting (default: 0.2)")
    parser.add_argument("--report", action="store_true", help="Generate report from existing scores only")
    args = parser.parse_args()

    if args.report:
        generate_report(args.threshold)
    else:
        score_all(batch_size=args.batch_size, threshold=args.threshold, fail_rate_limit=args.fail_rate_limit)


if __name__ == "__main__":
    main()
