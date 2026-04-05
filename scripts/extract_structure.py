"""
Structured extraction pass for EB-1A AAO decisions.

Uses Claude Sonnet to split each raw decision into structured components:
- Petitioner background/facts
- Evidence submitted per criterion
- AAO analysis per criterion
- Outcome reasoning
- Legal citations

These structured extractions feed into format_training_data.py to produce
LoRA fine-tuning examples.

Usage:
    python scripts/extract_structure.py
    python scripts/extract_structure.py --batch-size 10
    python scripts/extract_structure.py --report
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FINAL_DIR = DATA_DIR / "final"
EXTRACTED_DIR = DATA_DIR / "extracted"
REPORT_PATH = DATA_DIR / "extraction_report.json"

EXTRACTION_PROMPT = """You are an immigration law expert. Extract structured components from this AAO decision on an EB-1A/O-1A extraordinary ability petition.

## Case Decision Text
Filename: {filename}

{case_text}

---

Extract the following into a JSON object. Be thorough — capture all substantive content, not just summaries.

Return ONLY a valid JSON object with these exact fields:

{{
  "petitioner_background": "Who is the petitioner? Field of work, nationality if mentioned, what classification they sought, basic facts about their career/qualifications. 2-4 sentences.",

  "evidence_per_criterion": [
    {{
      "criterion": "awards",
      "evidence_submitted": "What specific evidence did the petitioner submit for this criterion? Include names of awards, documents, dates, organizations. Be specific.",
      "aao_analysis": "What did the AAO say about this evidence? Why did it meet or fail the criterion? Include the reasoning, not just the conclusion.",
      "met": false
    }}
  ],

  "outcome": "sustain or dismiss or remand",
  "outcome_reasoning": "The AAO's overall reasoning for the final decision. Include the final merits determination reasoning if present. 2-5 sentences.",

  "legal_citations": ["List every case, statute, or regulation cited (e.g., 'Kazarian v. USCIS, 580 F.3d 1030', '8 C.F.R. § 204.5(h)(3)', 'INA § 203(b)(1)(A)')"],

  "fraud_or_procedural_issues": "Any fraud findings, misrepresentation, procedural problems, or null if none."
}}

Rules:
- Only include criteria that were actually discussed in the decision. Do not invent criteria.
- For "criterion", use exactly one of: "awards", "membership", "published material", "judging", "original contributions", "scholarly articles", "exhibition", "leading role", "high salary", "commercial success"
- For evidence_submitted and aao_analysis, be detailed — these will be used as training data. Capture the substance, not just "the petitioner submitted letters."
- If the decision has a final merits determination section, capture that reasoning in outcome_reasoning.
- Return ONLY the JSON object. No markdown fences, no extra text."""


def get_unextracted_cases() -> list[Path]:
    """Find final cases that haven't been extracted yet."""
    extracted = set()
    if EXTRACTED_DIR.exists():
        extracted = {p.stem for p in EXTRACTED_DIR.glob("*.json")}

    all_cases = sorted(FINAL_DIR.glob("*.json"))
    return [p for p in all_cases if p.stem not in extracted]


def extract_single_case(case_path: Path) -> dict | None:
    """Extract structure from a single case via Claude CLI."""
    case_data = json.loads(case_path.read_text(encoding="utf-8"))
    filename = case_data["filename"]
    case_text = case_data["text"]

    # Truncate very long cases to avoid token limits
    max_chars = 50000
    if len(case_text) > max_chars:
        case_text = case_text[:max_chars] + "\n\n[TRUNCATED]"

    prompt = EXTRACTION_PROMPT.format(filename=filename, case_text=case_text)

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "sonnet", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"  Claude CLI error for {filename}: {result.stderr[:200]}")
            return None

        output = result.stdout.strip()

        # Handle markdown code fences
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()

        parsed = json.loads(output)

        # Carry over metadata from the scored case
        parsed["filename"] = filename
        parsed["score"] = case_data["score"]
        parsed["original_outcome"] = case_data["outcome"]
        parsed["original_criteria"] = case_data["criteria_discussed"]

        return parsed

    except subprocess.TimeoutExpired:
        print(f"  Timeout extracting {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Invalid JSON from Claude for {filename}: {e}")
        print(f"  Raw output: {output[:300]}")
        return None
    except Exception as e:
        print(f"  Error extracting {filename}: {e}")
        return None


def extract_all(batch_size: int = 10, fail_rate_limit: float = 0.2):
    """Extract structure from all unextracted cases."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    unextracted = get_unextracted_cases()
    if not unextracted:
        print("All cases already extracted.")
        return

    print(f"Found {len(unextracted)} cases to extract (batch size: {batch_size})")
    print(f"Will halt if failure rate exceeds {fail_rate_limit:.0%} (checked every 50 results)")

    scored_count = 0
    failed_count = 0
    halted = False

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(extract_single_case, path): path
            for path in unextracted
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            path = futures[future]
            result = future.result()

            if result is not None:
                out_path = EXTRACTED_DIR / f"{path.stem}.json"
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                scored_count += 1
            else:
                failed_count += 1

            total_processed = scored_count + failed_count
            if total_processed >= 50 and total_processed % 50 == 0:
                current_fail_rate = failed_count / total_processed
                if current_fail_rate > fail_rate_limit:
                    print(f"\nHALTING: Failure rate {current_fail_rate:.0%} "
                          f"({failed_count}/{total_processed}) exceeds {fail_rate_limit:.0%} limit.")
                    print("Cancelling remaining tasks...")
                    for f in futures:
                        f.cancel()
                    halted = True
                    break

    print(f"\nExtracted: {scored_count}, Failed: {failed_count}")
    if halted:
        print("Run was halted early. Diagnose failures before retrying.")
    else:
        generate_report()


def generate_report():
    """Generate a summary report of extraction results."""
    extracted_files = sorted(EXTRACTED_DIR.glob("*.json"))
    if not extracted_files:
        print("No extracted files found.")
        return

    total = len(extracted_files)
    criteria_counts = {}
    outcomes = {"sustain": 0, "dismiss": 0, "remand": 0, "other": 0}
    avg_criteria_per_case = 0
    has_fraud = 0
    citation_counts = {}

    for path in extracted_files:
        data = json.loads(path.read_text())

        outcome = data.get("outcome", "other").lower()
        if outcome in outcomes:
            outcomes[outcome] += 1
        else:
            outcomes["other"] += 1

        criteria = data.get("evidence_per_criterion", [])
        avg_criteria_per_case += len(criteria)
        for c in criteria:
            name = c.get("criterion", "unknown")
            criteria_counts[name] = criteria_counts.get(name, 0) + 1

        if data.get("fraud_or_procedural_issues"):
            has_fraud += 1

        for cite in data.get("legal_citations", []):
            # Normalize common citations
            key = cite.strip()[:80]
            citation_counts[key] = citation_counts.get(key, 0) + 1

    report = {
        "total_extracted": total,
        "outcomes": outcomes,
        "avg_criteria_per_case": round(avg_criteria_per_case / total, 1) if total else 0,
        "criteria_frequency": dict(sorted(criteria_counts.items(), key=lambda x: -x[1])),
        "cases_with_fraud_issues": has_fraud,
        "top_citations": dict(sorted(citation_counts.items(), key=lambda x: -x[1])[:20]),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n--- Extraction Report ---")
    print(f"Total extracted: {total}")
    print(f"Outcomes: {outcomes}")
    print(f"Avg criteria per case: {report['avg_criteria_per_case']}")
    print(f"Cases with fraud/procedural issues: {has_fraud}")
    print(f"Report saved to {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Extract structure from AAO decisions using Claude")
    parser.add_argument("--batch-size", type=int, default=10, help="Concurrent extraction agents (default: 10)")
    parser.add_argument("--fail-rate-limit", type=float, default=0.2, help="Max failure rate before halting (default: 0.2)")
    parser.add_argument("--report", action="store_true", help="Generate report from existing extractions only")
    args = parser.parse_args()

    if args.report:
        generate_report()
    else:
        extract_all(batch_size=args.batch_size, fail_rate_limit=args.fail_rate_limit)


if __name__ == "__main__":
    main()
