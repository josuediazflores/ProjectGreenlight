"""
Prepare a clean held-out benchmark from cases that were never used in training.

We have 4,643 EB-1A AAO decisions in data/extracted_text/. The training pipeline
only used 1,467 of them (the scored + deduplicated subset in data/final/). The
remaining ~3,176 cases were never scored, never deduplicated, never extracted,
and never seen by the model in any form.

This script picks a random sample of those completely-unseen cases and runs
structured extraction on them, producing a benchmark dataset for evaluating
the fine-tuned LoRA adapter on truly out-of-distribution data.

Usage:
    python scripts/prepare_benchmark.py --n 50 --model opus
    python scripts/prepare_benchmark.py --n 100 --model sonnet --batch-size 5
"""

import argparse
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_TEXT_DIR = DATA_DIR / "extracted_text"
FINAL_DIR = DATA_DIR / "final"
BENCHMARK_DIR = DATA_DIR / "benchmark"
BENCHMARK_EXTRACTED_DIR = BENCHMARK_DIR / "extracted"


# Same prompt as extract_structure.py — we want consistency between training
# extractions and benchmark extractions for fair evaluation.
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
- For evidence_submitted and aao_analysis, be detailed — these will be used as benchmark ground truth.
- If the decision has a final merits determination section, capture that reasoning in outcome_reasoning.
- Return ONLY the JSON object. No markdown fences, no extra text."""


def get_unseen_cases() -> list[Path]:
    """Find raw text cases that were never used in training.

    Returns paths to .txt files in extracted_text/ that have no corresponding
    file in data/final/ (the scored+deduped subset used for training).
    """
    used_stems = {p.stem for p in FINAL_DIR.glob("*.json")}
    all_text_cases = sorted(RAW_TEXT_DIR.glob("*.txt"))
    return [p for p in all_text_cases if p.stem not in used_stems]


def extract_single_case(text_path: Path, model: str) -> dict | None:
    """Extract structure from a single case via Claude CLI."""
    filename = text_path.stem
    case_text = text_path.read_text(encoding="utf-8", errors="replace")

    # Truncate very long cases to avoid token limits
    max_chars = 50000
    if len(case_text) > max_chars:
        case_text = case_text[:max_chars] + "\n\n[TRUNCATED]"

    prompt = EXTRACTION_PROMPT.format(filename=filename, case_text=case_text)

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model, "-p", prompt],
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
        parsed["filename"] = filename
        parsed["source"] = "benchmark_unseen"

        return parsed

    except subprocess.TimeoutExpired:
        print(f"  Timeout extracting {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Invalid JSON from Claude for {filename}: {e}")
        return None
    except Exception as e:
        print(f"  Error extracting {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of cases to extract (default: 50)")
    parser.add_argument("--model", type=str, default="opus", choices=["opus", "sonnet", "haiku"],
                        help="Claude model to use (default: opus)")
    parser.add_argument("--batch-size", type=int, default=5, help="Concurrent extraction agents (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case sampling (default: 42)")
    args = parser.parse_args()

    BENCHMARK_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # Find unseen cases
    unseen = get_unseen_cases()
    print(f"Total cases in extracted_text/: {len(list(RAW_TEXT_DIR.glob('*.txt')))}")
    print(f"Cases used in training (data/final/): {len(list(FINAL_DIR.glob('*.json')))}")
    print(f"Unseen cases (never scored, never trained): {len(unseen)}")

    # Skip cases already extracted as benchmark
    already_done = {p.stem for p in BENCHMARK_EXTRACTED_DIR.glob("*.json")}
    unseen = [p for p in unseen if p.stem not in already_done]
    if already_done:
        print(f"Already extracted: {len(already_done)}")

    # Random sample
    random.seed(args.seed)
    sample = random.sample(unseen, min(args.n, len(unseen)))
    print(f"Sampling {len(sample)} cases for benchmark extraction with {args.model}")
    print(f"Running {args.batch_size} concurrent agents")

    # Extract concurrently
    extracted_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {executor.submit(extract_single_case, path, args.model): path for path in sample}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting benchmark"):
            path = futures[future]
            result = future.result()

            if result is not None:
                out_path = BENCHMARK_EXTRACTED_DIR / f"{path.stem}.json"
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                extracted_count += 1
            else:
                failed_count += 1

    print(f"\nExtracted: {extracted_count}, Failed: {failed_count}")
    print(f"Benchmark data: {BENCHMARK_EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
