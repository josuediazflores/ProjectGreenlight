"""
Deduplicate scored cases using TF-IDF cosine similarity.

Computes TF-IDF vectors for all cases that passed quality scoring,
finds near-duplicates via cosine similarity, and removes them
(keeping the higher-scored case).

Usage:
    python scripts/deduplicate.py
    python scripts/deduplicate.py --similarity-threshold 0.95 --score-threshold 7.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_DIR = DATA_DIR / "extracted_text"
SCORED_DIR = DATA_DIR / "scored"
FINAL_DIR = DATA_DIR / "final"
DEDUP_REPORT_PATH = DATA_DIR / "dedup_report.json"


def load_scored_cases(score_threshold: float) -> list[dict]:
    """Load cases that passed quality scoring."""
    cases = []
    for path in sorted(SCORED_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("auto_reject"):
            continue
        if data.get("score", 0) < score_threshold:
            continue

        text_path = TEXT_DIR / f"{data['filename']}.txt"
        if not text_path.exists():
            continue

        data["text"] = text_path.read_text(encoding="utf-8")
        cases.append(data)

    return cases


def find_duplicates(
    cases: list[dict],
    threshold: float = 0.95,
) -> list[tuple[int, int, float]]:
    """Find duplicate pairs above the similarity threshold using TF-IDF."""
    texts = [c["text"] for c in cases]

    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    print("Computing pairwise cosine similarity...")
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Find pairs above threshold (upper triangle only)
    pairs = []
    n = len(cases)
    for i in tqdm(range(n), desc="Finding duplicates"):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                pairs.append((i, j, float(sim)))

    return pairs


def deduplicate(similarity_threshold: float = 0.95, score_threshold: float = 7.0):
    """Run the full deduplication pipeline."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_scored_cases(score_threshold)
    if not cases:
        print("No cases passed quality scoring. Run score_quality.py first.")
        return

    print(f"Loaded {len(cases)} cases above score threshold {score_threshold}")

    duplicate_pairs = find_duplicates(cases, similarity_threshold)

    print(f"Found {len(duplicate_pairs)} duplicate pairs above {similarity_threshold} threshold")

    # Determine which cases to remove (keep higher-scored one)
    to_remove = set()
    pair_details = []
    for i, j, sim in duplicate_pairs:
        score_i = cases[i].get("score", 0)
        score_j = cases[j].get("score", 0)

        remove_idx = j if score_i >= score_j else i
        keep_idx = i if remove_idx == j else j
        to_remove.add(remove_idx)

        pair_details.append({
            "kept": cases[keep_idx]["filename"],
            "removed": cases[remove_idx]["filename"],
            "similarity": round(sim, 4),
            "kept_score": score_i if keep_idx == i else score_j,
            "removed_score": score_j if remove_idx == j else score_i,
        })

    # Copy surviving cases to final/
    surviving = []
    for idx, case in enumerate(cases):
        if idx in to_remove:
            continue

        output = {
            "filename": case["filename"],
            "score": case.get("score", 0),
            "outcome": case.get("outcome"),
            "criteria_discussed": case.get("criteria_discussed", []),
            "summary": case.get("summary", ""),
            "text": case["text"],
        }
        out_path = FINAL_DIR / f"{case['filename']}.json"
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        surviving.append(case["filename"])

    # Save report
    report = {
        "input_cases": len(cases),
        "duplicate_pairs_found": len(duplicate_pairs),
        "cases_removed": len(to_remove),
        "cases_surviving": len(surviving),
        "similarity_threshold": similarity_threshold,
        "score_threshold": score_threshold,
        "pair_details": pair_details,
    }
    DEDUP_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n--- Deduplication Report ---")
    print(f"Input cases: {len(cases)}")
    print(f"Duplicate pairs: {len(duplicate_pairs)}")
    print(f"Cases removed: {len(to_remove)}")
    print(f"Cases surviving: {len(surviving)}")
    print(f"Final dataset saved to {FINAL_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate scored AAO cases")
    parser.add_argument("--similarity-threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for duplicates (default: 0.95)")
    parser.add_argument("--score-threshold", type=float, default=7.0,
                        help="Minimum quality score to include (default: 7.0)")
    args = parser.parse_args()

    deduplicate(
        similarity_threshold=args.similarity_threshold,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
