"""
Human evaluation utilities.

Creates a blank evaluation template for two raters and computes
inter-annotator agreement (Cohen's kappa) after ratings are entered.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from configs.config import NUM_HUMAN_EVAL_SAMPLES, RESULTS_DIR, SEED


CRITERIA = ["fluency", "relevance", "answerability"]


def create_evaluation_template(
    samples_tsv: Path,
    output_path: Path = RESULTS_DIR / "human_evaluation.csv",
    n_samples: int = NUM_HUMAN_EVAL_SAMPLES,
    seed: int = SEED,
) -> None:
    """Generate a blank CSV template for human evaluation.

    The template has columns:
        id, source, reference, greedy, beam,
        rater1_fluency, rater1_relevance, rater1_answerability,
        rater2_fluency, rater2_relevance, rater2_answerability

    Rater columns are left empty — humans must fill them with Yes/No.
    """
    import random

    # Read samples
    rows = []
    with open(samples_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    # Select a reproducible subset
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = indices[:min(n_samples, len(rows))]
    selected.sort()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id", "source", "reference", "greedy", "beam",
            "rater1_fluency", "rater1_relevance", "rater1_answerability",
            "rater2_fluency", "rater2_relevance", "rater2_answerability",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, idx in enumerate(selected, 1):
            row = rows[idx]
            writer.writerow({
                "id": i,
                "source": row["source"],
                "reference": row["reference"],
                "greedy": row["greedy"],
                "beam": row["beam"],
                # Leave rating columns empty for humans
                "rater1_fluency": "",
                "rater1_relevance": "",
                "rater1_answerability": "",
                "rater2_fluency": "",
                "rater2_relevance": "",
                "rater2_answerability": "",
            })

    print(f"[HumanEval] Template with {len(selected)} samples → {output_path}")
    print("[HumanEval] Both raters must fill Yes/No in the rater columns.")


def compute_human_eval_results(
    eval_csv: Path = RESULTS_DIR / "human_evaluation.csv",
) -> Dict:
    """Compute % yes and Cohen's kappa from a filled-in evaluation CSV.

    Returns a dict with per-criterion results.
    """
    from sklearn.metrics import cohen_kappa_score

    rows = []
    with open(eval_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = {}

    for criterion in CRITERIA:
        r1_key = f"rater1_{criterion}"
        r2_key = f"rater2_{criterion}"

        r1_ratings = []
        r2_ratings = []
        for row in rows:
            r1 = row.get(r1_key, "").strip().lower()
            r2 = row.get(r2_key, "").strip().lower()
            if r1 and r2:
                r1_ratings.append(1 if r1 == "yes" else 0)
                r2_ratings.append(1 if r2 == "yes" else 0)

        if not r1_ratings:
            results[criterion] = {
                "rater1_pct_yes": "NOT YET COMPUTED",
                "rater2_pct_yes": "NOT YET COMPUTED",
                "cohens_kappa": "NOT YET COMPUTED",
                "n_rated": 0,
            }
            continue

        r1_pct = sum(r1_ratings) / len(r1_ratings) * 100
        r2_pct = sum(r2_ratings) / len(r2_ratings) * 100
        kappa = cohen_kappa_score(r1_ratings, r2_ratings)

        results[criterion] = {
            "rater1_pct_yes": round(r1_pct, 1),
            "rater2_pct_yes": round(r2_pct, 1),
            "cohens_kappa": round(kappa, 3),
            "n_rated": len(r1_ratings),
        }

    return results
