"""Generate sample outputs for human evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import RESULTS_DIR
from src.evaluation.human_eval import create_evaluation_template

if __name__ == "__main__":
    samples_tsv = RESULTS_DIR / "samples.tsv"
    if not samples_tsv.exists():
        print("Run scripts/evaluate.py first to generate samples.tsv")
        sys.exit(1)
    create_evaluation_template(samples_tsv)
