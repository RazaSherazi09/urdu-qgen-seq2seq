"""
Aggregate results into the tables required by the assignment.

Creates:
    results/dataset_statistics.csv
    results/model_configuration.csv
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    BATCH_SIZE,
    BEAM_WIDTH,
    BIDIRECTIONAL,
    CHECKPOINT_DIR,
    DROPOUT,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    RESULTS_DIR,
    SEED,
    SP_MODEL_TYPE,
    VOCAB_SIZE,
)


def create_model_config_table() -> None:
    """Save model configuration as a CSV table."""
    rows = [
        ("Encoder layers", NUM_LAYERS),
        ("Encoder bidirectional", BIDIRECTIONAL),
        ("Decoder layers", NUM_LAYERS),
        ("Embedding dim", EMBEDDING_DIM),
        ("Hidden size", HIDDEN_SIZE),
        ("Dropout", DROPOUT),
        ("Attention", "Luong (general)"),
        ("Tokenizer", f"SentencePiece {SP_MODEL_TYPE}"),
        ("Vocabulary size", VOCAB_SIZE),
        ("Optimizer", "Adam"),
        ("Learning rate", LEARNING_RATE),
        ("Batch size", BATCH_SIZE),
        ("Max epochs", EPOCHS),
        ("Gradient clipping", 1.0),
        ("Beam width", BEAM_WIDTH),
        ("Seed", SEED),
        ("Parameter count", "NOT YET COMPUTED"),
    ]

    path = RESULTS_DIR / "model_configuration.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerows(rows)
    print(f"[Results] Model configuration → {path}")


if __name__ == "__main__":
    create_model_config_table()
    print("[Results] Done.")
