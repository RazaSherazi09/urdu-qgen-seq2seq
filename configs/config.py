"""
Central configuration for the Urdu Question Generation project.

All hyperparameters, file paths, and special token IDs are defined here
so that no magic numbers are scattered across the codebase.
"""

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────
# Resolve project root relative to this config file's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Random Seed ───────────────────────────────────────────────────────
SEED = 42

# ── Dataset ───────────────────────────────────────────────────────────
DATASET_NAME = "uqa/UQA"
WIKI_DATASET_NAME = "uqa/Wiki-UQA"

# Urdu sentence delimiters used to split context into sentences
SENTENCE_DELIMITERS = ["\u06D4", "\u061F", "!"]   # ۔  ؟  !

# Length thresholds (whitespace-separated token counts)
MAX_SOURCE_WORDS = 60
MAX_TARGET_WORDS = 25

# ── File Paths ────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = DATA_DIR / "train.tsv"
VALID_FILE = DATA_DIR / "valid.tsv"
WIKI_VALID_FILE = DATA_DIR / "wiki_valid.tsv"

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
SP_MODEL_PATH = TOKENIZER_DIR / "ur_sp.model"
SP_VOCAB_PATH = TOKENIZER_DIR / "ur_sp.vocab"
SP_CORPUS_PATH = TOKENIZER_DIR / "sp_corpus.txt"
SP_MODEL_PREFIX = str(TOKENIZER_DIR / "ur_sp")

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── SentencePiece ─────────────────────────────────────────────────────
VOCAB_SIZE = 8000
SP_MODEL_TYPE = "unigram"
CHARACTER_COVERAGE = 1.0
USER_DEFINED_SYMBOLS = ["<ans>", "</ans>"]

# Special token IDs — these are fixed during SentencePiece training
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# ── Model Architecture ───────────────────────────────────────────────
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True   # encoder only; decoder is always unidirectional

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 12
TEACHER_FORCING_RATIO = 1.0
GRAD_CLIP = 1.0       # max gradient norm for clip_grad_norm_

# ── Decoding ──────────────────────────────────────────────────────────
BEAM_WIDTH = 5
MAX_DECODE_LENGTH = 25

# ── Evaluation ────────────────────────────────────────────────────────
NUM_HUMAN_EVAL_SAMPLES = 50
NUM_SAVED_SAMPLES = 50

# ── Debug ─────────────────────────────────────────────────────────────
DEBUG_SUBSET_SIZE = 10_000
DEBUG_EPOCHS = 3
