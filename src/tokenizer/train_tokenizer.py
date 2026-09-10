"""
Train a SentencePiece unigram tokenizer on the UQA *training* data only.

The tokenizer corpus is built from training sources and targets exclusively.
Validation and Wiki-UQA data are never included — this prevents any form
of data leakage into the subword vocabulary.
"""

import csv
from pathlib import Path

import sentencepiece as spm

from configs.config import (
    CHARACTER_COVERAGE,
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    SP_CORPUS_PATH,
    SP_MODEL_PREFIX,
    SP_MODEL_TYPE,
    TRAIN_FILE,
    USER_DEFINED_SYMBOLS,
    VOCAB_SIZE,
)


def build_corpus(train_tsv: Path, corpus_path: Path) -> None:
    """Write a plain-text corpus file from training source + target lines.

    Each line in the corpus contains either a source sentence or a target
    question from the *training* split. The tokenizer will learn its
    vocabulary from this corpus alone.
    """
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_tsv.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_tsv}. "
            "Run data preparation first."
        )

    count = 0
    with open(train_tsv, "r", encoding="utf-8") as fin, \
         open(corpus_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            fout.write(row["source"] + "\n")
            fout.write(row["target"] + "\n")
            count += 1

    print(f"[Tokenizer] Built corpus with {count * 2} lines → {corpus_path}")


def train_sentencepiece(
    train_tsv: Path = TRAIN_FILE,
    corpus_path: Path = SP_CORPUS_PATH,
    model_prefix: str = SP_MODEL_PREFIX,
    vocab_size: int = VOCAB_SIZE,
) -> None:
    """Train a SentencePiece unigram model.

    Configuration matches the assignment specification exactly:
    - vocab_size = 8000
    - model_type = unigram
    - character_coverage = 1.0
    - user_defined_symbols = ["<ans>", "</ans>"]
    - pad_id=0, unk_id=1, bos_id=2, eos_id=3
    """
    # Step 1: build corpus from training data only
    build_corpus(train_tsv, corpus_path)

    # Step 2: train SentencePiece
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=SP_MODEL_TYPE,
        character_coverage=CHARACTER_COVERAGE,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )

    print(f"[Tokenizer] Trained SentencePiece model → {model_prefix}.model")
    print(f"[Tokenizer] Vocabulary → {model_prefix}.vocab")
    print(f"[Tokenizer] Vocab size = {vocab_size}, type = {SP_MODEL_TYPE}")


if __name__ == "__main__":
    train_sentencepiece()
