"""
Data preparation for Urdu Question Generation.

Loads the UQA (and optionally Wiki-UQA) dataset, extracts sentence-level
source–target pairs, marks the answer span with <ans>…</ans>, applies
length filtering, and writes train.tsv / valid.tsv / wiki_valid.tsv.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from configs.config import (
    DATASET_NAME,
    WIKI_DATASET_NAME,
    MAX_SOURCE_WORDS,
    MAX_TARGET_WORDS,
    SENTENCE_DELIMITERS,
    DATA_DIR,
    TRAIN_FILE,
    VALID_FILE,
    WIKI_VALID_FILE,
)


# ── helpers ───────────────────────────────────────────────────────────

def _build_delimiter_pattern() -> re.Pattern:
    """Compile a regex that splits text on Urdu sentence delimiters
    while *keeping* each delimiter attached to the sentence it ends."""
    escaped = [re.escape(d) for d in SENTENCE_DELIMITERS]
    # Look-behind: split right after a delimiter character
    return re.compile(r"(?<=[" + "".join(escaped) + r"])\s*")


_SPLIT_RE = _build_delimiter_pattern()


def split_into_sentences(text: str) -> List[Tuple[int, int, str]]:
    """Split *text* into sentences and return (start, end, sentence) triples.

    Character offsets are absolute positions in *text*.
    The delimiter character belongs to the sentence that precedes it, so
    the exclusive end offset sits right after the delimiter.
    """
    parts = _SPLIT_RE.split(text)
    sentences: List[Tuple[int, int, str]] = []
    offset = 0
    for part in parts:
        if not part:
            continue
        # Find the actual start of *part* in the original text
        idx = text.find(part, offset)
        if idx == -1:
            idx = offset
        start = idx
        end = start + len(part)
        sentences.append((start, end, part))
        offset = end
    return sentences


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space and strip."""
    return " ".join(text.split())


def _process_split(
    dataset_split,
    split_name: str,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Process one split of the UQA dataset and return (pairs, stats).

    Each pair is ``{"source": ..., "target": ...}``.
    """
    stats: Dict[str, int] = {
        "raw_rows": 0,
        "answerable_rows": 0,
        "offset_mismatches": 0,
        "valid_pairs_before_filter": 0,
        "removed_by_source_length": 0,
        "removed_by_target_length": 0,
        "final_pairs": 0,
    }
    length_stats: Dict[str, List[int]] = {"src_lengths": [], "tgt_lengths": []}

    pairs: List[Dict[str, str]] = []

    for row in dataset_split:
        stats["raw_rows"] += 1

        # ── Step 1: skip unanswerable rows ───────────────────────
        # Support both:
        # 1. Current UQA flat format:
        #       {"answer": "...", "answer_start": 123}
        #
        # 2. SQuAD-style nested format:
        #       {"answers": {"text": [...], "answer_start": [...]}}

        if "answer" in row and "answer_start" in row:
            answer_text = row["answer"]
            answer_start = row["answer_start"]

            if not answer_text or answer_start is None:
                continue

        elif "answers" in row:
            answers = row.get("answers", {})

            texts = answers.get("text", [])
            starts = answers.get("answer_start", [])

            if not texts or not starts:
                continue

            answer_text = texts[0]
            answer_start = starts[0]

        else:
            continue

        stats["answerable_rows"] += 1
        context = row["context"]
        question = row["question"]

        # ── Step 2: find the sentence containing the answer ──────
        sentences = split_into_sentences(context)
        target_sentence = None
        relative_start: Optional[int] = None

        for sent_start, sent_end, sent_text in sentences:
            if sent_start <= answer_start < sent_end:
                relative_start = answer_start - sent_start
                target_sentence = sent_text
                break

        if target_sentence is None or relative_start is None:
            stats["offset_mismatches"] += 1
            continue

        # ── Step 3: verify offset alignment ──────────────────────
        extracted = target_sentence[relative_start: relative_start + len(answer_text)]
        if extracted != answer_text:
            stats["offset_mismatches"] += 1
            continue

        # ── Step 4: insert <ans> … </ans> markers ────────────────
        before = target_sentence[:relative_start]
        after = target_sentence[relative_start + len(answer_text):]
        source = f"{before}<ans> {answer_text} </ans>{after}"

        source = normalize_whitespace(source)
        target = normalize_whitespace(question)

        stats["valid_pairs_before_filter"] += 1

        # ── Step 5: length filtering ─────────────────────────────
        src_word_count = len(source.split())
        tgt_word_count = len(target.split())

        if src_word_count > MAX_SOURCE_WORDS:
            stats["removed_by_source_length"] += 1
            continue
        if tgt_word_count > MAX_TARGET_WORDS:
            stats["removed_by_target_length"] += 1
            continue

        pairs.append({"source": source, "target": target})
        length_stats["src_lengths"].append(src_word_count)
        length_stats["tgt_lengths"].append(tgt_word_count)

    stats["final_pairs"] = len(pairs)

    # Compute summary length stats
    if length_stats["src_lengths"]:
        import statistics
        stats["mean_source_length"] = round(statistics.mean(length_stats["src_lengths"]), 2)
        stats["mean_target_length"] = round(statistics.mean(length_stats["tgt_lengths"]), 2)
        stats["min_source_length"] = min(length_stats["src_lengths"])
        stats["max_source_length"] = max(length_stats["src_lengths"])
        stats["min_target_length"] = min(length_stats["tgt_lengths"])
        stats["max_target_length"] = max(length_stats["tgt_lengths"])

    return pairs, stats


def _write_tsv(pairs: List[Dict[str, str]], path: Path) -> None:
    """Write source–target pairs to a tab-separated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "target"])
        for pair in pairs:
            writer.writerow([pair["source"], pair["target"]])
    print(f"[Data] Saved {len(pairs)} pairs → {path}")


def _print_stats(split_name: str, stats: Dict) -> None:
    """Pretty-print preprocessing statistics."""
    print(f"\n{'=' * 50}")
    print(f"  {split_name} Split Statistics")
    print(f"{'=' * 50}")
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label:.<40} {value}")
    print()


# ── public API ────────────────────────────────────────────────────────

def prepare_uqa_data() -> Tuple[Dict, Dict]:
    """Download UQA, preprocess train and validation, save TSVs.

    Returns (train_stats, valid_stats).
    """
    print("[Data] Loading UQA dataset …")
    dataset = load_dataset(DATASET_NAME)

    train_pairs, train_stats = _process_split(dataset["train"], "train")
    valid_pairs, valid_stats = _process_split(dataset["validation"], "validation")

    _write_tsv(train_pairs, TRAIN_FILE)
    _write_tsv(valid_pairs, VALID_FILE)

    _print_stats("Train", train_stats)
    _print_stats("Validation", valid_stats)

    return train_stats, valid_stats


def prepare_wiki_uqa_data() -> Dict:
    """Download Wiki-UQA, preprocess, save TSV.

    Returns wiki_stats.
    """
    print("[Data] Loading Wiki-UQA dataset …")
    try:
        dataset = load_dataset(WIKI_DATASET_NAME)
    except Exception as e:
        print(f"[Data] Could not load Wiki-UQA ({e}). "
              "Skipping — you can prepare it manually later.")
        return {}

    # Wiki-UQA may have only a validation or test split
    split_name = "validation" if "validation" in dataset else list(dataset.keys())[0]
    wiki_pairs, wiki_stats = _process_split(dataset[split_name], split_name)

    _write_tsv(wiki_pairs, WIKI_VALID_FILE)
    _print_stats("Wiki-UQA", wiki_stats)
    return wiki_stats


if __name__ == "__main__":
    prepare_uqa_data()
    prepare_wiki_uqa_data()
