"""
PyTorch Dataset and collate function for Urdu question generation.

Each item is a dictionary of token-ID tensors ready for batching.
The collate function pads sequences, creates masks, and returns
batch-first tensors.
"""

import csv
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from configs.config import BOS_ID, EOS_ID, PAD_ID


class QGenDataset(Dataset):
    """Loads a source–target TSV and tokenizes with a SentencePiece model.

    Parameters
    ----------
    tsv_path : Path
        Tab-separated file with ``source`` and ``target`` columns.
    tokenizer : object
        An ``UrduTokenizer`` instance (or anything with an
        ``encode_ids(text) -> List[int]`` method).

    Item format
    -----------
    Each ``__getitem__`` returns::

        {
            "src_ids":    List[int],   # [BOS, ...tokens..., EOS]
            "tgt_ids":    List[int],   # [BOS, ...tokens..., EOS]
            "src_length": int,
            "tgt_length": int,
        }
    """

    def __init__(self, tsv_path: Path, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pairs: List[Dict[str, str]] = []

        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {tsv_path}. "
                "Run data preparation first (scripts/prepare_data.py)."
            )

        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.pairs.append({"source": row["source"], "target": row["target"]})

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        pair = self.pairs[idx]

        src_ids = [BOS_ID] + self.tokenizer.encode_ids(pair["source"]) + [EOS_ID]
        tgt_ids = [BOS_ID] + self.tokenizer.encode_ids(pair["target"]) + [EOS_ID]

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_length": len(src_ids),
            "tgt_length": len(tgt_ids),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a list of dataset items into batch-first tensors.

    Returns
    -------
    dict with keys:
        src : LongTensor  [batch_size, max_src_len]
            Padded source token IDs.
        tgt : LongTensor  [batch_size, max_tgt_len]
            Padded target token IDs.
        src_lengths : LongTensor  [batch_size]
            Actual (un-padded) source lengths.
        tgt_lengths : LongTensor  [batch_size]
            Actual (un-padded) target lengths.
        src_mask : BoolTensor  [batch_size, max_src_len]
            True where the source position is a PAD token (for masking
            attention over padding).
    """
    # Sort batch by descending source length — needed for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x["src_length"], reverse=True)

    src_lengths = [item["src_length"] for item in batch]
    tgt_lengths = [item["tgt_length"] for item in batch]

    max_src = max(src_lengths)
    max_tgt = max(tgt_lengths)

    src_padded = []
    tgt_padded = []

    for item in batch:
        src = item["src_ids"] + [PAD_ID] * (max_src - item["src_length"])
        tgt = item["tgt_ids"] + [PAD_ID] * (max_tgt - item["tgt_length"])
        src_padded.append(src)
        tgt_padded.append(tgt)

    src_tensor = torch.LongTensor(src_padded)        # [B, max_src_len]
    tgt_tensor = torch.LongTensor(tgt_padded)        # [B, max_tgt_len]
    src_len_tensor = torch.LongTensor(src_lengths)    # [B]
    tgt_len_tensor = torch.LongTensor(tgt_lengths)    # [B]

    # Mask is True at PAD positions (used to mask attention scores)
    src_mask = src_tensor == PAD_ID                   # [B, max_src_len]

    return {
        "src": src_tensor,
        "tgt": tgt_tensor,
        "src_lengths": src_len_tensor,
        "tgt_lengths": tgt_len_tensor,
        "src_mask": src_mask,
    }
