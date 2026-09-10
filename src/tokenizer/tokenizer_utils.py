"""
Wrapper around a trained SentencePiece model for convenient
encoding / decoding of Urdu text.
"""

from pathlib import Path
from typing import List

import sentencepiece as spm

from configs.config import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SP_MODEL_PATH,
    UNK_ID,
)


class UrduTokenizer:
    """Thin wrapper around ``sentencepiece.SentencePieceProcessor``.

    Exposes simple methods and special-token properties so that the rest
    of the codebase never interacts with SentencePiece directly.

    Parameters
    ----------
    model_path : Path
        Path to the trained ``.model`` file.
    """

    def __init__(self, model_path: Path = SP_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found at {model_path}. "
                "Train the tokenizer first (src/tokenizer/train_tokenizer.py)."
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

    # ── special IDs ───────────────────────────────────────────────────

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    # ── encoding ──────────────────────────────────────────────────────

    def encode(self, text: str) -> List[str]:
        """Encode text into subword string pieces."""
        return self.sp.EncodeAsPieces(text)

    def encode_ids(self, text: str) -> List[int]:
        """Encode text into integer token IDs."""
        return self.sp.EncodeAsIds(text)

    # ── decoding ──────────────────────────────────────────────────────

    def decode(self, ids: List[int]) -> str:
        """Decode a list of integer IDs back into text.

        Automatically strips BOS, EOS, and PAD tokens before decoding
        so that the output is clean display text.
        """
        # Filter out special control tokens
        filtered = [
            tok_id for tok_id in ids
            if tok_id not in (self.pad_id, self.bos_id, self.eos_id)
        ]
        return self.sp.DecodeIds(filtered)

    def decode_pieces(self, pieces: List[str]) -> str:
        """Decode a list of subword string pieces back into text."""
        return self.sp.DecodePieces(pieces)

    # ── utilities ─────────────────────────────────────────────────────

    def id_to_piece(self, token_id: int) -> str:
        """Return the string piece for a single token ID."""
        return self.sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Return the token ID for a string piece."""
        return self.sp.PieceToId(piece)

    def is_unk(self, token_id: int) -> bool:
        """Check whether a token ID is the unknown token."""
        return token_id == self.unk_id

    def verify_round_trip(self, text: str) -> bool:
        """Encode then decode and check if the original text is recovered."""
        ids = self.encode_ids(text)
        reconstructed = self.sp.DecodeIds(ids)
        return reconstructed == text

    def show_example(self, text: str) -> None:
        """Print a full tokenization example for inspection."""
        pieces = self.encode(text)
        ids = self.encode_ids(text)
        reconstructed = self.sp.DecodeIds(ids)
        print(f"  Original:      {text}")
        print(f"  Pieces:        {pieces}")
        print(f"  IDs:           {ids}")
        print(f"  Reconstructed: {reconstructed}")
        print(f"  Round-trip OK: {reconstructed == text}")
        print()
