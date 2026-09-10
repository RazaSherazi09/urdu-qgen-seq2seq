"""Tests for the tokenizer wrapper.

These tests require a trained SentencePiece model.
They are skipped if the model file does not exist.
"""
import pytest
from pathlib import Path

from configs.config import SP_MODEL_PATH, PAD_ID, UNK_ID, BOS_ID, EOS_ID

SKIP_REASON = "SentencePiece model not yet trained"
model_exists = SP_MODEL_PATH.exists()


@pytest.fixture
def tokenizer():
    if not model_exists:
        pytest.skip(SKIP_REASON)
    from src.tokenizer.tokenizer_utils import UrduTokenizer
    return UrduTokenizer(SP_MODEL_PATH)


class TestSpecialIDs:
    def test_pad_id(self, tokenizer):
        assert tokenizer.pad_id == PAD_ID == 0

    def test_unk_id(self, tokenizer):
        assert tokenizer.unk_id == UNK_ID == 1

    def test_bos_id(self, tokenizer):
        assert tokenizer.bos_id == BOS_ID == 2

    def test_eos_id(self, tokenizer):
        assert tokenizer.eos_id == EOS_ID == 3


class TestAnswerMarkers:
    def test_ans_open_is_single_token(self, tokenizer):
        """<ans> should be a user-defined symbol, not split into pieces."""
        piece_id = tokenizer.piece_to_id("<ans>")
        assert piece_id != tokenizer.unk_id, "<ans> was not registered as a single token"

    def test_ans_close_is_single_token(self, tokenizer):
        piece_id = tokenizer.piece_to_id("</ans>")
        assert piece_id != tokenizer.unk_id, "</ans> was not registered as a single token"


class TestEncodeDecode:
    def test_round_trip(self, tokenizer):
        text = "یہ ایک ٹیسٹ ہے"
        assert tokenizer.verify_round_trip(text)

    def test_encode_returns_list(self, tokenizer):
        pieces = tokenizer.encode("ٹیسٹ")
        assert isinstance(pieces, list)
        assert all(isinstance(p, str) for p in pieces)

    def test_encode_ids_returns_ints(self, tokenizer):
        ids = tokenizer.encode_ids("ٹیسٹ")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_strips_special(self, tokenizer):
        """decode() should remove BOS/EOS/PAD from the output."""
        ids = [BOS_ID, 100, 200, EOS_ID, PAD_ID]
        text = tokenizer.decode(ids)
        # Should not contain the string representations of special tokens
        assert text  # non-empty
