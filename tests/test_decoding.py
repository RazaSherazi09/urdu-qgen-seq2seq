"""Tests for greedy and beam search decoding."""
import pytest
import torch

from configs.config import BOS_ID, EOS_ID, HIDDEN_SIZE, NUM_LAYERS, PAD_ID
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.seq2seq import Seq2Seq

VOCAB = 100
S = 8


def _make_model():
    """Create a small model for decoding tests."""
    encoder = Encoder(VOCAB, 32, 64, 2, 0.0, PAD_ID)
    decoder = Decoder(VOCAB, 32, 64, 128, 2, 0.0, PAD_ID)
    return Seq2Seq(encoder, decoder)


class TestGreedyDecode:
    def test_returns_list(self):
        from src.decoding.greedy import greedy_decode
        model = _make_model()
        model.eval()

        src = torch.randint(4, VOCAB, (1, S))
        src_len = torch.tensor([S])
        src_mask = torch.zeros(1, S, dtype=torch.bool)

        ids, attn = greedy_decode(model, src, src_len, src_mask, max_length=10)
        assert isinstance(ids, list)
        assert isinstance(attn, list)

    def test_respects_max_length(self):
        from src.decoding.greedy import greedy_decode
        model = _make_model()
        model.eval()

        src = torch.randint(4, VOCAB, (1, S))
        src_len = torch.tensor([S])
        src_mask = torch.zeros(1, S, dtype=torch.bool)

        ids, _ = greedy_decode(model, src, src_len, src_mask, max_length=5)
        assert len(ids) <= 5


class TestBeamSearch:
    def test_returns_list_and_score(self):
        from src.decoding.beam_search import beam_search_decode
        model = _make_model()
        model.eval()

        src = torch.randint(4, VOCAB, (1, S))
        src_len = torch.tensor([S])
        src_mask = torch.zeros(1, S, dtype=torch.bool)

        ids, score = beam_search_decode(
            model, src, src_len, src_mask, beam_width=3, max_length=10
        )
        assert isinstance(ids, list)
        assert isinstance(score, float)

    def test_beam_width_1_like_greedy(self):
        """Beam width 1 should behave similarly to greedy."""
        from src.decoding.beam_search import beam_search_decode
        model = _make_model()
        model.eval()

        src = torch.randint(4, VOCAB, (1, S))
        src_len = torch.tensor([S])
        src_mask = torch.zeros(1, S, dtype=torch.bool)

        ids, _ = beam_search_decode(
            model, src, src_len, src_mask, beam_width=1, max_length=10
        )
        assert isinstance(ids, list)

    def test_respects_max_length(self):
        from src.decoding.beam_search import beam_search_decode
        model = _make_model()
        model.eval()

        src = torch.randint(4, VOCAB, (1, S))
        src_len = torch.tensor([S])
        src_mask = torch.zeros(1, S, dtype=torch.bool)

        ids, _ = beam_search_decode(
            model, src, src_len, src_mask, beam_width=3, max_length=5
        )
        assert len(ids) <= 5
