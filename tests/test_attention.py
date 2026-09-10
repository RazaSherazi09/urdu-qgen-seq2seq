"""Tests for the Luong attention module."""
import pytest
import torch

from src.model.attention import LuongAttention


@pytest.fixture
def attention():
    return LuongAttention(encoder_hidden_size=1024, decoder_hidden_size=512)


class TestAttentionShapes:
    def test_context_shape(self, attention):
        B, S = 4, 10
        decoder_hidden = torch.randn(B, 512)
        encoder_outputs = torch.randn(B, S, 1024)
        src_mask = torch.zeros(B, S, dtype=torch.bool)

        context, weights = attention(decoder_hidden, encoder_outputs, src_mask)
        assert context.shape == (B, 1024)

    def test_weight_shape(self, attention):
        B, S = 4, 10
        decoder_hidden = torch.randn(B, 512)
        encoder_outputs = torch.randn(B, S, 1024)
        src_mask = torch.zeros(B, S, dtype=torch.bool)

        _, weights = attention(decoder_hidden, encoder_outputs, src_mask)
        assert weights.shape == (B, S)

    def test_weights_sum_to_one(self, attention):
        B, S = 4, 10
        decoder_hidden = torch.randn(B, 512)
        encoder_outputs = torch.randn(B, S, 1024)
        src_mask = torch.zeros(B, S, dtype=torch.bool)

        _, weights = attention(decoder_hidden, encoder_outputs, src_mask)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)


class TestAttentionMasking:
    def test_masked_positions_get_zero_weight(self, attention):
        """PAD positions must receive zero attention probability."""
        B, S = 2, 6
        decoder_hidden = torch.randn(B, 512)
        encoder_outputs = torch.randn(B, S, 1024)

        # Mask the last 2 positions for both examples
        src_mask = torch.zeros(B, S, dtype=torch.bool)
        src_mask[:, -2:] = True

        _, weights = attention(decoder_hidden, encoder_outputs, src_mask)

        # Masked positions should have zero (or near-zero) weight
        masked_weights = weights[:, -2:]
        assert (masked_weights < 1e-6).all(), \
            f"Masked positions should get ~0 weight, got {masked_weights}"

    def test_all_masked_except_one(self, attention):
        """If only one position is unmasked, it should get all attention."""
        B, S = 1, 5
        decoder_hidden = torch.randn(B, 512)
        encoder_outputs = torch.randn(B, S, 1024)

        src_mask = torch.ones(B, S, dtype=torch.bool)
        src_mask[0, 2] = False  # Only position 2 is unmasked

        _, weights = attention(decoder_hidden, encoder_outputs, src_mask)
        assert torch.allclose(weights[0, 2], torch.tensor(1.0), atol=1e-5)
