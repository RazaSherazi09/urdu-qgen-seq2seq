"""Tests for model tensor shapes through encoder, decoder, and full seq2seq."""
import pytest
import torch

from configs.config import EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, PAD_ID
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.seq2seq import Seq2Seq

VOCAB = 500  # small vocab for testing
B = 4        # batch size
S = 12       # source length
T = 8        # target length


@pytest.fixture
def encoder():
    return Encoder(
        vocab_size=VOCAB, embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        dropout=0.0, pad_id=PAD_ID,
    )


@pytest.fixture
def decoder():
    return Decoder(
        vocab_size=VOCAB, embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE, encoder_hidden_size=HIDDEN_SIZE * 2,
        num_layers=NUM_LAYERS, dropout=0.0, pad_id=PAD_ID,
    )


@pytest.fixture
def seq2seq(encoder, decoder):
    return Seq2Seq(encoder, decoder)


class TestEncoderShapes:
    def test_encoder_output_shape(self, encoder):
        src = torch.randint(0, VOCAB, (B, S))
        src_lengths = torch.full((B,), S)

        enc_out, hidden, cell = encoder(src, src_lengths)

        # Bidirectional output: [B, S, 2*hidden_size]
        assert enc_out.shape == (B, S, HIDDEN_SIZE * 2)

    def test_encoder_hidden_shape(self, encoder):
        src = torch.randint(0, VOCAB, (B, S))
        src_lengths = torch.full((B,), S)

        _, hidden, cell = encoder(src, src_lengths)

        # After projection: [num_layers, B, hidden_size]
        assert hidden.shape == (NUM_LAYERS, B, HIDDEN_SIZE)
        assert cell.shape == (NUM_LAYERS, B, HIDDEN_SIZE)


class TestDecoderShapes:
    def test_decoder_step_shapes(self, decoder):
        enc_hidden = HIDDEN_SIZE * 2
        input_token = torch.randint(0, VOCAB, (B,))
        hidden = torch.randn(NUM_LAYERS, B, HIDDEN_SIZE)
        cell = torch.randn(NUM_LAYERS, B, HIDDEN_SIZE)
        encoder_outputs = torch.randn(B, S, enc_hidden)
        src_mask = torch.zeros(B, S, dtype=torch.bool)

        logits, new_h, new_c, attn = decoder.forward_step(
            input_token, hidden, cell, encoder_outputs, src_mask
        )

        assert logits.shape == (B, VOCAB)
        assert new_h.shape == (NUM_LAYERS, B, HIDDEN_SIZE)
        assert new_c.shape == (NUM_LAYERS, B, HIDDEN_SIZE)
        assert attn.shape == (B, S)


class TestSeq2SeqShapes:
    def test_full_forward(self, seq2seq):
        src = torch.randint(0, VOCAB, (B, S))
        tgt = torch.randint(0, VOCAB, (B, T))
        src_lengths = torch.full((B,), S)
        src_mask = torch.zeros(B, S, dtype=torch.bool)

        outputs, attentions = seq2seq(
            src, src_lengths, src_mask, tgt, teacher_forcing_ratio=1.0
        )

        # outputs: [B, T-1, vocab_size]
        assert outputs.shape == (B, T - 1, VOCAB)
        # attentions: [B, T-1, S]
        assert attentions.shape == (B, T - 1, S)

    def test_outputs_are_finite(self, seq2seq):
        src = torch.randint(4, VOCAB, (B, S))  # avoid special tokens
        tgt = torch.randint(4, VOCAB, (B, T))
        src_lengths = torch.full((B,), S)
        src_mask = torch.zeros(B, S, dtype=torch.bool)

        outputs, _ = seq2seq(src, src_lengths, src_mask, tgt)
        assert torch.isfinite(outputs).all()
