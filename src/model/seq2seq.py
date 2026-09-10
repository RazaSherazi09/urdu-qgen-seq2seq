"""
Seq2Seq wrapper that ties the Encoder and Decoder together.

Training vs Inference
---------------------
During **training** with teacher forcing (ratio = 1.0), the decoder
receives the ground-truth previous token at every step. This makes
training stable because the decoder never has to recover from its own
mistakes.

During **inference** (teacher_forcing_ratio = 0.0), the decoder feeds
its own predicted token back as input. This is the regime used for
greedy decoding and beam search.

The gap between training and inference behaviour is called "exposure bias".
We do not implement scheduled sampling here because the assignment only
requires standard teacher forcing.
"""

import random

import torch
import torch.nn as nn

from src.model.encoder import Encoder
from src.model.decoder import Decoder


class Seq2Seq(nn.Module):
    """Encoder–Decoder wrapper with teacher forcing support.

    Parameters
    ----------
    encoder : Encoder
    decoder : Decoder
    """

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple:
        """Full forward pass for training.

        Parameters
        ----------
        src : LongTensor [B, S]
            Padded source token IDs.
        src_lengths : LongTensor [B]
            Actual source lengths (descending).
        src_mask : BoolTensor [B, S]
            True at PAD positions.
        tgt : LongTensor [B, T]
            Padded target token IDs (starts with BOS).
        teacher_forcing_ratio : float
            Probability of using the true previous token instead of
            the model's prediction. Default 1.0 (full teacher forcing)
            for training; set to 0.0 for inference.

        Returns
        -------
        outputs : Tensor [B, T-1, vocab_size]
            Vocabulary logits at each target position (excluding the
            first BOS token, which is given as initial input).
        attentions : Tensor [B, T-1, S]
            Attention weights at each decoding step.
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        # Tensor to store decoder outputs
        # We predict tgt_len-1 tokens (positions 1..T-1 given BOS at 0)
        outputs = torch.zeros(
            batch_size, tgt_len - 1, vocab_size, device=src.device
        )
        attentions = torch.zeros(
            batch_size, tgt_len - 1, src.size(1), device=src.device
        )

        # ── Encode ────────────────────────────────────────────────
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # ── Decode step by step ───────────────────────────────────
        # First input to the decoder is the BOS token (column 0 of tgt)
        input_token = tgt[:, 0]  # [B]

        for t in range(tgt_len - 1):
            logits, hidden, cell, attn_weights = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, src_mask
            )

            outputs[:, t, :] = logits
            attentions[:, t, :attn_weights.size(1)] = attn_weights

            # Decide next input: teacher forcing or own prediction
            if random.random() < teacher_forcing_ratio:
                # Use ground-truth token at position t+1
                input_token = tgt[:, t + 1]
            else:
                # Use the model's top prediction
                input_token = logits.argmax(dim=1)

        return outputs, attentions
