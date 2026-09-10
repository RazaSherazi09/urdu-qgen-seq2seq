"""
Luong (general) attention mechanism — implemented from scratch.

Conceptual steps:
    1. Transform the decoder hidden state with a learnable matrix W_a
       so that it lives in the same space as the encoder outputs.
    2. Compute alignment scores via a dot product:
           score(s_t, h_i) = s_t^T  W_a  h_i
       In practice we do:  (s_t W_a) · h_i^T  via a batched matmul.
    3. Mask padding positions to -inf so they receive zero probability.
    4. Softmax over source positions → attention weights α.
    5. Weighted sum of encoder outputs → context vector c_t.

The "general" variant of Luong attention uses a bilinear form, which
gives the model more expressive power than pure dot-product attention
while remaining simple to implement and explain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """Luong general attention.

    Parameters
    ----------
    encoder_hidden_size : int
        Width of each encoder output vector (2*hidden_size for a
        bidirectional encoder, i.e. 1024).
    decoder_hidden_size : int
        Width of the decoder hidden state (512).
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
    ) -> None:
        super().__init__()

        # W_a projects the decoder state into encoder output space
        # so that the dot product is well-defined.
        self.W_a = nn.Linear(decoder_hidden_size, encoder_hidden_size, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple:
        """Compute attention context and weights for one decoder step.

        Parameters
        ----------
        decoder_hidden : Tensor [B, decoder_hidden_size]
            Decoder hidden state at the current time step (top layer).
        encoder_outputs : Tensor [B, S, encoder_hidden_size]
            All encoder outputs (bidirectional concatenation).
        src_mask : BoolTensor [B, S]
            True at positions that are PAD tokens and should be ignored.

        Returns
        -------
        context : Tensor [B, encoder_hidden_size]
            Attention-weighted sum of encoder outputs.
        attn_weights : Tensor [B, S]
            Attention probability distribution over source positions.
        """
        # Step 1: project decoder hidden → encoder space
        # query: [B, encoder_hidden_size]
        query = self.W_a(decoder_hidden)

        # Step 2: compute alignment scores via batched dot product
        # query.unsqueeze(1): [B, 1, encoder_hidden_size]
        # encoder_outputs.transpose(1,2): [B, encoder_hidden_size, S]
        # scores: [B, 1, S] → squeeze → [B, S]
        scores = torch.bmm(
            query.unsqueeze(1), encoder_outputs.transpose(1, 2)
        ).squeeze(1)

        # Step 3: mask padding positions so they get zero attention
        scores = scores.masked_fill(src_mask, float("-inf"))

        # Step 4: softmax → attention weights  [B, S]
        attn_weights = F.softmax(scores, dim=1)

        # Step 5: weighted sum → context vector
        # attn_weights.unsqueeze(1): [B, 1, S]
        # encoder_outputs: [B, S, encoder_hidden_size]
        # context: [B, 1, encoder_hidden_size] → squeeze → [B, encoder_hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights
