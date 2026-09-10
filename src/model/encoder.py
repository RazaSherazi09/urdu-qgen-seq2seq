"""
Bidirectional LSTM Encoder for the Urdu question generation seq2seq model.

Architecture (per assignment spec):
    - 2-layer bidirectional LSTM
    - embedding_dim = 256
    - hidden_size = 512 per direction (1024 concatenated)
    - dropout = 0.3

The encoder converts a padded source sequence into:
    1. encoder_outputs:  [B, S, 2*hidden_size]  — used by attention
    2. hidden, cell:     [num_layers, B, hidden_size] — initial decoder state

Because the encoder is bidirectional, the raw LSTM produces hidden states
of size 2*hidden_size. The decoder is unidirectional with hidden_size=512,
so we must project the encoder's final states down. We concatenate the
forward and backward states for each layer and pass them through a linear
layer, which lets the decoder start from a meaningful initial state.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """2-layer bidirectional LSTM encoder.

    Parameters
    ----------
    vocab_size : int
        Size of the source vocabulary (including special tokens).
    embedding_dim : int
        Dimensionality of token embeddings (256).
    hidden_size : int
        Hidden size *per direction* (512).
    num_layers : int
        Number of LSTM layers (2).
    dropout : float
        Dropout probability (0.3).
    pad_id : int
        Padding token ID, used in the embedding layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_id: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Project concatenated forward+backward hidden states → decoder size
        # Each layer's forward hidden [B, H] and backward hidden [B, H]
        # are concatenated to [B, 2H], then projected to [B, H].
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple:
        """Encode a padded source batch.

        Parameters
        ----------
        src : LongTensor [B, S]
            Padded source token IDs.
        src_lengths : LongTensor [B]
            Actual (un-padded) source lengths, in descending order.

        Returns
        -------
        encoder_outputs : Tensor [B, S, 2*hidden_size]
            Concatenated forward + backward outputs at every position.
        hidden : Tensor [num_layers, B, hidden_size]
            Projected initial hidden state for the decoder.
        cell : Tensor [num_layers, B, hidden_size]
            Projected initial cell state for the decoder.
        """
        # src: [B, S] → embedded: [B, S, embedding_dim]
        embedded = self.dropout(self.embedding(src))

        # Packing avoids wasted computation on padding tokens.
        # src_lengths must be on CPU for pack_padded_sequence.
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # packed_outputs contains forward+backward hidden states
        # hidden: [num_layers*2, B, hidden_size]
        # cell:   [num_layers*2, B, hidden_size]
        packed_outputs, (hidden, cell) = self.lstm(packed)

        # Unpack → encoder_outputs: [B, S, 2*hidden_size]
        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        # ── Project bidirectional states for the decoder ──────────
        # hidden shape from bidirectional LSTM: [num_layers*2, B, H]
        # The layout is [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]
        # We need [num_layers, B, H] for the unidirectional decoder.

        # Reshape to [num_layers, 2, B, H] then concat directions → [num_layers, B, 2H]
        batch_size = src.size(0)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)

        # Concatenate forward (idx 0) and backward (idx 1) for each layer
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        # hidden, cell: [num_layers, B, 2*hidden_size]

        # Project down to decoder hidden size
        hidden = torch.tanh(self.fc_hidden(hidden))  # [num_layers, B, hidden_size]
        cell = torch.tanh(self.fc_cell(cell))         # [num_layers, B, hidden_size]

        return encoder_outputs, hidden, cell
