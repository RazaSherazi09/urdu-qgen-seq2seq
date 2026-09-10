"""
Unidirectional LSTM Decoder with Luong attention.

Architecture (per assignment spec):
    - 2-layer unidirectional LSTM
    - embedding_dim = 256
    - hidden_size = 512
    - dropout = 0.3

At every generation step the decoder:
    1. Embeds the previous target token → [B, embedding_dim]
    2. Runs one LSTM step → decoder_output [B, hidden_size]
    3. Computes Luong attention over encoder outputs → context [B, enc_hidden]
    4. Concatenates decoder_output and context → [B, hidden_size + enc_hidden]
    5. Projects through a linear layer → [B, hidden_size]  (the "attentional hidden state")
    6. Projects to vocabulary logits → [B, vocab_size]

The decoder returns vocabulary logits and attention weights so that
attention heatmaps can be visualized later.
"""

import torch
import torch.nn as nn

from src.model.attention import LuongAttention


class Decoder(nn.Module):
    """2-layer unidirectional LSTM decoder with Luong attention.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size.
    embedding_dim : int
        Token embedding dimensionality (256).
    hidden_size : int
        Decoder LSTM hidden size (512).
    encoder_hidden_size : int
        Width of encoder outputs (2*encoder_hidden for bidirectional = 1024).
    num_layers : int
        Number of LSTM layers (2).
    dropout : float
        Dropout probability (0.3).
    pad_id : int
        Padding token ID.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        encoder_hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_id: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = LuongAttention(encoder_hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Combine decoder output and attention context → attentional hidden state
        self.concat_linear = nn.Linear(
            hidden_size + encoder_hidden_size, hidden_size
        )

        # Project attentional hidden state → vocabulary logits
        self.output_linear = nn.Linear(hidden_size, vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple:
        """Run one decoding step.

        Parameters
        ----------
        input_token : LongTensor [B]
            Previous token (teacher-forced or predicted).
        hidden : Tensor [num_layers, B, hidden_size]
            LSTM hidden state.
        cell : Tensor [num_layers, B, hidden_size]
            LSTM cell state.
        encoder_outputs : Tensor [B, S, encoder_hidden_size]
            All encoder outputs for attention.
        src_mask : BoolTensor [B, S]
            True at PAD positions.

        Returns
        -------
        logits : Tensor [B, vocab_size]
            Un-normalised vocabulary scores.
        hidden : Tensor [num_layers, B, hidden_size]
            Updated hidden state.
        cell : Tensor [num_layers, B, hidden_size]
            Updated cell state.
        attn_weights : Tensor [B, S]
            Attention distribution for this step.
        """
        # Step 1: embed [B] → [B, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))

        # Step 2: LSTM step
        # lstm_out: [B, 1, hidden_size]
        lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        lstm_out = lstm_out.squeeze(1)  # [B, hidden_size]

        # Step 3: attention
        # Use the top-layer hidden state for attention scoring
        context, attn_weights = self.attention(
            lstm_out, encoder_outputs, src_mask
        )

        # Step 4: combine decoder output and context
        # concat: [B, hidden_size + encoder_hidden_size]
        concat = torch.cat([lstm_out, context], dim=1)

        # Step 5: attentional hidden state [B, hidden_size]
        attentional = torch.tanh(self.concat_linear(concat))
        attentional = self.dropout(attentional)

        # Step 6: vocabulary logits [B, vocab_size]
        logits = self.output_linear(attentional)

        return logits, hidden, cell, attn_weights
