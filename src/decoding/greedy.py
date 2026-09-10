"""
Greedy decoding for the seq2seq model.

At inference we do NOT use teacher forcing: the decoder feeds its own
previous prediction back as input at each step.

Procedure:
    1. Encode the source sequence.
    2. Initialize the decoder with encoder's projected hidden/cell states.
    3. Feed BOS as the first token.
    4. At each step, take the argmax of the vocabulary logits.
    5. Feed the predicted token back as next input.
    6. Stop when EOS is produced or max_length is reached.
"""

from typing import List, Tuple

import torch

from configs.config import BOS_ID, EOS_ID, MAX_DECODE_LENGTH, UNK_ID


@torch.no_grad()
def greedy_decode(
    model,
    src_ids: torch.Tensor,
    src_length: torch.Tensor,
    src_mask: torch.Tensor,
    max_length: int = MAX_DECODE_LENGTH,
) -> Tuple[List[int], List[torch.Tensor]]:
    """Generate a sequence using greedy (argmax) decoding.

    Parameters
    ----------
    model : Seq2Seq
        The trained model in eval mode.
    src_ids : LongTensor [1, S]
        Source token IDs (single example, batched as size 1).
    src_length : LongTensor [1]
        Actual source length.
    src_mask : BoolTensor [1, S]
        Padding mask for source.
    max_length : int
        Maximum number of tokens to generate.

    Returns
    -------
    token_ids : List[int]
        Generated token IDs (excluding BOS, including EOS if produced).
    attention_weights : List[Tensor]
        Attention weight vectors [S] for each generated step.
    """
    model.eval()
    device = src_ids.device

    # Encode
    encoder_outputs, hidden, cell = model.encoder(src_ids, src_length)

    # Start with BOS
    input_token = torch.tensor([BOS_ID], device=device)  # [1]

    token_ids: List[int] = []
    attention_weights: List[torch.Tensor] = []

    for _ in range(max_length):
        logits, hidden, cell, attn = model.decoder.forward_step(
            input_token, hidden, cell, encoder_outputs, src_mask
        )

        # Greedy: take argmax
        predicted_id = logits.argmax(dim=1).item()

        token_ids.append(predicted_id)
        attention_weights.append(attn.squeeze(0).cpu())  # [S]

        if predicted_id == EOS_ID:
            break

        # Feed prediction back as next input
        input_token = torch.tensor([predicted_id], device=device)

    return token_ids, attention_weights


def count_unks(token_ids: List[int]) -> int:
    """Count <unk> tokens in a generated sequence."""
    return sum(1 for t in token_ids if t == UNK_ID)
