"""
Beam search decoding for the seq2seq model.

Beam search maintains the top-k (beam_width) most probable hypotheses
at each step and expands them in parallel. This finds better sequences
than greedy decoding at the cost of higher computation.

We use **log probabilities** throughout to avoid numerical underflow
that would happen with multiplying many small probabilities.

Length normalisation divides each hypothesis score by its length so
that shorter sequences are not unfairly favoured.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F

from configs.config import BOS_ID, EOS_ID, MAX_DECODE_LENGTH, UNK_ID


@dataclass
class Hypothesis:
    """A single beam-search hypothesis."""
    tokens: List[int] = field(default_factory=list)
    log_prob: float = 0.0
    hidden: torch.Tensor = None  # type: ignore[assignment]
    cell: torch.Tensor = None    # type: ignore[assignment]
    finished: bool = False

    @property
    def score(self) -> float:
        """Length-normalised score.

        Dividing by length prevents the search from preferring very short
        sequences (which accumulate fewer negative log-prob terms).
        We add 1 to avoid division by zero for empty hypotheses.
        """
        return self.log_prob / max(len(self.tokens), 1)


@torch.no_grad()
def beam_search_decode(
    model,
    src_ids: torch.Tensor,
    src_length: torch.Tensor,
    src_mask: torch.Tensor,
    beam_width: int = 5,
    max_length: int = MAX_DECODE_LENGTH,
) -> Tuple[List[int], float]:
    """Generate a sequence using beam search.

    Parameters
    ----------
    model : Seq2Seq
        The trained model in eval mode.
    src_ids : LongTensor [1, S]
    src_length : LongTensor [1]
    src_mask : BoolTensor [1, S]
    beam_width : int
        Number of hypotheses to keep at each step (default 5).
    max_length : int
        Maximum generation length.

    Returns
    -------
    best_tokens : List[int]
        Token IDs of the best completed hypothesis (excl. BOS, incl. EOS).
    best_score : float
        Length-normalised log probability of the best hypothesis.
    """
    model.eval()
    device = src_ids.device

    # ── Encode (once) ─────────────────────────────────────────────
    encoder_outputs, hidden, cell = model.encoder(src_ids, src_length)
    # encoder_outputs: [1, S, enc_hidden]
    # hidden, cell: [num_layers, 1, dec_hidden]

    # ── Initialise beam with BOS ──────────────────────────────────
    initial = Hypothesis(
        tokens=[],
        log_prob=0.0,
        hidden=hidden,
        cell=cell,
        finished=False,
    )
    active_hypotheses: List[Hypothesis] = [initial]
    completed: List[Hypothesis] = []

    for step in range(max_length):
        if not active_hypotheses:
            break

        all_candidates: List[Hypothesis] = []

        for hyp in active_hypotheses:
            # Determine the input token for this hypothesis
            if not hyp.tokens:
                input_id = BOS_ID
            else:
                input_id = hyp.tokens[-1]

            input_token = torch.tensor([input_id], device=device)

            logits, new_hidden, new_cell, _ = model.decoder.forward_step(
                input_token, hyp.hidden, hyp.cell, encoder_outputs, src_mask
            )

            # Convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=1).squeeze(0)  # [vocab_size]

            # Take top-k candidates from this hypothesis
            topk_log_probs, topk_ids = log_probs.topk(beam_width)

            for i in range(beam_width):
                token_id = topk_ids[i].item()
                token_log_prob = topk_log_probs[i].item()

                new_hyp = Hypothesis(
                    tokens=hyp.tokens + [token_id],
                    log_prob=hyp.log_prob + token_log_prob,
                    hidden=new_hidden,
                    cell=new_cell,
                    finished=(token_id == EOS_ID),
                )
                all_candidates.append(new_hyp)

        # Separate completed and active hypotheses
        active_hypotheses = []
        for cand in all_candidates:
            if cand.finished:
                completed.append(cand)
            else:
                active_hypotheses.append(cand)

        # Keep only the top beam_width active hypotheses by score
        active_hypotheses.sort(key=lambda h: h.score, reverse=True)
        active_hypotheses = active_hypotheses[:beam_width]

        # Early stop: if we have enough completed hypotheses
        if len(completed) >= beam_width:
            break

    # ── Select best hypothesis ────────────────────────────────────
    # If no hypothesis finished, use the best active one
    if not completed:
        completed = active_hypotheses

    completed.sort(key=lambda h: h.score, reverse=True)
    best = completed[0]

    return best.tokens, best.score
