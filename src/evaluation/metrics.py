"""
Automatic evaluation metrics for Urdu question generation.

Metrics:
    - BLEU-4   via sacrebleu (corpus-level)
    - ROUGE-L  via rouge_score (F-measure, no stemmer — this is Urdu)
    - Perplexity = exp(mean cross-entropy) from teacher-forced evaluation
    - <unk> rate = fraction of generated token IDs that are UNK
"""

import math
from typing import List

import sacrebleu
from rouge_score import rouge_scorer

from configs.config import UNK_ID


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Corpus-level BLEU-4 using sacrebleu.

    Parameters
    ----------
    hypotheses : List[str]
        System-generated questions (detokenized text).
    references : List[str]
        Reference questions (detokenized text).

    Returns
    -------
    float
        BLEU-4 score.

    Notes
    -----
    sacrebleu expects references as a list-of-lists (one list per
    reference set). Since we have a single reference per example,
    we wrap the entire references list in another list.
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def compute_rouge_l(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Mean ROUGE-L F-measure.

    We disable the stemmer because English stemming rules should not
    be applied to Urdu text. This is documented in the assignment.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    f_scores = []
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        f_scores.append(scores["rougeL"].fmeasure)
    return sum(f_scores) / max(len(f_scores), 1)


def compute_perplexity(mean_cross_entropy: float) -> float:
    """Compute perplexity from the mean cross-entropy loss.

    Perplexity = exp(mean_CE).

    This is a teacher-forced metric computed over the reference targets —
    it does not depend on greedy vs beam decoding. The *mean_cross_entropy*
    should come from the validation loop with ignore_index=PAD_ID.
    """
    return math.exp(mean_cross_entropy)


def compute_unk_rate(
    all_token_ids: List[List[int]],
) -> float:
    """Fraction of generated token IDs that are <unk>.

    Computed from raw integer IDs *before* detokenisation to ensure we
    correctly identify the SentencePiece UNK token regardless of how
    it is rendered in text.
    """
    total = 0
    unks = 0
    for ids in all_token_ids:
        for tok in ids:
            total += 1
            if tok == UNK_ID:
                unks += 1
    return unks / max(total, 1)
