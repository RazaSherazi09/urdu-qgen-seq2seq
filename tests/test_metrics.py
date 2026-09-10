"""Tests for evaluation metrics."""
import pytest
import math

from src.evaluation.metrics import compute_bleu, compute_rouge_l, compute_perplexity, compute_unk_rate
from configs.config import UNK_ID


class TestBLEU:
    def test_perfect_match(self):
        hyps = ["یہ ایک سوال ہے"]
        refs = ["یہ ایک سوال ہے"]
        score = compute_bleu(hyps, refs)
        assert score > 50  # Should be high for exact match

    def test_no_overlap(self):
        hyps = ["الف ب پ"]
        refs = ["ت ث ج چ"]
        score = compute_bleu(hyps, refs)
        assert score < 5  # Should be very low

    def test_empty_hypothesis(self):
        hyps = [""]
        refs = ["کچھ متن"]
        score = compute_bleu(hyps, refs)
        assert score == 0.0


class TestROUGEL:
    def test_perfect_match(self):
        hyps = ["یہ ایک سوال ہے"]
        refs = ["یہ ایک سوال ہے"]
        score = compute_rouge_l(hyps, refs)
        assert score > 0.99

    def test_empty_hypothesis(self):
        hyps = [""]
        refs = ["کچھ متن"]
        score = compute_rouge_l(hyps, refs)
        assert score == 0.0


class TestPerplexity:
    def test_known_value(self):
        ce = 2.0
        ppl = compute_perplexity(ce)
        assert abs(ppl - math.exp(2.0)) < 0.01

    def test_zero_ce(self):
        assert compute_perplexity(0.0) == pytest.approx(1.0)


class TestUnkRate:
    def test_no_unks(self):
        ids = [[4, 5, 6], [7, 8]]
        assert compute_unk_rate(ids) == 0.0

    def test_all_unks(self):
        ids = [[UNK_ID, UNK_ID]]
        assert compute_unk_rate(ids) == 1.0

    def test_empty_input(self):
        assert compute_unk_rate([]) == 0.0

    def test_mixed(self):
        ids = [[UNK_ID, 5, 6, UNK_ID]]  # 2 out of 4
        assert compute_unk_rate(ids) == 0.5
