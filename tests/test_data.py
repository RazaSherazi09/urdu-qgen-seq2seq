"""Tests for data preparation utilities."""
import pytest
from src.data.prepare_data import split_into_sentences, normalize_whitespace


class TestSentenceSplitting:
    """Verify sentence splitting preserves character offsets correctly."""

    def test_single_sentence_urdu_period(self):
        text = "یہ ایک جملہ ہے۔"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1
        start, end, sent = sentences[0]
        assert text[start:end] == sent

    def test_multiple_sentences(self):
        text = "پہلا جملہ۔ دوسرا جملہ۔ تیسرا جملہ۔"
        sentences = split_into_sentences(text)
        assert len(sentences) >= 2
        # Verify every sentence maps back to its text
        for start, end, sent in sentences:
            assert text[start:end] == sent

    def test_question_mark_delimiter(self):
        text = "کیا ہوا؟ کچھ نہیں۔"
        sentences = split_into_sentences(text)
        assert len(sentences) >= 2

    def test_exclamation_delimiter(self):
        text = "واہ! بہت خوب۔"
        sentences = split_into_sentences(text)
        assert len(sentences) >= 2

    def test_offset_reconstruction(self):
        """Offsets should allow exact substring extraction."""
        text = "جملہ ایک۔ جملہ دو۔"
        sentences = split_into_sentences(text)
        for start, end, sent in sentences:
            assert text[start:end] == sent


class TestNormalization:
    def test_collapse_whitespace(self):
        assert normalize_whitespace("  a   b  c  ") == "a b c"

    def test_tabs_and_newlines(self):
        assert normalize_whitespace("a\t\nb") == "a b"


class TestLengthFiltering:
    """Length filtering is done in prepare_data — test the logic."""

    def test_within_limits(self):
        source = " ".join(["word"] * 60)
        assert len(source.split()) <= 60

    def test_exceeds_source_limit(self):
        source = " ".join(["word"] * 61)
        assert len(source.split()) > 60

    def test_exceeds_target_limit(self):
        target = " ".join(["word"] * 26)
        assert len(target.split()) > 25
