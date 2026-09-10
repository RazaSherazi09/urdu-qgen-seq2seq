"""
Streamlit frontend for Urdu Question Generation.

Usage:
    streamlit run frontend/app.py

The app loads the SentencePiece tokenizer and model checkpoint once
at startup (cached), then lets the user enter an Urdu sentence and
an answer span to generate questions via greedy and beam decoding.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import BEAM_WIDTH, BEST_MODEL_PATH, SP_MODEL_PATH


# ── Model Loading (cached — runs once) ────────────────────────────

@st.cache_resource
def load_generator():
    """Load the QuestionGenerator once and cache it across reruns."""
    from src.decoding.inference import QuestionGenerator
    try:
        generator = QuestionGenerator(
            model_path=BEST_MODEL_PATH,
            sp_model_path=SP_MODEL_PATH,
        )
        return generator
    except FileNotFoundError as e:
        st.error(str(e))
        return None


def mark_answer(sentence: str, answer: str) -> str:
    """Insert <ans> and </ans> markers around the answer in the sentence.

    If the answer appears multiple times, marks the first occurrence and
    displays a warning. Returns empty string if answer is not found.
    """
    if answer not in sentence:
        return ""

    count = sentence.count(answer)
    if count > 1:
        st.warning(
            f"The answer '{answer}' appears {count} times in the sentence. "
            "Marking the **first** occurrence."
        )

    idx = sentence.index(answer)
    before = sentence[:idx]
    after = sentence[idx + len(answer):]
    return f"{before}<ans> {answer} </ans>{after}"


# ── Streamlit App ─────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Urdu Question Generator",
        page_icon="❓",
        layout="centered",
    )

    st.title("❓ Urdu Question Generator")
    st.markdown(
        "Enter an Urdu sentence and highlight the answer span. "
        "The model generates questions using **greedy** and **beam search** decoding."
    )

    st.divider()

    # Input fields
    sentence = st.text_area(
        "Urdu Sentence",
        placeholder="دریائے سندھ کی لمبائی 3,180 کلومیٹر ہے۔",
        height=100,
    )

    answer = st.text_input(
        "Answer Span",
        placeholder="3,180 کلومیٹر",
    )

    beam_width = st.slider(
        "Beam Width", min_value=3, max_value=5, value=BEAM_WIDTH
    )

    generate_btn = st.button("🔍 Generate Question", type="primary", use_container_width=True)

    st.divider()

    if generate_btn:
        # ── Validation ────────────────────────────────────────────
        if not sentence.strip():
            st.error("Please enter an Urdu sentence.")
            return
        if not answer.strip():
            st.error("Please enter the answer span.")
            return

        marked = mark_answer(sentence.strip(), answer.strip())
        if not marked:
            st.error(
                f"Answer **'{answer}'** was not found in the sentence. "
                "Please check that the answer text exactly matches a substring."
            )
            return

        st.markdown("**Marked Source:**")
        st.code(marked, language=None)

        # ── Load model ────────────────────────────────────────────
        generator = load_generator()
        if generator is None:
            return

        # ── Generate ──────────────────────────────────────────────
        with st.spinner("Generating questions…"):
            greedy_q = generator.generate_greedy(marked)
            beam_q = generator.generate_beam(marked, beam_width=beam_width)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Greedy")
            st.success(greedy_q if greedy_q else "_(empty output)_")

        with col2:
            st.subheader(f"Beam (k={beam_width})")
            st.success(beam_q if beam_q else "_(empty output)_")


if __name__ == "__main__":
    main()
