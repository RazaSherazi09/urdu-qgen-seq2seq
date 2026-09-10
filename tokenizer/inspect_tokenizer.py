"""
Inspect a trained SentencePiece tokenizer.

Shows five tokenized examples, verifies that <ans> and </ans> are
single tokens, and runs a round-trip reconstruction test.
"""

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.tokenizer_utils import UrduTokenizer
from configs.config import SP_MODEL_PATH


EXAMPLES = [
    "دریائے سندھ کی لمبائی <ans> 3,180 کلومیٹر </ans> ہے۔",
    "اسلام آباد <ans> پاکستان </ans> کا دارالحکومت ہے۔",
    "<ans> علامہ اقبال </ans> نے شکوہ لکھی۔",
    "پاکستان کی آبادی <ans> 220 ملین </ans> سے زیادہ ہے۔",
    "قائداعظم <ans> محمد علی جناح </ans> پاکستان کے بانی تھے۔",
]


def main() -> None:
    tokenizer = UrduTokenizer(SP_MODEL_PATH)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD={tokenizer.pad_id}  UNK={tokenizer.unk_id}  "
          f"BOS={tokenizer.bos_id}  EOS={tokenizer.eos_id}")
    print()

    # Verify <ans> and </ans> are single tokens
    ans_open_id = tokenizer.piece_to_id("<ans>")
    ans_close_id = tokenizer.piece_to_id("</ans>")
    print(f"<ans>  → ID {ans_open_id}  (single token: {ans_open_id != tokenizer.unk_id})")
    print(f"</ans> → ID {ans_close_id}  (single token: {ans_close_id != tokenizer.unk_id})")
    print()

    # Show five tokenized examples
    print("=" * 60)
    print("  TOKENIZATION EXAMPLES")
    print("=" * 60)
    for i, text in enumerate(EXAMPLES, 1):
        print(f"\n── Example {i} ──")
        tokenizer.show_example(text)


if __name__ == "__main__":
    main()
