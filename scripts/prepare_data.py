"""Prepare UQA and Wiki-UQA data — CLI entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prepare_data import prepare_uqa_data, prepare_wiki_uqa_data

if __name__ == "__main__":
    prepare_uqa_data()
    prepare_wiki_uqa_data()
