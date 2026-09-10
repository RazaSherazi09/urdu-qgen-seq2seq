"""
Evaluate the best checkpoint on UQA-valid and Wiki-UQA.

Generates greedy + beam outputs, computes all metrics, saves results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    BEAM_WIDTH,
    BEST_MODEL_PATH,
    BIDIRECTIONAL,
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PAD_ID,
    RESULTS_DIR,
    SEED,
    SP_MODEL_PATH,
    VALID_FILE,
    WIKI_VALID_FILE,
)
from src.data.dataset import QGenDataset, collate_fn
from src.evaluation.evaluate import (
    evaluate_model,
    save_automatic_metrics,
    save_samples,
)
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq
from src.tokenizer.tokenizer_utils import UrduTokenizer
from src.training.train import validate_epoch
from src.training.utils import get_device, load_checkpoint, set_seed
import torch.nn as nn


def main() -> None:
    set_seed(SEED)
    device = get_device()

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = UrduTokenizer(SP_MODEL_PATH)
    vocab_size = tokenizer.vocab_size

    # ── Model ─────────────────────────────────────────────────────
    enc_hidden = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_id=PAD_ID,
    )
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        encoder_hidden_size=enc_hidden,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_id=PAD_ID,
    )
    model = Seq2Seq(encoder, decoder).to(device)
    load_checkpoint(BEST_MODEL_PATH, model, device=device)

    # ── Datasets ──────────────────────────────────────────────────
    valid_dataset = QGenDataset(VALID_FILE, tokenizer)
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
    )

    # Compute validation loss for perplexity
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    val_loss = validate_epoch(model, valid_loader, criterion, device)
    print(f"[Eval] UQA validation loss: {val_loss:.4f}")

    all_metrics = []

    # ── UQA Validation ────────────────────────────────────────────
    uqa_metrics, uqa_gen = evaluate_model(
        model, tokenizer, valid_dataset, device, val_loss,
        dataset_label="UQA-valid", beam_width=BEAM_WIDTH,
    )
    all_metrics.append(uqa_metrics["greedy"])
    all_metrics.append(uqa_metrics["beam"])

    # Save samples from UQA validation
    save_samples(uqa_gen, RESULTS_DIR / "samples.tsv")

    # ── Wiki-UQA ──────────────────────────────────────────────────
    wiki_valid_path = Path(WIKI_VALID_FILE)
    if wiki_valid_path.exists():
        wiki_dataset = QGenDataset(wiki_valid_path, tokenizer)
        wiki_loader = DataLoader(
            wiki_dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn,
        )
        wiki_loss = validate_epoch(model, wiki_loader, criterion, device)
        print(f"[Eval] Wiki-UQA loss: {wiki_loss:.4f}")

        wiki_metrics, _ = evaluate_model(
            model, tokenizer, wiki_dataset, device, wiki_loss,
            dataset_label="Wiki-UQA", beam_width=BEAM_WIDTH,
        )
        all_metrics.append(wiki_metrics["greedy"])
        all_metrics.append(wiki_metrics["beam"])
    else:
        print(f"[Eval] Wiki-UQA file not found ({wiki_valid_path}), skipping.")

    # ── Save ──────────────────────────────────────────────────────
    save_automatic_metrics(all_metrics, RESULTS_DIR / "automatic_metrics.csv")

    print("\n[Eval] Done! Results saved to results/")


if __name__ == "__main__":
    main()
