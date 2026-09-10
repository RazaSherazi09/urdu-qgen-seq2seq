"""
Full training on the complete UQA training set.

Intended to be run on Colab/Kaggle with a GPU runtime.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    BIDIRECTIONAL,
    DROPOUT,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    PAD_ID,
    SEED,
    SP_MODEL_PATH,
    TRAIN_FILE,
    VALID_FILE,
)
from src.data.dataset import QGenDataset, collate_fn
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq
from src.tokenizer.tokenizer_utils import UrduTokenizer
from src.training.train import train_model
from src.training.utils import count_parameters, get_device, set_seed


def main() -> None:
    set_seed(SEED)
    device = get_device()

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = UrduTokenizer(SP_MODEL_PATH)
    vocab_size = tokenizer.vocab_size
    print(f"[Train] Vocab size: {vocab_size}")

    # ── Data ──────────────────────────────────────────────────────
    train_dataset = QGenDataset(TRAIN_FILE, tokenizer)
    valid_dataset = QGenDataset(VALID_FILE, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    print(f"[Train] Training examples:   {len(train_dataset):,}")
    print(f"[Train] Validation examples:  {len(valid_dataset):,}")

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

    print(f"[Train] Trainable parameters: {count_parameters(model):,}")

    # ── Train ─────────────────────────────────────────────────────
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "bidirectional": BIDIRECTIONAL,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED,
    }

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_path=BEST_MODEL_PATH,
        config=config,
    )


if __name__ == "__main__":
    main()
