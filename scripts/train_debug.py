"""
Debug training on a 10k-example subset.

This script verifies the entire pipeline before committing to a full
GPU training run:
    1. Data loads correctly
    2. Tokenizer encodes/decodes
    3. Batches have correct tensor shapes
    4. Forward pass succeeds
    5. Loss is finite
    6. Gradients are finite
    7. Optimizer step works
    8. Loss decreases over 2–3 epochs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from configs.config import (
    BATCH_SIZE,
    BIDIRECTIONAL,
    CHECKPOINT_DIR,
    DEBUG_EPOCHS,
    DEBUG_SUBSET_SIZE,
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_SIZE,
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
    print(f"[Debug] Vocab size: {vocab_size}")

    # ── Data ──────────────────────────────────────────────────────
    train_dataset = QGenDataset(TRAIN_FILE, tokenizer)
    valid_dataset = QGenDataset(VALID_FILE, tokenizer)

    # Subset for debug
    n_train = min(DEBUG_SUBSET_SIZE, len(train_dataset))
    n_valid = min(DEBUG_SUBSET_SIZE // 5, len(valid_dataset))
    train_sub = Subset(train_dataset, list(range(n_train)))
    valid_sub = Subset(valid_dataset, list(range(n_valid)))

    train_loader = DataLoader(
        train_sub, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_sub, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    print(f"[Debug] Train subset: {n_train}  Valid subset: {n_valid}")

    # ── Quick shape check ─────────────────────────────────────────
    batch = next(iter(train_loader))
    print(f"[Debug] Batch shapes:")
    print(f"  src:         {batch['src'].shape}")
    print(f"  tgt:         {batch['tgt'].shape}")
    print(f"  src_lengths: {batch['src_lengths'].shape}")
    print(f"  src_mask:    {batch['src_mask'].shape}")

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

    print(f"[Debug] Trainable parameters: {count_parameters(model):,}")

    # ── Quick forward pass check ──────────────────────────────────
    model.train()
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)
    src_lengths = batch["src_lengths"]
    src_mask = batch["src_mask"].to(device)

    outputs, attentions = model(src, src_lengths, src_mask, tgt)
    print(f"[Debug] Forward output shape: {outputs.shape}")
    print(f"[Debug] Attention shape: {attentions.shape}")

    assert torch.isfinite(outputs).all(), "Non-finite values in forward output!"
    print("[Debug] ✓ Forward pass OK — outputs are finite")

    # ── Train ─────────────────────────────────────────────────────
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "bidirectional": BIDIRECTIONAL,
    }

    debug_checkpoint = CHECKPOINT_DIR / "debug_model.pt"
    logs = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=DEBUG_EPOCHS,
        checkpoint_path=debug_checkpoint,
        config=config,
    )

    # ── Verify loss decreased ─────────────────────────────────────
    first_loss = logs[0]["train_loss"]
    last_loss = logs[-1]["train_loss"]
    print(f"\n[Debug] First epoch loss: {first_loss:.4f}")
    print(f"[Debug] Last epoch loss:  {last_loss:.4f}")

    if last_loss < first_loss:
        print("[Debug] ✓ Loss decreased — model is learning")
    else:
        print("[Debug] ⚠ Loss did NOT decrease — investigate implementation")


if __name__ == "__main__":
    main()
