"""
Training loop for the Urdu question generation seq2seq model.

Key design choices:
    - Adam optimizer with lr = 3e-4
    - CrossEntropyLoss with ignore_index = PAD_ID (padding never counted)
    - Gradient clipping (max norm 1.0) for LSTM stability
    - Best checkpoint saved only when validation loss improves
    - Epoch-level logging of train/val loss, LR, and elapsed time
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    EPOCHS,
    GRAD_CLIP,
    LEARNING_RATE,
    PAD_ID,
    SEED,
    TEACHER_FORCING_RATIO,
)
from src.training.utils import Timer, count_parameters, save_checkpoint


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    device: torch.device,
    teacher_forcing_ratio: float = TEACHER_FORCING_RATIO,
) -> float:
    """Train for one epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        src = batch["src"].to(device)              # [B, S]
        tgt = batch["tgt"].to(device)              # [B, T]
        src_lengths = batch["src_lengths"]          # [B] — stays on CPU for packing
        src_mask = batch["src_mask"].to(device)     # [B, S]

        optimizer.zero_grad()

        # outputs: [B, T-1, vocab_size]
        outputs, _ = model(src, src_lengths, src_mask, tgt, teacher_forcing_ratio)

        # Reshape for CrossEntropyLoss:
        #   predictions: [(B*(T-1)), vocab_size]
        #   targets:     [(B*(T-1))]
        # The target is tgt[:, 1:] because position 0 is BOS (given as input).
        output_dim = outputs.size(-1)
        outputs_flat = outputs.reshape(-1, output_dim)
        targets_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(outputs_flat, targets_flat)
        loss.backward()

        # Gradient clipping prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate on the validation set and return mean loss.

    Uses torch.no_grad() to save memory and avoid building the
    computation graph that would be needed for backpropagation.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_lengths = batch["src_lengths"]
            src_mask = batch["src_mask"].to(device)

            # Full teacher forcing during validation — we want to
            # measure the model's loss on the correct target sequence.
            outputs, _ = model(src, src_lengths, src_mask, tgt, teacher_forcing_ratio=1.0)

            output_dim = outputs.size(-1)
            outputs_flat = outputs.reshape(-1, output_dim)
            targets_flat = tgt[:, 1:].reshape(-1)

            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    grad_clip: float = GRAD_CLIP,
    teacher_forcing_ratio: float = TEACHER_FORCING_RATIO,
    checkpoint_path: Path = BEST_MODEL_PATH,
    seed: int = SEED,
    config: Optional[Dict] = None,
) -> List[Dict]:
    """Full training loop with validation and best-checkpoint saving.

    Parameters
    ----------
    model : nn.Module
        The Seq2Seq model, already on *device*.
    train_loader, valid_loader : DataLoader
        Data loaders with the custom collate_fn.
    device : torch.device
    epochs : int
    learning_rate : float
    grad_clip : float
    teacher_forcing_ratio : float
    checkpoint_path : Path
        Where to save the best checkpoint.
    seed : int
    config : dict, optional
        Configuration dict to store inside the checkpoint.

    Returns
    -------
    logs : List[Dict]
        One dict per epoch with keys:
        epoch, train_loss, val_loss, lr, elapsed, best.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val_loss = float("inf")
    logs: List[Dict] = []
    timer = Timer()

    print(f"\n{'=' * 60}")
    print(f"  Training — {epochs} epochs, lr={learning_rate}, "
          f"batch={train_loader.batch_size}")
    print(f"  Trainable parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    total_timer = Timer()
    total_timer.start()

    for epoch in range(1, epochs + 1):
        timer.start()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            grad_clip, device, teacher_forcing_ratio,
        )
        val_loss = validate_epoch(model, valid_loader, criterion, device)

        elapsed = timer.elapsed_str()
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path, model, optimizer,
                epoch, best_val_loss, config or {}, seed,
            )

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": learning_rate,
            "elapsed": elapsed,
            "best": is_best,
        }
        logs.append(log_entry)

        marker = " ★" if is_best else ""
        print(
            f"  Epoch {epoch:>2}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"time={elapsed}{marker}"
        )

    total_time = total_timer.elapsed_str()
    best_epoch = min(logs, key=lambda x: x["val_loss"])["epoch"]

    print(f"\n{'=' * 60}")
    print(f"  Training complete")
    print(f"  Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f})")
    print(f"  Total time: {total_time}")
    print(f"{'=' * 60}\n")

    # Save training logs as JSON for later plotting
    log_path = CHECKPOINT_DIR / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"[Train] Logs saved → {log_path}")

    return logs
