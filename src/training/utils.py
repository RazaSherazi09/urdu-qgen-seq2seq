"""
Training utilities: reproducibility, device selection, checkpoint I/O.
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Note: exact bitwise reproducibility on GPU is not always guaranteed
    because some CUDA operations are non-deterministic by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Encourage deterministic behaviour where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select the best available device and print diagnostic info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available — using CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
    seed: int,
) -> None:
    """Save a training checkpoint with everything needed to resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "seed": seed,
        },
        path,
    )
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint. Returns the metadata dict.

    If *optimizer* is provided its state is restored too.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. "
            "Train the model first or download the checkpoint from the "
            "Hugging Face Model repository."
        )

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        f"[Checkpoint] Loaded from {path}  "
        f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['best_val_loss']:.4f})"
    )
    return checkpoint


class Timer:
    """Simple wall-clock timer for epoch timing."""

    def __init__(self) -> None:
        self._start: float = 0.0

    def start(self) -> None:
        self._start = time.time()

    def elapsed(self) -> float:
        """Return elapsed seconds since last start()."""
        return time.time() - self._start

    def elapsed_str(self) -> str:
        """Return elapsed time as 'Xm Ys'."""
        secs = self.elapsed()
        mins = int(secs // 60)
        secs = secs % 60
        return f"{mins}m {secs:.1f}s"
