"""
Visualization helpers for the Urdu question generation project.

Generates:
    1. Training vs validation loss curve
    2. Source-length histogram
    3. Target-length histogram
    4. Attention heatmap (source × generated tokens)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from configs.config import CHECKPOINT_DIR, FIGURES_DIR

# Use a non-interactive backend when running headless (e.g. Colab/servers)
matplotlib.use("Agg")

# Make plots look clean
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_loss_curve(
    log_path: Path = CHECKPOINT_DIR / "training_log.json",
    save_path: Path = FIGURES_DIR / "loss_curve.png",
) -> None:
    """Plot training vs validation loss per epoch."""
    with open(log_path) as f:
        logs = json.load(f)

    epochs = [e["epoch"] for e in logs]
    train_losses = [e["train_loss"] for e in logs]
    val_losses = [e["val_loss"] for e in logs]

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, "o-", label="Train Loss")
    ax.plot(epochs, val_losses, "s-", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Loss curve → {save_path}")


def plot_length_histogram(
    lengths: List[int],
    title: str,
    xlabel: str,
    save_path: Path,
    bins: int = 30,
) -> None:
    """Plot a length distribution histogram."""
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Add mean line
    mean_val = np.mean(lengths)
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.1f}")
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Histogram → {save_path}")


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    source_tokens: List[str],
    generated_tokens: List[str],
    save_path: Path = FIGURES_DIR / "attention_heatmap.png",
    title: str = "Attention Heatmap",
) -> None:
    """Plot an attention heatmap for a single example.

    Parameters
    ----------
    attention_matrix : ndarray [gen_len, src_len]
        Attention weights. Row i is the attention distribution when
        generating token i.
    source_tokens : List[str]
        Subword tokens on the Y axis (source sequence).
    generated_tokens : List[str]
        Subword tokens on the X axis (generated sequence).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Transpose so that source is on the Y axis and generated on X
    im = ax.imshow(attention_matrix.T, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(generated_tokens)))
    ax.set_xticklabels(generated_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(source_tokens)))
    ax.set_yticklabels(source_tokens, fontsize=8)

    ax.set_xlabel("Generated Tokens")
    ax.set_ylabel("Source Tokens")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Attention heatmap → {save_path}")
