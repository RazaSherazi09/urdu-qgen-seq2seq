"""
Model evaluation pipeline.

Evaluates the best checkpoint on UQA validation and Wiki-UQA using both
greedy and beam decoding. Produces four metric rows and saves samples.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    BEAM_WIDTH,
    BEST_MODEL_PATH,
    BOS_ID,
    EOS_ID,
    MAX_DECODE_LENGTH,
    NUM_SAVED_SAMPLES,
    PAD_ID,
    RESULTS_DIR,
    SEED,
    SP_MODEL_PATH,
    TRAIN_FILE,
    VALID_FILE,
    WIKI_VALID_FILE,
)
from src.data.dataset import QGenDataset, collate_fn
from src.decoding.beam_search import beam_search_decode
from src.decoding.greedy import greedy_decode, count_unks
from src.evaluation.metrics import (
    compute_bleu,
    compute_perplexity,
    compute_rouge_l,
    compute_unk_rate,
)
from src.tokenizer.tokenizer_utils import UrduTokenizer
from src.training.train import validate_epoch
from src.training.utils import set_seed
import torch.nn as nn


def _generate_all(
    model,
    dataset: QGenDataset,
    tokenizer: UrduTokenizer,
    device: torch.device,
    beam_width: int = BEAM_WIDTH,
    max_length: int = MAX_DECODE_LENGTH,
) -> Dict[str, list]:
    """Generate greedy and beam outputs for every example in dataset.

    Returns dict with keys: sources, references, greedy_texts, beam_texts,
    greedy_ids, beam_ids.
    """
    model.eval()
    results = {
        "sources": [],
        "references": [],
        "greedy_texts": [],
        "beam_texts": [],
        "greedy_ids": [],
        "beam_ids": [],
    }

    for i in range(len(dataset)):
        pair = dataset.pairs[i]
        item = dataset[i]

        src_ids = torch.LongTensor([item["src_ids"]]).to(device)
        src_len = torch.LongTensor([item["src_length"]])
        src_mask = (src_ids == PAD_ID)

        # Greedy
        g_ids, _ = greedy_decode(model, src_ids, src_len, src_mask, max_length)
        g_clean = [t for t in g_ids if t != EOS_ID]
        g_text = tokenizer.decode(g_clean)

        # Beam
        b_ids, _ = beam_search_decode(model, src_ids, src_len, src_mask, beam_width, max_length)
        b_clean = [t for t in b_ids if t != EOS_ID]
        b_text = tokenizer.decode(b_clean)

        results["sources"].append(pair["source"])
        results["references"].append(pair["target"])
        results["greedy_texts"].append(g_text)
        results["beam_texts"].append(b_text)
        results["greedy_ids"].append(g_ids)
        results["beam_ids"].append(b_ids)

    return results


def evaluate_model(
    model,
    tokenizer: UrduTokenizer,
    valid_dataset: QGenDataset,
    device: torch.device,
    val_loss: float,
    dataset_label: str = "UQA-valid",
    beam_width: int = BEAM_WIDTH,
) -> Dict[str, Dict]:
    """Evaluate a model on one dataset with both decoding methods.

    Returns a dict with keys "greedy" and "beam", each containing:
    bleu4, rouge_l, unk_rate. Perplexity is shared (teacher-forced).
    """
    gen = _generate_all(model, valid_dataset, tokenizer, device, beam_width)

    ppl = compute_perplexity(val_loss)

    metrics = {}
    for method, texts_key, ids_key in [
        ("greedy", "greedy_texts", "greedy_ids"),
        ("beam", "beam_texts", "beam_ids"),
    ]:
        bleu = compute_bleu(gen[texts_key], gen["references"])
        rouge = compute_rouge_l(gen[texts_key], gen["references"])
        unk = compute_unk_rate(gen[ids_key])

        metrics[method] = {
            "dataset": dataset_label,
            "decoding": method,
            "bleu4": round(bleu, 2),
            "rouge_l": round(rouge, 4),
            "perplexity": round(ppl, 2),
            "unk_rate": round(unk, 4),
        }

    return metrics, gen


def save_samples(
    gen_results: Dict,
    output_path: Path,
    n_samples: int = NUM_SAVED_SAMPLES,
    seed: int = SEED,
) -> None:
    """Save a fixed random subset of samples to TSV."""
    import random
    rng = random.Random(seed)

    total = len(gen_results["sources"])
    indices = list(range(total))
    rng.shuffle(indices)
    selected = sorted(indices[:min(n_samples, total)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "reference", "greedy", "beam"])
        for idx in selected:
            writer.writerow([
                gen_results["sources"][idx],
                gen_results["references"][idx],
                gen_results["greedy_texts"][idx],
                gen_results["beam_texts"][idx],
            ])

    print(f"[Eval] Saved {len(selected)} samples → {output_path}")


def save_automatic_metrics(
    all_metrics: List[Dict],
    output_path: Path,
) -> None:
    """Save the automatic metrics table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "decoding", "bleu4", "rouge_l", "perplexity", "unk_rate"],
        )
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)
    print(f"[Eval] Metrics saved → {output_path}")
