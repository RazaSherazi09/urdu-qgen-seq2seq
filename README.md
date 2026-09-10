# Urdu Question Generation — From-Scratch Seq2Seq

A sentence-level Urdu question generation system built from scratch using a
bidirectional LSTM encoder–decoder with Luong attention. No pretrained models,
embeddings, or Transformer components are used.

## Problem

Given an Urdu sentence with a marked answer span:

```
دریائے سندھ کی لمبائی <ans> 3,180 کلومیٹر </ans> ہے۔
```

Generate a natural Urdu question whose answer is the marked span:

```
دریائے سندھ کی لمبائی کتنی ہے؟
```

## Architecture

```
Source sentence + <ans> markers
        ↓
┌─────────────────────┐
│  Embedding (256)    │
│  Bidirectional LSTM │  2 layers, hidden 512
│  (Encoder)          │
└─────────┬───────────┘
          │ encoder outputs [B, S, 1024]
          │ projected hidden/cell [2, B, 512]
          ↓
┌─────────────────────┐
│  Embedding (256)    │
│  Unidirectional LSTM│  2 layers, hidden 512
│  Luong Attention    │
│  (Decoder)          │
└─────────┬───────────┘
          │ vocab logits [B, 8000]
          ↓
    Generated question
```

## Dataset

- **UQA** — Urdu Question Answering dataset (train + validation)
- **Wiki-UQA** — Out-of-domain evaluation set

## Data Preparation

1. Load UQA from Hugging Face (`uqa/UQA`)
2. Filter answerable rows
3. Split context into sentences using Urdu delimiters (۔ ؟ !)
4. Locate sentence containing the answer via character offsets
5. Verify offset alignment
6. Insert `<ans>` / `</ans>` markers
7. Apply length filtering (source ≤ 60 words, target ≤ 25 words)
8. Save `train.tsv` and `valid.tsv`

## Tokenizer

- **SentencePiece** unigram model
- Vocabulary: 8,000
- Character coverage: 1.0
- User-defined symbols: `<ans>`, `</ans>`
- Special tokens: PAD=0, UNK=1, BOS=2, EOS=3
- Trained on **training data only** (no validation leakage)

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Encoder layers | 2 (bidirectional) |
| Decoder layers | 2 (unidirectional) |
| Embedding dim | 256 |
| Hidden size | 512 |
| Dropout | 0.3 |
| Attention | Luong (general) |
| Optimizer | Adam (lr=3e-4) |
| Batch size | 64 |
| Gradient clipping | 1.0 |
| Beam width | 5 |
| Parameter count | *Computed at runtime* |

## Training

- Teacher forcing (ratio 1.0)
- CrossEntropyLoss with `ignore_index=PAD`
- Best checkpoint selected by validation loss
- Debug on 10k subset first, then full training with GPU

## Evaluation

| Dataset | Decoding | BLEU-4 | ROUGE-L | Perplexity | UNK Rate |
|---------|----------|--------|---------|------------|----------|
| UQA-valid | Greedy | — | — | — | — |
| UQA-valid | Beam | — | — | — | — |
| Wiki-UQA | Greedy | — | — | — | — |
| Wiki-UQA | Beam | — | — | — | — |

> Results will be populated after the final training run.

## Frontend

Streamlit app (`frontend/app.py`):
- Enter Urdu sentence + answer span
- View greedy and beam search outputs

```bash
streamlit run frontend/app.py
```

## Hugging Face

- **Model Repository**: [INSERT LINK] — Contains `ur_sp.model`, `ur_sp.vocab`, `best_model.pt`, `config.json`
- **Space**: [INSERT LINK] — Live demo

## Repository Structure

```
urdu-qgen-seq2seq/
├── configs/config.py          # All hyperparameters
├── src/
│   ├── data/                  # Data preparation & PyTorch Dataset
│   ├── tokenizer/             # SentencePiece training & wrapper
│   ├── model/                 # Encoder, Attention, Decoder, Seq2Seq
│   ├── decoding/              # Greedy, Beam Search, Inference API
│   ├── evaluation/            # Metrics, evaluation pipeline, human eval
│   ├── training/              # Training loop, utilities
│   └── visualization.py       # Plots and heatmaps
├── scripts/                   # CLI entry points
├── notebooks/                 # Execution notebooks (01–06)
├── tests/                     # Unit tests
├── frontend/app.py            # Streamlit frontend
├── docs/                      # Design notes, viva, blog, LinkedIn
├── results/                   # Metrics, samples, human eval
└── figures/                   # Generated plots
```

## Reproduction

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/urdu-qgen-seq2seq.git
cd urdu-qgen-seq2seq
pip install -r requirements.txt

# 2. Prepare data
python scripts/prepare_data.py

# 3. Train tokenizer
python -c "from src.tokenizer.train_tokenizer import train_sentencepiece; train_sentencepiece()"

# 4. Debug training (CPU ok)
python scripts/train_debug.py

# 5. Full training (GPU required)
python scripts/train_full.py

# 6. Evaluate
python scripts/evaluate.py

# 7. Frontend
streamlit run frontend/app.py
```

## Team Members

- [MEMBER 1]
- [MEMBER 2]

## Limitations

- No pretrained embeddings — the model starts from random initialisation
- LSTM architecture (no Transformer) — limited long-range modelling
- UQA is a translated dataset — may contain translation artifacts
- Greedy decoding can produce repetitive or generic outputs
- Beam search can sometimes prefer safe but bland questions
- Wiki-UQA performance expected to be lower due to domain shift
