"""
High-level inference API for question generation.

``QuestionGenerator`` loads the tokenizer and model checkpoint once
and exposes simple ``generate_greedy`` / ``generate_beam`` methods.
This class is shared by the frontend and evaluation scripts so that
model-loading logic is not duplicated.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch

from configs.config import (
    BEAM_WIDTH,
    BEST_MODEL_PATH,
    BIDIRECTIONAL,
    DROPOUT,
    EMBEDDING_DIM,
    EOS_ID,
    HIDDEN_SIZE,
    MAX_DECODE_LENGTH,
    NUM_LAYERS,
    PAD_ID,
    SP_MODEL_PATH,
    VOCAB_SIZE,
)
from src.decoding.beam_search import beam_search_decode
from src.decoding.greedy import greedy_decode
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq
from src.tokenizer.tokenizer_utils import UrduTokenizer
from src.training.utils import load_checkpoint


class QuestionGenerator:
    """Reusable inference wrapper.

    Parameters
    ----------
    model_path : Path
        Path to the best checkpoint (``.pt`` file).
    sp_model_path : Path
        Path to the SentencePiece ``.model`` file.
    device : torch.device or None
        Device to run inference on. Defaults to CUDA if available,
        otherwise CPU.
    """

    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        sp_model_path: Path = SP_MODEL_PATH,
        device: Optional[torch.device] = None,
    ) -> None:
        # ── Device ────────────────────────────────────────────────
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ── Tokenizer ─────────────────────────────────────────────
        self.tokenizer = UrduTokenizer(sp_model_path)
        actual_vocab = self.tokenizer.vocab_size

        # ── Model ─────────────────────────────────────────────────
        enc_hidden = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
        encoder = Encoder(
            vocab_size=actual_vocab,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            pad_id=PAD_ID,
        )
        decoder = Decoder(
            vocab_size=actual_vocab,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            encoder_hidden_size=enc_hidden,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            pad_id=PAD_ID,
        )
        self.model = Seq2Seq(encoder, decoder).to(device)

        # ── Load checkpoint ───────────────────────────────────────
        load_checkpoint(model_path, self.model, device=device)
        self.model.eval()

    def _prepare_input(self, source_text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize source text and create model-ready tensors."""
        from configs.config import BOS_ID, EOS_ID

        ids = [BOS_ID] + self.tokenizer.encode_ids(source_text) + [EOS_ID]
        src = torch.LongTensor([ids]).to(self.device)       # [1, S]
        src_len = torch.LongTensor([len(ids)])               # [1]
        src_mask = (src == PAD_ID)                           # [1, S]
        return src, src_len, src_mask

    def generate_greedy(self, source_text: str) -> str:
        """Generate a question using greedy decoding."""
        src, src_len, src_mask = self._prepare_input(source_text)
        token_ids, _ = greedy_decode(
            self.model, src, src_len, src_mask,
            max_length=MAX_DECODE_LENGTH,
        )
        # Remove EOS if present
        if token_ids and token_ids[-1] == EOS_ID:
            token_ids = token_ids[:-1]
        return self.tokenizer.decode(token_ids)

    def generate_beam(
        self, source_text: str, beam_width: int = BEAM_WIDTH,
    ) -> str:
        """Generate a question using beam search."""
        src, src_len, src_mask = self._prepare_input(source_text)
        token_ids, _ = beam_search_decode(
            self.model, src, src_len, src_mask,
            beam_width=beam_width,
            max_length=MAX_DECODE_LENGTH,
        )
        # Remove EOS if present
        if token_ids and token_ids[-1] == EOS_ID:
            token_ids = token_ids[:-1]
        return self.tokenizer.decode(token_ids)

    def generate_both(
        self, source_text: str, beam_width: int = BEAM_WIDTH,
    ) -> Tuple[str, str]:
        """Return (greedy_question, beam_question) for convenience."""
        greedy_q = self.generate_greedy(source_text)
        beam_q = self.generate_beam(source_text, beam_width)
        return greedy_q, beam_q
