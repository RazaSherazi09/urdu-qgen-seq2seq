
import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm


# =========================================================
# Configuration — must match Experiment 2
# =========================================================

EMB_DIM = 256
HID_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
DROPOUT = 0.3

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

MAX_TGT_LEN = 30

ANS_OPEN = "<ans>"
ANS_CLOSE = "</ans>"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Model definitions
# =========================================================

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hid_dim,
        n_layers,
        dropout
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            input_dim,
            emb_dim,
            padding_idx=PAD_IDX
        )

        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):

        embedded = self.dropout(
            self.embedding(src)
        )

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        packed_outputs, (hidden, cell) = self.rnn(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        # hidden:
        # [n_layers * 2, batch, hid_dim]

        hidden = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        cell = torch.cat(
            (cell[-2], cell[-1]),
            dim=1
        )

        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)

        # Replicate projected state for decoder layers
        hidden = hidden.unsqueeze(0).repeat(
            DEC_LAYERS, 1, 1
        )

        cell = cell.unsqueeze(0).repeat(
            DEC_LAYERS, 1, 1
        )

        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear(
            enc_hid_dim * 2 + dec_hid_dim,
            dec_hid_dim
        )

        self.v = nn.Linear(
            dec_hid_dim,
            1,
            bias=False
        )

    def forward(self, hidden, encoder_outputs, mask):

        # hidden:
        # [dec_layers, batch, hid_dim]

        hidden_last = hidden[-1]

        src_len = encoder_outputs.size(1)

        hidden_rep = hidden_last.unsqueeze(1).repeat(
            1,
            src_len,
            1
        )

        energy = torch.tanh(
            self.attn(
                torch.cat(
                    (hidden_rep, encoder_outputs),
                    dim=2
                )
            )
        )

        attention = self.v(energy).squeeze(2)

        attention = attention.masked_fill(
            mask == 0,
            -1e10
        )

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        hid_dim,
        n_layers,
        dropout
    ):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            output_dim,
            emb_dim,
            padding_idx=PAD_IDX
        )

        self.attention = BahdanauAttention(
            hid_dim,
            hid_dim
        )

        self.rnn = nn.LSTM(
            emb_dim + hid_dim * 2,
            hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        self.fc_out = nn.Linear(
            emb_dim + hid_dim * 2 + hid_dim,
            output_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_tok,
        hidden,
        cell,
        encoder_outputs,
        mask
    ):

        # input_tok: [batch]

        input_tok = input_tok.unsqueeze(1)

        embedded = self.dropout(
            self.embedding(input_tok)
        )

        a = self.attention(
            hidden,
            encoder_outputs,
            mask
        )

        a = a.unsqueeze(1)

        weighted = torch.bmm(
            a,
            encoder_outputs
        )

        rnn_input = torch.cat(
            (embedded, weighted),
            dim=2
        )

        output, (hidden, cell) = self.rnn(
            rnn_input,
            (hidden, cell)
        )

        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(
            torch.cat(
                (output, weighted, embedded),
                dim=1
            )
        )

        return prediction, hidden, cell, a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        device
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        return (src != PAD_IDX)

    def forward(self, src, src_lengths, trg):

        batch_size = src.size(0)
        trg_len = trg.size(1)

        output_dim = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size,
            trg_len,
            output_dim,
            device=self.device
        )

        encoder_outputs, hidden, cell = self.encoder(
            src,
            src_lengths
        )

        mask = self.create_mask(src)

        input_tok = trg[:, 0]

        for t in range(1, trg_len):

            output, hidden, cell, _ = self.decoder(
                input_tok,
                hidden,
                cell,
                encoder_outputs,
                mask
            )

            outputs[:, t] = output

            input_tok = trg[:, t]

        return outputs


# =========================================================
# Model + tokenizer loading
# =========================================================

@st.cache_resource
def load_model_and_tokenizer():

    tokenizer = spm.SentencePieceProcessor(
        model_file="ur_sp.model"
    )

    vocab_size = tokenizer.get_piece_size()

    encoder = Encoder(
        vocab_size,
        EMB_DIM,
        HID_DIM,
        ENC_LAYERS,
        DROPOUT
    ).to(DEVICE)

    decoder = Decoder(
        vocab_size,
        EMB_DIM,
        HID_DIM,
        DEC_LAYERS,
        DROPOUT
    ).to(DEVICE)

    model = Seq2Seq(
        encoder,
        decoder,
        DEVICE
    ).to(DEVICE)

    checkpoint = torch.load(
        "best_model.pt",
        map_location=DEVICE
    )

    # Support either a raw state_dict or a checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


# =========================================================
# Decoding
# =========================================================

def prepare_source(tokenizer, sentence):

    ids = (
        [BOS_IDX]
        + tokenizer.encode(sentence, out_type=int)
        + [EOS_IDX]
    )

    src = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    src_lengths = torch.tensor(
        [len(ids)],
        dtype=torch.long
    )

    return src, src_lengths


@torch.no_grad()
def greedy_decode(
    model,
    tokenizer,
    sentence,
    max_len=MAX_TGT_LEN
):

    src, src_lengths = prepare_source(
        tokenizer,
        sentence
    )

    encoder_outputs, hidden, cell = model.encoder(
        src,
        src_lengths
    )

    mask = model.create_mask(src)

    input_tok = torch.tensor(
        [BOS_IDX],
        dtype=torch.long,
        device=DEVICE
    )

    output_ids = []

    for _ in range(max_len):

        prediction, hidden, cell, _ = model.decoder(
            input_tok,
            hidden,
            cell,
            encoder_outputs,
            mask
        )

        next_tok = prediction.argmax(
            dim=1
        )

        token_id = next_tok.item()

        if token_id == EOS_IDX:
            break

        if token_id not in (
            PAD_IDX,
            BOS_IDX
        ):
            output_ids.append(token_id)

        input_tok = next_tok

    return tokenizer.decode(output_ids)


@torch.no_grad()
def beam_search_decode(
    model,
    tokenizer,
    sentence,
    beam_width=5,
    max_len=MAX_TGT_LEN,
    length_penalty=0.7
):

    src, src_lengths = prepare_source(
        tokenizer,
        sentence
    )

    encoder_outputs, hidden, cell = model.encoder(
        src,
        src_lengths
    )

    mask = model.create_mask(src)

    # tokens, score, hidden, cell, finished
    beams = [
        (
            [BOS_IDX],
            0.0,
            hidden,
            cell,
            False
        )
    ]

    for _ in range(max_len):

        candidates = []

        for tokens, score, h, c, finished in beams:

            if finished:
                candidates.append(
                    (
                        tokens,
                        score,
                        h,
                        c,
                        True
                    )
                )
                continue

            input_tok = torch.tensor(
                [tokens[-1]],
                dtype=torch.long,
                device=DEVICE
            )

            prediction, new_h, new_c, _ = model.decoder(
                input_tok,
                h,
                c,
                encoder_outputs,
                mask
            )

            log_probs = torch.log_softmax(
                prediction,
                dim=1
            ).squeeze(0)

            top_log_probs, top_indices = log_probs.topk(
                beam_width
            )

            for k in range(beam_width):

                token_id = top_indices[k].item()
                token_score = top_log_probs[k].item()

                candidates.append(
                    (
                        tokens + [token_id],
                        score + token_score,
                        new_h,
                        new_c,
                        token_id == EOS_IDX
                    )
                )

        candidates.sort(
            key=lambda x:
                x[1] /
                (len(x[0]) ** length_penalty),
            reverse=True
        )

        beams = candidates[:beam_width]

        if all(
            beam[4]
            for beam in beams
        ):
            break

    best = max(
        beams,
        key=lambda x:
            x[1] /
            (len(x[0]) ** length_penalty)
    )

    token_ids = [
        token_id
        for token_id in best[0]
        if token_id not in (
            BOS_IDX,
            EOS_IDX,
            PAD_IDX
        )
    ]

    return tokenizer.decode(token_ids)


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="Urdu Question Generation",
    page_icon="Q",
    layout="centered"
)

st.title(
    "Urdu Question Generation"
)

st.caption(
    "BiLSTM Encoder + Bahdanau Attention + LSTM Decoder"
)

st.markdown(
    """
Generate an Urdu question from a context sentence and an
answer span.

Wrap the answer with `<ans>` and `</ans>`.
"""
)

st.markdown(
    """
**Example**

`دریائے سندھ کی لمبائی <ans> 3,180 کلومیٹر </ans> ہے۔`
"""
)

sentence = st.text_area(
    "Input sentence",
    height=130,
    placeholder=(
        "Enter an Urdu context with the answer marked "
        "using <ans> ... </ans>"
    )
)

beam_width = st.slider(
    "Beam width",
    min_value=2,
    max_value=8,
    value=5
)

if st.button(
    "Generate Question",
    type="primary"
):

    if not sentence.strip():

        st.warning(
            "Please enter an input sentence."
        )

    elif (
        ANS_OPEN not in sentence
        or ANS_CLOSE not in sentence
    ):

        st.error(
            "Please wrap the answer span in "
            "`<ans> ... </ans>`."
        )

    else:

        try:

            model, tokenizer = (
                load_model_and_tokenizer()
            )

            with st.spinner(
                "Generating question..."
            ):

                greedy_question = greedy_decode(
                    model,
                    tokenizer,
                    sentence
                )

                beam_question = beam_search_decode(
                    model,
                    tokenizer,
                    sentence,
                    beam_width=beam_width
                )

            st.subheader("Generated Questions")

            st.write(
                "**Greedy decoding:**"
            )

            st.success(
                greedy_question
                if greedy_question
                else "No question generated."
            )

            st.write(
                "**Beam search:**"
            )

            st.success(
                beam_question
                if beam_question
                else "No question generated."
            )

            with st.expander(
                "Input details"
            ):

                answer_match = (
                    sentence.split(ANS_OPEN, 1)[1]
                    .split(ANS_CLOSE, 1)[0]
                    .strip()
                )

                st.write(
                    "**Answer span:**",
                    answer_match
                )

                st.write(
                    "**Device:**",
                    str(DEVICE)
                )

        except Exception as e:

            st.error(
                f"Generation failed: {e}"
            )
