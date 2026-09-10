# Viva Preparation Notes

## Architecture Questions

### Q: Why did we use a bidirectional encoder?

**A:** A bidirectional encoder reads the source sentence in both forward and
backward directions. Each encoder output position then captures context from
both sides. This is essential for question generation because the answer span
can appear anywhere in the sentence, and understanding it requires both
preceding and following context. For example, in "دریائے سندھ کی لمبائی
\<ans\> 3,180 کلومیٹر \</ans\> ہے", the word "لمبائی" (length) before the
answer is crucial context.

### Q: Why isn't the decoder bidirectional?

**A:** The decoder generates tokens autoregressively — it predicts each token
based on all previously generated tokens plus the encoder's representation.
If it were bidirectional, it would need to know future tokens, which are not
yet generated at inference time. Teacher forcing during training provides
ground-truth previous tokens, but the decoder still only looks backward.

### Q: What exactly happens inside attention?

**A:** At each decoder step:
1. The decoder's current hidden state `s_t` is projected by a learned
   matrix `W_a` into the encoder's output space.
2. Alignment scores are computed: `score_i = (s_t @ W_a) · h_i` for each
   encoder output `h_i`.
3. Padding positions are masked to -inf.
4. Softmax normalises scores into a probability distribution α over source
   positions.
5. Context vector = Σ(α_i × h_i) — a weighted sum of encoder outputs.
6. The context is concatenated with the decoder output and projected to
   produce vocabulary logits.

### Q: Why Luong instead of Bahdanau attention?

**A:** Luong attention computes scores after the decoder LSTM step (it uses
the current decoder output). Bahdanau computes scores before the LSTM step
(using the previous hidden state). Luong is simpler to implement and verify
because the decoder output is directly available. Both work well; we chose
Luong for clarity.

## Training Questions

### Q: Why teacher forcing?

**A:** Without teacher forcing, the decoder feeds its own predictions back as
input. In early training, predictions are essentially random — each wrong
prediction propagates errors to subsequent steps, making learning very
difficult. Teacher forcing provides the correct previous token, allowing the
model to learn meaningful patterns from the start.

### Q: What is exposure bias?

**A:** During training with teacher forcing, the decoder always receives
correct input. During inference, it receives its own (possibly incorrect)
predictions. This mismatch means the decoder never learned to recover from
its own mistakes during training, which can cause error accumulation at
inference. We don't mitigate this (no scheduled sampling) as the assignment
only requires standard teacher forcing.

### Q: Why CrossEntropyLoss with ignore_index=PAD?

**A:** Padding tokens are artificial — they only exist to make sequences in a
batch the same length. If we included padding in the loss, the model would be
rewarded for predicting PAD at padding positions, which teaches it nothing
useful and distorts the loss signal.

### Q: Why gradient clipping?

**A:** LSTMs can suffer from exploding gradients, especially with long
sequences. Gradient clipping caps the gradient norm at a threshold (1.0),
preventing extremely large updates that could destabilise training.

## Model Shape Questions

### Q: What are the key tensor shapes?

| Tensor | Shape |
|--------|-------|
| Source IDs | `[B, S]` |
| Embedded source | `[B, S, 256]` |
| Encoder outputs | `[B, S, 1024]` |
| Encoder hidden | `[2, B, 512]` (after projection) |
| Attention weights | `[B, S]` |
| Context vector | `[B, 1024]` |
| Decoder step output | `[B, 512]` |
| Vocabulary logits | `[B, 8000]` |

### Q: Why is the encoder output width 1024?

**A:** The encoder is bidirectional with hidden_size=512. Forward and backward
hidden states are concatenated at each position: 512 + 512 = 1024.

### Q: How are the encoder's final states converted for the decoder?

**A:** The LSTM returns hidden states shaped `[num_layers*2, B, 512]`
(2 directions). We reshape to `[num_layers, 2, B, 512]`, concatenate the
forward (idx 0) and backward (idx 1) → `[num_layers, B, 1024]`, then
project through a linear layer → `[num_layers, B, 512]` to match the
decoder's hidden size.

## Decoding Questions

### Q: How does beam search work?

**A:** Beam search maintains `k` (beam_width=5) hypotheses at each step:
1. For each active hypothesis, compute the next-token probability distribution.
2. For each hypothesis, consider the top-k next tokens.
3. This gives up to `k × k` candidates; keep only the top-k by cumulative
   log probability.
4. If a hypothesis generates EOS, move it to a "completed" list.
5. Continue until all hypotheses finish or max_length is reached.
6. Return the completed hypothesis with the best length-normalised score.

### Q: Why log probabilities?

**A:** Multiplying many small probabilities causes numerical underflow
(values become so small they round to zero). Adding log probabilities avoids
this: `log(p1 × p2) = log(p1) + log(p2)`.

### Q: Why length normalisation?

**A:** Without it, shorter sequences accumulate fewer (negative) log
probability terms and thus have higher scores. Length normalisation divides
the total log probability by the sequence length, removing this bias.

## Evaluation Questions

### Q: What does perplexity measure?

**A:** Perplexity = exp(mean cross-entropy). It measures how "surprised" the
model is by the reference targets. A perplexity of 10 means the model is,
on average, as uncertain as if it were choosing uniformly among 10 tokens.
Lower is better. It's computed via teacher forcing — it doesn't depend on
greedy vs beam decoding.

### Q: Why ROUGE-L and not ROUGE-1/ROUGE-2?

**A:** ROUGE-L measures the longest common subsequence (LCS), which captures
word-order similarity without requiring exact contiguous n-gram matches.
This is more forgiving for paraphrased questions that use the same words in a
slightly different order.

## Data Questions

### Q: Why is validation data excluded from tokenizer training?

**A:** If the tokenizer's vocabulary is influenced by validation data, it
would have an unfair advantage on validation examples. The subword vocabulary
would "fit" validation text better than unseen text, inflating validation
metrics and hiding generalisation issues.

## Potential Debugging Issues to Watch For

- **Dimension mismatch**: Encoder output (1024) ≠ decoder hidden (512);
  need the projection layer.
- **Pack/unpack with wrong lengths**: src_lengths must be on CPU and sorted
  in descending order for `pack_padded_sequence`.
- **Attention masking**: Forgetting to mask → PAD tokens receive attention
  probability → garbage context vectors.
- **Teacher forcing at inference**: Accidentally using target tokens during
  decoding → artificially good but unrealistic outputs.
- **Device mismatch**: Moving some tensors to GPU but forgetting others →
  PyTorch runtime error.
- **Loss not decreasing**: Check that padding is ignored, teacher forcing is
  enabled, and embeddings are not frozen.

## Project Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Vocab size | 8000 |
| Embedding dim | 256 |
| Hidden size | 512 |
| Num layers | 2 |
| Dropout | 0.3 |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Epochs | 12 |
| Beam width | 5 |
| Max source words | 60 |
| Max target words | 25 |
| Max decode length | 25 |
