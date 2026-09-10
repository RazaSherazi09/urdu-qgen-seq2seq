# Design Notes

## Why Seq2Seq?

Question generation from a source sentence is a sequence-to-sequence problem:
the input is a variable-length sequence (the marked sentence) and the output
is another variable-length sequence (the question). Encoder–decoder models
are the standard architecture for this class of problems.

## Why Sentence-Level Source?

The full context paragraph can be hundreds of words. Using only the sentence
that contains the answer gives the model a focused input and makes training
much more efficient. The sentence provides exactly the information needed to
generate a question about the answer.

## Why LSTM?

LSTMs are well-suited for sequential modelling and are explicitly required
by the assignment. They handle variable-length sequences via gating mechanisms
(forget, input, output gates) that control information flow, addressing the
vanishing gradient problem that affects vanilla RNNs.

## Why Bidirectional Encoder?

A bidirectional encoder reads the source in both directions. This means every
encoder output has information about both the preceding and following context.
This is important because the answer span can appear anywhere in the sentence,
and the model needs surrounding context from both sides to understand it.

## Why Unidirectional Decoder?

The decoder generates tokens left-to-right. Making it bidirectional would
require knowing future tokens, which contradicts the autoregressive generation
process. The decoder must predict each token based only on previously generated
tokens and the encoder's representation.

## Why Attention?

Without attention, the decoder must compress the entire source into a single
fixed-size vector (the final hidden state). This information bottleneck
limits performance, especially for longer sequences. Attention lets the decoder
look at all encoder outputs and focus on the most relevant parts at each
generation step.

## Why Luong (General) Attention?

Luong general attention uses a learnable bilinear form:
    score(s_t, h_i) = s_t^T W_a h_i

This is simple to implement, easy to explain, and gives the model enough
expressive power to learn meaningful alignments. The "general" variant is
preferred over pure dot-product because the encoder and decoder have different
hidden sizes (1024 vs 512).

## Why SentencePiece?

SentencePiece operates directly on raw Unicode text without a pre-tokenizer.
This is ideal for Urdu, which has complex morphology and no clear word
boundaries in connected script. It handles the full Unicode range natively.

## Why Unigram?

The unigram model learns a probabilistic subword vocabulary that maximises
the likelihood of the training corpus. It typically gives better coverage
for morphologically rich languages compared to BPE.

## Why Vocabulary 8000?

8000 subwords provide a good balance between vocabulary coverage and model
size. Too small → many words split into tiny pieces → longer sequences.
Too large → rare subwords with insufficient training examples.

## Why Padding Mask?

Padding tokens are artificial (they exist only to make batches rectangular).
If the attention mechanism assigns probability to padding positions, the
model will compute meaningless context vectors. We mask padding to -inf
before softmax so these positions receive exactly zero attention.

## Why Teacher Forcing?

During training, the decoder receives the correct previous target token
(from the ground truth) rather than its own prediction. This stabilises
training because the decoder doesn't need to recover from compounding
errors in early epochs when it predicts poorly.

## What is Exposure Bias?

Teacher forcing creates a discrepancy: during training the decoder always
sees correct input, but during inference it sees its own (possibly wrong)
predictions. This is called exposure bias. We don't address it here because
scheduled sampling is not required by the assignment.

## Why Greedy Decoding?

Greedy decoding selects the highest-probability token at each step. It's fast
and simple. It serves as a baseline against beam search.

## Why Beam Search?

Beam search maintains multiple hypotheses and explores more of the output
space. It often finds better overall sequences than greedy because it
avoids greedy's tendency to commit to locally optimal but globally suboptimal
choices.

## Why BLEU?

BLEU measures n-gram overlap between generated and reference text. BLEU-4
considers unigrams through 4-grams, rewarding both word choice and word order.

## Why ROUGE-L?

ROUGE-L measures the longest common subsequence between generated and
reference text. It captures sentence-level structure similarity without
requiring exact n-gram matches.

## Why Perplexity?

Perplexity = exp(mean cross-entropy) measures how well the model predicts
the reference targets. Lower perplexity means the model assigns higher
probability to the correct tokens. Unlike BLEU, it doesn't require
generation — it's a teacher-forced metric.

## Why Wiki-UQA?

Wiki-UQA is from a different domain than the UQA training data. Evaluating
on it reveals how well the model generalises beyond its training distribution.
Performance degradation is expected and informative.

## Why Can Beam Search Sometimes Hurt?

Beam search optimises for high probability, which can favour generic, safe
outputs over more specific but less probable ones. A greedy path might take
a "risky" but correct word that beam search prunes early. This is especially
visible when the model is uncertain.

## How Does a Transformer Differ?

Transformers use self-attention instead of recurrence. This allows parallel
computation (faster training) and direct modelling of long-range dependencies.
However, Transformers require positional encoding and are typically more
parameter-hungry. Our LSTM approach is simpler, easier to debug, and
sufficient for this sentence-level task.
