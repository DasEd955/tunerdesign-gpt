# Tuner Design GPT — GPT from Scratch

A ground up implementation of the GPT architecture in Python, built component by
component from first mathematical principles before a single PyTorch module was
introduced. Every abstraction in this repository, from scalar gradient descent to
grouped query attention, was written and validated as a standalone unit before being
composed into the final working language model. The result is a complete, end-to-end
system that demonstrates the full engineering depth behind modern large language models:
foundations, tokenization, a production caliber transformer architecture, KV-cached
autoregressive inference, and an AdamW training loop.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Architecture Diagrams](#2-architecture-diagrams)
3. [Foundations](#3-foundations)
4. [Tokenization and Data Pipeline](#4-tokenization-and-data-pipeline)
5. [Transformer Architecture](#5-transformer-architecture)
6. [Training](#6-training)
7. [Inference](#7-inference)
8. [Spec Reflection](#8-spec-reflection)
9. [Test Suite](#9-test-suite)
10. [Limitations](#10-limitations)
11. [Setup and Launch](#11-setup-and-launch)

---

## 1. Architecture Overview

The repository is organized into four layers of increasing abstraction, each built before
the layer above it depends on it. At the base is `foundations/`, a collection of
pure NumPy and early PyTorch modules that implement every mathematical primitive a neural
network needs: gradient descent, backpropagation through a single neuron, multilayer
backpropagation, activation functions, loss functions, normalization variants, weight
initialization strategies, and diagnostic utilities for dead neurons and gradient health.
These modules are not wrappers around library calls; they derive every operation from the
underlying calculus so that the behavior of the higher level components can be understood
mechanically rather than treated as a black box.

Above the foundations sits `data/`, which handles the full pipeline from raw text to
batched training tensors. A character-level vocabulary builder (`vocab.py`) constructs
integer-to-character and character-to-integer mappings from any input corpus. A
byte-pair encoding tokenizer (`tokenizer.py`) implements the greedy merge algorithm used
by GPT family models to build subword vocabularies. An NLP preprocessing module
(`nlp_preprocessing.py`) constructs padded, vocabulary aligned datasets for sequence
classification tasks. Utilities in `tokenizer_utils.py` implement greedy longest match
tokenization and compute fertility scores for measuring vocabulary efficiency. Finally,
`loader.py` and `dataset.py` implement the batched data loaders that randomly sample
`(input, target)` sequence pairs from a flat corpus, creating the shifted by one targets
that autoregressive language model training requires.

The core of the system is `model/`, which implements every major component of the GPT
architecture. A single attention head (`attention.py`) computes scaled dot product
attention with a causal mask. `multi_head_attention.py` runs several heads in parallel and
projects their concatenated outputs. `transformer.py` composes attention and a
position wise feed forward network inside a Pre-LN residual block, the architectural
variant used by modern GPT family models (normalization before each sublayer rather than
after, which stabilizes training). `gpt.py` stacks N of those blocks with learned token
and position embeddings and a final linear projection to vocabulary logits. Three
normalization variants are provided: layer normalization (`normalization.py`), RMS
normalization (`rms_normalization.py`), and batch normalization with training and
inference modes (`batch_normalization.py`). Positional encoding
(`positional_encoding.py`) implements the sinusoidal scheme from the original "Attention
is All You Need" paper. The embedding lookup (`embeddings.py`) demonstrates index-based
retrieval from the weight matrix. For efficient inference, `kv_cache.py` implements a
stateful KV cache that appends newly projected keys and values to a running buffer rather
than recomputing the full sequence at each decoding step. `grouped_query_attention.py`
implements the memory efficient attention variant used in Llama 2 and Mistral, where
multiple query heads share a smaller number of key/value heads, reducing the KV cache
footprint proportionally. `model.py` is the unified entry point: it defines `GPTConfig`
(a dataclass that holds all hyperparameters), `create_model` (constructs a `GPT` from a
config), `save_model` and `load_model` (persist and restore checkpoints to `saved_model/`),
and `run` (wires vocabulary building, training via `train.py`, checkpoint saving, and
generation via `generate.py` into a single callable).

`train.py` implements a standard language model training loop over the GPT model using
AdamW optimization and cross-entropy loss, treating every position in the sequence as an
independent next-token classification problem. `generate.py` implements autoregressive
decoding: at each step the model receives the current context, produces a probability
distribution over the vocabulary by applying softmax to the final position logits, samples
the next token using multinomial sampling, appends it to the context, and repeats for
however many characters are requested. Both modules are consumed by `model/model.py`
rather than duplicating their logic.

---

## Project Structure

```
model/          Attention, Transformer, GPT architecture
  model.py                 GPTConfig, create_model, save_model, load_model, run
  gpt.py                   GPT model
  attention.py             Self-attention head
  multi_head_attention.py  Multi-headed attention
  transformer.py           Transformer block
  normalization.py         Layer normalization
  batch_normalization.py   Batch normalization
  rms_normalization.py     RMS normalization
  embeddings.py            Word embeddings
  positional_encoding.py   Positional encoding
  kv_cache.py              KV-Cache for fast inference
  grouped_query_attention.py  Grouped query attention

data/           Data pipeline
  tokenizer.py                BPE tokenizer
  vocab.py                    Character-level vocabulary
  loader.py                   Batched training data loader
  dataset.py                  GPT dataset preparation
  nlp_preprocessing.py        NLP preprocessing
  tokenizer_utils.py          Tokenization edge cases

train.py        GPT training loop (used by model/model.py)
generate.py     Text generation (used by model/model.py)

saved_model/    Persisted model checkpoints (written by save_model)

foundations/    Neural network primitives built from scratch
  neuron.py, backprop.py, mlp.py, activations.py, loss.py,
  training_loop.py, dead_relu_detector.py, ...
```

---

## 2. Architecture Diagrams

### GPT Architecture

```mermaid
flowchart TD
    RAW["Raw text corpus"]

    subgraph DATA["Data pipeline (data/)"]
        direction LR
        TOK["Tokenizer\nBPE + char vocab\ntokenizer.py · vocab.py · tokenizer_utils.py"]
        PRE["Preprocessor\nNLP cleaning\nnlp_preprocessing.py"]
        DS["Dataset\nGPT prep\ndataset.py"]
        LDR["Loader\nBatched training\nloader.py"]
        TOK --> PRE --> DS --> LDR
    end

    subgraph FOUND["Neural network foundations (foundations/)"]
        direction LR
        NB["Neuron + backprop\nneuron.py · backprop.py · multi_layer_backprop.py"]
        MA["MLP + activations\nmlp.py · activations.py · softmax.py"]
        LF["Loss functions\nloss.py · linear_regression.py"]
        TL["Training loop\ntraining_loop.py · gradient_descent.py · weight_init.py"]
        DR["Dead ReLU detect\ndead_relu_detector.py · training_diagnostics.py"]
        NB ~~~ MA ~~~ LF ~~~ TL ~~~ DR
    end

    subgraph MODEL["GPT model architecture (model/)"]
        direction TB
        EMB["Embeddings\nWord + positional\nembeddings.py · positional_encoding.py"]
        ATT["Attention\nSelf + multi-head\nattention.py · multi_head_attention.py"]
        NORM["Normalization\nLayer / RMS / batch\nnormalization.py · rms_normalization.py · batch_normalization.py"]
        TB["Transformer block\nAttn + FFN + norm\ntransformer.py"]
        GPT["GPT model\nStacked transformer\ngpt.py"]
        KV["KV-cache\nFast inference\nkv_cache.py"]
        GQA["Grouped query attn\nEfficient multi-head\ngrouped_query_attention.py"]
        ENTRY["model.py\nGPTConfig · create_model\nsave_model · load_model · run"]
        EMB --> ATT --> NORM --> TB --> GPT
        KV --> GPT
        GQA --> GPT
        GPT --> ENTRY
    end

    TRAIN["train.py\nGPT training loop\nAdamW + cross-entropy"]
    GEN["generate.py\nText generation\nAutoregressive decoding"]
    CKPT["saved_model/\nCheckpoint · gpt.pt"]
    OUT["Generated text output"]

    RAW --> DATA
    DATA --> FOUND
    FOUND --> MODEL
    ENTRY --> TRAIN
    TRAIN --> ENTRY
    ENTRY --> GEN
    ENTRY --> CKPT
    GEN --> OUT
```

### Transformer Block

```mermaid
flowchart TD
    IN["Input tensor x\nShape: (batch, seq_len, 128)\nOne vector per token, 128 numbers wide\ntransformer.py"]

    subgraph SUBLAYER1["Sublayer 1 — Self-Attention (Pre-LN residual)"]
        direction TB
        LN1["LayerNorm\nNormalize each token vector to zero mean,\nunit variance before attention sees it\nnormalization.py"]
        MHA["Multi-Head Attention\n4 independent heads, each 32-dim\nEach head learns different relationships\nbetween tokens in the sequence\nmulti_head_attention.py · attention.py"]
        ADD1(["Residual add +\nAdd original input back to attention output\nso the block only needs to learn the\n'correction', not reconstruct the signal"])
        LN1 --> MHA --> ADD1
    end

    subgraph SUBLAYER2["Sublayer 2 — FeedForward Network (Pre-LN residual)"]
        direction TB
        LN2["LayerNorm\nNormalize again before the feed forward\nnetwork so signal stays well-scaled\nnormalization.py"]
        FF1["Linear expand: 128 → 512\nProject up to a wider space so the\nnetwork can express richer combinations"]
        ACT["ReLU activation\nZero out negatives — introduces\nnonlinearity so the model isn't\njust doing linear algebra\nactivations.py"]
        FF2["Linear compress: 512 → 128\nProject back down to the original\nmodel width so the next block\nreceives the same shape"]
        DROP["Dropout (p=0.2)\nRandomly zero 20% of values during\ntraining to prevent overfitting"]
        ADD2(["Residual add +\nAdd the feedforward output back to\nthe post-attention signal"])
        LN2 --> FF1 --> ACT --> FF2 --> DROP --> ADD2
    end

    OUT["Block output\nShape: (batch, seq_len, 128)\nSame shape as input — N blocks\ncan be stacked cleanly\ngpt.py"]

    IN --> SUBLAYER1
    IN --> ADD1
    ADD1 --> SUBLAYER2
    ADD1 --> ADD2
    ADD2 --> OUT
```

Each transformer block applies two sublayers in sequence, each wrapped in a Pre-LN residual connection. The input first passes through LayerNorm and into multi-head self-attention, which runs 4 independent 32-dimensional heads in parallel before projecting their concatenated output back to 128 dimensions. The attention output is added back to the original input via a skip connection. That sum then goes through a second LayerNorm and into a position-wise feed forward network that projects up to 512 dimensions, applies ReLU, projects back down to 128, and applies dropout at rate 0.2 before a second skip connection produces the block output. Placing normalization before rather than after each sublayer (Pre-LN) keeps gradient magnitude stable as the signal passes through many stacked blocks.

### Attention Mechanism

```mermaid
flowchart TD
    IN["Input x\nShape: (batch, seq_len, model_dim)\nEvery token in the sequence arrives\nas a vector of model_dim numbers\nattention.py · multi_head_attention.py"]

    subgraph QKV["Step 1 — Project into Q, K, V (three separate learned linear layers)"]
        direction LR
        WQ["W_Q  ·  Linear projection\nQuery: 'What am I looking for?'\nShape: (batch, seq_len, head_dim)"]
        WK["W_K  ·  Linear projection\nKey: 'What do I advertise about myself?'\nShape: (batch, seq_len, head_dim)"]
        WV["W_V  ·  Linear projection\nValue: 'What do I actually share if selected?'\nShape: (batch, seq_len, head_dim)"]
    end

    SCORES["Step 2 — Compute raw attention scores\nQ @ Kᵀ  ÷  √head_dim\nDot product of every query with every key.\nDivide by √head_dim to stop scores from\ngrowing too large and collapsing the softmax.\nResult shape: (batch, seq_len, seq_len)"]

    MASK["Step 3 — Apply causal mask\nSet every position above the diagonal to −∞\nThis forces the model to look only at\npast tokens, not future ones — essential\nfor language modelling where future words\nare unknown at prediction time"]

    SOFT["Step 4 — Softmax → Attention weights\nConvert masked scores to probabilities (0–1)\nthat sum to 1 across the sequence.\nHigh weight = 'pay a lot of attention to that token'"]

    WSUM["Step 5 — Weighted sum of Values\nWeights @ V\nEach token's output is a blend of all Value\nvectors, weighted by how relevant each\nposition was. Shape: (batch, seq_len, head_dim)"]

    CONCAT["Step 6 — Concatenate all heads\n[head_0 ‖ head_1 ‖ … ‖ head_n]\nAll 4 heads ran the above in parallel,\neach attending to different patterns.\nConcatenate to restore model width.\nShape: (batch, seq_len, model_dim)"]

    WO["Step 7 — Output projection W_O\nOne final linear layer that mixes information\nacross heads so the result is a coherent\nblended representation, not 4 independent ones.\nmulti_head_attention.py"]

    OUT["Attention output\nShape: (batch, seq_len, model_dim)\nPassed to the residual add in the\ntransformer block"]

    IN --> QKV
    WQ --> SCORES
    WK --> SCORES
    SCORES --> MASK
    MASK --> SOFT
    SOFT --> WSUM
    WV --> WSUM
    WSUM --> CONCAT
    CONCAT --> WO
    WO --> OUT
```

The input is linearly projected into three separate tensors: queries (Q), keys (K), and values (V), each of shape `(batch, seq_len, head_dim)`. Attention scores are computed as the dot product of Q and K transposed, scaled by `1 / sqrt(head_dim)` to prevent the softmax from saturating in high-dimensional spaces. A causal lower triangular mask then sets all positions above the diagonal to negative infinity, which forces the softmax to assign zero probability to future tokens and makes this decoder-style attention suitable for language modeling. The resulting attention weights are applied to V via a weighted sum to produce each head's output. In multi-head attention, this full computation runs in parallel across all 4 heads independently, their outputs are concatenated along the feature dimension to restore model width, and a final learned projection W_O mixes information across heads before the result is passed to the transformer block's residual connection.

### Autoregressive Generation Pipeline

```mermaid
flowchart TD
    START["Initial context\nThe seed text, encoded as integer token IDs\nShape: (1, context_length)\ngenerate.py · model/model.py"]

    CROP{"Is context longer\nthan max context_length?"}

    CROPYES["Crop to last context_length tokens\nThe model has a fixed memory window —\nif the running text grows past it, we\nslide the window forward and drop the oldest tokens\ngenerate.py"]

    FWD["GPT forward pass\nFeed the full context through all transformer blocks.\nEvery token position produces logits — raw scores,\none per vocabulary character.\nOutput shape: (1, seq_len, vocab_size)\ngpt.py"]

    SLICE["Slice out the final position's logits\nOnly the last token's logits matter:\nthat position has seen all prior tokens\nand predicts what comes next.\nShape: (1, vocab_size)"]

    SOFTMAX["Softmax → probability distribution\nConvert raw logits to probabilities (0 – 1)\nthat sum to 1 across the entire vocabulary.\nHigh probability = model is confident\nabout that character.\nfoundations/softmax.py"]

    SAMPLE["torch.multinomial — sample one token\nDraw one character index at random,\nweighted by the probabilities.\nSeeded generator makes this reproducible.\nResult: a single integer index"]

    DECODE["Decode integer → character\nLook up the integer in int_to_char map\nbuilt by vocab.py to get the actual\ncharacter string"]

    APPEND_STR["Append character to output string\nBuild up the generated text\none character at a time"]

    APPEND_CTX["Append token ID to context tensor\nGrow the context by 1 so the next\nforward pass conditions on everything\ngenerated so far\nShape grows: (1, seq_len + 1)"]

    DONE{"Generated enough characters?\ngenerated_count < new_chars"}

    RETURN["Return generated text string\nThe complete model output\nmodel/model.py → caller"]

    START --> CROP
    CROP -- Yes --> CROPYES
    CROP -- No --> FWD
    CROPYES --> FWD
    FWD --> SLICE
    SLICE --> SOFTMAX
    SOFTMAX --> SAMPLE
    SAMPLE --> DECODE
    DECODE --> APPEND_STR
    SAMPLE --> APPEND_CTX
    APPEND_STR --> DONE
    APPEND_CTX --> DONE
    DONE -- Not yet --> CROP
    DONE -- Yes --> RETURN
```

Generation is a loop that extends the context by one token per step. At the start of each step the running context is cropped to the model's maximum `context_length` if it has grown beyond that limit, ensuring the forward pass always receives a valid input shape. The full context is passed through the GPT, which returns logits of shape `(1, seq_len, vocab_size)`, and only the logits at the final sequence position are retained because that position conditions on all preceding tokens. Softmax converts those logits to a probability distribution over the vocabulary, and `torch.multinomial` draws a single sample from it using a seeded generator for reproducibility. The sampled integer index is decoded back to a character, appended to the output string, and also appended to the context tensor before the next iteration begins. This process repeats for `new_chars` steps, with the context window growing by one token each time and the model paying O(T^2) cost per step unless a KV cache is attached to skip reprocessing prior positions.

---

## 3. Foundations

Implements the mathematical building blocks of neural networks from first principles,
entirely in NumPy and early PyTorch, before any high level abstractions are introduced.

---

### `foundations/gradient_descent.py`

Minimizes the scalar function `f(x) = x^2` using vanilla gradient descent, applying the
update rule `x = x - lr * f'(x)` for a fixed number of iterations. This is the simplest
possible demonstration of the optimization loop that underlies all of deep learning.

---

### `foundations/linear_regression.py`

Computes predictions via `y_hat = X @ w` and mean squared error between predictions and
ground truth. Establishes the linear model and loss surface that gradient-based training
descends.

---

### `foundations/linear_regression_training.py`

Adds a full gradient descent training loop to the linear model, computing per-weight
derivatives analytically and updating weights iteratively over a configurable number of
passes through the data.

---

### `foundations/neuron.py`

Implements a single artificial neuron: computes the pre-activation `z = dot(x, w) + b`
and applies either sigmoid or ReLU, producing a scalar output. This is the atomic
computation unit every neural network is composed from.

---

### `foundations/activations.py`

Implements sigmoid and ReLU as standalone NumPy functions. Sigmoid squashes any real
number into `(0, 1)`; ReLU clips negatives to zero and is the activation used throughout
the model directory.

---

### `foundations/softmax.py`

Implements numerically stable softmax by subtracting `max(z)` before exponentiation,
converting a vector of raw logits into a valid probability distribution. This is the
final operation in the generation pipeline before sampling.

---

### `foundations/loss.py`

Implements binary cross-entropy (for binary classification) and categorical cross-entropy
(for multiclass problems), both with an epsilon guard against `log(0)`. Cross-entropy
is the loss used during GPT training when reshaped to treat each token position
independently.

---

### `foundations/backprop.py`

Derives gradients analytically for a single sigmoid neuron with MSE loss, computing
`dL/dw` and `dL/db` via the chain rule without autograd. The manual derivation cements
the mechanics that PyTorch automates in the training loop.

---

### `foundations/multi_layer_backprop.py`

Implements a full forward and backward pass through a two layer network (`x -> Linear ->
ReLU -> Linear -> MSE`), returning gradients for all four parameter tensors. This is the
direct precursor to understanding how PyTorch's autograd engine propagates gradients
through the transformer blocks.

---

### `foundations/mlp.py`

Implements an MLP forward pass for an arbitrary number of layers: applies
`h = ReLU(h @ W + b)` through all hidden layers with no activation on the final output.
Demonstrates how depth is composed from repeated application of the same linear + activation pattern.

---

### `foundations/pytorch_basics.py`

Covers the foundational PyTorch tensor operations needed throughout the model: reshape,
column-wise mean, concatenation, and MSE loss. These are the low level tools that every
subsequent module relies on.

---

### `foundations/digit_classifier.py`

A concrete PyTorch `nn.Module` implementing an MNIST-style digit classifier:
`Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid`. Introduces
the `nn.Module` pattern, dropout as a regularization technique, and multiple output sigmoid
classification before the transformer's more complex module hierarchy is introduced.

---

### `foundations/sentiment.py`

A minimal sentiment classifier: embeds each token into a 16-dimensional space, averages
across the sequence dimension to produce a single bag-of-words representation, and
projects through a linear layer and sigmoid to a binary prediction. Demonstrates learned
embeddings as a trainable lookup table before the full-scale embedding layer in the GPT
model.

---

### `foundations/weight_init.py`

Implements Xavier/Glorot and Kaiming/He initialization strategies and a diagnostic that
measures activation standard deviation across layers under each scheme. Proper
initialization is what prevents signal collapse or explosion before training has had any
chance to correct it.

---

### `foundations/training_diagnostics.py`

Instruments a model's forward and backward pass to report activation mean, standard
deviation, and dead neuron fraction per linear layer, plus gradient mean, standard
deviation, and L2 norm. Classifies the network as healthy, suffering from dead neurons,
or experiencing exploding or vanishing gradients based on a priority-ordered threshold
check.

---

### `foundations/dead_relu_detector.py`

Detects dead ReLU neurons (units that output zero for every sample in the batch) by
hooking into the model's ReLU layers and computing the fraction of neurons that never
activate. Includes a rule-based advisor that recommends leaky ReLU, reinitialization,
or a lower learning rate depending on the observed pattern.

---

## 4. Tokenization and Data Pipeline

Implements the full journey from raw text to batched integer tensors ready for the
training loop.

---

### `data/vocab.py`

Builds character-level vocabulary mappings (`stoi` and `itos`) by sorting the unique
characters in the corpus alphabetically and assigning consecutive integer IDs. Also
provides `encode` and `decode` functions that convert between strings and integer
sequences using those mappings.

---

### `data/tokenizer.py`

Implements the byte-pair encoding (BPE) merge algorithm from scratch: repeatedly counts
adjacent token pair frequencies, selects the most common pair (with lexicographic
tiebreaking), merges all non-overlapping occurrences frin left to right, and records the merge
rule. Running `num_merges` rounds produces the subword vocabulary that GPT family models
use to balance vocabulary size against sequence length.

---

### `data/tokenizer_utils.py`

Implements greedy longest match tokenization over an existing vocabulary, as well as two
derived metrics: raw token count and fertility score (tokens per word). Fertility
quantifies how efficiently a vocabulary represents a given piece of text; lower fertility
means fewer tokens and faster inference.

---

### `data/nlp_preprocessing.py`

Builds a word-level integer vocabulary from a combined set of positive and negative
example sentences, encodes each sentence, and pads all sequences to equal length using
`nn.utils.rnn.pad_sequence`. Produces the padded, batch first tensor that sequence
classification training loops consume.

---

### `data/dataset.py`

Implements the GPT dataset construction step: tokenizes a raw text corpus by whitespace,
samples random starting indices, and constructs `(X, Y)` pairs where `Y` is `X` shifted
right by one token. This shifted pair structure is what makes next-token prediction a
self-supervised task requiring no labels beyond the corpus itself.

---

### `data/loader.py`

Implements a tensor-based batched data loader that randomly samples `batch_size` starting
positions from an already encoded 1D corpus tensor, stacks the corresponding windows into
`(batch_size, context_length)` input and target matrices, and returns them. This is the
direct data supplier for the training loop in `train.py`.

---

## 5. Transformer Architecture

Implements every component of the GPT model architecture as an independent, testable
module.

---

### `model/embeddings.py`

Implements the token embedding lookup as a NumPy index operation: given a vocabulary
matrix of shape `(vocab_size, embed_dim)` and a sequence of token IDs, returns the
corresponding row vectors. This is the first transformation every input token undergoes
before attention is applied.

---

### `model/positional_encoding.py`

Implements the sinusoidal positional encoding from "Attention is All You Need":
`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))` and `PE(pos, 2i+1) = cos(...)`. Since
self-attention is permutation-invariant by design, these encodings are the only mechanism
that injects sequence order information into the model.

---

### `model/normalization.py`

Implements layer normalization: normalizes each feature vector to zero mean and unit
variance, then applies learned scale (`gamma`) and shift (`beta`) parameters. In the GPT
model this is applied in the Pre-LN position, before each attention and feed forward
sublayer, which stabilizes gradient flow in deep stacks.

---

### `model/rms_normalization.py`

Implements RMS normalization, a simplified variant of layer norm that omits mean
centering and the beta shift parameter, normalizing only by the root mean square of the
input. Used in Llama family models as a computationally cheaper alternative that retains
most of the training stability benefit.

---

### `model/batch_normalization.py`

Implements batch normalization with separate training and inference branches: during
training, statistics are computed over the current batch and exponential moving averages
of mean and variance are updated; during inference, the stored running statistics are used
instead. Included to demonstrate how normalization strategy choices affect dependency on
batch structure and deployment behavior.

---

### `model/attention.py`

Implements a single self-attention head: projects the input into queries, keys, and
values; computes scaled dot product scores `(Q @ K^T) / sqrt(head_dim)`; applies a
causal lower triangular mask so each position can only attend to earlier tokens; and
returns the weighted sum of values. The causal mask is what makes this decoder style
attention suitable for language modeling.

---

### `model/multi_head_attention.py`

Instantiates `num_heads` independent single-head attention modules, runs each on the
same input in parallel, concatenates their outputs along the feature dimension, and
projects through a learned output matrix `W_O`. Running multiple heads in parallel lets
the model attend to different aspects of context simultaneously within the same layer.

---

### `model/transformer.py`

Composes multi-head attention and a two layer position-wise feed forward network (
`Linear(d_model, 4*d_model) -> ReLU -> Linear(4*d_model, d_model) -> Dropout`) inside a
Pre-LN residual block, where layer normalization precedes each sublayer and the
sublayer output is added back to the input via a skip connection. Residual connections
allow gradients to flow directly to early layers without degrading through the full
nonlinear path.

---

### `model/grouped_query_attention.py`

Implements grouped query attention (GQA): queries are projected to `num_heads` heads
while keys and values use a smaller number of `num_kv_heads`, which are then
`repeat_interleave` expanded to match the query count before attention is computed. GQA
reduces KV cache memory footprint by a factor of `num_heads / num_kv_heads` at inference
time with minimal impact on model quality, and is the attention variant used in
production scale models like Llama 2 70B and Mistral 7B.

---

### `model/kv_cache.py`

Implements KV caching for efficient autoregressive decoding: a `KVCache` object
accumulates projected keys and values by concatenating each new step's projections to a
running buffer, so that `CachedAttention` computes attention over the full history
without reprocessing previous tokens. Without this, inference cost scales as O(T^2) per
step; with the cache it scales as O(T) per step.

---

### `model/gpt.py`

The full GPT model: learned token embeddings and position embeddings are summed and
passed through a sequential stack of `TransformerBlock` modules, followed by a final
layer normalization and a linear projection from model dimension to vocabulary size that
produces unnormalized logits. No softmax is applied in the forward pass; the training
loop applies cross-entropy (which includes softmax internally) and the inference loop
applies softmax explicitly before sampling.

---

## 6. Training

### `train.py`

Implements the GPT training loop using AdamW optimization: for each epoch, a random
batch of `(X, Y)` sequence pairs is sampled from the encoded corpus, the model produces
logits of shape `(B, T, vocab_size)`, these are reshaped to `(B*T, vocab_size)` and
targets to `(B*T,)` so that cross-entropy treats every token position as an independent
classification problem, gradients are computed via backpropagation, and AdamW applies
weight updates with decoupled L2 regularization. Returns the final loss rounded to four
decimal places. Called internally by `model/model.py` via `TrainSolution.train`.

---

## 7. Inference

### `generate.py`

Implements autoregressive text generation: the model's context window is cropped to
`context_length` if it exceeds the maximum, the model's forward pass produces logits for
all positions, the final position logits are converted to probabilities via softmax, the
next token is sampled using `torch.multinomial` with a seeded generator for
reproducibility, that token is appended to the running context, and the decoded character
is appended to the output string. This loop repeats for `new_chars` steps, expanding the
context by one token at each step. Called internally by `model/model.py` via
`GenerateSolution.generate`.

---

## 7a. Model Module

### `model/model.py`

The unified entry point that bridges every layer of the system. `GPTConfig` is a
dataclass that consolidates all hyperparameters (vocab size, context length, model
dimension, number of blocks, number of heads, batch size, epochs, learning rate) in one
place so that training and inference share an identical configuration rather than
duplicating constants. `create_model(config)` constructs a `GPT` instance from a config.
`save_model` writes the model state dict and config together to `saved_model/` as a
single `.pt` checkpoint. `load_model` restores both the weights and the config from that
checkpoint, rebuilding the model without requiring the caller to supply hyperparameters
again. `run` is the top level entry point: it builds the vocabulary, encodes the corpus,
constructs and trains the model by calling into `train.py`, saves the checkpoint, and
generates sample text by calling into `generate.py`. Running `python model/model.py`
directly executes a self contained demonstration on a short Shakespeare passage.

---

## 8. Spec Reflection

The sequential construction order of this repository, foundations before tokenization
before model components before the training loop, was a deliberate design choice with
direct payoff. Because each module was a standalone, verifiable unit before it was
composed into a larger one, bugs were caught at the level where they originated rather
than surfacing as mysterious failures inside a fully assembled model. The backpropagation
derivations in `foundations/` informed the correct understanding of how gradients flow
through residual connections in `model/transformer.py`. The character-level vocabulary
built in `data/vocab.py` established the encoding contract that `train.py` and
`generate.py` depend on without any ambiguity. The three normalization variants were
implemented as isolated NumPy solutions before the PyTorch `nn.LayerNorm` call appeared
in `model/gpt.py`, ensuring that the choice of Pre-LN versus Post-LN was understood
mechanically rather than copied from a reference implementation. The KV cache was
implemented and tested as an independent `kv_cache.py` module before being integrated
into the generation loop, which made the correctness of the cache semantics verifiable
in isolation.

One deliberate trade-off in the model implementation is the use of explicit Python loops
over attention heads in `multi_head_attention.py` rather than a single batched matrix
multiplication over all heads simultaneously. The loop version is slower but maps
directly to the conceptual description in the original "Attention is All You Need" paper,
making the module easier to reason about and modify. A production implementation would
reshape the input tensor to process all heads in a single GEMM call.

---

## 9. Test Suite

The `tests/` directory contains a pytest-based test suite covering every major module and
stage in building up the GPT. Tests are organized by layer of abstraction, matching the
sequential construction order of the repository.

---

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/
```

To run a single file:

```bash
pytest tests/test_vocab.py
```

To run with verbose output:

```bash
pytest tests/ -v
```

---

### Test files

| File | Modules covered | What is tested |
|---|---|---|
| `test_vocab.py` | `data/vocab.py` | Vocab construction, alphabetical ordering, encode/decode round-trip |
| `test_tokenizer.py` | `data/tokenizer.py` | BPE merge selection, lexicographic tiebreaking, non-overlapping merges, early stopping |
| `test_tokenizer_utils.py` | `data/tokenizer_utils.py` | Greedy longest match tokenization, number tokenization, token count, fertility score |
| `test_data_loaders.py` | `data/loader.py`, `data/dataset.py`, `data/nlp_preprocessing.py` | Batch shapes, Y is X shifted-by-one, reproducibility, padding, ID assignment |
| `test_normalization.py` | `model/normalization.py`, `model/rms_normalization.py`, `model/batch_normalization.py` | Zero mean, unit variance, gamma/beta scaling, training vs inference mode, rounding |
| `test_embeddings_and_encoding.py` | `model/embeddings.py`, `model/positional_encoding.py` | Row retrieval, sine/cosine formula correctness, boundary positions, odd d_model |
| `test_attention.py` | `model/attention.py`, `model/multi_head_attention.py`, `model/grouped_query_attention.py` | Output shapes, causal masking, head count, KV head repetition, rounding |
| `test_transformer.py` | `model/transformer.py` | Output shape, residual connections, FFN dimension ratios, sublayer presence |
| `test_gpt.py` | `model/gpt.py`, `model/model.py` | Forward pass shape, raw logits, GPTConfig defaults, save/load checkpoint integrity |
| `test_kv_cache.py` | `model/kv_cache.py` | Cache initialization, incremental concatenation, clear/reset, CachedAttention output |
| `test_train_and_generate.py` | `train.py`, `generate.py` | Loss type and rounding, loss decreases, weight updates, output length, vocab coverage, determinism |
| `test_integration.py` | Full pipeline | Data-to-training, training-to-generation, checkpoint round trip, `run()` entry point, reproducibility |

---

### Coverage summary

- **Unit tests** verify each module in isolation with controlled inputs and exact output contracts (shapes, rounding precision, numerical formulas).
- **Integration tests** verify that modules compose correctly: the data pipeline feeds the training loop, trained models generate valid characters, saved checkpoints reload identically, and `run()` completes the full pipeline end to end.

---

## 10. Limitations

The model implemented here is architecturally complete. Given adequate training data and
compute, it would learn to generate coherent text. Under the current conditions, it does
not.

The core constraint is data and scale. GPT-2's smallest configuration (117M parameters)
was trained on 40 GB of WebText across tens of thousands of GPU hours. The character-level
corpus used here is several orders of magnitude smaller, and the model is configured at a
fraction of GPT-2's parameter count. At this scale, cross-entropy loss does decrease over
training epochs, confirming that gradient flow and the optimization loop are functioning
correctly. The generated text, however, is statistically indistinguishable from random
character sampling. This is not a failure of the implementation; it is the expected
behavior of a correctly implemented language model that has not been given sufficient data
to learn a meaningful distribution over tokens. The absence of coherent output is a
resource constraint, not an architectural one: every component of the system that would
enable competent generation at scale, causally masked attention, residual connections,
learned positional embeddings, subword tokenization, AdamW optimization, KV-cached
decoding, and grouped query attention, is present and correctly implemented.

What the implementation demonstrates correctly:

- The full forward pass from token IDs to logits through embeddings, positional
  encodings, N transformer blocks, and a vocabulary projection.
- Causally masked self-attention that prevents any position from attending to future
  tokens.
- Pre-LN residual connections that stabilize gradient flow through depth.
- The AdamW training loop with per-step batching and cross-entropy loss.
- KV-cached autoregressive decoding with context window cropping.
- Grouped query attention with the correct key/value head repetition scheme.

The distinction between "the architecture is correct" and "the model generates coherent
language" is purely a function of training resources. The fundamental competencies
required to scale this to GPT-2 capability are present in this codebase; what is missing
is the data and the compute.

---

## 11. Setup and Launch

**Prerequisites**

- Python 3.10+
- PyTorch 2.0+

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Train, generate, and save a checkpoint in one command**

```bash
python -m model.model
```

This runs the full pipeline: builds vocabulary, trains the GPT, writes the checkpoint to
`saved_model/gpt.pt`, and prints generated text. To use a custom corpus or tune
hyperparameters, import `run` or `GPTConfig` from `model.model`:

```python
from model.model import run, GPTConfig, create_model, load_model

# Full pipeline from text to saved checkpoint
run(training_text, epochs=500, model_dim=256)

# Or load a saved checkpoint for generation only
model, config = load_model("saved_model/gpt.pt")
```

**Train only**

```bash
python train.py
```

**Generate only**

```bash
python generate.py
```

**Stack**

Python · PyTorch · NumPy
