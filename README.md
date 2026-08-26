# HowLLMsWork

**A from-scratch, educational implementation of Transformer and Large Language Model fundamentals in Python.**

HowLLMsWork is a learning-oriented implementation that builds the core path from tokens to next-token generation step by step.

The goal is not to reproduce the scale or performance of production LLMs. The goal is to make the internal mechanics of a language model explicit, inspectable, testable, and understandable.

> **Focus:** understand the mathematics, data flow, training loop, autoregressive inference, sampling, and KV-cache mechanics by implementing them directly.

---

## 1. The Big Picture

The project follows the conceptual pipeline:

```
Text
  ↓
Tokenization
  ↓
Vocabulary
  ↓
Training examples
  ↓
Token embeddings
  ↓
Positional information
  ↓
Q / K / V projections
  ↓
Scaled dot-product attention
  ↓
Multi-head attention
  ↓
Residual connection + normalization
  ↓
Feed-forward network
  ↓
Transformer decoder
  ↓
Vocabulary projection
  ↓
Logits
  ↓
Cross-entropy loss
  ↓
Backpropagation
  ↓
Parameter updates
  ↓
Next-token prediction
  ↓
Autoregressive generation
  ↓
Sampling
  ↓
KV cache
  ↓
Prefill / decode
```

The repository implements these ideas as small Python components rather than hiding them behind a high-level deep-learning framework.

---

## 2. What Is Implemented

### Tokenization and data

- Vocabulary management
- Tokenization
- Language-model examples
- Causal next-token targets
- Batching

### Language models

- Simple context language model
- Positional context language model
- Transformer language model
- Cached Transformer language model

### Transformer internals

- Token embeddings
- Positional encoding
- Query / Key / Value projection
- Scaled dot-product attention
- Multi-head attention
- Residual connections
- Layer normalization
- Feed-forward networks
- Transformer decoder block
- Vocabulary projection

### Training

- Causal language-model objective
- Cross-entropy loss
- Gradient computation
- Backpropagation through Transformer components
- Parameter updates
- Training loops
- Evaluation

### Inference

- Next-token prediction
- Greedy generation
- Temperature sampling
- Top-K sampling
- Top-P / nucleus sampling
- Pluggable sampling strategies
- Generation backends

### Efficient autoregressive inference

- KV cache
- Cached attention
- Cached multi-head attention
- Cached Transformer backbone
- Prefill / decode API
- Cached generation
- Sliding context-window handling

---

## 3. The Mathematics

### 3.1 Token embeddings

A token ID is mapped to a learned vector:

$$e_i = E[i]$$

where:

- $E \in \mathbb{R}^{V \times d}$ is the embedding matrix
- $V$ is the vocabulary size
- $d$ is the model dimension
- $e_i \in \mathbb{R}^{d}$ is the representation of token $i$

For a sequence of $n$ tokens:

$$X = \begin{bmatrix} e_1 \\ e_2 \\ \vdots \\ e_n \end{bmatrix} \in \mathbb{R}^{n \times d}$$

---

### 3.2 Positional information

Self-attention by itself does not inherently encode token order, so positional information is added to token representations.

A classical sinusoidal positional encoding is:

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

and the input to the Transformer can be written as:

$$H^{(0)} = X + PE$$

The repository also contains simplified positional mechanisms in some educational components. The important idea is the same: the model needs information about **where** a token occurs.

---

### 3.3 Query, Key, and Value projections

For hidden states $X$, the attention projections are:

$$Q = XW_Q$$

$$K = XW_K$$

$$V = XW_V$$

where:

$$W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$$

and $d_h$ is the per-head dimension.

These projections create three different views of the same hidden states:

- **Query:** what this position is looking for
- **Key:** what this position offers for matching
- **Value:** what information this position contributes

---

### 3.4 Scaled dot-product attention

The core attention equation is:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

The scaling factor prevents the dot products from growing too large as the dimension increases.

For a causal decoder, future positions must not be visible. Conceptually this is implemented with a causal mask:

$$A = \text{softmax}\left(\frac{QK^\top + M}{\sqrt{d_h}}\right)$$

where masked future positions receive a value approaching $-\infty$ before softmax.

---

### 3.5 Softmax

Given logits $z_1,\dots,z_V$:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

The resulting probabilities satisfy:

$$0 \le p_i \le 1$$

and:

$$\sum_{i=1}^{V} p_i = 1$$

---

### 3.6 Multi-head attention

Instead of using one attention operation, a Transformer uses multiple heads:

$$\text{head}_i = \text{Attention}(Q_i,K_i,V_i)$$

The heads are concatenated:

$$H = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)$$

and projected:

$$O = HW_O$$

The project contains both a trainable multi-head implementation and a cached incremental version.

---

### 3.7 Residual connections

A sublayer output is combined with its input:

$$R = X + F(X)$$

This allows information to flow through the network while making deeper optimization easier.

The repository explicitly models residual operations because they are an important part of the Transformer computation graph.

---

### 3.8 Layer normalization

For a vector $x$ of dimension $d$:

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$$

$$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2$$

Then:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$$

where $\gamma$ and $\beta$ are learnable parameters.

---

### 3.9 Feed-forward network

A Transformer feed-forward block can be represented as:

$$\text{FFN}(x) = W_2 \, \phi(W_1x+b_1) + b_2$$

where $\phi$ is a non-linear activation.

Conceptually, the attention mechanism mixes information **between positions**, while the feed-forward network transforms each position's representation.

---

### 3.10 Transformer block

A simplified decoder block follows the pattern:

$$H_1 = \text{LayerNorm}\left(X + \text{Attention}(X)\right)$$

followed by:

$$H_2 = H_1 + \text{FFN}(H_1)$$

and another normalization step depending on the exact block formulation.

The project intentionally keeps these operations explicit so the intermediate values can be inspected.

---

### 3.11 Vocabulary projection

The final hidden representation is projected into vocabulary space:

$$Z = HW_{out} + b_{out}$$

where:

- $H \in \mathbb{R}^{n \times d}$
- $W_{out} \in \mathbb{R}^{d \times V}$

giving:

$$Z \in \mathbb{R}^{n \times V}$$

Each row contains the logits for predicting the next token at that position.

---

### 3.12 Causal language-model objective

For a sequence:

```
tokens:  x₁ x₂ x₃ x₄
```

the training objective shifts the sequence:

```
input:   x₁ x₂ x₃
target:  x₂ x₃ x₄
```

The model learns:

$$P(x_t \mid x_1,\ldots,x_{t-1})$$

for each position $t$.

The complete autoregressive objective is:

$$\mathcal{L} = -\sum_{t=1}^{n}\log P(x_t \mid x_{<t})$$

Usually the mean is taken over the training positions.

---

### 3.13 Cross-entropy loss

For a target token $y$ and predicted probability distribution $p$:

$$\mathcal{L} = -\log p_y$$

For a sequence of $n$ predictions:

$$\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n}\log p_{t,y_t}$$

A lower loss means the model assigns higher probability to the correct target tokens.

---

### 3.14 Backpropagation

Training computes gradients of the loss with respect to model parameters:

$$\frac{\partial \mathcal{L}}{\partial \theta}$$

The parameters are then updated using a simple gradient-descent rule:

$$\theta \leftarrow \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$$

where $\eta$ is the learning rate.

The repository includes explicit backward implementations for important Transformer components so the gradient flow can be studied rather than treated as a black box.

---

## 4. From Logits to a Generated Token

At inference time the model produces a vocabulary-sized logit vector:

$$z = [z_1,z_2,\ldots,z_V]$$

A deterministic approach chooses:

$$\hat{y} = \arg\max_i z_i$$

This is the basic greedy strategy.

But generation does not have to be deterministic.

---

### 4.1 Temperature sampling

Temperature rescales logits:

$$z_i' = \frac{z_i}{T}$$

and probabilities become:

$$p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

Interpretation:

- $T < 1$: sharper distribution
- $T = 1$: original distribution
- $T > 1$: flatter distribution

The project demonstrates this effect explicitly.

---

### 4.2 Top-K sampling

Top-K sampling keeps only the $K$ highest-scoring tokens.

Let $S_K$ be the set of those tokens:

$$S_K = \text{TopK}(z,K)$$

All other token probabilities are removed and the remaining probabilities are renormalized:

$$p_i = 0 \quad \text{for } i \notin S_K$$

Then sampling occurs only from the reduced candidate set.

---

### 4.3 Top-P sampling

Top-P, or nucleus sampling, sorts candidates by probability and retains the smallest set whose cumulative probability reaches $p$:

$$\sum_{i \in S_p} p_i \ge p$$

Tokens outside the nucleus are removed before sampling.

This allows the candidate set to adapt to the shape of the current distribution.

---

## 5. Autoregressive Generation

Suppose the prompt is:

```
the cat drinks milk
```

Generation proceeds iteratively:

```
prompt
   ↓
Transformer
   ↓
next-token logits
   ↓
sampling strategy
   ↓
new token
   ↓
append token
   ↓
repeat
```

Mathematically:

$$x_{t+1} \sim P(\cdot \mid x_1,\ldots,x_t)$$

The sequence is therefore generated one token at a time.

The repository contains both direct generation and cached generation paths.

---

## 6. KV Cache

A major cost of autoregressive generation is repeatedly recomputing keys and values for tokens that have already been processed.

Without a KV cache:

```
Step 1 → process token 1
Step 2 → process tokens 1..2 again
Step 3 → process tokens 1..3 again
Step 4 → process tokens 1..4 again
...
```

With a KV cache:

```
Prompt
  ↓
Prefill
  ↓
store K/V
  ↓
new token
  ↓
compute only new K/V
  ↓
append to cache
  ↓
attend against cached K/V
```

For each attention head, the cache stores:

$$K_{cache} = \begin{bmatrix} K_1 \\ K_2 \\ \vdots \\ K_t \end{bmatrix}$$

and:

$$V_{cache} = \begin{bmatrix} V_1 \\ V_2 \\ \vdots \\ V_t \end{bmatrix}$$

For the new query $Q_t$:

$$\text{Attention}(Q_t,K_{cache},V_{cache}) = \text{softmax}\left(\frac{Q_tK_{cache}^\top}{\sqrt{d_h}}\right)V_{cache}$$

The key point is that previous $K$ and $V$ vectors do not have to be recomputed for every generated token.

---

## 7. Prefill and Decode

The repository makes the two inference phases explicit.

### Prefill

The complete prompt is processed:

```
prompt = [x₁, x₂, x₃, x₄]
     ↓
cache contains K/V for all prompt positions
```

The engine returns the logits needed to generate the next token.

### Decode

Only a newly generated token is processed:

```
x₅
 ↓
new Q/K/V
 ↓
append K/V
 ↓
attend over cached history
 ↓
predict x₆
```

This gives the conceptual interface:

```
prefill(prompt)
      ↓
next logits
      ↓
decode(next_token)
      ↓
next logits
      ↓
decode(...)
```

---

## 8. Sliding Context Window

The cache cannot grow without bounds when a fixed context size is being enforced.

For a context size of 4:

```
[0, 1, 2, 3]
```

then, after generation:

```
[1, 2, 3, 4]
```

then:

```
[2, 3, 4, 5]
```

The repository keeps the low-level Transformer backbone bounded while the generation layer replays the latest context window when necessary.

This makes the distinction explicit:

```
Backbone
    ↓
fixed context contract

Generation layer
    ↓
sliding-window policy
```

That separation is intentional.

---

## 9. Project Architecture

```
HowLLMsWork/
│
├── src/
│   │
│   ├── tokenization/
│   │   ├── tokenizer.py
│   │   └── vocabulary.py
│   │
│   ├── attention/
│   │   ├── qkv_projection.py
│   │   ├── scaled_dot_product.py
│   │   ├── multi_head.py
│   │   ├── kv_cache.py
│   │   ├── cached_attention.py
│   │   └── cached_multi_head.py
│   │
│   ├── llm/
│   │   ├── language_model.py
│   │   ├── simple_language_model.py
│   │   ├── positional_language_model.py
│   │   ├── transformer_backbone.py
│   │   ├── transformer_language_model.py
│   │   ├── cached_transformer_backbone.py
│   │   └── cached_transformer_language_model.py
│   │
│   ├── training/
│   │   ├── dataset.py
│   │   ├── batch.py
│   │   ├── causal_examples.py
│   │   ├── language_model_objective.py
│   │   ├── language_model_training.py
│   │   ├── model_evaluation.py
│   │   ├── positional_language_model_training.py
│   │   ├── transformer_training_bridge.py
│   │   └── transformer_session_client.py
│   │
│   ├── inference/
│   │   ├── next_token.py
│   │   ├── sampling.py
│   │   ├── sampling_strategy.py
│   │   ├── top_k_sampling.py
│   │   ├── top_p_sampling.py
│   │   ├── transformer_inference.py
│   │   ├── cached_transformer_inference.py
│   │   ├── generator.py
│   │   ├── cached_generator.py
│   │   ├── prefill_decode.py
│   │   ├── generation_backend.py
│   │   ├── legacy_generation_backend.py
│   │   ├── cached_generation_backend.py
│   │   └── unified_generator.py
│   │
│   └── experiments/
│       └── step-by-step demonstrations
│
└── tests/
    └── automated unit and integration tests
```

---

## 10. Recommended Learning Order

The easiest way to understand the repository is to follow the concepts in this order.

### Step 1 — Tokenization

Start with:

```
src/tokenization/
src/experiments/tokenization_demo.py
```

Understand how text becomes token IDs.

### Step 2 — Causal prediction

Explore:

```
src/training/dataset.py
src/training/causal_examples.py
```

Understand why the input and target sequences are shifted.

### Step 3 — Simple language models

Explore:

```
src/llm/simple_language_model.py
src/llm/positional_language_model.py
```

This gives a simpler baseline before introducing attention.

### Step 4 — Attention mathematics

Study:

```
src/attention/qkv_projection.py
src/attention/scaled_dot_product.py
src/attention/multi_head.py
```

Recommended demonstrations:

```
qkv_projection_demo.py
scaled_attention_demo.py
multi_head_attention_demo.py
```

### Step 5 — Transformer language model

Move to:

```
src/llm/transformer_language_model.py
```

Then inspect the Transformer backbone and decoder components represented in the project.

### Step 6 — Training

Explore:

```
src/training/language_model_objective.py
src/training/language_model_training.py
src/training/transformer_training_bridge.py
```

The central loop is:

```
forward
  ↓
loss
  ↓
gradient
  ↓
parameter update
  ↓
repeat
```

### Step 7 — Next-token prediction

Explore:

```
src/inference/next_token.py
src/experiments/next_token_demo.py
```

### Step 8 — Generation and sampling

Study:

```
src/inference/generator.py
src/inference/sampling_strategy.py
```

Then compare:

```
Greedy
Temperature
Top-K
Top-P
```

### Step 9 — KV cache

Finally study:

```
src/attention/kv_cache.py
src/attention/cached_attention.py
src/attention/cached_multi_head.py
```

and:

```
src/inference/prefill_decode.py
src/inference/cached_generator.py
```

This is where the project moves from basic Transformer mechanics into the mechanics of efficient autoregressive inference.

---

## 11. Experiments

The repository contains focused executable demonstrations.

### Tokenization

```bash
python -m src.experiments.tokenization_demo
```

### Objective

```bash
python -m src.experiments.objective_demo
```

### Q/K/V projection

```bash
python -m src.experiments.qkv_projection_demo
```

### Scaled attention

```bash
python -m src.experiments.scaled_attention_demo
```

### Multi-head attention

```bash
python -m src.experiments.multi_head_attention_demo
```

### Next-token prediction

```bash
python -m src.experiments.next_token_demo
```

### Temperature sampling

```bash
python -m src.experiments.temperature_demo
```

### Top-K sampling

```bash
python -m src.experiments.top_k_demo
```

### Top-P sampling

```bash
python -m src.experiments.top_p_demo
```

### KV cache

```bash
python -m src.experiments.kv_cache_demo
```

### Prefill / decode

```bash
python -m src.experiments.prefill_decode_demo
```

### Cached generation

```bash
python -m src.experiments.cached_generation_demo
```

### Cached sampling

```bash
python -m src.experiments.cached_sampling_demo
```

### End-to-end training

```bash
python -m src.experiments.end_to_end_transformer_training
```

Other experiments are available under `src/experiments/`.

---

## 12. Testing

The project is heavily test-driven.

Run the complete test suite:

```bash
python -m pytest
```

The current repository contains **209 automated tests** covering:

- tokenization
- datasets
- causal examples
- model objectives
- Transformer components
- Q/K/V projection
- attention
- multi-head attention
- KV cache
- cached inference
- prefill/decode
- generation
- sampling
- sliding-window generation
- integration boundaries

---

## 13. Code Quality

The repository uses:

- Python 3.12+
- NumPy
- pytest
- Ruff
- mypy

Run all quality checks:

```bash
python -m pytest
python -m ruff check .
python -m mypy src
```

At the time this README was prepared, the repository baseline was:

```
209 passed
Ruff: clean
Mypy: clean
```

---

## 14. Design Philosophy

The project follows a few principles.

### Explicit computation

Important mathematical operations are implemented as explicit Python components rather than hidden behind a high-level model API.

### Small contracts

Major parts expose simple boundaries such as:

```
token_ids → hidden states
hidden states → logits
logits → token
```

### Testable components

Attention, sampling, cache behavior, training objectives, and generation paths are individually tested.

### Educational clarity

The project intentionally favors readability and inspectability over production-scale optimizations.

### Separation of concerns

The repository separates:

```
Model
Training
Inference
Generation
Sampling
Caching
```

This makes it easier to reason about where a particular behavior belongs.

---

## 15. What This Project Is Not

This is **not** a production-scale LLM.

It does not attempt to reproduce the scale or engineering complexity of systems such as GPT-class foundation models.

It does not provide:

- large-scale distributed training
- GPU kernels
- CUDA optimization
- tensor parallelism
- pipeline parallelism
- mixed-precision production training
- billion-parameter models
- web-scale datasets
- production inference serving
- fault-tolerant distributed infrastructure

Those are different engineering problems.

The purpose of this repository is to understand the **algorithmic and mathematical foundations** beneath those systems.

---

## 16. What Is Simplified

Some implementations in the project are intentionally simplified so that the underlying idea remains visible.

For example:

- datasets are small and educational
- dimensions are tiny compared with real LLMs
- some positional mechanisms are simplified
- some Transformer components are implemented for clarity rather than maximum performance
- generation backends are designed to demonstrate architecture rather than production serving
- the KV-cache sliding-window behavior uses replay at the generation layer rather than implementing a fully optimized production cache eviction strategy

These choices are deliberate.

The project is about understanding the mechanism, not pretending that a small NumPy implementation is equivalent to a production model stack.

---

## 17. Why Implement It From Scratch?

Using a high-level framework makes it easy to call a language model.

Building the important pieces yourself forces the full computation to remain visible:

```
token
  ↓
embedding
  ↓
position
  ↓
Q/K/V
  ↓
attention
  ↓
multi-head composition
  ↓
residual + normalization
  ↓
feed-forward
  ↓
logits
  ↓
probabilities
  ↓
loss
  ↓
gradients
  ↓
parameter update
```

That makes it possible to inspect not only the final prediction, but also the intermediate representations and transformations that produced it.

---

## 18. Current Project Status

The core educational implementation is complete enough to demonstrate the main Transformer-to-LLM pipeline:

- ✅ Tokenization
- ✅ Causal language modeling
- ✅ Embeddings
- ✅ Positional information
- ✅ Q/K/V projection
- ✅ Scaled dot-product attention
- ✅ Multi-head attention
- ✅ Residual connections
- ✅ Layer normalization
- ✅ Feed-forward network
- ✅ Transformer language model
- ✅ Cross-entropy objective
- ✅ Backpropagation
- ✅ Training
- ✅ Next-token prediction
- ✅ Greedy generation
- ✅ Temperature sampling
- ✅ Top-K sampling
- ✅ Top-P sampling
- ✅ KV cache
- ✅ Cached attention
- ✅ Prefill / decode
- ✅ Cached generation
- ✅ Sliding context-window generation
- ✅ Automated tests
- ✅ Static type checking
- ✅ Linting

---

## 19. A Compact Mental Model

If you remember only one thing from this repository, it should be this:

$$\boxed{\text{LLM} = \text{Transformer} + \text{Language-Model Objective} + \text{Training} + \text{Autoregressive Inference}}$$

And during generation:

$$\boxed{x_{t+1} \sim P(\cdot \mid x_1,\ldots,x_t)}$$

with Transformer inference providing the probability distribution and the generation strategy deciding how the next token is selected.

For efficient autoregressive decoding:

$$\boxed{\text{KV Cache} \Rightarrow \text{reuse previous Key/Value states}}$$

That is the central journey this repository is designed to make understandable.

---

## 20. License

Add an open-source license before publishing the repository publicly.

For example, choose an appropriate license such as MIT if that matches your intended usage and contribution model.

Before publishing, add a `LICENSE` file to the repository and update this section with the chosen license.

---

## Verification snapshot

This README describes the repository as it stood after the latest validation run:

```
Tests: 209 passed
Ruff: clean
Mypy: clean
```

The numbers should be refreshed whenever the implementation changes materially.
