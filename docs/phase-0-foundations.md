# Phase 0 — Foundations: How Language Models Work

**Status:** Complete  
**Goal:** Deep understanding of what an LLM is, how it processes input, how it learns, and what fine-tuning actually changes inside the model.

---

## 1. The One Thing a Language Model Does

A language model has exactly one job:

> **Given everything so far, predict the most likely next token.**

That's it. Everything else — answering questions, writing code, summarizing documents — is just this single operation repeated many times. Understanding this is the foundation for understanding everything that follows.

---

## 2. What is a Token?

Before the model sees any text, the text has to be converted into numbers. The model cannot read strings — it only works with numbers.

**Tokenization** is the process of splitting text into chunks called tokens, then mapping each chunk to a number from a fixed vocabulary.

### Real example

Take the sentence: `The capital of France is Paris`

The tokenizer might split it like this:

```
"The"     → 464
"capital" → 3361
"of"      → 286
"France"  → 4881
"is"      → 318
"Paris"   → 6342
```

Each number is a **token ID** — just an index into the model's vocabulary (which might have 32,000 to 150,000 entries depending on the model).

### Key points about tokens

- Tokens are not always full words. "unbelievable" might become `["un", "believ", "able"]` — three tokens
- Spaces are often part of the token: `" the"` and `"the"` can be different tokens
- Numbers, punctuation, code symbols all have their own tokens
- A rough rule of thumb: **1 token ≈ 0.75 words** in English

The tokenizer is a fixed lookup table — it does not learn or change. It is set when the model is built and never changes.

---

## 3. From Token IDs to Vectors — Embeddings

Token IDs are just integers. The model cannot do useful math on integers directly. So each token ID gets converted into a **dense vector** — a list of floating point numbers.

This is the **embedding layer** — a lookup table where each row is the vector for one token.

### What does a vector represent?

A vector of 1536 numbers (for example) is a point in 1536-dimensional space. Through training, tokens with similar meanings end up at similar positions in that space. 

Rough intuition: words that appear in similar contexts end up with similar vectors. This is classically demonstrated with standalone word vectors (word2vec, 2013):
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```
In modern LLM embedding layers the same principle holds — similar tokens cluster together — but the arithmetic is noisier because these embeddings are one layer inside a deep contextual model, not standalone distributional vectors.

### Embedding dimensions are model-specific

| Model | Hidden / Embedding dimension |
|---|---|
| Qwen 2.5 1.5B | 1536 |
| Llama 3.2 1B | 2048 |
| Llama 3.2 3B | 3072 |
| GPT-4 (estimated) | ~12288 |

Larger dimension = more capacity to represent meaning, but more memory and compute.

---

## 4. Through the Transformer — Layer by Layer

Once tokens are embedded, they pass through a stack of **transformer layers**. A 1.5B model might have 28 layers. Each layer does two things: **self-attention** and a **feed-forward network**.

### 4a. Self-Attention — How Tokens Talk to Each Other

This is the most important mechanism in modern LLMs. It is how the model figures out which tokens are relevant to each other.

**The problem it solves:** when processing the token "it" in the sentence "The cat sat because it was tired" — the model needs to know that "it" refers to "cat". Self-attention is what makes that connection.

**How it works — step by step:**

Each token's embedding is projected into three different vectors using three learned weight matrices:

```
Q (Query)  = embedding × W_Q    "What am I looking for?"
K (Key)    = embedding × W_K    "What do I contain?"
V (Value)  = embedding × W_V    "What do I actually pass forward?"
```

Then for each token, we compute how much attention it should pay to every other token:

```
Step 1: Score = Q · Kᵀ          (dot product of query with all keys)
Step 2: Scale = Score / √(d_k)   (divide by square root of dimension, for stability)
Step 3: Mask  = apply causal mask (set future positions to -infinity)
Step 4: Weights = softmax(Scale)  (convert to probabilities that sum to 1)
Step 5: Output = Weights · V      (weighted sum of value vectors)
```

The output for each token is now a mix of all previous tokens' information, weighted by how relevant they are.

### Causal Masking — Why Decoder Models Only Look Backwards

The models we work with (Qwen, Llama, GPT) are **decoder-only** models. They are designed to generate text left to right. When processing token 5, they must not be allowed to "see" token 6, 7, 8 — that would be cheating.

This is enforced by **causal masking**: before softmax, we set all future positions to -∞. After softmax, -∞ becomes 0 — those tokens get zero attention weight.

```
Attention mask for a 5-token sequence:

Token  1  2  3  4  5
  1  [ ✓  ✗  ✗  ✗  ✗ ]    Token 1 can only see itself
  2  [ ✓  ✓  ✗  ✗  ✗ ]    Token 2 can see 1 and itself
  3  [ ✓  ✓  ✓  ✗  ✗ ]    Token 3 can see 1, 2, itself
  4  [ ✓  ✓  ✓  ✓  ✗ ]
  5  [ ✓  ✓  ✓  ✓  ✓ ]    Last token sees everything
```

This is why generation is left-to-right — each new token can only build on what came before.

### Multi-Head Attention

In practice, attention is not done once — it is done in parallel with many different Q, K, V projections. Each is called a "head". A model might have 16 or 32 heads running in parallel.

Each head can learn to attend to different kinds of relationships — one head might focus on syntax, another on coreference, another on positional proximity. Their outputs are concatenated and projected back down.

### 4b. Feed-Forward Network

After attention, each token's vector passes through a small two-layer neural network independently. This is where the model stores factual associations — the "knowledge" is mostly in these layers.

```
FFN(x) = activation( x · W₁ + b₁ ) · W₂ + b₂
```

The intermediate layer is typically larger than the hidden size. Modern models vary — Qwen 2.5 1.5B (hidden size 1536) uses an intermediate size of 8960 with a SwiGLU gated activation (two up-projection matrices) rather than the classic 4× rule from the original Transformer paper.

### Layer Norm and Residual Connections

Each sub-layer (attention and FFN) has a **residual connection** — the input is added back to the output. This keeps gradients flowing during training and prevents the signal from collapsing.

**Layer normalization** stabilizes the values before each sub-layer.

```
output = x + SubLayer(RMSNorm(x))
```

Modern models like Llama and Qwen use **Pre-RMSNorm** — the normalisation is applied before each sub-layer, not after the residual add. This differs from the Post-LayerNorm in the original 2017 Transformer paper. RMSNorm is a simpler, faster variant of LayerNorm.

---

## 5. The Full Inference Flow

```
Input text: "The capital of France is"
        │
        ▼
┌─────────────────────┐
│     TOKENIZER       │  "The"→464, "capital"→3361, "of"→286, "France"→4881, "is"→318
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   EMBEDDING LAYER   │  Each token ID → vector of 1536 floats
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   TRANSFORMER       │
│   LAYER 1           │  Self-attention + FFN
│   LAYER 2           │  Self-attention + FFN
│   ...               │
│   LAYER 28          │  Self-attention + FFN
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   OUTPUT PROJECTION │  Final vector → logits (one number per vocab token)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      SOFTMAX        │  Logits → probabilities (all sum to 1.0)
└─────────────────────┘
        │
        ▼
   "Paris" (probability 0.94)
   "Lyon"  (probability 0.02)
   "Rome"  (probability 0.01)
   ...
        │
        ▼
   Select next token → "Paris"
        │
        ▼
   Append to input → "The capital of France is Paris"
   Repeat from tokenizer...
```

**Key point:** during all of this, no weights change. The model is frozen. This is **inference**.

---

## 6. Training — Where Learning Happens

Training runs the same forward pass, but then adds three extra steps.

### What training data looks like

A training example is just text. During pretraining, the model is shown something like:

```
"The capital of France is Paris and the official language is French"
```

It is asked to predict each token given the previous ones:

```
Input:  "The"                     → Predict: "capital"
Input:  "The capital"             → Predict: "of"
Input:  "The capital of"          → Predict: "France"
Input:  "The capital of France"   → Predict: "is"
Input:  "The capital of France is"→ Predict: "Paris"
...
```

One sequence of N tokens gives N-1 training examples. This is extremely data-efficient.

### Step 1 — Forward Pass (same as inference)

Run the input through the model, get a probability distribution over the vocabulary.

### Step 2 — Calculate Loss

Compare what the model predicted against what actually came next in the data.

**Cross-entropy loss** measures this. For each position, it looks at the probability the model assigned to the correct next token:

```
If the correct token is "Paris" and the model said:
  P("Paris") = 0.94  → Loss is low  (model was confident and right)
  P("Paris") = 0.01  → Loss is high (model was confident but wrong)

Loss = -log( P(correct token) )
```

The negative log means: if the model gives the right token probability 1.0, loss = 0. If it gives it probability near 0, loss → infinity.

The total loss for a batch is the average across all positions.

### Step 3 — Backpropagation

Now we know the loss — a single number saying how wrong the model was. But we need to know *which weights* caused the error and *which direction* to move them.

Backpropagation calculates a **gradient** for every single weight — a number saying:

> "If you increase this weight by a tiny amount, the loss will go up by X (or down by X)"

It does this by working backwards through the network using the chain rule from calculus. This is automatic — PyTorch and MLX handle it.

### Step 4 — Gradient Descent

Once we have the gradient for every weight, we update each weight:

```
new_weight = old_weight - (learning_rate × gradient)
```

If the gradient is positive (increasing the weight increases loss), we decrease the weight.
If the gradient is negative (increasing the weight decreases loss), we increase the weight.

The **learning rate** controls the size of each step. Too large → weights overshoot and training is unstable. Too small → training is very slow.

**Adam optimizer** extends this by keeping a running average of past gradients (momentum) and past squared gradients (variance), which makes the updates more stable and adaptive per weight.

---

## 7. The Full Training Flow

```
Training data: "The capital of France is Paris"
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   FORWARD PASS                      │
│                                                     │
│  Tokenize → Embed → 28 Transformer Layers → Logits │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│   LOSS CALCULATION  │  Compare predicted vs actual next tokens
│   Cross-entropy     │  One loss value per token position
│   Average → scalar  │  Single number: how wrong was the model?
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  BACKPROPAGATION    │  Work backwards through every layer
│                     │  Compute gradient for every weight
│                     │  "How much did each weight contribute to this error?"
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  WEIGHT UPDATE      │  new_weight = old_weight - lr × gradient
│  (Adam optimizer)   │  Apply to every trainable weight
└─────────────────────┘
        │
        ▼
   Loss goes down slightly
   Repeat with next batch...
   After millions of batches: model predicts well
```

---

## 8. Inference vs Training — Side by Side

| | Inference | Training |
|---|---|---|
| Weights | Frozen — never change | Updated after every batch |
| Loss | Not calculated | Calculated at every step |
| Backpropagation | Does not happen | Happens after every forward pass |
| Gradients | Not stored | Stored for every trainable weight |
| Memory | Weights only | Weights + gradients + optimizer states |
| Speed | Fast | Much slower |
| Goal | Produce output | Reduce loss |

---

## 9. What Fine-Tuning Is and What It Changes

Pretraining trains the model on trillions of tokens for months. Fine-tuning continues training on a small, specific dataset — thousands to millions of examples — for hours.

**What changes:**
- Weight values shift slightly toward the patterns in the fine-tuning data
- The model's output distribution for specific inputs changes
- It becomes more likely to respond in a particular format, style or domain

**What does not change:**
- The architecture (number of layers, attention heads, dimensions)
- The tokenizer and vocabulary
- The general language understanding built in pretraining
- The total number of parameters

**What fine-tuning is not:**
- It does not "add" knowledge like inserting rows into a database
- It does not guarantee the model will learn a new fact reliably
- It shifts probabilities — it makes certain outputs more likely for certain inputs

**Why start from pretrained instead of scratch:**

| | Pretraining | Fine-tuning |
|---|---|---|
| Data | Trillions of tokens | Thousands to millions of examples |
| Compute | Months on hundreds of GPUs | Hours on one machine |
| Cost | Millions of dollars | Near zero |
| What it builds | General language understanding | Task-specific behaviour |

---

## 10. Memory Cost of Training

Understanding where memory goes matters for every decision in this project.

For full fine-tuning with Adam optimizer in mixed precision:

```
Per parameter cost breakdown:
  FP16 weights          = 2 bytes
  FP16 gradients        = 2 bytes
  FP32 master weights   = 4 bytes   (kept for numerical stability)
  FP32 Adam momentum    = 4 bytes
  FP32 Adam variance    = 4 bytes
  ─────────────────────────────────
  Total                 ≈ 16 bytes per parameter

  + activations (depends on batch size and sequence length)
```

For a **1.5B parameter model**:
```
  16 bytes × 1,500,000,000 = 24,000,000,000 bytes = ~24 GB
```

Your machine has 16 GB total. macOS takes ~2-3 GB. You have ~12-13 GB usable.

Full fine-tuning of 1.5B model = impossible on this machine.

This is exactly why LoRA and QLoRA exist — which is Phase 1.

---

## Summary of Key Concepts

| Concept | One-line definition |
|---|---|
| Token | Smallest unit of text the model processes; mapped to an integer ID |
| Embedding | A dense vector representing a token; learned during training |
| Self-attention | Mechanism that computes relevance between all token pairs using Q, K, V matrices |
| Causal masking | Prevents tokens from attending to future positions; enforces left-to-right generation |
| Feed-forward layer | Per-token transformation after attention; stores most of the model's factual knowledge |
| Loss | A number measuring how wrong the model's prediction was; minimised during training |
| Cross-entropy | The specific loss function used for next-token prediction |
| Backpropagation | Algorithm that computes gradients by working backwards through the network |
| Gradient | Per-weight number saying how much that weight contributed to the loss |
| Learning rate | Controls how large each weight update step is |
| Adam | Optimizer that adapts the learning rate per weight using momentum and variance |
| Fine-tuning | Continuing training on a small task-specific dataset, starting from pretrained weights |
