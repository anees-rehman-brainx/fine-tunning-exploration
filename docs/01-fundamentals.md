# Phase 1 — Fine-Tuning Fundamentals

**Status:** UNDERSTOOD  
**Goal:** Deep understanding of what an LLM is, how it processes input, how it learns, and what fine-tuning actually changes inside the model.

---

## 1.1 The One Thing a Language Model Does

A language model has exactly one job:

> **Given everything so far, predict the most likely next token.**

That is it. Everything else — answering questions, writing code, summarizing documents — is this single operation repeated many times. Understanding this is the foundation for understanding everything that follows.

---

## 1.2 Tokens and Tokenization

Before the model sees any text, the text has to be converted into numbers. The model cannot read strings — it only works with numbers.

**Tokenization** is the process of splitting text into chunks called tokens, then mapping each chunk to a number from a fixed vocabulary.

**Example:** `The capital of France is Paris`

```
"The"     → 464
"capital" → 3361
"of"      → 286
"France"  → 4881
"is"      → 318
"Paris"   → 6342
```

Each number is a **token ID** — just an index into the model's vocabulary (which might have 32,000 to 150,000 entries).

Key points:
- Tokens are not always full words. "unbelievable" might become `["un", "believ", "able"]` — three tokens
- Spaces are often part of the token: `" the"` and `"the"` can be different tokens
- Numbers, punctuation, and code symbols all have their own tokens
- Rough rule of thumb: **1 token ≈ 0.75 words** in English
- The tokenizer is a fixed lookup table — it does not learn or change

---

## 1.3 Embeddings — From Token IDs to Vectors

Token IDs are integers. The model cannot do useful math on integers directly. Each token ID gets converted into a **dense vector** — a list of floating point numbers.

This is the **embedding layer**: a lookup table where each row is the vector for one token.

A vector of 1536 numbers (for example) is a point in 1536-dimensional space. Through training, tokens with similar meanings end up at similar positions in that space.

Classic word-vector intuition:
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

In modern LLM embeddings the same principle holds, but the arithmetic is noisier because these embeddings are one layer inside a deep contextual model.

**Embedding dimensions by model:**

| Model | Hidden / Embedding dimension |
|---|---|
| Qwen 2.5 1.5B | 1536 |
| Llama 3.2 1B | 2048 |
| Llama 3.2 3B | 3072 |
| GPT-4 (estimated) | ~12288 |

Larger dimension = more capacity to represent meaning, but more memory and compute.

---

## 1.4 Through the Transformer — Layer by Layer

Tokens pass through a stack of **transformer layers**. A 1.5B model might have 28 layers. Each layer does two things: **self-attention** and a **feed-forward network**.

### Self-Attention — How Tokens Talk to Each Other

This is the most important mechanism in modern LLMs. It is how the model figures out which tokens are relevant to each other.

**The problem it solves:** when processing "it" in "The cat sat because it was tired" — the model needs to know that "it" refers to "cat". Self-attention makes that connection.

**How it works:**

Each token's embedding is projected into three vectors using three learned weight matrices:

```
Q (Query)  = embedding × W_Q    "What am I looking for?"
K (Key)    = embedding × W_K    "What do I contain?"
V (Value)  = embedding × W_V    "What do I actually pass forward?"
```

Then for each token, compute how much attention it should pay to every other token:

```
Step 1: Score = Q · Kᵀ          (dot product of query with all keys)
Step 2: Scale = Score / √(d_k)   (divide by square root of dimension, for stability)
Step 3: Mask  = apply causal mask (set future positions to -infinity)
Step 4: Weights = softmax(Scale)  (convert to probabilities that sum to 1)
Step 5: Output = Weights · V      (weighted sum of value vectors)
```

The output for each token is a mix of all previous tokens' information, weighted by how relevant they are.

### Causal Masking — Why Decoder Models Only Look Backwards

The models we work with (Qwen, Llama, GPT) are **decoder-only** models. They generate text left to right. When processing token 5, they must not be allowed to see token 6, 7, 8 — that would be cheating.

**Causal masking:** before softmax, set all future positions to -∞. After softmax, -∞ becomes 0 — those tokens get zero attention weight.

```
Attention mask for a 5-token sequence:

Token  1  2  3  4  5
  1  [ ✓  ✗  ✗  ✗  ✗ ]    Token 1 can only see itself
  2  [ ✓  ✓  ✗  ✗  ✗ ]    Token 2 can see 1 and itself
  3  [ ✓  ✓  ✓  ✗  ✗ ]
  4  [ ✓  ✓  ✓  ✓  ✗ ]
  5  [ ✓  ✓  ✓  ✓  ✓ ]    Last token sees everything
```

### Multi-Head Attention

Attention is done in parallel with many different Q, K, V projections. Each is called a "head". A model might have 16 or 32 heads running in parallel.

Each head can learn different relationships — one for syntax, another for coreference, another for positional proximity. Their outputs are concatenated and projected back down.

### Feed-Forward Network

After attention, each token's vector passes through a small two-layer neural network independently. This is where the model stores factual associations — the "knowledge" is mostly in these layers.

```
FFN(x) = activation( x · W₁ + b₁ ) · W₂ + b₂
```

The intermediate layer is typically larger than the hidden size. Qwen 2.5 1.5B (hidden size 1536) uses an intermediate size of 8960 with a SwiGLU gated activation.

### Residual Connections and Layer Norm

Each sub-layer (attention and FFN) has a **residual connection** — the input is added back to the output. This keeps gradients flowing and prevents the signal from collapsing.

Modern models like Llama and Qwen use **Pre-RMSNorm** — normalization is applied before each sub-layer, not after the residual add.

```
output = x + SubLayer(RMSNorm(x))
```

---

## 1.5 The Full Inference Flow

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
   Append to input → "The capital of France is Paris"
   Repeat...
```

During all of this, no weights change. The model is frozen. This is **inference**.

---

## 1.6 Training — Where Learning Happens

Training runs the same forward pass, then adds three extra steps.

### What Training Data Looks Like

A training example is text. The model is shown:

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

### Step 1 — Forward Pass

Same as inference: run input through the model, get a probability distribution.

### Step 2 — Calculate Loss (Cross-Entropy)

Compare what the model predicted against what actually came next.

**Cross-entropy loss** looks at the probability the model assigned to the correct next token:

```
If the correct token is "Paris":
  P("Paris") = 0.94  → Loss is low  (model was confident and right)
  P("Paris") = 0.01  → Loss is high (model was wrong)

Loss = -log( P(correct token) )
```

The negative log means: if the model gives the right token probability 1.0, loss = 0. If it gives it probability near 0, loss → infinity. Total loss for a batch is the average across all positions.

### Step 3 — Backpropagation

We know the loss — a single number saying how wrong the model was. But we need to know *which weights* caused the error and *which direction* to move them.

Backpropagation calculates a **gradient** for every single weight:

> "If you increase this weight by a tiny amount, the loss will go up by X (or down by X)"

It does this by working backwards through the network using the chain rule from calculus. PyTorch handles this automatically.

### Step 4 — Gradient Descent

Once we have the gradient for every weight, update each weight:

```
new_weight = old_weight - (learning_rate × gradient)
```

If the gradient is positive (increasing the weight increases loss), we decrease the weight.
If the gradient is negative (increasing the weight decreases loss), we increase the weight.

The **learning rate** controls the size of each step. Too large → weights overshoot and training is unstable. Too small → training is very slow.

**Adam optimizer** extends this by keeping a running average of past gradients (momentum) and past squared gradients (variance), which makes the updates more stable and adaptive per weight.

---

## 1.7 Training vs Inference — Side by Side

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

## 1.8 What Fine-Tuning Is and What It Changes

Pretraining trains the model on trillions of tokens for months. Fine-tuning continues training on a small, specific dataset — thousands to millions of examples — for hours.

**What changes:**
- Weight values shift slightly toward patterns in the fine-tuning data
- The model's output distribution for specific inputs changes
- It becomes more likely to respond in a particular format, style, or domain

**What does not change:**
- Architecture (number of layers, attention heads, dimensions)
- Tokenizer and vocabulary
- General language understanding built in pretraining
- Total number of parameters

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

## 1.9 Prompting vs RAG vs Fine-Tuning vs Tools

| Approach | Use when |
|---|---|
| **Prompting** | The model already knows how to do the task; you just need to instruct it correctly. Zero-shot or few-shot. No training required. |
| **RAG** | The model needs access to specific documents, current information, or domain knowledge it cannot memorize. Retrieval at inference time. |
| **Tools** | The model needs live data: databases, APIs, calculated results, real-time information. Function calling at inference time. |
| **Fine-tuning** | The model needs to change its behavior: format, style, tone, domain-specific response patterns, following specific instructions consistently. |

**Fine-tuning is not the right choice when:**
- The base model already does the task well with a good prompt
- You need current information (use RAG or tools instead)
- You have very few examples (use few-shot prompting instead)
- You want the model to know new facts (fine-tuning is unreliable for this — the model memorizes patterns, not facts precisely)

**Fine-tuning is the right choice when:**
- You need a consistent output format or style across many different inputs
- You want the model to follow a specific instruction style that is hard to express in a prompt
- You are building a domain-specific assistant with a large high-quality dataset
- You need faster/cheaper inference without long system prompts

---

## 1.10 Memory Cost of Training — Full Fine-Tuning

For full fine-tuning with Adam optimizer in mixed precision:

```
Per parameter cost:
  FP16 weights          = 2 bytes
  FP16 gradients        = 2 bytes
  FP32 master weights   = 4 bytes   (kept for numerical stability during updates)
  FP32 Adam momentum    = 4 bytes
  FP32 Adam variance    = 4 bytes
  ─────────────────────────────────
  Total                 ≈ 16 bytes per parameter

  + activations (scales with batch size and sequence length)
```

For a **1.5B parameter model**:
```
  16 bytes × 1,500,000,000 = 24,000,000,000 bytes = ~24 GB
```

Machine has 16 GB total. macOS takes ~2–3 GB. Usable: ~12–13 GB.

Full fine-tuning of 1.5B model = impossible on this machine. This is exactly why LoRA and QLoRA exist.

---

## 1.11 Overfitting

Overfitting happens when the model learns the training data too well — it memorizes specific examples rather than learning generalizable patterns.

**Signs:**
- Training loss keeps going down
- Validation loss stops going down or starts going up

**Why it matters:** a model that overfits performs well on training data but poorly on new prompts.

**Prevention:**
- Enough training data
- Validation set that is never used for training
- Early stopping — stop training when validation loss stops improving
- Lower learning rate
- Fewer epochs

**Key distinction:** training loss and validation loss should both go down during healthy training. When they diverge, that is an overfitting signal.
