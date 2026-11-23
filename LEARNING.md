# Learning Notes: LLM Fine-Tuning

Consolidated reference for all studied concepts. Organized to match the ROADMAP.md phase structure. Notes are written after discussion — not from memory or assumption.

---

## Part 1 — Fine-Tuning Fundamentals

### 1.1 The One Thing a Language Model Does

A language model has exactly one job:

> **Given everything so far, predict the most likely next token.**

That is it. Everything else — answering questions, writing code, summarizing documents — is this single operation repeated many times. Understanding this is the foundation for understanding everything that follows.

---

### 1.2 Tokens and Tokenization

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

### 1.3 Embeddings — From Token IDs to Vectors

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

### 1.4 Through the Transformer — Layer by Layer

Tokens pass through a stack of **transformer layers**. A 1.5B model might have 28 layers. Each layer does two things: **self-attention** and a **feed-forward network**.

#### Self-Attention — How Tokens Talk to Each Other

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

#### Causal Masking — Why Decoder Models Only Look Backwards

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

#### Multi-Head Attention

Attention is done in parallel with many different Q, K, V projections. Each is called a "head". A model might have 16 or 32 heads running in parallel.

Each head can learn different relationships — one for syntax, another for coreference, another for positional proximity. Their outputs are concatenated and projected back down.

#### Feed-Forward Network

After attention, each token's vector passes through a small two-layer neural network independently. This is where the model stores factual associations — the "knowledge" is mostly in these layers.

```
FFN(x) = activation( x · W₁ + b₁ ) · W₂ + b₂
```

The intermediate layer is typically larger than the hidden size. Qwen 2.5 1.5B (hidden size 1536) uses an intermediate size of 8960 with a SwiGLU gated activation.

#### Residual Connections and Layer Norm

Each sub-layer (attention and FFN) has a **residual connection** — the input is added back to the output. This keeps gradients flowing and prevents the signal from collapsing.

Modern models like Llama and Qwen use **Pre-RMSNorm** — normalization is applied before each sub-layer, not after the residual add.

```
output = x + SubLayer(RMSNorm(x))
```

---

### 1.5 The Full Inference Flow

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

### 1.6 Training — Where Learning Happens

Training runs the same forward pass, then adds three extra steps.

#### What Training Data Looks Like

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

#### Step 1 — Forward Pass

Same as inference: run input through the model, get a probability distribution.

#### Step 2 — Calculate Loss (Cross-Entropy)

Compare what the model predicted against what actually came next.

**Cross-entropy loss** looks at the probability the model assigned to the correct next token:

```
If the correct token is "Paris":
  P("Paris") = 0.94  → Loss is low  (model was confident and right)
  P("Paris") = 0.01  → Loss is high (model was wrong)

Loss = -log( P(correct token) )
```

The negative log means: if the model gives the right token probability 1.0, loss = 0. If it gives it probability near 0, loss → infinity. Total loss for a batch is the average across all positions.

#### Step 3 — Backpropagation

We know the loss — a single number saying how wrong the model was. But we need to know *which weights* caused the error and *which direction* to move them.

Backpropagation calculates a **gradient** for every single weight:

> "If you increase this weight by a tiny amount, the loss will go up by X (or down by X)"

It does this by working backwards through the network using the chain rule from calculus. PyTorch handles this automatically.

#### Step 4 — Gradient Descent

Once we have the gradient for every weight, update each weight:

```
new_weight = old_weight - (learning_rate × gradient)
```

If the gradient is positive (increasing the weight increases loss), we decrease the weight.
If the gradient is negative (increasing the weight decreases loss), we increase the weight.

The **learning rate** controls the size of each step. Too large → weights overshoot and training is unstable. Too small → training is very slow.

**Adam optimizer** extends this by keeping a running average of past gradients (momentum) and past squared gradients (variance), which makes the updates more stable and adaptive per weight.

---

### 1.7 Training vs Inference — Side by Side

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

### 1.8 What Fine-Tuning Is and What It Changes

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

### 1.9 Prompting vs RAG vs Fine-Tuning vs Tools

A framework for choosing the right approach:

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

### 1.10 Memory Cost of Training — Full Fine-Tuning

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

### 1.11 Overfitting

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

---

## Part 2 — LoRA

### 2.1 The Core Insight

LoRA (Low-Rank Adaptation) keeps every original weight in the model completely frozen and adds small trainable adapter matrices alongside selected layers. No original weight is ever touched. Only the adapters train.

**The key observation:** when you fine-tune a large model, the weight changes (deltas — how much each weight moves) have **very low rank**. Even though there are millions of weights, the information needed to adapt the model to a new task lives in a much smaller dimensional subspace.

LoRA exploits this by representing the weight update as a product of two small matrices instead of updating the full matrix directly.

---

### 2.2 How LoRA Works — The Math

Take any weight matrix W in the model — say the Query projection, size 1536 × 1536.

**Full fine-tuning would do:**
```
W_new = W_original + ΔW       (update the matrix directly — expensive)
```

**LoRA does:**
```
Keep W_original completely frozen — never updated, ever.

Add two new small matrices:
    A: shape (rank × input_dim)    — thin and wide, randomly initialized
    B: shape (output_dim × rank)   — tall and thin, initialized to ZERO

Effective computation:
    output = W · input  +  (α/r) · B · A · input
             ───────────   ──────────────────────
             frozen base       trainable adapter
```

**Why B starts at zero:** at step one, B·A = 0, so the adapter contributes nothing. The model starts behaving exactly like the base model. Training gradually moves B and A away from zero toward what reduces the loss. This means fine-tuning always starts from a stable baseline.

---

### 2.3 The Mathematical Trick — Why Two Matrices

The weight update ΔW for a 1536×1536 matrix = 2,359,296 numbers.

LoRA decomposes it as B × A:
```
B (1536 × rank) × A (rank × 1536) = ΔW (1536 × 1536)

At rank 16:
  B: 1536 × 16 =  24,576 numbers
  A: 16 × 1536 =  24,576 numbers
  Total:          49,152 numbers   ←  2% of the original 2.3M
```

This works because fine-tuning weight changes are genuinely low rank — the decomposition captures most of what matters.

---

### 2.4 Rank — What It Controls

Rank is the size of the bottleneck dimension — the "bandwidth" of change allowed:

| Rank | Parameters (1536×1536 layer) | Use case |
|---|---|---|
| 4 | ~12K | Very fast, simple tasks, may underfit |
| 8 | ~25K | Simple tasks, common default |
| 16 | ~50K | Most common starting point, good balance |
| 32 | ~100K | Complex tasks needing more capacity |
| 64 | ~200K | High capacity, approaches full fine-tuning |

Higher rank = more expressive = more parameters = more memory = slower training.

**Start at rank 16** for most tasks. Go lower if the task is simple or memory is tight. Go higher if rank 16 underfits (validation loss does not come down).

---

### 2.5 Alpha — What It Controls

Alpha (α) scales the adapter output. The effective scale is **α/r** (alpha divided by rank).

```
output = W · input  +  (α/r) · B · A · input
```

Dividing by r keeps the scale stable as rank changes. The common convention:

```
alpha = 2 × rank

rank=8,  alpha=16  →  scale = 2.0
rank=16, alpha=32  →  scale = 2.0
rank=32, alpha=64  →  scale = 2.0
```

This means the effective adapter contribution is consistent regardless of which rank you chose. You can then tune rank independently of the scaling.

**Practical implication:** if you are not sure, set `alpha = 2 * rank` and leave it alone while you experiment with rank.

---

### 2.6 Target Modules — Which Layers Get Adapters

Most commonly the attention projections in every transformer layer:
- `q_proj` (query)
- `k_proj` (key)
- `v_proj` (value)
- `o_proj` (output)

Optionally also the feed-forward layers (`up_proj`, `down_proj`, `gate_proj`). More layers = more capacity, more memory.

**Why attention projections first:** these are where contextual relationships are learned. They are the most task-sensitive parts of the model.

**Common PEFT configuration:**
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

---

### 2.7 Memory Comparison — LoRA vs Full Fine-Tuning (1.5B model)

Adapter parameter count at rank 16 targeting q/k/v/o across 28 layers (using 1536×1536 as a uniform approximation):
```
Per layer: 4 modules × (16×1536 + 1536×16) = 4 × 49,152 = 196,608 params
28 layers: 196,608 × 28 ≈ 5.5M adapter params
```

Memory breakdown:
```
                    Full FT         LoRA (rank 16, q/k/v/o, 28 layers)
                    ───────         ────────────────────────────────────
Base weights:        3 GB            3 GB    (same — frozen, still loads)
Gradients (FP16):    3 GB           ~11 MB   (5.5M × 2 bytes)
Master copy (FP32):  6 GB           ~22 MB   (5.5M × 4 bytes)
Adam momentum:       6 GB           ~22 MB   (5.5M × 4 bytes)
Adam variance:       6 GB           ~22 MB   (5.5M × 4 bytes)
─────────────────────────────────────────────────────────────────────
Total:             ~24 GB          ~3.1 GB
```

The frozen base still occupies memory — but needs no gradients or optimizer state. Only the adapter parameters do. Total adapter overhead (gradients + optimizer states) is ~77 MB, not gigabytes.

---

### 2.8 Saving, Loading, and Merging Adapters

**Saving:**
After training, the adapter (A and B matrices for each targeted layer) is saved. At rank 16 targeting q/k/v/o only on a 1.5B model, the adapter file is ~11 MB (5.5M params × 2 bytes FP16). Including FFN layers pushes it higher. The base model weights are not in this file.

**Loading — keeping separate:**
Load the base model, then load the adapter on top with `PeftModel.from_pretrained`. The adapter is applied at inference time. This allows one base model with many task adapters swapped on top.

**Loading — merging:**
Compute `W_new = W + (α/r) · B · A` and write it into W. Result is a standard model with no adapter overhead. Inference is identical speed to the base model. The adapter is now baked in and cannot be removed without reloading the base.

**When to merge:** when you have one primary task and want inference to be as fast as possible. When you want to distribute the model without exposing the adapter structure.

**When to keep separate:** when you have multiple task adapters, want to switch between them, or want to share just the adapter file rather than a full model copy.

---

### 2.9 LoRA Trade-Offs Summary

| Dimension | LoRA |
|---|---|
| Memory | Low — ~3.5 GB for 1.5B model at rank 16 |
| Speed | Fast — gradient work is tiny (adapters only) |
| Quality | Close to full fine-tuning on narrow tasks |
| Complexity | Low — one rank and alpha to set |
| Forgetting risk | Low — base weights never change |
| Output size | Small adapter file (MB), not a full model copy |
| Feasible on M1 16 GB | Yes — comfortable for 1.5B, possible for 3B |

---

## Part 3 — QLoRA

*Not yet formally documented. Topics studied conceptually are listed below. Full notes will be added after discussion and verification.*

### Topics Studied (to be formally documented)

- What quantization is — representing weights in fewer bits to save memory
- FP32, FP16, BF16 — what each sacrifices in precision
- 4-bit quantization and why it is challenging for weights
- NF4 (NormalFloat4) — why it is information-theoretically optimal for normally-distributed model weights
- Double quantization — quantizing the quantization constants themselves
- The dequantize-compute-quantize cycle — why QLoRA can still compute accurate gradients
- What is still in FP16 during QLoRA (the LoRA adapters and optimizer states)
- LoRA vs QLoRA — when to use each

### Placeholder Summary

QLoRA extends LoRA by loading the base model in 4-bit (instead of 16-bit), reducing base model memory from 3 GB to ~0.75 GB for a 1.5B model. The adapters still train in 16-bit. This enables fine-tuning much larger models (7B, 13B) on the same hardware.

The trade-off: 4-bit quantization introduces some precision loss in the base. For most tasks the quality difference from LoRA is small but measurable.

---

## Part 4 — Practical Implementation

*Not yet started. Notes will be added during Phase 4 implementation.*

---

## Part 5 — Evaluation and Experimentation

*Not yet started. Notes will be added during Phase 5.*

---

## Part 6 — Production Integration

*Not yet started. Notes will be added during Phase 6.*

---

## Quick Reference

### Key Concept Definitions

| Concept | One-line definition |
|---|---|
| Token | Smallest unit of text the model processes; mapped to an integer ID |
| Embedding | A dense vector representing a token; learned during training |
| Self-attention | Mechanism that computes relevance between all token pairs using Q, K, V matrices |
| Causal masking | Prevents tokens from attending to future positions; enforces left-to-right generation |
| Feed-forward layer | Per-token transformation after attention; stores most of the model's factual knowledge |
| Loss | A number measuring how wrong the model's prediction was; minimized during training |
| Cross-entropy | The specific loss function used for next-token prediction |
| Backpropagation | Algorithm that computes gradients by working backwards through the network |
| Gradient | Per-weight number saying how much that weight contributed to the loss |
| Learning rate | Controls how large each weight update step is |
| Adam | Optimizer that adapts the learning rate per weight using momentum and variance |
| Fine-tuning | Continuing training on a small task-specific dataset, starting from pretrained weights |
| SFT | Supervised Fine-Tuning — training on (prompt, correct response) pairs |
| PEFT | Parameter-Efficient Fine-Tuning — only training a small subset of parameters |
| LoRA | Low-Rank Adaptation — trains small A and B matrices alongside frozen base weights |
| Rank | LoRA bottleneck dimension; controls adapter expressiveness |
| Alpha | LoRA scale factor; effective scale is α/r |
| Adapter | The A and B matrices added by LoRA; can be saved separately from the base |
| Quantization | Representing weights in fewer bits (e.g. 4-bit instead of 16-bit) to save memory |
| QLoRA | LoRA on a 4-bit quantized base model |
| NF4 | NormalFloat4 — 4-bit data type optimized for normally-distributed model weights |
