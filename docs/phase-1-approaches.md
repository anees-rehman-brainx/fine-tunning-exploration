# Phase 1 — Fine-Tuning Approaches

**Status:** In Progress  
**Goal:** Deep understanding of all four fine-tuning approaches — full fine-tuning, LoRA, QLoRA, and DPO — with trade-offs in memory, speed, quality and complexity.

---

## What Fine-Tuning Is

A pretrained model has seen trillions of tokens. It knows language, facts, and reasoning — but it has no preference for how to respond. It was trained on one objective only: predict the next token on massive general text.

Fine-tuning continues training on a small, specific dataset to teach the model a particular behaviour, format, or skill. It does not add knowledge like a database insert — it shifts the probability distributions so the model becomes more likely to respond in a specific way for specific inputs.

**What changes:** weight values shift slightly toward patterns in the fine-tuning data.  
**What stays the same:** architecture, tokenizer, vocabulary, general language understanding.

---

## The Four Approaches — Overview

```
                    FINE-TUNING APPROACHES
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
   What you train:   What you train:   What you train on:
   ALL weights       SOME weights      PREFERENCE PAIRS
          │                │                 │
   Full Fine-Tuning   LoRA / QLoRA          DPO
                           │
                    ┌──────┴──────┐
                    │             │
              Base in 16-bit  Base in 4-bit
                    │             │
                  LoRA          QLoRA
```

Three fundamentally different ideas:
1. **Full fine-tuning** — update every weight in the model
2. **LoRA / QLoRA** — freeze the base, train tiny adapter matrices (differ only in how the base is stored in memory)
3. **DPO** — do not train on correct answers, train on which of two answers is better

---

## Approach 1 — Full Fine-Tuning

### What it is

The most straightforward approach. Every single weight in the model is updated during training — exactly like pretraining, just on a smaller task-specific dataset.

### What has to live in memory simultaneously

Training is not just loading the model. For every single weight, 5 things must exist in memory at the same time:

| What | Precision | Bytes per param | Why it exists |
|---|---|---|---|
| Weights | FP16 | 2 | The model itself |
| Gradients | FP16 | 2 | Backprop result — how much each weight contributed to the loss |
| Master weight copy | FP32 | 4 | FP16 is too imprecise for update math — optimizer keeps a full copy |
| Adam momentum | FP32 | 4 | Running average of past gradients — smooths noisy updates |
| Adam variance | FP32 | 4 | Running average of squared gradients — adapts learning rate per weight |
| **Total** | | **~16 bytes** | Before activations |

**Plus activations** — every intermediate value from the forward pass must be kept for backpropagation. These scale with batch size and sequence length.

### Real numbers for a 1.5B model

```
Weights (FP16):          1.5B × 2 bytes  =   3 GB
Gradients (FP16):        1.5B × 2 bytes  =   3 GB
Master copy (FP32):      1.5B × 4 bytes  =   6 GB
Adam momentum (FP32):    1.5B × 4 bytes  =   6 GB
Adam variance (FP32):    1.5B × 4 bytes  =   6 GB
─────────────────────────────────────────────────
Static total:                              24 GB
```

Machine available: ~12 GB usable on 16 GB M1. Full fine-tuning of 1.5B = not possible.

### Problems beyond memory

**Catastrophic forgetting:** when every weight updates on a narrow dataset, the model optimizes hard for the new task and overwrites what it knew before. Fine-tune on SQL only → the model gets worse at Python, general reasoning, everything else.

**Storage cost:** every fine-tuned version is a full copy of all weights. 5 task-specific models = 5 × 3 GB = 15 GB of files. Scales poorly.

### When full fine-tuning is used

- Large compute budgets (datacenter A100s / H100s where 24 GB is trivial)
- Task is very different from pretraining (e.g. medical imaging reports, legal documents)
- Large dataset where LoRA would underfit
- Maximum quality is required and budget allows

### Trade-offs

| Dimension | Full Fine-Tuning |
|---|---|
| Memory | Very high — ~16 bytes/param (~24 GB for 1.5B) |
| Speed | Slowest — gradients and optimizer updates for all weights |
| Quality | Highest ceiling — every weight can move |
| Complexity | Low code, high infrastructure |
| Forgetting risk | Highest |
| Output size | Full model copy per task |
| Feasible on M1 16GB | ❌ Not above ~500M params |

---

## Approach 2 — LoRA (Low-Rank Adaptation)

### What it is

LoRA keeps every original weight in the model completely frozen and adds small trainable adapter matrices alongside selected layers. No original weight is ever touched. Only the adapters train.

### The key insight

When you fine-tune a large model, the weight changes (deltas — how much each weight moves) have **very low rank**. Even though there are millions of weights, the information needed to adapt the model to a new task lives in a much smaller dimensional space. LoRA exploits this by representing the weight update as a product of two small matrices instead of updating the full matrix directly.

### How it works

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

**Why B starts at zero:** at step one, B·A = 0, so the adapter contributes nothing. The model starts behaving exactly like the base model. Training gradually moves B and A away from zero toward what reduces the loss.

### The mathematical trick — why two matrices

The weight update ΔW for a 1536×1536 matrix = 2.3 million numbers.

LoRA decomposes it as B × A:
```
B (1536 × rank) × A (rank × 1536) = ΔW (1536 × 1536)

At rank 16:
  B: 1536 × 16 =  24,576 numbers
  A: 16 × 1536 =  24,576 numbers
  Total:          49,152 numbers   ←  2% of the original 2.3M
```

This works because fine-tuning weight changes are genuinely low rank — the decomposition captures most of what matters.

### What rank controls

Rank is the size of the bottleneck dimension — the "bandwidth" of change allowed:

| Rank | Parameters (1536×1536 layer) | Use case |
|---|---|---|
| 4 | ~12K | Very fast, simple tasks, may underfit |
| 8 | ~25K | Simple tasks, common default |
| 16 | ~50K | Most common starting point, good balance |
| 32 | ~100K | Complex tasks needing more capacity |
| 64 | ~200K | High capacity, approaches full fine-tuning |

Higher rank = more expressive = more parameters = more memory.

### What alpha controls

Alpha (α) scales the adapter output. The effective scale is α/r.

```
output = W · input  +  (α/r) · B · A · input
```

Dividing by r keeps the scale stable as rank changes. Common starting point: **alpha = 2 × rank**.

```
rank=8,  alpha=16  →  scale = 2.0
rank=16, alpha=32  →  scale = 2.0
rank=32, alpha=64  →  scale = 2.0
```

### Which layers get adapters

Most commonly the attention projections in every transformer layer:
- Q projection (query)
- K projection (key)
- V projection (value)
- O projection (output)

Optionally also feed-forward layers (up, down, gate projections). More layers = more capacity, more memory.

### After training — two options

**Keep separate:** adapter files are tiny (50–200 MB at rank 16 for a 1.5B model). One base model, many task adapters swapped on top.

**Merge:** compute `W_new = W + (α/r) · B · A` and write it into W. Result is a standard model with no adapter overhead. Inference is identical speed to the base.

### Memory comparison — LoRA vs Full Fine-Tuning (1.5B model)

```
                    Full FT         LoRA (rank 16)
                    ───────         ──────────────
Base weights:        3 GB            3 GB  (same — still loads, but frozen)
Gradients:           3 GB           ~50 MB (adapters only)
Master copy:         6 GB          ~100 MB (adapters only)
Adam momentum:       6 GB          ~100 MB (adapters only)
Adam variance:       6 GB          ~100 MB (adapters only)
─────────────────────────────────────────────────────────────
Total:             ~24 GB          ~3.5 GB
```

The frozen base still occupies memory — but needs no gradients or optimizer state. Only adapter parameters do.

### Trade-offs

| Dimension | LoRA |
|---|---|
| Memory | Low — ~3.5 GB for 1.5B model at rank 16 |
| Speed | Fast — gradient work is tiny (adapters only) |
| Quality | Close to full fine-tuning on narrow tasks |
| Complexity | Low — one rank and alpha to set |
| Forgetting risk | Low — base weights never change |
| Output size | Small adapter file (MB), not a full model copy |
| Feasible on M1 16GB | ✅ Comfortable for 1.5B, possible for 3B |

---

## Approach 3 — QLoRA

*In progress — to be added after discussion.*

---

## Approach 4 — DPO

*In progress — to be added after discussion.*

---

## Comparison Table

*To be completed after all approaches are discussed.*
