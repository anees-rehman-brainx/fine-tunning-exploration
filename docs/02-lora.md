# Phase 2 — LoRA

**Status:** UNDERSTOOD (conceptual)  
**Goal:** Understand LoRA from first principles well enough to configure, debug, and explain every component.

---

## 2.1 The Core Insight

LoRA (Low-Rank Adaptation) keeps every original weight in the model completely frozen and adds small trainable adapter matrices alongside selected layers. No original weight is ever touched. Only the adapters train.

**The key observation:** when you fine-tune a large model, the weight changes (deltas — how much each weight moves) have **very low rank**. Even though there are millions of weights, the information needed to adapt the model to a new task lives in a much smaller dimensional subspace.

LoRA exploits this by representing the weight update as a product of two small matrices instead of updating the full matrix directly.

---

## 2.2 How LoRA Works — The Math

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

## 2.3 The Mathematical Trick — Why Two Matrices

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

## 2.4 Rank — What It Controls

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

## 2.5 Alpha — What It Controls

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

**Practical implication:** set `alpha = 2 * rank` and leave it alone while experimenting with rank.

---

## 2.6 Target Modules — Which Layers Get Adapters

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

## 2.7 Memory Comparison — LoRA vs Full Fine-Tuning (1.5B model)

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

## 2.8 Saving, Loading, and Merging Adapters

**Saving:**
After training, the adapter (A and B matrices for each targeted layer) is saved. At rank 16 targeting q/k/v/o only on a 1.5B model, the adapter file is ~11 MB (5.5M params × 2 bytes FP16). Including FFN layers pushes it higher. The base model weights are not in this file.

**Loading — keeping separate:**
Load the base model, then load the adapter on top with `PeftModel.from_pretrained`. The adapter is applied at inference time. This allows one base model with many task adapters swapped on top.

**Loading — merging:**
Compute `W_new = W + (α/r) · B · A` and write it into W. Result is a standard model with no adapter overhead. Inference is identical speed to the base model. The adapter is now baked in and cannot be removed without reloading the base.

**When to merge:** when you have one primary task and want inference to be as fast as possible, or when distributing the model without exposing the adapter structure.

**When to keep separate:** when you have multiple task adapters, want to switch between them, or want to share just the adapter file rather than a full model copy.

---

## 2.9 LoRA Trade-Offs Summary

| Dimension | LoRA |
|---|---|
| Memory | Low — ~3.1 GB for 1.5B model at rank 16 (q/k/v/o only) |
| Speed | Fast — gradient work is tiny (adapters only) |
| Quality | Close to full fine-tuning on narrow tasks |
| Complexity | Low — one rank and alpha to set |
| Forgetting risk | Low — base weights never change |
| Output size | Small adapter file (~11 MB at rank 16), not a full model copy |
| Feasible on M1 16 GB | Yes — comfortable for 1.5B, possible for 3B |
