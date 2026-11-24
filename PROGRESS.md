# Progress Tracker

Statuses: **NOT STARTED** | **IN PROGRESS** | **UNDERSTOOD** | **PRACTICALLY COMPLETED** | **NEEDS REVISION**

A concept is UNDERSTOOD only after it has been demonstrated in conversation. An implementation is PRACTICALLY COMPLETED only after actual execution and verification.

---

## Current Position

**Active phase:** Phase 4 complete — moving to Phase 5 (Evaluation and Experimentation)

**Last completed:** Phase 4 — full end-to-end LoRA fine-tuning pipeline executed and verified on 2026-11-24.

**What was built:** Organizational Procedure Assistant — a fine-tuned Qwen2.5-0.5B-Instruct model that takes workplace situations and returns structured procedures. Trained on 167 examples, evaluated on 10 held-out prompts. Result: 5.0/6.0 average score, PASS. See `practical/results/RESULTS.md`.

**Hardware note:** Mac M1, 16 GB unified memory. ~12–13 GB usable. LoRA on 0.5B model: ~3 GB active. For QLoRA on larger models, use Google Colab or a hosted GPU environment.

---

## Phase 1 — Fine-Tuning Fundamentals

| Topic | Status |
|---|---|
| LLM tokenization and vocabulary | UNDERSTOOD |
| Embeddings — from token IDs to vectors | UNDERSTOOD |
| Next-token prediction | UNDERSTOOD |
| Self-attention mechanism | UNDERSTOOD |
| Causal masking | UNDERSTOOD |
| Feed-forward layers | UNDERSTOOD |
| Transformer layer stack | UNDERSTOOD |
| Model weights and what they encode | UNDERSTOOD |
| Cross-entropy loss | UNDERSTOOD |
| Backpropagation | UNDERSTOOD |
| Gradient descent | UNDERSTOOD |
| Learning rate | UNDERSTOOD |
| Adam optimizer | UNDERSTOOD |
| Batches and epochs | UNDERSTOOD |
| Overfitting — detection and prevention | UNDERSTOOD |
| What fine-tuning changes vs what stays the same | UNDERSTOOD |
| Full fine-tuning memory cost (~16 bytes/param) | UNDERSTOOD |
| Why full FT of 1.5B is not feasible on 16 GB | UNDERSTOOD |
| Prompting vs RAG vs tools vs fine-tuning | UNDERSTOOD |
| Supervised Fine-Tuning (SFT) format | UNDERSTOOD |
| PEFT overview | UNDERSTOOD |

---

## Phase 2 — LoRA

| Topic | Status |
|---|---|
| Core LoRA insight — why fine-tuning deltas are low rank | UNDERSTOOD |
| LoRA architecture — A and B matrices, frozen base | UNDERSTOOD |
| Why B is initialized to zero | UNDERSTOOD |
| Low-rank decomposition — B × A approximating ΔW | UNDERSTOOD |
| Matrix dimensions and parameter counts at different ranks | UNDERSTOOD |
| Rank — what it controls, how to choose | UNDERSTOOD |
| Alpha — α/r scaling, common conventions | UNDERSTOOD |
| Target modules — which layers get adapters | UNDERSTOOD |
| LoRA memory saving vs full fine-tuning | UNDERSTOOD |
| How A and B learn during training | UNDERSTOOD |
| Saving adapters — what an adapter file contains | UNDERSTOOD |
| Loading adapters — keeping separate vs merging | UNDERSTOOD |
| LoRA dropout | IN PROGRESS |

---

## Phase 3 — QLoRA

| Topic | Status |
|---|---|
| What quantization is | IN PROGRESS |
| FP32, FP16, BF16 precision trade-offs | IN PROGRESS |
| 8-bit quantization | IN PROGRESS |
| 4-bit quantization challenges | IN PROGRESS |
| NF4 — why it is optimal for model weights | IN PROGRESS |
| Double quantization | IN PROGRESS |
| Dequantize-compute-quantize cycle | IN PROGRESS |
| QLoRA memory layout — what is in 4-bit vs 16-bit | IN PROGRESS |
| LoRA vs QLoRA — when to use each | IN PROGRESS |
| Memory estimation for QLoRA | NOT STARTED |

*Note: These topics have been studied but not yet formally documented or verified in conversation. Status is IN PROGRESS until confirmed.*

---

## Phase 4 — Practical Implementation

| Topic / Task | Status |
|---|---|
| Python environment setup | PRACTICALLY COMPLETED |
| PyTorch basics — tensors, devices | PRACTICALLY COMPLETED |
| Hugging Face Transformers — model loading | PRACTICALLY COMPLETED |
| Hugging Face PEFT — LoraConfig | PRACTICALLY COMPLETED |
| Hugging Face TRL — SFTTrainer (SFTConfig) | PRACTICALLY COMPLETED |
| Hugging Face Datasets | PRACTICALLY COMPLETED |
| Model selection — Qwen2.5-0.5B-Instruct | PRACTICALLY COMPLETED |
| Dataset preparation (12 issue types, chat JSONL) | PRACTICALLY COMPLETED |
| Train/validation split (167 train / 30 val) | PRACTICALLY COMPLETED |
| LoRA configuration — r=16, alpha=32, q_proj+v_proj | PRACTICALLY COMPLETED |
| Inspect trainable parameters (1.08M / 495M = 0.22%) | PRACTICALLY COMPLETED |
| Run training, monitor loss (3.3 → 2.2 over 3 epochs) | PRACTICALLY COMPLETED |
| Save adapter (adapter_v1/, ~16 MB) | PRACTICALLY COMPLETED |
| Load adapter and verify | PRACTICALLY COMPLETED |
| Run inference — base vs fine-tuned comparison | PRACTICALLY COMPLETED |
| API version debugging (TRL 1.x + transformers 5.x) | PRACTICALLY COMPLETED |
| Extend to QLoRA | NOT STARTED |

---

## Phase 5 — Evaluation and Experimentation

| Topic / Task | Status |
|---|---|
| Zero-shot baseline evaluation | NOT STARTED |
| Few-shot baseline evaluation | NOT STARTED |
| Fine-tuned model evaluation | NOT STARTED |
| Rank experiment (compare rank 4, 16, 64) | NOT STARTED |
| Alpha experiment | NOT STARTED |
| Learning rate experiment | NOT STARTED |
| Overfitting detection from loss curves | NOT STARTED |
| Failure analysis | NOT STARTED |

---

## Phase 6 — Production Integration

| Topic / Task | Status |
|---|---|
| Local inference server setup | NOT STARTED |
| Node.js integration | NOT STARTED |
| Adapter versioning | NOT STARTED |
| Latency measurement | NOT STARTED |

---

## Phase 7 — Capstone Project

| Task | Status |
|---|---|
| Define task and design overall system | NOT STARTED |
| Build fine-tuned component | NOT STARTED |
| Build RAG component | NOT STARTED |
| Wire tools and APIs | NOT STARTED |
| Node.js backend integration | NOT STARTED |
| Final evaluation and documentation | NOT STARTED |

---

## Topics Needing Revision

None flagged yet. If a comprehension check reveals a gap, it will be listed here.

---

## Practical Exercises Completed

### Phase 4 — Organizational Procedure Assistant (2026-11-24)

**Task:** Fine-tune Qwen2.5-0.5B-Instruct to respond to workplace situations with structured procedures.

**Approach:** LoRA (rank=16, alpha=32) on q_proj and v_proj. 167 training examples across 12 issue types. SFTTrainer via TRL. Training on Apple M1 MPS.

**Result:** Average evaluation score 5.0/6.0 on 10 held-out prompts. PASS. Structured output with ISSUE TYPE, SEVERITY, numbered steps, escalation triggers, and documentation requirements generated consistently.

**Key lessons from debugging:** TRL 1.x and transformers 5.x introduced breaking API changes — SFTConfig replaces TrainingArguments, `processing_class` replaces `tokenizer=`, `dtype` replaces `torch_dtype`, and `apply_chat_template` now returns a BatchEncoding dict. All fixed and documented in the scripts.

**Artifacts:** `practical/adapter_v1/`, `practical/results/RESULTS.md`

---

## Concepts Marked Understood After Demonstration

The topics marked UNDERSTOOD in Phase 1 and Phase 2 above were confirmed through conversation and comprehension checks before this tracking document was created. They should be revisited during Phase 4 implementation — practical exposure often surfaces gaps that conversation does not.
