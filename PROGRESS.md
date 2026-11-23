# Progress Tracker

Statuses: **NOT STARTED** | **IN PROGRESS** | **UNDERSTOOD** | **PRACTICALLY COMPLETED** | **NEEDS REVISION**

A concept is UNDERSTOOD only after it has been demonstrated in conversation. An implementation is PRACTICALLY COMPLETED only after actual execution and verification.

---

## Current Position

**Active phase:** Moving to Phase 4 — Practical LoRA Implementation

**Last completed:** Phase 2 (LoRA) — conceptual understanding confirmed.

**Next activity:** Set up the Python environment, choose a small pretrained model, prepare a small dataset, and run the first end-to-end LoRA fine-tuning pipeline.

**Hardware note:** Mac M1, 16 GB unified memory. ~12–13 GB usable. LoRA on 1.5B model is feasible. For QLoRA on larger models, use Google Colab or a hosted GPU environment.

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
| Python environment setup | NOT STARTED |
| PyTorch basics — tensors, devices | NOT STARTED |
| Hugging Face Transformers — model loading | NOT STARTED |
| Hugging Face PEFT — LoraConfig | NOT STARTED |
| Hugging Face TRL — SFTTrainer | NOT STARTED |
| Hugging Face Datasets | NOT STARTED |
| Model selection (small pretrained model) | NOT STARTED |
| Dataset preparation (format as chat JSONL) | NOT STARTED |
| Train/validation split | NOT STARTED |
| LoRA configuration and setup | NOT STARTED |
| Inspect trainable parameters | NOT STARTED |
| Run training, monitor loss | NOT STARTED |
| Save adapter | NOT STARTED |
| Load adapter and verify | NOT STARTED |
| Run inference — base vs fine-tuned comparison | NOT STARTED |
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

None yet. Phase 4 is the first hands-on implementation.

---

## Concepts Marked Understood After Demonstration

The topics marked UNDERSTOOD in Phase 1 and Phase 2 above were confirmed through conversation and comprehension checks before this tracking document was created. They should be revisited during Phase 4 implementation — practical exposure often surfaces gaps that conversation does not.
