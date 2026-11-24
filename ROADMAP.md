# Roadmap: LLM Fine-Tuning Curriculum

This is the full curriculum organized into phases. Each phase has clear topics, expected outcomes, and checkpoint questions. We do not move to the next phase until the checkpoint is answered confidently.

---

## Current Status

| Phase | Topic | Status |
|---|---|---|
| Phase 1 | Fine-Tuning Fundamentals | UNDERSTOOD |
| Phase 2 | LoRA | UNDERSTOOD |
| Phase 3 | QLoRA | IN PROGRESS |
| Phase 4 | Practical Implementation | PRACTICALLY COMPLETED |
| Phase 5 | Evaluation and Experimentation | NOT STARTED |
| Phase 6 | Production Integration | NOT STARTED |
| Phase 7 | Capstone Project | NOT STARTED |

**Current position:** Phase 4 complete (2026-11-24). Full end-to-end LoRA pipeline built and executed. Qwen2.5-0.5B-Instruct fine-tuned on Organizational Procedure Assistant task, evaluated at 5.0/6.0, PASS. See `practical/results/RESULTS.md`. Next: Phase 5 (Evaluation and Experimentation) — controlled hyperparameter experiments, rank comparison, failure analysis.

---

## Phase 1 — Fine-Tuning Fundamentals

**Goal:** Understand what fine-tuning is, why it exists, what it changes in the model, and when it is the right choice versus alternatives.

### Topics

1. How language models work — tokenization, embeddings, next-token prediction
2. What model weights and parameters are
3. Training mechanics — loss, cross-entropy, backpropagation, gradient descent, learning rate
4. Batches, epochs, and the training loop
5. Overfitting — what it is and how to detect it
6. What fine-tuning changes versus what stays the same
7. Supervised Fine-Tuning (SFT) — what the data looks like, what the model learns
8. Full fine-tuning versus PEFT — memory cost comparison, trade-offs
9. Prompting versus RAG versus tool calling versus fine-tuning — when each is the right choice
10. Memory cost of training — why ~16 bytes per parameter for Adam in mixed precision

### Checkpoint — be able to explain:

- [x] Why generating text is a prediction problem over a fixed vocabulary
- [x] What a weight is and why there are billions of them
- [x] What loss means and why we want it to go down
- [x] Why fine-tuning from a pretrained base is cheaper than training from scratch
- [x] What changes in the model after fine-tuning and what stays the same
- [x] When you would choose prompting, RAG, tools, or fine-tuning for a given task
- [x] Why full fine-tuning of a 1.5B model requires ~24 GB and cannot run on 16 GB

**Output:** Notes in LEARNING.md, Part 1.

---

## Phase 2 — LoRA

**Goal:** Understand LoRA from first principles well enough to configure, debug, and explain every component.

### Topics

1. The core insight — why fine-tuning weight changes are low rank
2. Low-rank decomposition — how B × A approximates ΔW
3. Adapter matrices A and B — shapes, initialization, what they learn
4. Rank — what it controls, how to choose it
5. Alpha — what it controls, the α/r scaling, common conventions
6. Target modules — which layers get adapters and why
7. Dropout — what it does in LoRA context
8. The training lifecycle — what is frozen, what is trainable, what optimizer state is allocated
9. Saving adapters — what an adapter file contains
10. Loading adapters — merging into base vs keeping separate
11. LoRA merge — when and why to merge, what the resulting model looks like

### Checkpoint — be able to explain:

- [x] What the low-rank insight actually is — why it is not obvious that fine-tuning deltas are low rank
- [x] Why B is initialized to zero and what happens at step one
- [x] What rank 16 means in terms of concrete parameter counts for a 1536×1536 layer
- [x] Why alpha/r is the effective scale and not just alpha
- [x] The memory saving from LoRA: what still takes memory, what does not
- [x] What is in an adapter file and what is not
- [x] The difference between keeping adapters separate and merging them

**Output:** Notes in LEARNING.md, Part 2.

---

## Phase 3 — QLoRA

**Goal:** Understand quantization well enough to explain why QLoRA enables larger models, what the trade-offs are, and how to use it correctly.

### Topics

1. What quantization is — representing weights in fewer bits
2. Data types — FP32, FP16, BF16 and what each sacrifices
3. 8-bit quantization (INT8) — how it works, what it loses
4. 4-bit quantization — the precision challenge, how NF4 solves it
5. NF4 (NormalFloat4) — what makes it information-theoretically optimal for normally-distributed weights
6. Double quantization — quantizing the quantization constants, why it matters
7. The dequantize-compute-quantize cycle — why QLoRA can still compute accurate gradients
8. QLoRA training — what is quantized, what is not, what the memory layout looks like
9. LoRA versus QLoRA — when to use each
10. GPU memory considerations — how to estimate feasibility before starting

### Checkpoint — be able to explain:

- [ ] Why 4-bit quantization of weights does not mean 4-bit training
- [ ] What NF4 does differently from uniform quantization
- [ ] Why double quantization saves meaningful memory
- [ ] What is still in FP16 or BF16 during QLoRA training (adapters, optimizer states)
- [ ] How to estimate whether a given model fits in a given memory budget with QLoRA
- [ ] When you would choose QLoRA over LoRA

**Output:** Notes in LEARNING.md, Part 3.

---

## Phase 4 — Practical Implementation

**Goal:** Build a working end-to-end LoRA fine-tuning pipeline. Extend it to QLoRA. Actually run it, not just read about it.

### Environment

Assess hardware before choosing the environment. Mac M1 with 16 GB can run LoRA on small models. Use Google Colab or a hosted GPU environment for QLoRA on larger models if needed.

### Topics and Steps

1. Python environment setup (uv or venv, pinned dependencies)
2. PyTorch fundamentals — tensors, devices, autograd
3. Hugging Face ecosystem — Transformers, PEFT, TRL, Datasets
4. Model selection — small pretrained model for initial experiments
5. Dataset preparation — loading, formatting as chat JSONL, train/validation split
6. LoRA configuration — `LoraConfig`, rank, alpha, target modules
7. Inspect trainable parameters — confirm adapter count is as expected
8. Training — `SFTTrainer` or `Trainer`, monitoring loss
9. Save adapter — what files are written, what they contain
10. Load adapter — `PeftModel.from_pretrained`
11. Run inference — compare fine-tuned vs base model on held-out prompts
12. Extend to QLoRA — `BitsAndBytesConfig`, 4-bit loading, same pipeline

### Checkpoint — be able to:

- [x] Set up the environment and explain what each library does
- [x] Prepare a dataset in the format the trainer expects
- [x] Configure LoRA and explain why each setting was chosen
- [x] Read the trainable parameter count output and verify it matches expectations
- [x] Monitor training loss and explain what it means
- [x] Save and reload an adapter and confirm outputs are identical
- [x] Run a side-by-side inference comparison and interpret the difference

**Output:** Working scripts in `practical/`, results in `practical/results/RESULTS.md`, workflow reference in `docs/04-training-workflow.md`.

---

## Phase 5 — Evaluation and Experimentation

**Goal:** Evaluate the fine-tuned model honestly and run controlled experiments to understand what each hyperparameter actually does.

### Topics

1. Baseline comparison — zero-shot base, few-shot base, fine-tuned
2. Evaluation datasets — held-out test set, never seen during training
3. Response quality metrics — exact match, F1, task-specific scoring
4. Training loss vs validation loss — what the curves tell you
5. Overfitting signals — when to stop training
6. Rank experiments — what changes between rank 4, 16, 64
7. Alpha experiments — effect of different α/r ratios
8. Learning rate experiments — underfitting vs overfitting signals
9. Experiment tracking — logging, naming runs, comparing results

### Checkpoint — be able to:

- [ ] Run the three-way comparison (zero-shot base, few-shot base, fine-tuned) and interpret results
- [ ] Explain what training loss going down but validation loss rising means
- [ ] Show the effect of rank on output quality and adapter size with real numbers
- [ ] Identify where the fine-tuned model is better and where it regressed

**Output:** Evaluation report, experiment logs in `practical/`.

---

## Phase 6 — Production Integration

**Goal:** Understand how fine-tuned models are served, integrated into applications, and maintained over time.

### Topics

1. Model serving — local inference server, Hugging Face Inference API
2. Adapter management — versioning, switching adapters at runtime
3. Inference APIs — REST endpoints, streaming
4. Node.js integration — calling the inference server from a TypeScript/Node.js backend
5. Latency and resource management — batch inference, quantization for serving
6. Deployment options — local, cloud, managed endpoints
7. Monitoring — tracking output quality in production
8. Versioning — managing adapter versions over time

**Output:** Integration example in `practical/`, notes in LEARNING.md, Part 6.

---

## Phase 7 — Capstone Project

**Goal:** Build a complete fine-tuned AI assistant that demonstrates the correct responsibilities of each component. Something that can be presented, defended, and deployed.

### What to Build

A fine-tuned AI assistant combining:

- A pretrained LLM as the base
- A custom SFT dataset
- LoRA (and optionally QLoRA for a larger base)
- Evaluation against a baseline
- RAG for documentation retrieval (retrieval-augmented generation for things the model should not need to memorize)
- Database or API tools for live information
- A system prompt for behavioral instructions
- A Python inference service
- A Node.js backend integration

### Why Each Component

| Component | Responsibility |
|---|---|
| Fine-tuning | Teaches format, style, and task-specific behavior |
| RAG | Supplies current or domain-specific documents the model cannot memorize |
| Tools | Provides live data (APIs, databases) the model cannot know |
| System prompt | Sets behavioral instructions and persona |

The capstone should make clear when to use each of these and why — not combine them blindly.

### Checkpoint — be able to:

- [ ] Explain why fine-tuning was used instead of only prompting or RAG
- [ ] Explain what each system component is responsible for
- [ ] Run a demo comparing the base model vs fine-tuned model on real prompts
- [ ] Describe what failed, what could be improved, and what you would do with more time

**Output:** Complete project in `practical/capstone/`, final documentation.

---

## Assignment Deliverables Mapping

| Assignment Step | Covered In |
|---|---|
| Research: approaches, trade-offs, feasibility | LEARNING.md Parts 1–3 |
| Framework choice and justification | Phase 4 notes |
| Model and dataset selection with justification | Phase 4 notes |
| Data preparation and train/validation split | Phase 4 scripts |
| Training artifacts (scripts, config, adapter) | `practical/` |
| Evaluation report with baseline comparison | Phase 5 output |
| Demo or walkthrough | Phase 7 / capstone |
