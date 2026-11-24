# LLM Fine-Tuning Exploration

A structured learning workspace for understanding and implementing LoRA fine-tuning on language models from first principles — not by following a tutorial, but by building, breaking, and fixing every piece of the pipeline.

**Phase 4 is complete.** A Qwen2.5-0.5B-Instruct model was fine-tuned on a custom dataset, evaluated against a held-out baseline, and scored 5.0/6.0 (PASS).

---

## What Was Built

**Task:** Organizational Procedure Assistant for a software house — takes a plain-text workplace situation and returns a structured procedure with issue classification, severity, escalation triggers, and documentation requirements.

**How:** LoRA fine-tuning (rank 16, alpha 32) on Qwen2.5-0.5B-Instruct using 167 training examples across 12 issue types. Trained locally on Apple M1 in ~16 minutes.

**Result:**

| | Base model (before) | Fine-tuned (after) |
|---|---|---|
| Output format | Generic markdown bullets | Structured uppercase headers |
| Severity | Always "High" | Differentiated (Critical / High / Medium) |
| Escalation path | Vague or missing | Explicit in every response |
| Documentation checklist | Missing | Required section in every response |
| Eval score | — | **5.0 / 6.0 average (PASS)** |

---

## Quick Start — Run the Trained Model

```bash
# From the project root
cd practical
source .venv/bin/activate      # activate the Python environment

# Interactive assistant (trained adapter already saved)
python inference.py --adapter adapter_v1
```

Type any workplace situation at the `Issue:` prompt. Type `quit` to exit.

To re-run the full training pipeline from scratch:

```bash
cd practical
source .venv/bin/activate

# 1. Record base model outputs before training
python evaluation/capture_baseline.py

# 2. Train (~16 min on Apple M1)
python train.py

# 3. Evaluate — prints before/after score sheet
python evaluation/evaluate.py --adapter adapter_v1

# 4. Test interactively
python inference.py --adapter adapter_v1
```

---

## Project Structure

```
.
├── README.md               ← this file
├── ROADMAP.md              ← full curriculum, phases, checkpoint questions
├── PROGRESS.md             ← topic-by-topic status tracker
│
├── docs/                   ← learning notes, one file per phase
│   ├── 00-glossary.md      ← quick-reference definitions for all key terms
│   ├── 01-fundamentals.md  ← LLM foundations: tokenization, training, fine-tuning basics
│   ├── 02-lora.md          ← LoRA: low-rank decomposition, rank, alpha, adapters
│   ├── 03-qlora.md         ← QLoRA: quantization, NF4, 4-bit training (in progress)
│   └── 04-training-workflow.md  ← end-to-end pipeline reference with all commands
│
└── practical/              ← all implementation code and artifacts
    ├── requirements.txt    ← 6 dependencies (torch, transformers, peft, trl, datasets, accelerate)
    ├── train.py            ← full LoRA training script
    ├── inference.py        ← interactive CLI to test the trained model
    │
    ├── dataset/
    │   ├── generate.py     ← generates the training dataset from the 12 procedures
    │   ├── train.jsonl     ← 167 training examples (chat JSONL format)
    │   └── validation.jsonl← 30 validation examples
    │
    ├── evaluation/
    │   ├── capture_baseline.py     ← run BEFORE training — records base model outputs
    │   ├── evaluate.py             ← run AFTER training — scores and compares
    │   ├── test_prompts.txt        ← 10 held-out prompts (never used in training)
    │   ├── baseline_outputs.jsonl  ← saved base model outputs (before training)
    │   └── finetuned_outputs.jsonl ← saved fine-tuned outputs (after training)
    │
    ├── results/
    │   └── RESULTS.md      ← full before/after comparison with training metrics
    │
    └── adapter_v1/         ← trained LoRA adapter (~16 MB, ready to use)
        ├── adapter_model.safetensors
        ├── adapter_config.json
        └── tokenizer.json
```

---

## Learning Curriculum

| Document | Contents |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Full curriculum with phases, topics, and checkpoint questions |
| [PROGRESS.md](PROGRESS.md) | Per-topic status: UNDERSTOOD / IN PROGRESS / NOT STARTED |
| [docs/00-glossary.md](docs/00-glossary.md) | Quick-reference definitions |
| [docs/01-fundamentals.md](docs/01-fundamentals.md) | Tokenization, embeddings, training mechanics, SFT vs full FT |
| [docs/02-lora.md](docs/02-lora.md) | LoRA from first principles: A/B matrices, rank, alpha, memory savings |
| [docs/03-qlora.md](docs/03-qlora.md) | QLoRA: NF4, double quantization, dequantize-compute cycle |
| [docs/04-training-workflow.md](docs/04-training-workflow.md) | Complete pipeline reference with commands, API gotchas, deployment options |

---

## Implementation Reference

| File | Purpose |
|---|---|
| [practical/train.py](practical/train.py) | LoRA training: load model → attach adapters → SFTTrainer → save |
| [practical/inference.py](practical/inference.py) | Load adapter and run interactive inference |
| [practical/dataset/generate.py](practical/dataset/generate.py) | Generate training dataset from procedure definitions |
| [practical/evaluation/capture_baseline.py](practical/evaluation/capture_baseline.py) | Record base model outputs before training |
| [practical/evaluation/evaluate.py](practical/evaluation/evaluate.py) | Score fine-tuned outputs, compare with baseline |
| [practical/results/RESULTS.md](practical/results/RESULTS.md) | Full before/after results with training metrics |

---

## Key Decisions Made

| Decision | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Small enough for M1 (16GB), instruction-tuned, strong chat template |
| Fine-tuning method | LoRA (not QLoRA) | No quantization needed at 0.5B, simpler pipeline, MPS compatible |
| LoRA rank | 16 | Standard starting point; 1.08M trainable params (0.22% of total) |
| LoRA alpha | 32 | Convention: alpha = 2 × rank, effective scale = 2.0 |
| Target modules | q_proj, v_proj | Attention key layers; sufficient for behavior shaping at this scale |
| Epochs | 3 | Enough to converge on a 167-example dataset without overfitting |
| Batch size | 2 (effective 8 with grad accum) | M1 memory constraint; gradient accumulation compensates |
| Dataset | 12 issue types, 167 examples | Covers main software house scenarios; enough variety for generalization |

---

## Hardware

- Machine: MacBook with Apple M1, 16 GB unified memory
- Backend: MPS (Metal Performance Shaders — Apple's GPU framework for PyTorch)
- Full fine-tuning of a 1.5B model: ~24 GB (not feasible on this machine)
- LoRA on 0.5B model: ~3 GB active (feasible)
- For QLoRA on larger models: use Google Colab or a hosted GPU

---

## Working Rules

1. No concept is documented until it has been explained and verified in conversation.
2. No implementation step is marked complete until the code actually ran and produced the expected output.
3. Status trackers (PROGRESS.md, ROADMAP.md) reflect actual state, not aspirational state.
4. Failures and debugging sessions are documented, not hidden.
5. Every phase ends with a checkpoint — understanding is verified before moving forward.
