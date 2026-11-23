# LLM Fine-Tuning Exploration

A structured learning workspace for understanding and implementing LoRA and QLoRA fine-tuning on language models. The goal is independent engineering competence: being able to design, implement, evaluate, troubleshoot, and deploy fine-tuning workflows without blindly following tutorials.

---

## Project Context

This project follows an assignment that requires:

- Researching fine-tuning approaches with trade-offs documented
- Selecting and justifying a base model and dataset
- Running a training pipeline locally (Apple Silicon, no cloud compute)
- Evaluating the fine-tuned model against a baseline
- Producing a working demo and final documentation

---

## Navigation

| Document | Purpose |
|---|---|
| [README.md](README.md) | This file — project overview, navigation, and working rules |
| [ROADMAP.md](ROADMAP.md) | Complete curriculum, phases, milestones, and current status |
| [PROGRESS.md](PROGRESS.md) | Topic tracker — what is understood, in progress, or needs revision |
| [docs/](docs/) | Learning notes, one file per phase |
| [practical/](practical/) | Implementation code, notebooks, datasets, and experiments |

### Learning Notes (docs/)

| File | Contents |
|---|---|
| [docs/00-glossary.md](docs/00-glossary.md) | Quick-reference definitions for all key concepts across all phases |
| [docs/01-fundamentals.md](docs/01-fundamentals.md) | LLM foundations: tokenization, embeddings, training, fine-tuning basics |
| [docs/02-lora.md](docs/02-lora.md) | LoRA: low-rank decomposition, rank, alpha, adapters, memory savings |
| [docs/03-qlora.md](docs/03-qlora.md) | QLoRA: quantization, NF4, 4-bit training (in progress) |
| [docs/04-training-workflow.md](docs/04-training-workflow.md) | End-to-end practical training workflow and implementation checklist |

---

## Hardware Context

- Machine: MacBook with Apple M1, 16 GB unified memory
- Usable for training: ~12–13 GB after macOS overhead
- Full fine-tuning of a 1.5B model requires ~24 GB — not feasible on this machine
- LoRA on 1.5B model: ~3.1 GB — feasible
- QLoRA extends feasibility to larger models by quantizing the base

---

## Working Method

Teaching cycle for each new concept:

1. Introduce the concept
2. Explain why it exists
3. Explain how it works
4. Give a concrete example
5. Connect it to a real implementation
6. Ask one comprehension question
7. Evaluate the answer
8. Correct any misconceptions
9. Move forward only when understanding is confirmed

One concept at a time. Never dump a full chapter in one response.

---

## Working Rules

1. **Research before anything.** No concept is documented from memory or assumption. Papers, official docs, real sources first.
2. **Discuss before concluding.** After research, talk it through until it makes sense — not just until there is a nod.
3. **Brainstorm before deciding.** For model, dataset, framework, or approach choices: lay out options, discuss trade-offs, the learner decides.
4. **Nothing goes into a doc until both agree.** Docs are finalized knowledge, not drafts in progress.
5. **Checkpoint before moving to the next phase.** Every phase has checkpoint questions. If the answers are not solid, we do not move on.
6. **Status is always current.** PROGRESS.md reflects where we actually are, not where we hoped to be.
7. **Commit at the end of each phase.** Git history is the real record of progress.
8. **Honest about failures and gaps.** If something does not work, we say so and document why.
9. **No skipping.** If something is unclear, we stop right there.
10. **The learner owns this.** The understanding, decisions, and final work belong to the learner.

---

## Learner Background

Senior Software Engineer, ~4 years MERN experience.

**Strengths:** Node.js, TypeScript, React, backend engineering, API architecture, database design, production application development, LLM integrations, RAG and agentic AI.

**Goal:** Become a stronger Generative AI / Full Stack AI Engineer.

ML and Deep Learning taught from zero, with focus on practical engineering rather than research mathematics.
