# Training Plan: Model Fine-Tuning Exploration

> **How to read this document**
> This is not a checklist to rush through. It is a learning roadmap.
> Every phase has a "you should be able to explain this" section.
> We do not move to the next phase until those questions can be answered confidently.
> The trainer (Claude) will research, fetch docs, run experiments and explain.
> You make the decisions. You understand the reasoning. You own the outcome.

---

## Our Goal

By the end of this project you will have:

- A deep understanding of *how* language models learn, and what fine-tuning actually does inside the model.
- Hands-on experience running a real training pipeline on your own machine.
- A tested, documented result you can present and defend to anyone.
- The ability to pick the right fine-tuning approach for any future task and explain why.

---

## Phases Overview

| Phase | What it covers | Output |
| --- | --- | --- |
| **Phase 0** | Foundations — how LLMs work, what training means | You can explain the basics in your own words |
| **Phase 1** | Fine-tuning approaches — full, LoRA, QLoRA, DPO | Research document (Step 1 of brief) |
| **Phase 2** | Hardware reality check — what actually fits on your machine | Feasibility section of research doc |
| **Phase 3** | Framework exploration — MLX, PEFT/TRL, LLaMA-Factory | Framework choice, justified |
| **Phase 4** | Model and dataset selection | Model + dataset choice, justified (Step 2 of brief) |
| **Phase 5** | Data preparation | Clean dataset, train/valid/test split, scripts |
| **Phase 6** | Trainer review | Review notes (Step 3 of brief) |
| **Phase 7** | Training run | Logged training run, adapter weights (Step 4 of brief) |
| **Phase 8** | Evaluation | Evaluation report (Step 5 of brief) |
| **Phase 9** | Demo and wrap-up | Demo, final documentation |

---

## Phase 0 — Foundations

**Purpose:** before we talk about fine-tuning methods, we need a shared mental model of what a language model is and what happens when you train one. Without this, everything in Phase 1 is just vocabulary.

### Topics we will cover

1. **What a language model actually does** — predicting the next token, probability distributions, temperature.
2. **What "weights" are** — numbers inside the model that encode what it has learned.
3. **What training does** — gradient descent, loss functions, what "loss going down" means in plain terms.
4. **What a pretrained model already knows** — why we start from a pretrained model and not from scratch.
5. **What fine-tuning changes** — which weights move, by how much, and why that changes behaviour.

### How we will work

- I explain one topic at a time, in plain language, with analogies.
- You ask questions until it clicks.
- We do not move on until you can re-explain it back.

### Checkpoint — you should be able to explain:

- [ ] Why generating text is a prediction problem.
- [ ] What a "weight" is and why there are billions of them.
- [ ] What "loss" means and why we want it to go down.
- [ ] Why fine-tuning from a pretrained base is cheaper than training from scratch.
- [ ] What changes in the model after fine-tuning and what stays the same.

---

## Phase 1 — Fine-Tuning Approaches

**Purpose:** understand the four main approaches from first principles, not just their names.

### Topics we will cover

1. **Full fine-tuning** — update every weight; why it costs so much memory; what "optimizer state" is.
2. **LoRA** — the core idea of low-rank matrices; why it saves memory; rank, alpha, what they control; what an adapter file is.
3. **QLoRA** — what quantization means; NF4 data type; why dequantizing on the fly works; what the trade-off is.
4. **DPO** — what "preference tuning" means; how it differs from SFT; what a (chosen, rejected) pair looks like; why it needs a reference model.
5. **When each one is the right tool** — a decision framework you can use for any future project.
6. **Fine-tuning vs prompting vs RAG** — not just trade-offs, but how to choose in a real situation.

### How we will work

- One approach per session.
- For each one: explain the idea, look at the original paper, work through a small concrete example.
- Compare against what we already know before moving to the next.

### Checkpoint — you should be able to explain:

- [ ] Why full fine-tuning uses about 16 bytes per parameter with Adam, and where those bytes go.
- [ ] What LoRA's low-rank matrix trick actually does, with a rough sketch.
- [ ] Why QLoRA can fit a bigger model in memory and what the cost of that is.
- [ ] What DPO trains the model to do differently from SFT.
- [ ] Given a task and a machine, which approach you would pick and why.
- [ ] When you would choose prompting or RAG instead of fine-tuning.

**Output of this phase:** research document, Step 1 of the brief.

---

## Phase 2 — Hardware Reality Check

**Purpose:** understand what your specific machine can and cannot do, with numbers, not guesses.

### Topics we will cover

1. **Apple M1 unified memory** — how CPU and GPU share one pool, what that means for training.
2. **Memory budget** — macOS overhead, how much is actually usable for training.
3. **Memory per approach** — work out the actual numbers for your machine and candidate model sizes.
4. **Speed reality** — training on Apple Silicon vs a GPU; what to expect; why patience matters.
5. **Pilot run** — a short training run to measure real memory and speed before committing to the full run.

### Checkpoint — you should be able to explain:

- [ ] Why Apple Silicon's unified memory is different from a GPU with VRAM.
- [ ] How much memory a 1.5B model in 16-bit takes, and how much is left for training.
- [ ] Which model sizes are feasible with which approaches on this machine.
- [ ] Why we do a pilot run before the full training.

**Output of this phase:** feasibility section added to the research document.

---

## Phase 3 — Framework Exploration

**Purpose:** understand the tools, try them briefly, and make an informed choice together.

### Tools we will look at

1. **MLX / mlx-lm** — Apple's own framework; native Silicon support; what it can and cannot do.
2. **Hugging Face PEFT + TRL** — the standard for the broader community; PyTorch on Mac.
3. **LLaMA-Factory** — a higher-level wrapper; what it hides.

### How we will work

- For each framework: look at the docs, understand the data format it expects, understand the training command.
- Short hands-on test: install it, run a tiny sanity check.
- Compare and decide.

### Checkpoint — you should be able to explain:

- [ ] Why MLX is different from PyTorch on Mac and what the trade-off is.
- [ ] What data format each framework expects.
- [ ] Which framework you chose and why.

**Output of this phase:** framework choice added to research document; environment installed and tested.

---

## Phase 4 — Model and Dataset Selection

**Purpose:** choose the base model and task, with full reasoning. This is a decision you make, not one I make for you.

### For the model, we will evaluate:

- Size vs memory budget (from Phase 2)
- License (can we use and share it freely?)
- Whether an MLX-compatible version exists
- Whether it already does something close to our task (baseline strength)
- How much headroom it leaves for improvement

### For the dataset and task, we will evaluate:

- What task is narrow enough to show a clear gain from fine-tuning
- Whether the dataset is high quality, well-licensed, and big enough
- Whether we can score the output automatically (important for clean evaluation)
- How much data preparation work it needs

### How we will work

- I research the candidates: look up model cards, dataset cards, run real examples.
- We look at actual examples from each dataset together.
- You decide.

### Checkpoint — you should be able to explain:

- [ ] Why you chose this model and not the others.
- [ ] Why you chose this task and dataset.
- [ ] What the data looks like and what good vs bad examples look like.

**Output of this phase:** model and dataset choices, fully justified, added to the research document (Step 2 of brief).

---

## Phase 5 — Data Preparation

**Purpose:** turn raw data into a clean, properly split dataset the training framework can use.

### Steps

1. Download the dataset, record the source and version.
2. Inspect real examples — look at length distribution, label quality, edge cases.
3. Clean: remove duplicates, broken rows, over-long examples.
4. Format: convert to the chat JSONL format the framework expects.
5. Split into train / validation / test with a fixed seed. Test set is locked away — never used for tuning.
6. Record counts, statistics, the seed, and the script so anyone can rebuild the data.

### Checkpoint — you should be able to explain:

- [ ] Why we separate train, validation and test sets and what each one is for.
- [ ] What data leakage is and how we avoid it.
- [ ] What the final JSONL examples look like and why they are formatted that way.
- [ ] How someone else could reproduce our dataset exactly.

**Output of this phase:** clean dataset files, prep script, data statistics. Data prep section added to research document.

---

## Phase 6 — Trainer Review

**Purpose:** present the full research document and plan, receive feedback, adjust before training.

This is Step 3 of the brief. At this point we will:

- Review the research document together.
- Review the model and dataset choices.
- Review the planned training configuration.
- Record any changes made as a result.

**Output of this phase:** trainer review notes, confirmed plan.

---

## Phase 7 — Training Run

**Purpose:** run the actual fine-tuning, monitor it, understand what is happening while it runs.

### Steps

1. Set up the training configuration (hyperparameters, logged).
2. Run a pilot: about 50-100 iterations, check memory and speed.
3. Adjust if needed.
4. Run the full training.
5. Monitor training loss and validation loss together — learn what the curves mean.
6. Save checkpoints, pick the best one.

### What we will watch and understand

- What the loss curve means and what overfitting looks like.
- Why we validate separately from training.
- What the adapter file is and what it contains.
- How to resume if the run is interrupted.

### Checkpoint — you should be able to explain:

- [ ] What each hyperparameter in the config controls.
- [ ] What the training loss curve is telling you.
- [ ] What it means when validation loss stops going down.
- [ ] What the saved adapter contains.

**Output of this phase:** training log with loss curves, adapter weights, training script and config.

---

## Phase 8 — Testing and Evaluation

**Purpose:** compare the fine-tuned model against the base model honestly and rigorously.

### What we will compare

1. Base model — zero-shot (no examples in the prompt).
2. Base model — few-shot (2-3 examples in the prompt).
3. Fine-tuned model — zero-shot.

### How we will evaluate

- At least one automatic metric (depends on task — exact match, F1, execution accuracy, etc.).
- Manual review of 20-30 examples — read them together and tag what went right and wrong.
- Honest discussion of failures, edge cases, and what the fine-tuning did not fix.
- A quick general-ability check — did training hurt the model on things outside the task?

### Checkpoint — you should be able to explain:

- [ ] Why we include the few-shot baseline (not just base vs fine-tuned).
- [ ] What the numbers mean and what they do not mean.
- [ ] Where the fine-tuned model is better, where it is worse, and why.
- [ ] What you would do differently with more time or data.

**Output of this phase:** evaluation report with metric table, example outputs, failure analysis, conclusions.

---

## Phase 9 — Demo and Wrap-Up

**Purpose:** show the model working, and document the full project so it stands alone.

### Steps

1. Build a short demo: run the fine-tuned model interactively on new prompts.
2. Compare live with the base model.
3. Final review of all documentation — does it tell the full story clearly?
4. Write up lessons learned.

**Output of this phase:** working demo, final polished documentation.

---

## Rules for how we work together

1. **You decide, I advise.** I will give you options and explain trade-offs. The choices are yours.
2. **No skipping.** If a phase is not understood, we do not move on. The checkboxes are real.
3. **Ask anything.** No question is too basic. The goal is understanding, not performance.
4. **Honest progress.** If something does not work, we say so and figure out why. No hiding failures.
5. **I research, you learn.** I will look up papers, fetch docs, run commands. You will understand what I find and why it matters.

---

## Current Status

| Phase | Status |
| --- | --- |
| Phase 0 — Foundations | Not started |
| Phase 1 — Fine-Tuning Approaches | Not started |
| Phase 2 — Hardware Reality Check | Not started |
| Phase 3 — Framework Exploration | Not started |
| Phase 4 — Model and Dataset Selection | Not started |
| Phase 5 — Data Preparation | Not started |
| Phase 6 — Trainer Review | Not started |
| Phase 7 — Training Run | Not started |
| Phase 8 — Testing and Evaluation | Not started |
| Phase 9 — Demo and Wrap-Up | Not started |

---

*We start with Phase 0 — Foundations. Ready when you are.*
