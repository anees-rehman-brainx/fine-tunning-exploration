# Task: Model Fine-Tuning Exploration

## Objective

Explore approaches to fine-tuning language models, then fine-tune a small open-source model locally on a dataset you select, and evaluate the result. The goal is to show that you can research, make justified technical decisions, and carry a small training project through to a tested outcome.

---

## Constraints

- Open-source models and free tools only.
- Training must run locally on a MacBook (Apple Silicon, no dedicated GPU). Choose a model size that fits the available memory.
- No paid APIs or cloud compute for the training itself.

---

## Steps

### 1. Research and Documentation

Research the main fine-tuning approaches (for example full fine-tuning, LoRA, QLoRA, and preference tuning such as DPO). Write a short document that covers:

- A summary of each approach, with trade-offs in memory, speed, quality, and complexity.
- Which approaches are feasible on your hardware, and why.
- When fine-tuning is the right choice compared with alternatives such as prompting or RAG.
- The tools or frameworks you considered (for example MLX, Hugging Face PEFT/TRL, LLaMA-Factory) and which you chose.

### 2. Model and Dataset Selection

- Pick a local base model and justify the choice (size, license, hardware fit, task suitability).
- Pick or build a dataset and justify the choice (relevance to the task, quality, size, license, format).
- Document the data preparation steps: cleaning, formatting, and the train/validation split.

### 3. Review with the Trainer

Before starting training, present your documentation, model choice, dataset choice, and training plan to the trainer. Record the feedback you receive and any changes you make as a result.

### 4. Training

- Fine-tune the selected model on the chosen dataset using your agreed approach.
- Log your configuration (hyperparameters, epochs/iterations, batch size) and training and validation loss.

### 5. Testing and Evaluation

- Test the fine-tuned model against the base model on held-out prompts.
- Use a clear evaluation method (for example exact match, a rubric, or side-by-side comparison) and report the results.
- Discuss strengths, failures, and limitations honestly.

---

## Deliverables

1. Documentation covering the research, decisions, and reasoning from steps 1-2.
2. Trainer review notes capturing the feedback and your response to it.
3. Training artifacts: scripts or configs, the dataset (or a link to it), and the adapter or fine-tuned model.
4. Evaluation report: base vs fine-tuned comparison, with examples and conclusions.
5. Short demo or walkthrough of the final model in action.

---

## Evaluation Criteria

- Depth and accuracy of the research
- Quality of the reasoning behind model, dataset, and method choices
- Data preparation quality
- Working training pipeline and reproducibility
- Rigor of testing and honesty about results
- Clarity of documentation and communication with the trainer
