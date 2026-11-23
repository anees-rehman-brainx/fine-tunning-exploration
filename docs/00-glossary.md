# Glossary — Key Concept Definitions

Quick reference for all concepts across Phases 1–3. One-line definitions for fast lookup during review or implementation.

---

| Concept | One-line definition |
|---|---|
| **Token** | Smallest unit of text the model processes; mapped to an integer ID |
| **Embedding** | A dense vector representing a token; learned during training |
| **Self-attention** | Mechanism that computes relevance between all token pairs using Q, K, V matrices |
| **Causal masking** | Prevents tokens from attending to future positions; enforces left-to-right generation |
| **Feed-forward layer** | Per-token transformation after attention; stores most of the model's factual knowledge |
| **Loss** | A number measuring how wrong the model's prediction was; minimized during training |
| **Cross-entropy** | The specific loss function used for next-token prediction |
| **Backpropagation** | Algorithm that computes gradients by working backwards through the network |
| **Gradient** | Per-weight number saying how much that weight contributed to the loss |
| **Learning rate** | Controls how large each weight update step is |
| **Adam** | Optimizer that adapts the learning rate per weight using momentum and variance |
| **Fine-tuning** | Continuing training on a small task-specific dataset, starting from pretrained weights |
| **SFT** | Supervised Fine-Tuning — training on (prompt, correct response) pairs |
| **PEFT** | Parameter-Efficient Fine-Tuning — only training a small subset of parameters |
| **LoRA** | Low-Rank Adaptation — trains small A and B matrices alongside frozen base weights |
| **Rank** | LoRA bottleneck dimension; controls adapter expressiveness |
| **Alpha** | LoRA scale factor; effective scale is α/r |
| **Adapter** | The A and B matrices added by LoRA; can be saved separately from the base |
| **Quantization** | Representing weights in fewer bits (e.g. 4-bit instead of 16-bit) to save memory |
| **QLoRA** | LoRA on a 4-bit quantized base model |
| **NF4** | NormalFloat4 — 4-bit data type optimized for normally-distributed model weights |
| **Double quantization** | Quantizing the quantization constants themselves; saves additional memory |
| **Overfitting** | Model memorizes training data; training loss falls but validation loss rises |
| **Epoch** | One full pass through the entire training dataset |
| **Batch** | A subset of training examples processed together before a weight update |
