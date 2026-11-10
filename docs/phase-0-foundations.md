# Phase 0 — Foundations

**Status:** Complete  
**Goal:** Shared mental model of how a language model works before touching any fine-tuning concept.

---

## How a language model works

### The core idea

A language model does one thing: given everything before, predict what comes next. This is called **next-token prediction**. Every model — GPT, Llama, Qwen, all of them — is built on this single idea applied billions of times on massive text.

### The pipeline (inference)

1. **Tokenization** — input text is split into tokens by the tokenizer. A token is roughly a word or sub-word. Each token is mapped to a token ID from the model's vocabulary (similar to how ASCII maps characters to numbers).

2. **Embedding** — each token ID is converted into a high-dimensional vector. This vector is the model's internal representation of that token. The number of dimensions (the hidden size) is model-specific — for example Qwen 2.5 1.5B uses 1536 dimensions, Llama 3.2 1B uses 2048.

3. **Attention layers** — the model runs through many transformer layers. The key operation in each is **self-attention**: every token's embedding is compared against other tokens using three learned matrices called Query, Key and Value. This computes a relevance score between token pairs — not geometric distance, but learned relevance. Softmax converts those scores into weights (0 to 1), and those weights decide how much information flows from each token into the next. This is how the model builds context. Note: the models we work with (Qwen, Llama) are **decoder-only** and use **causal masking** — each token can only attend to itself and tokens that came before it, never future tokens. This is why these models generate text left to right.

4. **Next token prediction** — after the final layer, the model outputs a probability distribution over its entire vocabulary. The token with the highest probability (or a sample from the distribution) becomes the next token.

5. **Decode** — the token ID is converted back to a string. The process repeats, now including the new token.

**During inference, the weights are completely frozen. No learning happens.**

### The pipeline (training)

Training runs the same forward pass, then adds three steps:

1. **Calculate loss** — compare the predicted next token against the actual next token from the training data. Loss is a number. High loss = the model was very wrong. Low loss = the model was close. The standard loss function for language models is cross-entropy.

2. **Backpropagation** — calculate the gradient: how much should each weight move, and in which direction, to reduce the loss. This flows backward through the entire network.

3. **Gradient descent step** — nudge every trainable weight slightly in the direction that reduces loss. Repeat for millions or billions of examples.

**Training changes the weights. Inference does not.**

### What weights contain

The weights are the numbers inside all the matrices of the model — the Q, K, V matrices in attention, the feed-forward layers, the embedding table. A 1.5B parameter model has 1.5 billion of these numbers. They encode everything the model learned during pretraining — language patterns, facts, reasoning, style.

When you download a model from Hugging Face, you are downloading a file of these numbers.

### Why we fine-tune instead of training from scratch

Pretraining a 1.5B model from scratch requires trillions of tokens, months of compute on hundreds of GPUs, and millions of dollars. A pretrained model's weights already encode general language understanding. Fine-tuning starts from those weights and shifts them further on a small, specific dataset — hours instead of months, thousands of examples instead of trillions.

### What fine-tuning changes

Fine-tuning does not load new knowledge like a database insert. It shifts the probability distributions — the model starts predicting different outputs for specific inputs. The architecture, vocabulary and general language understanding stay the same. What changes is which outputs the model considers likely for your task.

---

## Checkpoint — confirmed understanding

- Tokenizer converts text to token IDs; token IDs map to embedding vectors
- Embedding dimensions are model-specific (not fixed across all models)
- Self-attention computes learned relevance between tokens using Q, K, V matrices — not raw distance
- Softmax converts attention scores to probabilities
- Loss = how wrong the predicted token probability distribution was vs. the actual next token, measured by cross-entropy; minimising loss improves predictions
- Backpropagation calculates gradients; gradient descent updates weights
- Backpropagation and weight updates happen only during **training**, not during inference
- Fine-tuning shifts weight values on a small dataset, starting from pretrained weights
