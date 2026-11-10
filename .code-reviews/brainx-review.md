---
diff_hash: c005bd1191756b7767fa3ecb6a4192046e0981561c4a6eb907ae3b2fc27abfc6
reviewed_at: 2026-11-10T15:16:00Z

## Summary

Two-file documentation commit: marks Phase 0 as complete in the training plan, and adds `docs/phase-0-foundations.md` — a new foundational document explaining how language models work (tokenization, embeddings, attention, training vs. inference, and why fine-tuning exists). No stack markers present; this is a pure documentation repo. Review applied generic correctness criteria.

The document is accurate and well-written at the foundational level. One medium technical inaccuracy and one low-severity wording issue are flagged below.

## Findings

### Critical (0)

### High (0)

### Medium (1)

- **docs/phase-0-foundations.md:23** — Attention described as comparing "every token against every other token," which is accurate for bidirectional models (BERT) but not for the decoder-only models this training plan focuses on (Llama, Qwen, GPT). Those use **causal (masked) self-attention**: each token can only attend to itself and tokens that came before it — not future tokens. This distinction matters for fine-tuning intuition because it explains why these models generate text left-to-right and why the KV-cache works. *(correctness)*
  - status: open
  - dev_reasoning: N/A
  - ai_reasoning: The sentence "every token's embedding is compared against every other token" is a correct description of full self-attention (encoder-only models), but all models named in this repo (Qwen 2.5, Llama 3.2) are decoder-only and use causal masking. A one-sentence qualifier ("…though in decoder-only models like Llama and Qwen, tokens can only attend to previous tokens, not future ones — this is called causal or masked self-attention") would make the document accurate for the actual models being studied.

### Low (1)

- **docs/phase-0-foundations.md:62** — Checkpoint bullet "Loss = difference between predicted and actual next token" uses "difference," which implies a simple subtraction (like MAE). The body text correctly names cross-entropy as the loss function; the checkpoint bullet should match — e.g. "Loss = how wrong the predicted token distribution was vs. the actual next token, measured by cross-entropy." *(correctness)*
  - status: open
  - dev_reasoning: N/A
  - ai_reasoning: "Difference" is technically imprecise for cross-entropy loss. The body already explains this correctly; the checkpoint is just slightly inconsistent. Very low risk — it's an educational document and the body text is accurate — but the checkpoint is meant to be a durable reference and should be precise.

## Reviewed files

- `TRAINING-PLAN.md` — status cell updated, no issues
- `docs/phase-0-foundations.md` — new file, reviewed in full
