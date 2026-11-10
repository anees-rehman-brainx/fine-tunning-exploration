---
diff_hash: c95c6f09f9f8cfaf651dc000d2ab00f836fdac5a4654abb01e6e9662fc5a33c0
reviewed_at: 2026-11-10T16:00:00Z

## Summary

Two-file commit: replaces the Phase 0 foundations doc with a substantially expanded version (~65 → ~419 lines). The expanded doc adds worked examples, ASCII diagrams of the full inference and training flows, a causal masking visualization, a memory-cost breakdown, and a summary table. No stack markers — pure documentation repo. Generic correctness criteria applied.

All three findings from the previous review (FFN intermediate size, LayerNorm formula, word2vec analogy) are resolved in the expanded doc. No new findings remain.

## Findings

### Critical (0)

### High (0)

### Medium (2)

- **docs/phase-0-foundations.md:147** — FFN intermediate size example is wrong for Qwen 2.5 1.5B. The sentence "so for a 1536-dim model the FFN expands to 6144 then contracts back to 1536" implies Qwen 2.5 1.5B (the model the document established has 1536 hidden dims) has an FFN intermediate of 6144. The actual `intermediate_size` in the Qwen 2.5 1.5B config is 8960 (~5.8× hidden, not 4×), and it uses a SwiGLU gated activation (two up-projection matrices) rather than the classic 2-layer FFN. The 4× heuristic predates SwiGLU architectures and no longer applies to the models being studied. *(correctness)*
  - status: fixed
  - dev_reasoning: N/A
  - ai_reasoning: The staged doc now explicitly states "Qwen 2.5 1.5B (hidden size 1536) uses an intermediate size of 8960 with a SwiGLU gated activation (two up-projection matrices) rather than the classic 4× rule from the original Transformer paper." The incorrect 6144 figure is gone and the actual value with SwiGLU context is now present.

- **docs/phase-0-foundations.md:156** — LayerNorm formula shown is Post-LayerNorm (`output = LayerNorm(x + SubLayer(x))`), which is the 2017 "Attention Is All You Need" style. Qwen 2.5 and Llama 3.2 — the two models this project works with — use **Pre-RMSNorm**: `output = x + SubLayer(RMSNorm(x))`. The formula is applied before the sub-layer, not after the residual add, and uses RMSNorm not LayerNorm. *(correctness)*
  - status: fixed
  - dev_reasoning: N/A
  - ai_reasoning: The staged doc now shows `output = x + SubLayer(RMSNorm(x))` and explicitly notes "Modern models like Llama and Qwen use Pre-RMSNorm — the normalisation is applied before each sub-layer, not after the residual add. This differs from the Post-LayerNorm in the original 2017 Transformer paper." Both the formula and the explanatory prose are correct.

### Low (1)

- **docs/phase-0-foundations.md:64** — The vector arithmetic analogy (`vector("king") ≈ vector("queen") - vector("woman") + vector("man")`) is a classic word2vec result from 2013. It is demonstrable on static word2vec embeddings but does not reliably hold in modern LLM embedding layers. *(correctness)*
  - status: fixed
  - dev_reasoning: N/A
  - ai_reasoning: The staged doc now attributes the arithmetic to "standalone word vectors (word2vec, 2013)" and adds "In modern LLM embedding layers the same principle holds — similar tokens cluster together — but the arithmetic is noisier because these embeddings are one layer inside a deep contextual model, not standalone distributional vectors." The analogy is correctly scoped and the qualifier is exactly what was needed.

## Reviewed files

- `.code-reviews/brainx-review.md` — updated review file, no issues
- `docs/phase-0-foundations.md` — expanded from ~65 to ~419 lines, reviewed in full; all prior findings resolved
