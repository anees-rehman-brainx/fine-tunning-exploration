---
diff_hash: 612092ce39ccee9564d3834d449cf1352fe28290e7fcba091ce2311a870f914f
reviewed_at: 2026-11-23T15:25:00Z

## Summary

Documentation reorganization: the monolithic `LEARNING.md` is split into four phase-scoped files under `docs/`. `docs/01-fundamentals.md` retains the Phase 1 content with adjusted heading levels and minor prose edits. `docs/02-lora.md` extracts the LoRA content with a corrected memory figure (3.5 GB → 3.1 GB, now consistent with the breakdown math). `docs/03-qlora.md` is a structured placeholder for QLoRA with a comparison table. `docs/04-training-workflow.md` is a new, detailed end-to-end implementation guide. `README.md` navigation is updated to point to the new structure. No code is involved — this is a pure documentation change.

## Findings

### Critical (0)

### High (0)

### Medium (1)

- **LEARNING.md (removed section)** — The "Quick Reference" glossary table (20+ key concept one-liners: Token, Embedding, Self-attention, Causal masking, LoRA, Rank, Alpha, QLoRA, NF4, etc.) was at the bottom of `LEARNING.md` and has been dropped from the reorganization. None of the four new docs files contain it. This was the only location where all Phase 1–3 concepts were defined in one place for fast lookup. *(correctness/completeness)*
  - status: open
  - dev_reasoning: N/A
  - ai_reasoning: The glossary is not in `docs/01-fundamentals.md`, `docs/02-lora.md`, `docs/03-qlora.md`, or `docs/04-training-workflow.md`. It was valuable consolidated reference material. If intentionally removed (e.g., planned to live in a future `docs/00-glossary.md`), that file should be created or the glossary moved to `docs/01-fundamentals.md` to preserve accessibility.

### Low (1)

- **docs/01-fundamentals.md:217** — Section 1.9 shortened "fine-tuning is unreliable for this — the model memorizes patterns, not facts precisely" to "fine-tuning is unreliable for this", dropping the clause that explains *why*. The removed clause is accurate and meaningful context for understanding the limitation. *(maintainability)*
  - status: open
  - dev_reasoning: N/A
  - ai_reasoning: "The model memorizes patterns, not facts precisely" is the key nuance — it explains why fine-tuning is unreliable for factual injection (as opposed to, say, style or format). Without it, the guidance reads as an unexplained assertion.

## Reviewed files

- `README.md` — updated navigation and corrected LoRA memory figure
- `LEARNING.md` → `docs/01-fundamentals.md` — renamed, Phase 1 content only, heading levels adjusted
- `docs/02-lora.md` — new file, Phase 2 LoRA content extracted from old LEARNING.md, memory figure corrected
- `docs/03-qlora.md` — new file, structured QLoRA placeholder with comparison table
- `docs/04-training-workflow.md` — new file, detailed end-to-end training workflow and implementation checklist
