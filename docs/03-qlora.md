# Phase 3 — QLoRA

**Status:** IN PROGRESS  
**Goal:** Understand quantization well enough to explain why QLoRA enables larger models, what the trade-offs are, and how to use it correctly.

---

*Topics studied conceptually but not yet formally documented or verified in conversation. Full notes will be added after discussion.*

---

## Topics to Document

- What quantization is — representing weights in fewer bits to save memory
- FP32, FP16, BF16 — what each sacrifices in precision
- 8-bit quantization (INT8) — how it works, what it loses
- 4-bit quantization — the precision challenge
- NF4 (NormalFloat4) — why it is information-theoretically optimal for normally-distributed model weights
- Double quantization — quantizing the quantization constants themselves
- The dequantize-compute-quantize cycle — why QLoRA can still compute accurate gradients
- What is still in FP16 during QLoRA (the LoRA adapters and optimizer states)
- LoRA vs QLoRA — when to use each
- Memory estimation — how to estimate whether a model fits a given memory budget with QLoRA

---

## Placeholder Summary

QLoRA extends LoRA by loading the base model in 4-bit (instead of 16-bit), reducing base model memory from 3 GB to ~0.75 GB for a 1.5B model. The adapters still train in 16-bit. This enables fine-tuning much larger models (7B, 13B) on the same hardware.

The trade-off: 4-bit quantization introduces some precision loss in the base. For most tasks the quality difference from LoRA is small but measurable.

**Quick comparison:**

| | LoRA | QLoRA |
|---|---|---|
| Base model precision | FP16 (~3 GB for 1.5B) | 4-bit NF4 (~0.75 GB for 1.5B) |
| Adapter precision | FP16 | FP16 |
| Optimizer states | FP32 | FP32 |
| Total memory (1.5B) | ~3.1 GB | ~1 GB |
| Feasible model size on 16 GB | Up to ~3B | Up to ~13B |
| Quality vs full FT | Very close | Slightly lower |
| Use when | RAM is sufficient | RAM is tight, need bigger model |
