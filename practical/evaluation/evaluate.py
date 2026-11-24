#!/usr/bin/env python3
"""
Post-training evaluation: run the same 10 test prompts through the fine-tuned
model and print a side-by-side comparison with the baseline.

Usage:
    python evaluate.py --adapter ../adapter_v1

Output:
    evaluation/finetuned_outputs.jsonl
    Printed score sheet
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_outputs.jsonl")

SYSTEM = (
    "You are an Organizational Procedure Assistant for a mid-sized software development company. "
    "When someone reports a situation, identify the issue type, state the severity, and provide "
    "the exact step-by-step procedure to follow — including escalation triggers and documentation requirements. "
    "Be specific, structured, and actionable."
)

EXPECTED_ISSUE_TYPES = [
    "Production Outage",
    "Client Escalation",
    "Invoice Overdue",
    "Key Developer Leaving",
    "Scope Change",
    "Post-Deploy Bug",
    "Security Incident",
    "Performance Issue",
    "QA Failure",
    "HR Issue",
]

REQUIRED_FORMAT_MARKERS = [
    "ISSUE TYPE:",
    "SEVERITY:",
    "ESCALATION",
    "DOCUMENTATION",
]

ROLE_NAMES = ["AM", "CTO", "Tech Lead", "Account Director", "DevOps", "PM", "HR Manager"]


def score_output(output: str) -> dict:
    output_upper = output.upper()
    return {
        "has_issue_type": "ISSUE TYPE:" in output_upper,
        "has_severity": "SEVERITY:" in output_upper,
        "has_escalation": "ESCALATION" in output_upper,
        "has_documentation": "DOCUMENTATION" in output_upper,
        "has_numbered_steps": any(f"\n{i}." in output for i in range(1, 6)),
        "mentions_role": any(role in output for role in ROLE_NAMES),
    }


def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        print(f"WARNING: No baseline found at {BASELINE_PATH}. Run capture_baseline.py first.")
        return {}
    with open(BASELINE_PATH) as f:
        return {r["id"]: r for r in (json.loads(l) for l in f)}


def load_model(adapter_path: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device, user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory")
    args = parser.parse_args()

    baseline = load_baseline()
    model, tokenizer, device = load_model(args.adapter)

    results = []
    scores = []

    for i, (prompt_id, base_data) in enumerate(sorted(baseline.items()), 1):
        prompt = base_data["prompt"]
        baseline_out = base_data["baseline_output"]

        print(f"\n{'='*60}")
        print(f"[{i}/10] {prompt}")
        print(f"\n--- BASE MODEL ---")
        print(baseline_out[:300])

        ft_out = generate(model, tokenizer, device, prompt)
        print(f"\n--- FINE-TUNED MODEL ---")
        print(ft_out[:300])

        s = score_output(ft_out)
        total = sum(s.values())
        scores.append(total)
        print(f"\nScore: {total}/6 — {s}")

        results.append({
            "id": i,
            "prompt": prompt,
            "baseline_output": baseline_out,
            "finetuned_output": ft_out,
            "scores": s,
        })

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    avg = sum(scores) / len(scores)
    print(f"Average score: {avg:.1f} / 6.0")
    print(f"Examples scoring 5+ / 6: {sum(1 for s in scores if s >= 5)} / {len(scores)}")

    if avg >= 5.0:
        print("\nResult: PASS — Training worked. Proceed to deployment.")
    elif avg >= 3.5:
        print("\nResult: PARTIAL — Identify weak issue types and add more examples.")
    else:
        print("\nResult: FAIL — Check dataset format and model loading. Retrain.")

    out_path = os.path.join(os.path.dirname(__file__), "finetuned_outputs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
