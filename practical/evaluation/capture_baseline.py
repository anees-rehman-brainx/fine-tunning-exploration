#!/usr/bin/env python3
"""
Capture baseline outputs from the base model BEFORE any fine-tuning.
Run this once. The saved outputs are your before-training benchmark.

Usage:
    python capture_baseline.py

Output:
    evaluation/baseline_outputs.jsonl
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM = (
    "You are an Organizational Procedure Assistant for a mid-sized software development company. "
    "When someone reports a situation, identify the issue type, state the severity, and provide "
    "the exact step-by-step procedure to follow — including escalation triggers and documentation requirements. "
    "Be specific, structured, and actionable."
)

TEST_PROMPTS = [
    "The website went offline 5 minutes ago. All 500 users are locked out.",
    "Client sent a formal email this morning saying they are reconsidering the entire contract.",
    "Invoice from 8 weeks ago is still unpaid. We have sent 3 reminders with no response.",
    "Senior developer just texted: she is resigning. Joining a startup next month.",
    "Client is asking for push notifications. It was not in the original scope.",
    "We just deployed and immediately got reports of the user profile page breaking.",
    "Someone sent me unusual logs — looks like external IPs are querying our DB.",
    "The dashboard is painfully slow for all users. Takes 12 seconds to load.",
    "Delivery was rejected — client says the mobile layout is unusable.",
    "Two developers had a serious argument. One filed a formal complaint with me.",
]


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device, user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]
    # transformers 5.x: apply_chat_template returns a BatchEncoding, not a tensor
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


def main():
    out_path = os.path.join(os.path.dirname(__file__), "baseline_outputs.jsonl")
    if os.path.exists(out_path):
        print(f"WARNING: {out_path} already exists. Overwriting.")

    print("Loading base model (downloads on first run)...\n")
    model, tokenizer, device = load_model()

    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] {prompt[:60]}...")
        output = generate(model, tokenizer, device, prompt)
        results.append({"id": i, "prompt": prompt, "baseline_output": output})
        print(f"Output preview: {output[:150]}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nBaseline saved: {out_path}")
    print("Do not modify this file. It is your before-training benchmark.")


if __name__ == "__main__":
    main()
