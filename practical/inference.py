#!/usr/bin/env python3
"""
Organizational Procedure Assistant — Interactive Inference

Loads the fine-tuned LoRA adapter and runs an interactive command-line session.

Usage:
    python inference.py                     # uses adapter_v1/ by default
    python inference.py --adapter adapter_v1

Type your issue description and press Enter.
Type 'quit' or Ctrl+C to exit.
"""

import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM = (
    "You are an Organizational Procedure Assistant for a mid-sized software development company. "
    "When someone reports a situation, identify the issue type, state the severity, and provide "
    "the exact step-by-step procedure to follow — including escalation triggers and documentation requirements. "
    "Be specific, structured, and actionable."
)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(adapter_path: str):
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    print("Ready.\n")
    return model, tokenizer, device


def classify(model, tokenizer, device, user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]
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
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="adapter_v1",
        help="Path to the LoRA adapter directory (default: adapter_v1)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"ERROR: Adapter not found at '{args.adapter}'")
        print("Run train.py first to generate the adapter.")
        return

    model, tokenizer, device = load_model(args.adapter)

    print("=" * 60)
    print("Organizational Procedure Assistant")
    print("=" * 60)
    print("Describe any organizational issue to get the procedure.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Issue: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        print("\nProcedure:\n")
        result = classify(model, tokenizer, device, user_input)
        print(result)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
