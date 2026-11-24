#!/usr/bin/env python3
"""
Organizational Procedure Assistant — LoRA Training Script

Trains a LoRA adapter on top of Qwen2.5-0.5B-Instruct using the
organizational procedures dataset.

Usage:
    # From the practical/ directory with the venv active:
    python train.py

Output:
    adapter_v1/adapter_model.safetensors  — trained LoRA weights (~11 MB)
    adapter_v1/adapter_config.json        — LoRA config
    adapter_v1/tokenizer.json             — tokenizer files

After training:
    python evaluation/evaluate.py --adapter adapter_v1
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Configuration — change these to adapt to a different model or dataset
# ---------------------------------------------------------------------------

MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_DATA    = "dataset/train.jsonl"
VAL_DATA      = "dataset/validation.jsonl"
OUTPUT_DIR    = "adapter_v1"
MAX_SEQ_LEN   = 1024

LORA_R        = 16
LORA_ALPHA    = 32     # convention: alpha = 2 × rank → effective scale = 2.0
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "v_proj"]   # attention modules; add k_proj/o_proj for more capacity

LR            = 2e-4
EPOCHS        = 3
BATCH_SIZE    = 2      # per device; reduce to 1 if OOM
GRAD_ACCUM    = 4      # effective batch = BATCH_SIZE × GRAD_ACCUM = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_examples(batch: dict, tokenizer) -> dict:
    """
    Apply the model's chat template to convert the messages array into
    a single formatted string the model is trained on.

    The chat template inserts the model's special tokens (e.g. <|im_start|>)
    so the model learns to generate within its own conversation format.
    """
    texts = []
    for msgs in batch["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Suppress a harmless tokenizer parallelism warning on Mac
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = get_device()
    print(f"\nDevice : {device}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Output : {OUTPUT_DIR}\n")

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        # Qwen models use eos as pad — required for batched training
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"   # pad on the right for causal LM

    # ------------------------------------------------------------------
    # 2. Base model
    # ------------------------------------------------------------------
    print("Loading base model (downloads on first run, ~1 GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        # Do not set device_map here — let the Trainer place the model
    )
    # Disable KV cache during training (incompatible with backprop)
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # 3. LoRA adapters
    # ------------------------------------------------------------------
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: trainable params ~5.5M out of ~462M (≈1.2%)

    # ------------------------------------------------------------------
    # 4. Dataset
    # ------------------------------------------------------------------
    print("\nLoading and formatting dataset...")
    train_raw = load_jsonl(TRAIN_DATA)
    val_raw   = load_jsonl(VAL_DATA)

    train_ds = Dataset.from_list(train_raw)
    val_ds   = Dataset.from_list(val_raw)

    train_ds = train_ds.map(
        lambda batch: format_examples(batch, tokenizer),
        batched=True,
        remove_columns=["messages"],
        desc="Formatting train",
    )
    val_ds = val_ds.map(
        lambda batch: format_examples(batch, tokenizer),
        batched=True,
        remove_columns=["messages"],
        desc="Formatting val",
    )

    print(f"Train examples : {len(train_ds)}")
    print(f"Val examples   : {len(val_ds)}")

    # Sanity check: print one formatted example
    print("\n--- Sample formatted input (first 300 chars) ---")
    print(train_ds[0]["text"][:300])
    print("---\n")

    # ------------------------------------------------------------------
    # 5. Training config (SFTConfig owns both training + SFT-specific args)
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        # fp16/bf16: disabled on MPS (not all ops supported)
        fp16=False,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,  # required on MPS
        # SFT-specific — moved here from SFTTrainer in TRL 1.x
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
    )

    # ------------------------------------------------------------------
    # 6. SFTTrainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,   # replaces tokenizer= in TRL 1.x
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("Starting training...\n")
    print("Watch for:")
    print("  train/loss going DOWN        = model is learning")
    print("  eval/loss going DOWN too     = healthy generalization")
    print("  eval/loss going UP           = overfitting — stop training\n")

    trainer.train()

    # ------------------------------------------------------------------
    # 8. Save adapter
    # ------------------------------------------------------------------
    print(f"\nSaving adapter to {OUTPUT_DIR}/...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Adapter saved.\n")

    # ------------------------------------------------------------------
    # 9. Reload and verify
    # ------------------------------------------------------------------
    print("Verifying: reloading adapter from disk...")
    from peft import PeftModel

    base_check = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    loaded = PeftModel.from_pretrained(base_check, OUTPUT_DIR)
    loaded.eval()

    test_msgs = [
        {"role": "system", "content": (
            "You are an Organizational Procedure Assistant for a mid-sized software development company."
        )},
        {"role": "user", "content": "Production is completely down. All users locked out."},
    ]
    verify_inputs = tokenizer.apply_chat_template(
        test_msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    verify_len = verify_inputs["input_ids"].shape[-1]
    verify_inputs = {k: v.to(device) for k, v in verify_inputs.items()}
    with torch.no_grad():
        out = loaded.generate(**verify_inputs, max_new_tokens=80, do_sample=False)
    decoded = tokenizer.decode(out[0][verify_len:], skip_special_tokens=True)

    print("\n--- Verification output (first 80 tokens) ---")
    print(decoded)
    print("---")
    print("\nTraining complete.")
    print(f"\nNext step:")
    print(f"  python evaluation/evaluate.py --adapter {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
