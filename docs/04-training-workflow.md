# Phase 4 — Training Workflow

**Status:** COMPLETED (2026-11-24)  
**Goal:** A complete, reusable reference for the full fine-tuning lifecycle — from environment setup through production deployment. Applies to any dataset and any task. Project-specific decisions for the Organizational Procedure Assistant are marked clearly.

---

## Overview

```
PHASE 0 — Environment Setup
  Install Python, create venv, install libraries, verify hardware

PHASE 1 — Pre-Training (Before You Touch the Model)
  Select model → validate dataset → capture baseline → checkpoint

PHASE 2 — Training
  Configure LoRA → configure trainer → run training → monitor → save

PHASE 3 — Post-Training Evaluation
  Run same test prompts → score outputs → compare → decide

PHASE 4 — Deployment (Making It Live)
  Local script → FastAPI server → merge and push (three options)
```

**Rule:** Never skip Phase 1. If you do not capture a baseline before training, you have no way to prove the training did anything.

---

## How This Works — The Complete Mental Model

Before touching any code, understand what is actually happening end to end. Every phase below maps to one part of this picture.

### Where does the model come from?

Models live on **Hugging Face Hub** — an open platform where research labs and companies publish pretrained model weights for free. Think of it like npm for AI models.

When your code calls `from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`, it:
1. Checks your local disk cache (`~/.cache/huggingface/hub/`)
2. If not cached: downloads the weights once (~1 GB for a 0.5B model)
3. On every subsequent run: reads from disk instantly, no internet needed

What is downloaded — the **base model** — already knows English, can follow instructions, write code, reason, and summarize. It was trained by the model's creators on trillions of tokens over months. You are not building a model from scratch. You are taking something that already works well and shifting its behaviour toward your specific task.

### What is fine-tuning actually doing?

The base model has weights — hundreds of millions of numbers that encode everything it has learned. Fine-tuning continues training it on your small, task-specific dataset. The model sees your (prompt, correct response) pairs many times, and its weights shift slightly toward producing outputs that look like your training data.

**What changes:** the weight values shift toward your task's patterns.  
**What does not change:** the architecture, the tokenizer, the general language understanding built in pretraining.

Fine-tuning is not like inserting rows into a database. It shifts probability distributions — it makes certain outputs more likely for certain inputs. The model does not "learn facts"; it learns patterns of behaviour.

### What is LoRA and why does it exist?

Full fine-tuning updates all weights simultaneously. For a 1.5B parameter model that requires ~24 GB of memory — more than a 16 GB Mac has. LoRA solves this by not touching the base weights at all.

Instead, LoRA adds small trainable matrices alongside a few key layers:

```
Normal layer output:  y = W · x           (W is frozen, 3 GB)
LoRA layer output:    y = W · x  +  (α/r) · B · A · x
                                    └────────────────┘
                                    only these 5.5M params train
```

`A` and `B` are tiny matrices. `A` starts random, `B` starts at zero (so the adapter adds nothing at the start of training — a clean baseline). As training proceeds, only `A` and `B` update. The base model `W` never changes.

Result: ~5.5M trainable parameters instead of 500M+. Memory drops from ~24 GB to ~3.1 GB.

### What happens during one training step?

```
1. FORWARD PASS
   Feed a training prompt into the model.
   It predicts the next token at every position in the response.

2. LOSS CALCULATION
   Compare what the model predicted against what your dataset says was correct.
   Cross-entropy loss: a single number measuring how wrong the prediction was.
   High loss = model was wrong. Loss of 0 = perfect prediction.

3. BACKPROPAGATION
   Work backwards through the network to compute a gradient for every
   trainable parameter (the LoRA A and B matrices only).
   Each gradient says: "increase this weight → loss goes up/down by this much."

4. WEIGHT UPDATE
   new_weight = old_weight − (learning_rate × gradient)
   Only the LoRA A and B matrices update. Base weights stay frozen.

5. REPEAT
   Next batch. Loss goes down a little each time.
   After hundreds of steps across the whole dataset, the adapter has learned.
```

### What is saved after training?

Only the adapter — not the base model (it was never modified):

```
adapter_v1/
  adapter_model.safetensors   ← the trained A and B matrices  (~11 MB)
  adapter_config.json         ← rank, alpha, target modules, base model name
  tokenizer.json              ← tokenizer files for convenience
```

The 1 GB base model sits in the Hugging Face cache unchanged. At inference time you load both together: base + adapter. You can share the 11 MB adapter file with anyone who has the same base model.

### How do you know if training worked?

You compare outputs on the same prompts, before and after:

```
BEFORE TRAINING (base model, zero-shot):
  Input:  "Production is down. All users locked out."
  Output: "You should contact your IT department and check the servers."
          → Generic. No structure. No role names. No escalation rule.

AFTER TRAINING (fine-tuned model, zero-shot):
  Input:  "Production is down. All users locked out."
  Output: "ISSUE TYPE: Production Outage (P1)
           SEVERITY: Critical
           IMMEDIATE (0–10 minutes):
           1. Post in #incidents Slack immediately..."
          → Structured. Company-specific. Actionable.
```

This is why you capture the baseline **before** training. Without it, you cannot show the difference.

### How does it go live?

Three options in increasing production-readiness:

```
Option A — Local script
  Load base + adapter in Python. Call a function. Get a response.
  Good for demos and personal use. No server needed.

Option B — FastAPI REST API
  Wrap Option A in an HTTP endpoint.
  POST /classify → { "procedure": "..." }
  Any tool — Slack bot, internal dashboard, Jira automation — can call it.
  Deploy on any server.

Option C — Merge and push
  Bake the adapter permanently into the base model weights.
  One combined model file (~1 GB). No adapter loading overhead.
  Push to Hugging Face as your own private model.
  Serve via any inference provider.
```

### The full picture

```
Hugging Face Hub
      ↓  download once (~1 GB), cached permanently
Base model on disk
      ↓  capture baseline FIRST — save what it outputs untrained
Baseline outputs saved  ←──────────────── comparison reference
      ↓
Attach LoRA adapters (5.5M trainable params, base frozen)
      ↓
Train on your dataset (forward → loss → backprop → update adapters)
  167 examples × 3 epochs × ~8 steps/epoch ≈ ~500 total steps
  ~15–30 minutes on Apple M1
      ↓
Save adapter to disk (~11 MB)
      ↓
Run same 10 test prompts through fine-tuned model
Compare against baseline → score sheet → PASS or RETRAIN
      ↓
Deploy as local script / FastAPI / merged model
```

---

## Phase 0 — Environment Setup

### 0.1 Python and Virtual Environment

```bash
# Check Python version — need 3.10 or higher
python3 --version

# Create a virtual environment in the practical/ folder
cd practical/
python3 -m venv .venv

# Activate it
source .venv/bin/activate   # Mac / Linux
# .venv\Scripts\activate    # Windows

# Confirm you are in the venv
which python  # should point inside practical/.venv/
```

### 0.2 Install Libraries

```bash
pip install torch transformers peft trl datasets accelerate
```

Library purposes:

| Library | What it does |
|---|---|
| `torch` | The tensor engine and autograd. Everything else sits on top of it. |
| `transformers` | Load pretrained models and tokenizers from Hugging Face. |
| `peft` | Add LoRA adapters, configure rank / alpha / target modules. |
| `trl` | `SFTTrainer` — handles the full SFT training loop. |
| `datasets` | Load, split, and process datasets in memory-efficient format. |
| `accelerate` | Handles multi-device and Apple MPS acceleration automatically. |

For QLoRA only (requires Linux + CUDA — does not work on Mac M1):
```bash
pip install bitsandbytes
```

### 0.3 Verify Hardware

```python
import torch

# Apple Silicon (MPS)
print(torch.backends.mps.is_available())   # should print True on M1/M2

# CUDA (NVIDIA GPU, not on Mac)
print(torch.cuda.is_available())

# Fallback: CPU (slow but always works)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
```

### 0.4 Verify Disk Space

Models and training artifacts need space:

| Item | Approximate size |
|---|---|
| `Qwen2.5-0.5B-Instruct` weights | ~1 GB |
| `Qwen2.5-1.5B-Instruct` weights | ~3 GB |
| LoRA adapter after training | ~11–50 MB |
| Training checkpoints (optional) | ~1 GB per checkpoint |

Minimum free disk: 5 GB. Recommended: 10 GB.

---

## Phase 1 — Pre-Training

Do all of this before writing a single line of training code.

### 1.1 Model Selection

Criteria for choosing a base model:

| Criterion | Why it matters |
|---|---|
| **Size vs hardware** | Model must fit in memory at training time (see LoRA memory table in `docs/02-lora.md`) |
| **Instruction-tuned** | An instruction-tuned base already responds to prompts — better SFT starting point |
| **Task alignment** | Pick a model pretrained on data similar to your task domain |
| **Community support** | Is it supported by PEFT and TRL? Check the PEFT docs. |
| **License** | Commercial use? Check the model card on Hugging Face. |

**Decision for this project:** `Qwen/Qwen2.5-0.5B-Instruct`  
Reason: small enough for M1 16 GB, instruction-tuned baseline, fully supported by PEFT and TRL.

### 1.2 Dataset Validation

Before training, confirm the dataset is correct:

```python
import json

# Load and inspect
with open("dataset/train.jsonl") as f:
    examples = [json.loads(line) for line in f]

# Check count
print(f"Total examples: {len(examples)}")

# Check structure
ex = examples[0]
assert "messages" in ex
assert len(ex["messages"]) == 3
assert ex["messages"][0]["role"] == "system"
assert ex["messages"][1]["role"] == "user"
assert ex["messages"][2]["role"] == "assistant"
print("Structure: OK")

# Check no empty messages
for i, ex in enumerate(examples):
    for msg in ex["messages"]:
        assert len(msg["content"].strip()) > 0, f"Empty content at example {i}"
print("No empty messages: OK")

# Check output format — all responses should start with ISSUE TYPE:
bad = [i for i, ex in enumerate(examples)
       if not ex["messages"][2]["content"].startswith("ISSUE TYPE:")]
print(f"Format check: {len(bad)} examples failed (should be 0)")
```

**What to verify before proceeding:**
- [ ] Example count matches expectations (train ≥ 100, validation ≥ 20)
- [ ] Every example has exactly 3 messages: system, user, assistant
- [ ] No empty content fields
- [ ] All assistant responses follow the expected output format
- [ ] Train and validation sets do not share identical user messages (data leak check)

**Data leak check:**
```python
with open("dataset/train.jsonl") as f:
    train_msgs = {json.loads(l)["messages"][1]["content"] for l in f}
with open("dataset/validation.jsonl") as f:
    val_msgs = {json.loads(l)["messages"][1]["content"] for l in f}

overlap = train_msgs & val_msgs
print(f"Leaked messages: {len(overlap)}")  # must be 0
```

### 1.3 Define and Save Your Test Prompts

Pick 10 prompts that are **not in the training set** and will not change. These are your evaluation set for before/after comparison. Write them to a file so they cannot drift.

**Decision for this project — test prompts saved at `practical/evaluation/test_prompts.txt`:**

```
1. The website went offline 5 minutes ago. All 500 users are locked out.
2. Client sent a formal email this morning saying they are reconsidering the entire contract.
3. Invoice from 8 weeks ago is still unpaid. We have sent 3 reminders with no response.
4. Senior developer just texted: she is resigning. Joining a startup next month.
5. Client is asking for push notifications. It was not in the original scope.
6. We just deployed and immediately got reports of the user profile page breaking.
7. Someone sent me unusual logs — looks like external IPs are querying our DB.
8. The dashboard is painfully slow for all users. Takes 12 seconds to load.
9. Delivery was rejected — client says the mobile layout is unusable.
10. Two developers had a serious argument. One filed a complaint with me.
```

These prompts cover all 10 unique issue types (excluding onboarding and invoice-only which we have covered in training). They are not in `train.jsonl` or `validation.jsonl`.

### 1.4 Capture the Baseline

Run the base model — untouched, no fine-tuning — on all 10 test prompts. Save the outputs.

```python
# practical/evaluation/capture_baseline.py
import json
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
    "Two developers had a serious argument. One filed a complaint with me.",
]

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
model.eval()

results = []
for prompt in TEST_PROMPTS:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=400, do_sample=False)
    decoded = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    results.append({"prompt": prompt, "baseline_output": decoded})
    print(f"Prompt: {prompt[:50]}...")
    print(f"Output: {decoded[:200]}\n---")

with open("evaluation/baseline_outputs.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("Baseline saved to evaluation/baseline_outputs.jsonl")
```

**What you will see:** The base model gives generic helpful advice — "Contact your IT department" or "Have a meeting" — not your specific company procedures with role names and escalation rules. That gap is exactly what training closes.

### 1.5 Phase 1 Checkpoint

Do not start training until all of these are true:

- [ ] Virtual environment active, all libraries installed
- [ ] `torch.backends.mps.is_available()` returns True (on Mac M1)
- [ ] Dataset validation passes with zero errors
- [ ] No data leak between train and validation sets
- [ ] 10 test prompts written to file and committed
- [ ] Baseline outputs saved to `evaluation/baseline_outputs.jsonl`
- [ ] Enough disk space confirmed

---

## Phase 2 — Training

### 2.1 LoRA Configuration

Key decisions and their meaning:

| Parameter | Our value | Why |
|---|---|---|
| `r` (rank) | 16 | Good balance — enough expressiveness for a 12-class procedure task |
| `lora_alpha` | 32 | Convention: alpha = 2 × rank. Effective scale = 32/16 = 2.0 |
| `target_modules` | `["q_proj", "v_proj"]` | Attention modules; sufficient for behavior/format learning |
| `lora_dropout` | 0.05 | Light regularization; prevents overfitting on small datasets |
| `bias` | `"none"` | Do not train bias terms — keeps adapter small |
| `task_type` | `CAUSAL_LM` | Decoder-only text generation |

For a larger task or higher-quality result: also target `k_proj` and `o_proj`.

### 2.2 Training Configuration

| Parameter | Our value | Why |
|---|---|---|
| `learning_rate` | 2e-4 | Standard for LoRA SFT; lower if you see loss spikes |
| `per_device_train_batch_size` | 2 | Small batch for 16 GB RAM; set to 1 if OOM |
| `gradient_accumulation_steps` | 4 | Effective batch = 2 × 4 = 8; simulates larger batch |
| `num_train_epochs` | 3 | 3 passes through 167 examples. Watch for overfitting. |
| `max_seq_length` | 1024 | Long enough for our procedure responses (~400 tokens) |
| `warmup_ratio` | 0.03 | Gradual ramp-up; prevents early instability |
| `lr_scheduler_type` | `cosine` | Decays LR smoothly to near zero |
| `eval_strategy` | `epoch` | Check validation loss after each epoch |
| `save_strategy` | `epoch` | Save a checkpoint after each epoch |
| `load_best_model_at_end` | `True` | Automatically picks the checkpoint with best eval loss |

### 2.3 The Training Script

**`practical/train.py`** — written and ready to run.

```python
# High-level structure of practical/train.py:
# 1. Detect device (MPS / CUDA / CPU)
# 2. Load tokenizer and base model (Qwen2.5-0.5B-Instruct)
# 3. Apply LoRA config via get_peft_model()
# 4. Load and format dataset (apply_chat_template → text field)
# 5. Configure SFTTrainer with TrainingArguments
# 6. trainer.train() — monitors train loss and eval loss per epoch
# 7. Save adapter to adapter_v1/
# 8. Reload adapter and run one verification prompt
```

Run it from the `practical/` directory with the venv active:

```bash
python train.py
```

### 2.4 What to Monitor During Training

Watch the terminal output. TRL prints a table every N steps:

```
Step  Training Loss  Validation Loss
 50   1.4200         —
100   0.9800         0.8500
150   0.6100         —
200   0.4200         0.4800     ← validation loss still going down = healthy
250   0.2900         0.5200     ← validation loss rising = early overfitting signal
```

**Healthy training:** both losses going down.  
**Overfitting:** training loss keeps falling, validation loss rises. Stop training and use the previous checkpoint.  
**Not learning:** training loss barely moves. Learning rate may be too low or dataset may have a formatting issue.

### 2.5 After Training — Save and Verify

```python
# Save the adapter (not the base model — it did not change)
model.save_pretrained("./adapter_v1")
tokenizer.save_pretrained("./adapter_v1")

# Immediately verify: reload and test one prompt
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(MODEL)
loaded = PeftModel.from_pretrained(base, "./adapter_v1")

# Run one prompt — confirm output looks trained, not random
```

**What gets saved:**

| File | What it is | Size |
|---|---|---|
| `adapter_model.safetensors` | The trained A and B LoRA matrices | ~11–50 MB |
| `adapter_config.json` | Rank, alpha, target modules, base model name | ~1 KB |
| `tokenizer.json` + related | Tokenizer files | ~2 MB |

The base model weights are NOT saved again — they are unchanged.

---

## Phase 3 — Post-Training Evaluation

### 3.1 Run the Same 10 Test Prompts

```python
# practical/evaluation/evaluate.py
# Same structure as capture_baseline.py but with the fine-tuned model loaded

from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
model = PeftModel.from_pretrained(base, "../adapter_v1").to(device)
model.eval()

# Run same TEST_PROMPTS, save to evaluation/finetuned_outputs.jsonl
```

### 3.2 Score Sheet — Side by Side

For each of the 10 prompts, fill this in:

| Prompt # | Base model output | Fine-tuned output | Issue type correct | Format correct | Roles named | Procedure accurate |
|---|---|---|---|---|---|---|
| 1 | | | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ |
| 2 | | | | | | |
| ... | | | | | | |

**Metrics explained:**

| Metric | What "correct" means |
|---|---|
| Issue type correct | Starts with `ISSUE TYPE: [exact name]` matching our 12 categories |
| Format correct | Has SEVERITY, numbered phase sections, ESCALATION TRIGGERS, DOCUMENTATION |
| Roles named | Uses AM, CTO, Tech Lead, Account Director — not generic "management" |
| Procedure accurate | Steps match what we defined in the dataset |

**Minimum threshold to consider training successful:** 8 out of 10 prompts score correct on all 4 metrics.

### 3.3 Decision After Evaluation

| Score | Decision |
|---|---|
| 9–10 / 10 | Training worked well. Proceed to deployment. |
| 7–8 / 10 | Acceptable. Identify which issue types are weak and add more training examples for them. |
| 5–6 / 10 | Partial. Likely a formatting issue in the dataset or the model is too small. Investigate. |
| < 5 / 10 | Did not work. Check: is the dataset format correct? Is the model loading correctly? Run the data validation again. |

---

## Phase 4 — Deployment

### 4.1 Option A — Local Inference Script

Simplest option. Good for personal use or demos.

```python
# practical/inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "./adapter_v1"
SYSTEM = "You are an Organizational Procedure Assistant..."

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
model = PeftModel.from_pretrained(base, ADAPTER).to(device)
model.eval()

def classify(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=500, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        msg = input("\nDescribe the issue: ")
        print(classify(msg))
```

### 4.2 Option B — FastAPI REST Server (Recommended for Internal Tools)

Wraps the model in an HTTP API. Any tool — Slack bot, internal dashboard, Jira automation, email system — can call it.

```python
# practical/server.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model once at startup
# (same loading code as Option A above)

class Request(BaseModel):
    message: str

class Response(BaseModel):
    procedure: str

@app.post("/classify", response_model=Response)
def classify_issue(req: Request):
    result = classify(req.message)
    return Response(procedure=result)
```

```bash
# Run the server
pip install fastapi uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```

```bash
# Call it from anywhere
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "Production is down. All users affected."}'
```

Deploy on: EC2, Fly.io, Railway, an office Mac mini, any Linux server.

### 4.3 Option C — Merge Adapter and Push to Hugging Face

After merging, the adapter no longer exists separately. The LoRA weights are permanently baked into the base model. The result behaves like your fine-tuned model at full inference speed with no adapter loading overhead.

```python
# Merge LoRA weights permanently into the base model
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
peft_model = PeftModel.from_pretrained(base, "./adapter_v1")

merged = peft_model.merge_and_unload()     # returns a plain AutoModelForCausalLM
merged.save_pretrained("./merged_model")   # save the full merged weights (~1 GB)

# Optional: push to your private Hugging Face repo
merged.push_to_hub("your-username/org-procedure-assistant-0.5b")
```

Use this when: you want to serve via Hugging Face Inference Endpoints, share the model, or deploy on a serverless inference provider.

**Comparison:**

| Option | Effort | Who can use it | Speed | Cost |
|---|---|---|---|---|
| A — Local script | Minimal | You only | Fine | Free |
| B — FastAPI server | Medium | Any internal service via HTTP | Fine | Server cost |
| C — Merge + push | Medium | Anyone with the HF model | Slightly faster | HF hosting |

---

## Full Implementation Checklist

### Phase 0 — Environment
- [ ] Python 3.10+ confirmed
- [ ] Virtual environment created and activated
- [ ] Libraries installed: transformers, peft, trl, datasets, torch, accelerate
- [ ] MPS available confirmed on Mac M1
- [ ] Disk space confirmed (5 GB minimum)

### Phase 1 — Pre-Training
- [ ] Model selected and justified
- [ ] Dataset validated: structure, counts, format, no data leak
- [ ] 10 held-out test prompts written to `evaluation/test_prompts.txt`
- [ ] Baseline outputs captured and saved to `evaluation/baseline_outputs.jsonl`

### Phase 2 — Training
- [ ] LoRA config defined: rank, alpha, target modules, dropout
- [ ] Training config defined: LR, batch size, epochs, scheduler
- [ ] `train.py` written and reviewed
- [ ] Training run started
- [ ] Loss curves monitored — no overfitting
- [ ] Adapter saved to `adapter_v1/`
- [ ] Adapter reloaded and verified post-save

### Phase 3 — Evaluation
- [ ] Fine-tuned outputs generated on all 10 test prompts
- [ ] Score sheet completed: issue type / format / roles / procedure
- [ ] Score ≥ 8/10 → proceed to deployment
- [ ] Score < 8/10 → investigate and retrain

### Phase 4 — Deployment
- [ ] Deployment option chosen (A / B / C)
- [ ] Inference script or server written and tested
- [ ] One real-world prompt tested end-to-end through the deployed system
- [ ] Results documented in PROGRESS.md

---

## This Project: Current Status

### Learning Completed

- [x] LoRA core mechanics — low-rank decomposition, A/B matrices, B initialized to zero
- [x] Rank and alpha — what they control, the 2× convention, effective scale (α/r)
- [x] Target modules — q_proj, v_proj, k_proj, o_proj and when to use each
- [x] Memory math — 5.5M adapter params, ~11 MB gradients, ~3.1 GB total vs ~24 GB full FT
- [x] Adapter lifecycle — save separately, load on top of base, merge permanently when ready
- [x] LoRA vs QLoRA — when each is appropriate, memory trade-offs
- [x] Fine-tuning vs RAG vs prompting — which solves which problem and why
- [x] Task selection — why Organizational Procedure Assistant is the right task for this machine and this learning goal
- [x] Dataset design — 12 issue types, 190 input variations, 7 edge cases, chat JSONL format
- [x] End-to-end training workflow — download, baseline, train, evaluate, deploy

### Code Built

- [x] `practical/dataset/generate.py` — dataset generator (12 procedures + all input messages)
- [x] `practical/dataset/train.jsonl` — 167 training examples, ready
- [x] `practical/dataset/validation.jsonl` — 30 validation examples, ready
- [x] `practical/evaluation/test_prompts.txt` — 10 held-out prompts locked for before/after comparison
- [x] `practical/evaluation/capture_baseline.py` — records base model outputs before training
- [x] `practical/evaluation/evaluate.py` — scores fine-tuned outputs after training, prints verdict
- [x] `practical/train.py` — full LoRA training script (load → LoRA → SFTTrainer → save → verify)
- [x] `practical/inference.py` — interactive CLI to test the trained model
- [x] `practical/requirements.txt` — all 6 dependencies pinned

### Pipeline Execution Status (2026-11-24)

```
✅ Dataset ready (167 train / 30 val)
✅ Dependencies installed (torch 2.x, transformers 5.24, trl 1.18, peft 0.14)
✅ All API breaking changes fixed (see below)
✅ capture_baseline.py — baseline_outputs.jsonl saved
✅ train.py — adapter_v1/ saved, 3 epochs, ~16 min on M1
✅ evaluate.py — 5.0/6.0 average, PASS, finetuned_outputs.jsonl saved
✅ inference.py — interactive test confirmed working
```

Full before/after comparison: `practical/results/RESULTS.md`

### API Breaking Changes Fixed (Real-World Library Version Mismatch)

When running against the installed versions (transformers 5.24, TRL 1.18), all scripts required fixes. These are documented here because they will recur on any fresh install.

**transformers 5.x:**
- `torch_dtype=` renamed to `dtype=` in `from_pretrained()`
- `apply_chat_template(..., return_tensors="pt")` now returns a `BatchEncoding` dict, not a raw tensor. Fix: add `return_dict=True`, then move all keys to device and unpack with `**inputs`

**TRL 1.x:**
- `TrainingArguments` → `SFTConfig` (owns both training args and SFT-specific args)
- `tokenizer=` → `processing_class=` in `SFTTrainer`
- `max_seq_length=` → `max_length=` (moved into `SFTConfig`)
- `dataset_text_field=` moved from `SFTTrainer` into `SFTConfig`
- `warmup_ratio=` → `warmup_steps=` (ratio removed from SFTConfig)
- `logging_dir=` removed from SFTConfig entirely

### Quick Reference — Run the Full Pipeline

```bash
# From the practical/ directory with venv active
source .venv/bin/activate

# Step 1: Capture what the base model does (run once, saves benchmark)
python evaluation/capture_baseline.py

# Step 2: Train (downloads model on first run, then cached)
python train.py

# Step 3: Evaluate — prints before/after score sheet
python evaluation/evaluate.py --adapter adapter_v1

# Step 4: Test interactively
python inference.py --adapter adapter_v1
```
