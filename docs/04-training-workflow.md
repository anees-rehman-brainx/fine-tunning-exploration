# Phase 4 — Practical Training Workflow

**Status:** NOT STARTED  
**Goal:** Understand and execute the complete end-to-end LoRA training pipeline — from downloading a model to comparing fine-tuned vs base outputs.

---

## The Big Picture

```
Hugging Face Hub (open-source model weights)
        ↓
  Download base model (Qwen2.5-0.5B, ~1 GB)
        ↓
  Freeze base weights + attach LoRA adapters
        ↓
  Load dataset (500–2000 labelled examples)
        ↓
  Train on Mac M1 (~10–30 mins for a small model)
        ↓
  Save adapter file (~11 MB)
        ↓
  Test: base model vs fine-tuned model on same prompts
        ↓
  Measure the difference
```

---

## Step 1 — Where the Model Comes From

Models are hosted on **Hugging Face Hub** (huggingface.co) — an open-source platform where researchers and organizations publish model weights for free.

When you run the training code, it downloads the model weights to your local machine (~500 MB to 3 GB depending on the model). These are pre-trained weights — the model already knows English, can reason, write code, etc. We are not building from scratch.

**Model we will use for the first run:** `Qwen/Qwen2.5-0.5B-Instruct` or similar small model.

Why:
- Small enough to download and train quickly on M1
- Instruction-tuned baseline (already responds to prompts)
- Supported by Hugging Face PEFT and TRL
- Clear before/after comparison possible

---

## Step 2 — What We Are Teaching It

We pick a narrow task where:
- The base model is "okay" but inconsistent
- Our dataset clearly demonstrates the right behavior
- The improvement is measurable

**Options for the first run:**

| Task | Dataset | Evaluation |
|---|---|---|
| Follow instructions in a specific format | 500–1000 synthetic examples | Does output match expected format? |
| Answer domain-specific questions | Custom Q&A pairs | Manual review + exact match |
| Classify text into categories | Labelled classification set | Accuracy % |

**Decision:** to be made before implementation starts.

---

## Step 3 — What the Dataset Looks Like

For instruction fine-tuning (SFT), each example is a (prompt, response) pair formatted as a conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

This is called **chat JSONL format** — one JSON object per line, each with a `messages` array.

The dataset is split:
- **Training set** (~80–90%): what the model trains on
- **Validation set** (~10–20%): used to monitor for overfitting, never trained on
- **Test set** (held out): used only for final evaluation, never seen during training

---

## Step 4 — The Training Loop (What Actually Happens)

```
For each batch of examples:

  1. FORWARD PASS
     Feed the prompt → model predicts the response token by token

  2. LOSS CALCULATION
     Compare predicted tokens vs actual tokens in dataset
     → Cross-entropy loss: how wrong was the prediction?

  3. BACKPROPAGATION
     Calculate gradient for every adapter parameter (A and B matrices only)
     "How much did each adapter weight contribute to this error?"

  4. WEIGHT UPDATE
     new_weight = old_weight - (learning_rate × gradient)
     Only the LoRA A and B matrices update — base model weights never change

  5. REPEAT
     Next batch → loss goes down slightly → repeat ~100–500 steps
```

What you watch during training:
- **Training loss** — should go down steadily
- **Validation loss** — should also go down; if it rises while training loss falls, overfitting is happening
- **Steps per second** — gives you an ETA

---

## Step 5 — What Gets Saved

After training, two things are on disk:

| File | What it is | Size |
|---|---|---|
| Adapter weights | The trained A and B matrices | ~11–50 MB |
| `adapter_config.json` | Rank, alpha, target modules, base model name | A few KB |

The base model weights are NOT saved again — they are unchanged and already on disk from the download.

---

## Step 6 — How We Test

Three-way comparison:

```
Same prompt given to all three:

1. BASE MODEL (zero-shot)
   → No fine-tuning, no examples in prompt
   → Shows what the pretrained model does by default

2. BASE MODEL (few-shot)
   → No fine-tuning, but 2–3 examples added to the prompt
   → Shows how much prompting alone can achieve

3. FINE-TUNED MODEL (zero-shot)
   → LoRA adapter loaded on top of base
   → Shows what training actually added
```

**What we look for:**
- Does the fine-tuned model follow the target format/behavior more consistently?
- Did it regress on anything the base model did well?
- Is the improvement significant or marginal?

**Evaluation methods:**
- **Manual review** — read 20–30 side-by-side outputs, tag what went right/wrong
- **Automatic scoring** — exact match, F1, or task-specific metric if the task allows it

---

## Step 7 — Extending to QLoRA

After the LoRA run works end-to-end, the same pipeline is extended to QLoRA by changing two things:

```python
# Load model in 4-bit instead of 16-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
)
```

Everything else — LoRA config, training, saving, evaluation — stays the same.

---

## Libraries We Will Use

| Library | What it does |
|---|---|
| `transformers` | Load pretrained models and tokenizers from Hugging Face |
| `peft` | Add LoRA adapters, configure rank/alpha/target modules |
| `trl` | `SFTTrainer` — handles the training loop for instruction fine-tuning |
| `datasets` | Load and process datasets |
| `torch` | The underlying tensor computation and autograd engine |
| `bitsandbytes` | 4-bit quantization for QLoRA (may need a Linux/CUDA environment) |

---

## Hardware Notes

| Setup | Model size | Approach | Expected time |
|---|---|---|---|
| Mac M1 16 GB | 0.5B–1.5B | LoRA (FP16) | 10–30 min |
| Mac M1 16 GB | 7B | QLoRA (4-bit) | Likely too slow or OOM |
| Google Colab (free T4) | Up to 7B | QLoRA | 30–60 min |
| Google Colab (A100) | Up to 13B | QLoRA | 20–40 min |

**Decision for first run:** use Mac M1 with a 0.5B model and LoRA. No cloud needed.

---

## Implementation Checklist

- [ ] Set up Python environment (uv or venv)
- [ ] Install libraries (transformers, peft, trl, datasets, torch)
- [ ] Load base model and tokenizer
- [ ] Inspect model architecture and identify target module names
- [ ] Prepare dataset in chat JSONL format
- [ ] Split into train/validation sets
- [ ] Configure LoRA (`LoraConfig`)
- [ ] Inspect trainable parameter count — confirm it matches expectations
- [ ] Run training with `SFTTrainer`
- [ ] Monitor training loss and validation loss
- [ ] Save adapter
- [ ] Reload adapter and confirm outputs match pre-save outputs
- [ ] Run three-way comparison (base zero-shot, base few-shot, fine-tuned zero-shot)
- [ ] Document results in PROGRESS.md
