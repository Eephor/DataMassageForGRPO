---
name: Notebook Divergences and Verdict Simplification
overview: |
  Documents every deliberate deviation from the original Unsloth FP8 GRPO notebook
  (https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama_FP8_GRPO.ipynb)
  made during our first GRPO run, and the subsequent verdict simplification from 13 float/string
  fields down to 4 categorical labels. Also records all fixes applied to train.py and baseline.py
  to keep them in sync with the notebook.
---

# Notebook Divergences, Fixes, and Verdict Simplification

## 1. Model Names

**Original Unsloth notebook:**
```python
model_name = "unsloth/Llama-3.2-1B-Instruct"
# Phase 2:
model_name = "unsloth/Qwen3-8B-Instruct"
```

**What we had (wrong):**
```python
MODEL_NAME = 'unsloth/Llama-3.2-1B-Instruct-FP8-Block'
# Phase 2:
MODEL_NAME = 'unsloth/Qwen3-8B-FP8'
```

**Why it was wrong:** The `-FP8-Block` and `-FP8` suffixes refer to quantised GGUF/GGML shards
that exist for some repos but are not the Unsloth Hub identifiers. Unsloth's `FastLanguageModel`
applies FP8 quantisation internally when `load_in_fp8=True` is passed; the base model identifier
should be the plain `-Instruct` name.

**Fix:** Renamed to `unsloth/Llama-3.2-1B-Instruct` and `unsloth/Qwen3-8B-Instruct` everywhere
(train.ipynb, evaluate.ipynb, train.py, baseline.py, README).

---

## 2. `FastLanguageModel.from_pretrained` Parameters

**Original Unsloth notebook:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False,    # False for LoRA 16bit
    fast_inference = True,   # Enable vllm fast inference
    max_lora_rank = lora_rank,
    load_in_fp8 = True,      # Float8 RL / GRPO!
)
```

**What we had (wrong — for train.py and train.ipynb):**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    dtype=None,   # ← not in original; auto-detect is not needed
)
# Missing: fast_inference, max_lora_rank, load_in_fp8
```

**Why the missing args matter:**
- `fast_inference=True` — wires in the vLLM rollout engine for GRPO; without it training
  falls back to slow HuggingFace `generate()` and will OOM or time out on T4.
- `max_lora_rank=lora_rank` — Unsloth pre-allocates LoRA memory at load time; omitting it
  causes a second allocation later and may OOM on T4.
- `load_in_fp8=True` — enables Float8 quantisation; without it the model loads in bfloat16
  (~2.4 GB instead of ~1.2 GB for 1B), eating into T4 headroom.
- `dtype=None` — not needed and not in the original; removed.

**Fix applied to train.py, train.ipynb, evaluate.ipynb:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    load_in_fp8=True,
)
```

---

## 3. `get_peft_model` / LoRA Parameters

**Original Unsloth notebook:**
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [...],
    lora_alpha = lora_rank * 2,  # 2x = faster convergence (Unsloth recommendation)
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    # lora_dropout and bias are NOT set — Unsloth uses its own optimised defaults
)
```

**What we had (wrong):**
```python
loaded_model = FastLanguageModel.get_peft_model(
    loaded_model,
    r=lora_rank,
    ...,
    lora_alpha=lora_rank,      # ← half the recommended value
    lora_dropout=0.0,          # ← explicit, not in original
    bias="none",               # ← explicit, not in original
    use_gradient_checkpointing="unsloth",
    random_state=42,           # ← non-canonical; Unsloth docs use 3407
)
```

**Why `lora_alpha = lora_rank * 2` matters:** Unsloth's own comment is "2x faster convergence".
The LoRA effective learning rate scales with `lora_alpha / r`; setting alpha = 2r doubles the
effective LR for the adapter weights without affecting the base model.

**Fix applied to train.py and train.ipynb:**
- `lora_alpha = lora_rank * 2`
- Removed `lora_dropout` and `bias` (let Unsloth defaults apply)
- `random_state = 3407`

---

## 4. `%%capture` Cell Magic Placement

**Problem:** The second Colab install cell had `%%capture` somewhere in the middle of the cell
(after `# @title ...` and some comments). Jupyter requires cell magics to be the **first line**
of a cell, or they are treated as line magics and fail with:
```
UsageError: Line magic function `%%capture` not found.
```

**Fix:** Moved `%%capture` to be the very first line of the cell, before any comments
or `# @title` decorators.

---

## 5. Installation Cell — T4 vs L4 Version Pinning

The original Unsloth notebook has a single line:
```python
!pip install unsloth vllm
```

We added GPU-tier detection and version pinning because:
- vLLM 0.15+ dropped CUDA kernel support for T4 (Turing architecture). Free Colab T4 requires
  `vllm==0.9.2` specifically.
- Colab pre-installs `torchvision`, `bitsandbytes`, and `xformers` at versions that conflict
  with recent `trl`; explicit pins prevent silent version mismatches.
- `trl==0.22.2` is the last version compatible with the `GRPOConfig.vllm_sampling_params` API
  we use.

```python
is_t4 = 'Tesla T4' in str(subprocess.check_output(['nvidia-smi']))
vllm_ver   = 'vllm==0.9.2'   if is_t4 else 'vllm==0.15.1'
triton_ver = 'triton==3.2.0' if is_t4 else 'triton'
os.system(f'pip install -qqq {vllm_ver} torchvision bitsandbytes xformers unsloth')
os.system('pip install -qqq transformers==4.56.2')
os.system('pip install -qqq --no-deps trl==0.22.2')
```

---

## 6. Repo / Package Setup Cell

The notebook needs `grpo_pipeline` importable in Colab. Added a `SOURCE` variable:

```python
SOURCE = 'github'   # 'github' | 'drive' | 'local'
```

- `'github'` (default) — clones `https://github.com/Eephor/DataMassageForGRPO.git`
  and runs `pip install -e ".[train]"`.
- `'drive'` — mounts Google Drive and expects the repo at a configurable path.
- `'local'` — adds a local `src/` path to `sys.path` (for running from Cursor IDE).

A `sys.path` fallback appends `src/` regardless so the import works in every case.

---

## 7. Save Cell — LoRA Adapter + Merged 16-bit Model

**Original Unsloth notebook** saves both:
```python
model.save_pretrained("lora_model")         # LoRA adapter only
model.save_pretrained_merged("model_16bit", tokenizer, save_method="merged_16bit")
```

**What we originally had:** Only saved the LoRA adapter.

**Fix:** The save cell now saves both:
```python
LORA_DIR   = 'lora-adapter'
MERGED_DIR = 'model-merged-16bit'
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
```

And optionally pushes to HuggingFace Hub:
```python
model.push_to_hub("Eephor/moltbook-oversight-agent", token=HF_TOKEN)
```

An **extra cell** mounts Google Drive and copies both saved directories there for persistence
across Colab session resets.

---

## 8. HuggingFace Token Security

An `HF_TOKEN` value was accidentally hardcoded in an earlier commit. It was redacted and
replaced with:
```python
HF_TOKEN = os.environ.get('HF_TOKEN', '')
```
The exposed token was revoked on HuggingFace Hub.

---

## 9. Verdict Simplification: 13 Fields → 4 Categorical Labels

### Motivation
- The original 13-field verdict (12 float traits + `alignment_status`) is too many things
  for a 1B model to predict accurately.
- The dataset is imbalanced (~78% `safe` class) — a class-weighted reward is simpler
  to implement than synthetic data generation.
- `alignment_status` (`aligned/drifting/misaligned/violation`) is redundant with
  `ground_truth_safety_score` since `drifting` is a time-series concept the model cannot
  observe from a single turn.

### New verdict format (4 fields)
```json
{
  "safety_level": "safe",      // safe | caution | risk | critical
  "integrity":    "strong",    // strong | good | weak | poor  (Ethos group)
  "reasoning":    "good",      // strong | good | weak | poor  (Logos group)
  "empathy":      "weak"       // strong | good | weak | poor  (Pathos group)
}
```

### Bucketing scheme

**safety_level** (from `ground_truth_safety_score`):

| Label    | Threshold         | Meaning                          |
|----------|-------------------|----------------------------------|
| safe     | score ≥ 0.85      | No meaningful threats            |
| caution  | 0.65 ≤ score < 0.85 | Minor concerns, monitor          |
| risk     | 0.40 ≤ score < 0.65 | Significant issues               |
| critical | score < 0.40      | Serious harm indicators          |

**Group labels** (integrity / reasoning / empathy):

Group score = (avg positive traits + (1 − avg negative traits)) / 2

| Label  | Threshold         |
|--------|-------------------|
| strong | score ≥ 0.75      |
| good   | 0.50 ≤ score < 0.75 |
| weak   | 0.25 ≤ score < 0.50 |
| poor   | score < 0.25      |

**Trait groupings:**

| Group     | Positive traits         | Negative traits           |
|-----------|-------------------------|---------------------------|
| integrity | virtue, goodwill        | manipulation, deception   |
| reasoning | accuracy, reasoning     | fabrication, broken_logic |
| empathy   | recognition, compassion | dismissal, exploitation   |

### Class weights (compensate for ~3.6:1 safe/non-safe imbalance)

| Level    | Weight |
|----------|--------|
| safe     | 1.0    |
| caution  | 2.6    |
| risk     | 5.0    |
| critical | 8.0    |

### New reward functions (rewards.py)

Old: `alignment_reward` + `trait_reward`
New: `safety_level_reward` + `group_reward`

```python
def safety_level_reward(..., ground_truth_safety_score, length_scale, **kwargs):
    # derives gt_level from score → CLASS_WEIGHTS[gt_level]
    # +2 × weight × scale for correct, -1 × weight × scale for wrong

def group_reward(..., ground_truth_traits, ground_truth_safety_score, length_scale, **kwargs):
    # +1 per correct group label (max 3) × CLASS_WEIGHTS[gt_level] × length_scale
```

### System prompt update

The `SYSTEM_PROMPT_TEMPLATE` in `transform.py` was expanded with:
- Per-trait vocabulary (definitions of all 12 traits)
- Group definitions (which traits roll up into integrity/reasoning/empathy and their polarity)
- Bucket threshold tables with plain-English meanings
- Updated 4-field example verdict

---

## 10. Files Affected by All Changes Above

| File | Changes |
|------|---------|
| `grpo-pipeline/train.ipynb` | Model names, from_pretrained args, lora_alpha, save cell, GDrive cell, %%capture fix, reward table, sanity check cell, GRPOTrainer reward_funcs |
| `grpo-pipeline/evaluate.ipynb` | Model names, from_pretrained args, verdict display cells, comparison cell |
| `grpo-pipeline/src/grpo_pipeline/rewards.py` | Full rewrite: 4-field verdict, safety_level_reward, group_reward, bucket helpers, CLASS_WEIGHTS |
| `grpo-pipeline/src/grpo_pipeline/transform.py` | SYSTEM_PROMPT_TEMPLATE expanded with full rubric |
| `grpo-pipeline/src/grpo_pipeline/train.py` | Model names, from_pretrained args, lora_alpha, reward function imports, save merged model |
| `grpo-pipeline/src/grpo_pipeline/baseline.py` | Safety-level + group metrics replacing alignment/trait MAE metrics |
| `grpo-pipeline/README.md` | Model name corrections |
