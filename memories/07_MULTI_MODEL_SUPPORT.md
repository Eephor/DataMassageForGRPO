# Memory 07 — Multi-Model Support: QUANT_MODE System

## Background

The pipeline originally targeted only `unsloth/Llama-3.2-1B-Instruct` (FP8, free T4).
Support for additional models was added incrementally across several sessions.
Each new model introduced a fundamentally different loading strategy, which led to
replacing the boolean `IS_FP8` flag with a string `QUANT_MODE` tri-state, and then
a quad-state.

---

## QUANT_MODE Detection Logic

`QUANT_MODE` is auto-derived from `MODEL_NAME` in both `train.ipynb` (constants cell)
and `train.py` (CLI). The detection precedence is:

```python
_name = MODEL_NAME.lower()
if 'gpt-oss' in _name or 'gpt_oss' in _name:
    QUANT_MODE = 'bf16'
elif 'deepseek' in _name or '-r1' in _name:
    QUANT_MODE = '4bit'
elif 'llama' in _name and '3b' in _name:
    QUANT_MODE = '16bit'
else:
    QUANT_MODE = 'fp8'
```

`IS_FP8 = QUANT_MODE == 'fp8'` is still computed for any legacy references.

---

## Model Matrix

| Option | Model | QUANT_MODE | GPU | Rollout engine | Notebook |
|--------|-------|------------|-----|----------------|----------|
| A | `unsloth/Llama-3.2-1B-Instruct` | `fp8` | T4 14 GB (free) | vLLM | [Unsloth Llama FP8 GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama_FP8_GRPO.ipynb) |
| B | `unsloth/Qwen3-8B` | `fp8` | L4 22 GB (Pro) | vLLM | [Unsloth Qwen3 8B FP8 GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_8B_FP8_GRPO.ipynb) |
| C | `unsloth/DeepSeek-R1-0528-Qwen3-8B` | `4bit` | L4/A100 | vLLM | [Unsloth DeepSeek R1 0528 GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb) |
| D | `unsloth/Llama-3.2-3B-Instruct` | `16bit` | T4/L4 | vLLM | [Unsloth Advanced Llama 3.2 3B GRPO LoRA](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_%283B%29_GRPO_LoRA.ipynb) |
| E | `unsloth/gpt-oss-20b-BF16` | `bf16` | A100/H100 80 GB | HF native | [Unsloth GPT-OSS 20B BF16 GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb) |

---

## Per-Mode Parameter Differences

### `"fp8"` — FP8 with vLLM (Options A & B)

Derived from: Unsloth FP8 GRPO notebooks for Llama-1B and Qwen3-8B.

```python
FastLanguageModel.from_pretrained(
    load_in_fp8=True,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
)
get_peft_model(r=LORA_RANK, lora_alpha=LORA_RANK * 2, ...)
GRPOConfig(
    vllm_sampling_params=SamplingParams(...),
    lr_scheduler_type='cosine',
    weight_decay=0.1,
    learning_rate=5e-6,
    gradient_accumulation_steps=1,  # FP8 small model
)
save_method = 'merged_16bit'
```

**Notes:**
- Qwen3-8B: `safe_apply_template` auto-injects `enable_thinking=False` (see Memory 06).
- `GRAD_ACCUM_STEPS=1` — small model fits in one step.

---

### `"16bit"` — Full-precision 16-bit LoRA with vLLM (Option D)

Derived from: [Unsloth Advanced Llama 3.2 (3B) GRPO LoRA notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_%283B%29_GRPO_LoRA.ipynb).

```python
FastLanguageModel.from_pretrained(
    load_in_fp8=False,
    load_in_4bit=False,
    gpu_memory_utilization=0.9,   # per 3B notebook
    fast_inference=True,
    max_lora_rank=LORA_RANK,
)
get_peft_model(r=LORA_RANK, lora_alpha=LORA_RANK, ...)  # alpha = rank, NOT rank*2
GRPOConfig(
    vllm_sampling_params=SamplingParams(...),
    lr_scheduler_type='cosine',
    weight_decay=0.1,
    learning_rate=5e-6,
    gradient_accumulation_steps=4,  # per 3B notebook
    max_grad_norm=1.0,
)
save_method = 'merged_16bit'
```

**Key difference from FP8:** `lora_alpha = LORA_RANK` (not `LORA_RANK * 2`).
The 3B Unsloth notebook does not use the 2× convergence trick.
`gpu_memory_utilization=0.9` is also new vs. FP8.

---

### `"4bit"` — BitsAndBytes 4-bit quantisation with vLLM (Option C)

Derived from: [Unsloth DeepSeek R1 0528 Qwen3 8B GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb).

```python
FastLanguageModel.from_pretrained(
    load_in_fp8=False,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
)
get_peft_model(r=LORA_RANK, lora_alpha=LORA_RANK * 2, ...)
GRPOConfig(
    vllm_sampling_params=SamplingParams(...),
    lr_scheduler_type='cosine',
    weight_decay=0.1,
    learning_rate=5e-6,
    gradient_accumulation_steps=4,
)
save_method = 'merged_16bit'
```

**Notes:**
- DeepSeek-R1 distilled on Qwen3 arch. Uses DeepSeek chat tokens
  (`<｜begin▁of▁sentence｜>` etc.), **not** Qwen3 ChatML.
- `safe_apply_template` does NOT inject `enable_thinking` — DS template doesn't
  have that kwarg. Natively generates `<think>` blocks regardless.

---

### `"bf16"` — BF16 with HF native generation, NO vLLM (Option E)

Derived from: [Unsloth GPT-OSS 20B BF16 GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb).

**Critical difference:** `fast_inference` is completely absent — GRPOTrainer uses
Hugging Face native generation for rollouts instead of vLLM.

```python
FastLanguageModel.from_pretrained(
    load_in_fp8=False,
    load_in_4bit=False,
    # NO fast_inference=True
    # NO max_lora_rank
)
get_peft_model(r=LORA_RANK, lora_alpha=LORA_RANK * 2, ...)
GRPOConfig(
    # NO vllm_sampling_params
    lr_scheduler_type='linear',   # per GPT-OSS notebook
    weight_decay=0.001,           # per GPT-OSS notebook
    learning_rate=5e-5,           # 10× higher than other models
    gradient_accumulation_steps=1,
    num_generations=2,            # memory constraint for 20B
)
save_method = 'mxfp4'            # OpenAI native precision
```

**Additional GPT-OSS consideration — `reasoning_effort`:**
GPT-OSS's chat template supports a `reasoning_effort` parameter. Without
`reasoning_effort='low'`, the model generates very long internal reasoning blocks
that bloat prompt lengths and waste VRAM. `safe_apply_template` auto-injects
`reasoning_effort='low'` using the same Jinja template inspection pattern as
`enable_thinking` for Qwen3:

```python
if tokenizer.chat_template and 'reasoning_effort' in tokenizer.chat_template:
    kwargs.setdefault('reasoning_effort', 'low')
```

**LORA_RANK default:** `8` (vs. `32` for other modes). The GPT-OSS notebook
suggests `4`; `8` is a middle ground for our oversight task. Increase for more
model capacity if VRAM allows.

---

## `save_method` per Mode

| QUANT_MODE | `save_method` | Notes |
|------------|---------------|-------|
| `fp8` | `merged_16bit` | Standard bfloat16 merged model |
| `16bit` | `merged_16bit` | Same |
| `4bit` | `merged_16bit` | Same |
| `bf16` | `mxfp4` | OpenAI's native micro-float precision (~4-bit); GPT-OSS specific |

`mxfp4` is auto-selected in the save cell when `QUANT_MODE == 'bf16'`.

---

## GRPOConfig Parameters by Mode

| Parameter | fp8 | 16bit | 4bit | bf16 |
|-----------|-----|-------|------|------|
| `vllm_sampling_params` | ✓ | ✓ | ✓ | **✗** |
| `learning_rate` | 5e-6 | 5e-6 | 5e-6 | **5e-5** |
| `lr_scheduler_type` | cosine | cosine | cosine | **linear** |
| `weight_decay` | 0.1 | 0.1 | 0.1 | **0.001** |
| `gradient_accumulation_steps` | 1 | 4 | 4 | 1 |
| `num_generations` | 4 | 4 | 4 | **2** |
| `max_steps` default | 500 | 500 | 500 | **600** |
| `max_grad_norm` | 1.0 | 1.0 | 1.0 | 1.0 |

---

## Files Changed

| File | Change |
|------|--------|
| `train.ipynb` — constants cell | `IS_FP8` → `QUANT_MODE`, conditional defaults for all training hyperparameters, Option E added |
| `train.ipynb` — model loading cell | Four-way branch on `QUANT_MODE`; bf16 omits `fast_inference`/`max_lora_rank` |
| `train.ipynb` — LoRA cell | Uses `LORA_ALPHA` constant (rank for 16bit, rank×2 for others) |
| `train.ipynb` — `safe_apply_template` cell | Added `reasoning_effort='low'` auto-injection |
| `train.ipynb` — GRPOConfig cell | Conditional `vllm_sampling_params`; uses `LR_SCHEDULER`/`WEIGHT_DECAY` constants |
| `train.ipynb` — save cell | Uses `_save_method` (`mxfp4` for bf16, `merged_16bit` for others) |
| `train.ipynb` — appendix cell | Updated to 5-option comparison table |
| `rewards.py` — `safe_apply_template` | Added `reasoning_effort` detection |
| `train.py` | All same changes mirrored; `effective_lr`/`lr_scheduler`/`weight_decay` derived from `quant_mode` |

---

## Commit History

| Commit | Change |
|--------|--------|
| `7bede25` | Add Llama-3.2-3B-Instruct support (16-bit LoRA) as Option D |
| `dc88a92` | Add GPT-OSS 20B (BF16) support as Option E |
