# Memory 06 — Llama / Qwen3 Tokenizer Compatibility

## Background

The codebase originally assumed Llama-style tokenization throughout. When
planning the switch to `unsloth/Qwen3-8B-Instruct`, a thorough audit found
two categories of incompatibility.

---

## Issue 1 — Qwen3 Thinking Mode Injects `<think>` Into the Prompt

### What happens

Qwen3's Jinja chat template, when called with `add_generation_prompt=True`,
appends `<think>\n` to the end of the formatted prompt **by default** (thinking
mode enabled). This means:

- The **model's completion** starts *inside* the `<think>` block.
- The opening `<think>` tag is part of the **prompt**, not the completion.
- `GRPOTrainer` passes only the completion to reward functions.
- `format_reward` looks for `<think>…</think>` in the completion → finds
  nothing → returns `0.0` for every step → training dies (zero reward = zero
  gradient signal), exactly like the first Llama run without warmup.

### Fix

Pass `enable_thinking=False` to `apply_chat_template` for models whose Jinja
template supports that parameter.

**Detection strategy:** inspect the template string itself — do not hard-code
model names. Any model whose template contains the literal string
`enable_thinking` gets the flag set; all others are unaffected.

```python
def safe_apply_template(tokenizer, msgs, **kwargs):
    if tokenizer.chat_template and 'enable_thinking' in tokenizer.chat_template:
        kwargs.setdefault('enable_thinking', False)
    return tokenizer.apply_chat_template(msgs, **kwargs)
```

An earlier version used `try/except TypeError` as a fallback. That was
replaced because using exceptions for control flow is a code smell and the
template-inspection approach is cleaner and equally robust.

### Why it does NOT affect Llama

Llama's chat template never injects `<think>` tokens. The old code calling
`tokenizer.apply_chat_template` directly was correct for Llama and produced
identical prompts to the new `safe_apply_template` call.

---

## Issue 2 — GRPOTrainer Re-Applies the Chat Template Internally

### What happens

When `dataset['prompt']` contains **message dicts** (list of
`{"role": ..., "content": ...}` objects), TRL's `GRPOTrainer` calls
`tokenizer.apply_chat_template` internally — **without** `enable_thinking=False`.
Any fix applied in the user-facing dataset-preparation cell would be bypassed.

### Fix

Pre-format all prompts as **strings** in the dataset-preparation cell (cell 12
of `train.ipynb`) using `safe_apply_template(..., add_generation_prompt=True)`.
When `dataset['prompt']` holds strings, GRPOTrainer tokenises them directly
and does **not** re-apply the chat template.

```python
def format_prompts(batch):
    return {'prompt': [
        safe_apply_template(
            tokenizer,
            [{'role': 'system', 'content': SYSTEM_PROMPT_TEMPLATE.format(author=a)}] + p,
            tokenize=False,
            add_generation_prompt=True,
        )
        for a, p in zip(batch['author'], batch['prompt'])
    ]}
```

### Why it does NOT affect Llama training

Llama's template produces the same string whether called via the old message-dict
path (GRPOTrainer internal) or the new pre-formatted string path. The two runs
of Llama training ([200-step run with warmup](043113eb-2e1b-4080-af5c-895987cc78b4))
were not affected.

---

## Issue 3 — `skip_special_tokens=True` Strips `<think>`/`</think>` on Qwen3

### What happens

On Qwen3, `<think>` (token ID 151668) and `</think>` (token ID 151669) are
**registered special tokens**. Calling `tokenizer.decode(..., skip_special_tokens=True)`
strips them from the output, breaking `extract_think` and `extract_verdict` in
all inference and evaluation paths.

### Fix

Use `skip_special_tokens=False` and manually strip only the model-specific
wrapper tokens (eos, bos):

```python
def decode_completion(tokenizer, output_ids):
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    for tok in filter(None, [tokenizer.eos_token, tokenizer.bos_token]):
        text = text.replace(tok, '')
    return text.strip()
```

This preserves `<think>`/`</think>` regardless of whether they are plain text
(Llama) or special tokens (Qwen3).

### Why it does NOT affect Llama training

Llama's tokenizer does not register `<think>` or `</think>` as special tokens.
The old `skip_special_tokens=True` and the new `skip_special_tokens=False`
produce identical decoded output for Llama completions.

---

## Files Changed

| File | Cell / Location | Change |
|------|----------------|--------|
| `train.ipynb` | Cell 12 | Added `safe_apply_template()` and `format_prompts()` (pre-formatted strings) |
| `train.ipynb` | Cell 13 | Use `sample['prompt']` directly (already a string) |
| `train.ipynb` | Cell 14 | Use `tokenizer.encode(p)` instead of `apply_chat_template` |
| `train.ipynb` | Cell 16 | Warmup: use `safe_apply_template` |
| `train.ipynb` | Cell 25 | Inference: use `safe_apply_template` + `decode_completion` |
| `evaluate.ipynb` | Cell 7 | Added `safe_apply_template()` and `decode_completion()` helpers |
| `evaluate.ipynb` | Cell 8 | Use `safe_apply_template` |
| `evaluate.ipynb` | Cell 9 | Use `decode_completion` |
| `evaluate.ipynb` | Cell 11 | Use `safe_apply_template` + `decode_completion` in batch loop |
| `evaluate.ipynb` | Cell 15 | `run_single()` uses `safe_apply_template` + `decode_completion` |

---

## Effect on Current Llama Training Runs

The 200-step and 150-step continuation runs were done with the **old code**
(message-dict prompts, `skip_special_tokens=True`). Both ran on Llama where
neither issue manifests. Training results, reward curves, and saved LoRA
adapters from those runs are **valid and unaffected**.

---

## Switching to Qwen3

When changing `MODEL_NAME` to `unsloth/Qwen3-8B-Instruct`:

1. `safe_apply_template` detects `enable_thinking` in the template → sets
   `enable_thinking=False` automatically. No code changes needed.
2. Pre-formatted string prompts bypass GRPOTrainer's internal `apply_chat_template`.
3. `decode_completion` preserves `<think>`/`</think>` tags in evaluation output.
4. All reward functions are pure text-based regex — unaffected by model choice.
5. Adjust `MAX_SEQ_LENGTH`, `LORA_RANK`, `BATCH_SIZE` down as needed for the
   larger model (requires Colab Pro L4 or A100, ~22 GB VRAM).
