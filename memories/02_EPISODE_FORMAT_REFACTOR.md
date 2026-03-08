---
name: Episode Format Refactor
overview: Refactor the GRPO dataset to support sequential episode-based training by adding turn metadata fields, stripping the redundant system prompt from the dataset rows (move to training-time injection), and clarifying output directory conventions.
todos:
  - id: models-turn-fields
    content: Add turn_index, total_turns, length_scale to GRPORecord in models.py; keep prompt as user-only list
    status: completed
  - id: transform-episode
    content: Populate turn metadata in transform.py; remove system prompt from prompt rows; keep SYSTEM_PROMPT_TEMPLATE as module-level constant for training-time injection
    status: completed
  - id: fix-tests
    content: Update _make_record fixture in test_split.py to include the new turn_index, total_turns, length_scale fields; verify all 10 tests still pass
    status: completed
  - id: readme-update
    content: "Update README.md: canonical output dir (transformed/), system prompt injection pattern, episode-based training loop explanation, split seed behaviour"
    status: completed
isProject: false
---

# Episode Format Refactor

## Three Changes

### 1. Add Episode Turn Metadata to `GRPORecord`

Three new fields in `[grpo-pipeline/src/grpo_pipeline/models.py](grpo-pipeline/src/grpo_pipeline/models.py)` on `GRPORecord`:

```python
turn_index: int       # 0-based position of this message in its thread
total_turns: int      # total messages in the thread
length_scale: float   # (turn_index + 1) / total_turns — reward weight for training
```

For batch records (single-message "threads"): `turn_index=0, total_turns=1, length_scale=1.0`.
For conversation records: populated from the sorted thread reconstruction in `transform_conversation_file`.

### 2. Strip System Prompt from Dataset Rows

Currently the full ~600-character system prompt is baked into every `prompt` row, varying only in `{author}`. This bloats the dataset by ~30% with redundant data.

**Change in `[grpo-pipeline/src/grpo_pipeline/models.py](grpo-pipeline/src/grpo_pipeline/models.py)`:**

- Add `target_author: str` as a top-level field on `GRPORecord` (separate from `author` which already exists — consolidate to one)
- The `prompt` field stores only the user-role message: `[{"role": "user", "content": "..."}]`

**Change in `[grpo-pipeline/src/grpo_pipeline/transform.py](grpo-pipeline/src/grpo_pipeline/transform.py)`:**

- Remove `SYSTEM_PROMPT` from dataset generation
- `build_grpo_record` only builds the user message and stores `author` as the target identifier
- Add a module-level `SYSTEM_PROMPT_TEMPLATE` constant (kept in source for reference/documentation) but not written into rows

At training time, the training script (future) prepends:

```python
{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(author=record["author"])}
```

before passing to `tokenizer.apply_chat_template()`. TRL supports this via `formatting_func` or a pre-processing step.

### 3. Canonical Output Directory

The `transformed/` dir at the repo root is the canonical output location, not `grpo-pipeline/output/`. Update the README to specify this convention and add a note to `transform.py`'s module docstring.

## Files Changed

- `[grpo-pipeline/src/grpo_pipeline/models.py](grpo-pipeline/src/grpo_pipeline/models.py)` — add `turn_index`, `total_turns`, `length_scale` to `GRPORecord`; `prompt` becomes user-only
- `[grpo-pipeline/src/grpo_pipeline/transform.py](grpo-pipeline/src/grpo_pipeline/transform.py)` — populate turn metadata; remove system prompt from rows; keep `SYSTEM_PROMPT_TEMPLATE` as a documented constant
- `[grpo-pipeline/tests/test_split.py](grpo-pipeline/tests/test_split.py)` — update fixture `_make_record` to include the new required fields
- `[grpo-pipeline/README.md](grpo-pipeline/README.md)` — document canonical output dir, the system prompt injection pattern, and the episode-based training loop concept

## How the Training Loop Uses This

The training script (not yet written) would:

1. Load `train.jsonl`
2. Group records by `thread_id`, sort each group by `turn_index`
3. For each thread (episode): iterate turns in order, call the model, compute reward × `length_scale`
4. After the last turn of a thread, the episode ends; move to the next thread

Batch records (`total_turns=1`) always get full reward weight — they have no context to build on.

## On the Split Randomness

The split is **deterministic but shuffled**. `split.py` uses `random.Random(seed=42).shuffle(thread_ids)`. Same seed always produces the same train/test partition. Pass `--seed N` to get a different split. The default 42 is documented in the README.
