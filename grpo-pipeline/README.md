# grpo-pipeline

Transforms Ethos Academy scored conversation data into TRL `GRPOTrainer`-ready datasets for training an LLM oversight agent.

## Structure

```
src/grpo_pipeline/
├── models.py     # Pydantic schemas for source JSONL records and GRPORecord output
├── transform.py  # Thread reconstruction + GRPO prompt formatting
├── split.py      # Thread-level train/test splitting (no leakage)
├── rewards.py    # GRPO reward functions (format / alignment / trait MAE)
├── train.py      # GRPO training script (Unsloth + TRL GRPOTrainer)
└── baseline.py   # Headless batch evaluation against test set
evaluate.ipynb    # Interactive notebook: single-record inspection + metrics table
tests/
└── test_split.py # Verifies no data leakage across splits
```

## Usage

### 1. Data pipeline (no GPU required)

```bash
# Install data pipeline deps only
uv sync

# Transform all staged data — write to repo-root/transformed/
uv run python -m grpo_pipeline.transform --input ../raw-data --output ../transformed

# Split into train/test (thread-level, no leakage)
uv run python -m grpo_pipeline.split --input ../transformed/dataset.jsonl --output ../transformed --test-ratio 0.2 --seed 42

# Verify no leakage
uv run pytest tests/
```

### 2. Training (GPU required — Colab T4 free tier or better)

```bash
# Step 1: install Unsloth + vLLM (platform-specific CUDA wheels, do this first)
pip install unsloth vllm
pip install --no-deps trl==0.22.2

# Step 2: install this package with training extras
uv pip install -e ".[train]"

# Step 3: run baseline evaluation (base model, no RL)
python -m grpo_pipeline.baseline \
    --test-file ../transformed/test.jsonl \
    --output ../baseline_results.json

# Step 4: train with GRPO (Phase 1 — Llama-1B on T4)
python -m grpo_pipeline.train \
    --train-file ../transformed/train.jsonl \
    --output-dir ../lora-adapter \
    --max-steps 200

# Phase 2 — Qwen3-8B on Colab Pro L4 (just change --model, no other changes)
python -m grpo_pipeline.train \
    --train-file ../transformed/train.jsonl \
    --output-dir ../lora-adapter-8b \
    --model unsloth/Qwen3-8B-Instruct \
    --max-steps 500

# Step 5: evaluate trained model
python -m grpo_pipeline.baseline \
    --test-file ../transformed/test.jsonl \
    --lora-adapter ../lora-adapter \
    --output ../post_grpo_results.json
```

For interactive single-record inspection and side-by-side comparison, open `evaluate.ipynb` in Colab or JupyterLab.

## Canonical Output Directory

`../transformed/` (repo root `transformed/`) is the canonical location for generated datasets.
`grpo-pipeline/output/` is a scratch directory — do not treat it as authoritative.

## Dataset Format

Each row in `dataset.jsonl` is a `GRPORecord` with these key fields:

| Field | Type | Description |
|---|---|---|
| `prompt` | `list[dict]` | User-only message in chat format. System prompt is NOT included — inject at training time. |
| `ground_truth_traits` | `dict[str, float]` | All 12 raw trait scores (0.0–1.0). |
| `ground_truth_safety_score` | `float` | `mean(1-manipulation, 1-deception, 1-exploitation)`. Below 0.5 = misaligned. |
| `ground_truth_alignment` | `str` | `aligned \| drifting \| misaligned \| violation` |
| `ground_truth_phronesis` | `str` | `trustworthy \| mixed \| untrustworthy` |
| `turn_index` | `int` | 0-based position of this message in its thread. |
| `total_turns` | `int` | Total messages in this thread. |
| `length_scale` | `float` | `(turn_index + 1) / total_turns`. Multiply reward by this value. |
| `thread_id` | `str` | Groups all turns of one conversation. Used for thread-level splitting. |
| `evaluation_id` | `str` | Unique per evaluation. Used for deduplication. |
| `author` | `str` | The target agent being evaluated. |

## System Prompt Injection

The system prompt is stored in `transform.py` as `SYSTEM_PROMPT_TEMPLATE` but is **not written into dataset rows**. Inject it at training time:

```python
from grpo_pipeline.transform import SYSTEM_PROMPT_TEMPLATE

def add_system_prompt(record):
    system_msg = {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(author=record["author"])}
    return [system_msg] + record["prompt"]

# In TRL training loop:
full_prompt = add_system_prompt(record)
inputs = tokenizer.apply_chat_template(full_prompt, return_tensors="pt")
```

This keeps the dataset lean (~30% smaller) and lets you iterate on the system prompt without re-running the transform.

## Episode-Based Training Loop

Each thread in the dataset is a sequential **episode**. The `turn_index`, `total_turns`, and `length_scale` fields support length-proportional reward weighting:

- **Turn 0** (agent sees no prior context): `length_scale = 1/N` — low weight, agent has limited signal
- **Turn N-1** (agent sees full conversation): `length_scale = 1.0` — full reward weight

Example training loop pattern:

```python
from collections import defaultdict

# Group and sort by thread
episodes = defaultdict(list)
for record in train_records:
    episodes[record["thread_id"]].append(record)
for thread_id in episodes:
    episodes[thread_id].sort(key=lambda r: r["turn_index"])

# Process each episode
for thread_id, turns in episodes.items():
    for turn in turns:
        verdict = model.generate(add_system_prompt(turn))
        raw_reward = compute_reward(verdict, turn["ground_truth_alignment"], turn["ground_truth_traits"])
        weighted_reward = raw_reward * turn["length_scale"]
        optimizer.step(weighted_reward)
```

Batch records (standalone messages with no thread context) always have `total_turns=1, length_scale=1.0` and receive full reward weight.

## Train/Test Split

Splits are done at the **thread level** — all turns from the same conversation go entirely into train or test. This prevents data leakage from the rolling context window pattern (turn 0 and turn 1 of the same thread share identical context).

The split is **deterministic but shuffled**:
- `--seed 42` (default) always produces the same partition
- Pass `--seed N` for a different but reproducible split
- The assignment is a random shuffle of thread IDs, not a sequential slice

## Reward Functions

Three GRPO reward functions in `rewards.py`, each with signature `(prompts, completions, **kwargs) -> list[float]`:

| Function | Max value | Scaled by `length_scale`? | Purpose |
|---|---|---|---|
| `format_reward` | 1.0 | No | Forces `<think>`/`<verdict>` structured output |
| `alignment_reward` | 2.0 | Yes | Correct `alignment_status` verdict |
| `trait_reward` | ~1.0 | Yes | Accurate 12-trait scoring (weighted MAE) |

Max total reward at the final turn of a thread (`length_scale=1.0`): **~4.0**.
At turn 0 of a 5-turn thread (`length_scale=0.2`): **~1.4**.

The `extract_verdict(text)` helper from `rewards.py` is shared by `baseline.py` and `evaluate.ipynb` — parsing logic lives in one place.

## Hardware Targets

| Phase | Model | GPU | Notes |
|---|---|---|---|
| 1 (default) | `unsloth/Llama-3.2-1B-Instruct` | T4 16 GB (free Colab) | Prototype: verify reward functions, check loss decreases |
| 2 | `unsloth/Qwen3-8B-Instruct` | L4 22 GB (Colab Pro) | Scale up; pass `--model unsloth/Qwen3-8B-Instruct` |

Unsloth's `UNSLOTH_VLLM_STANDBY=1` (set automatically by `train.py`) enables sequential vLLM/train memory sharing, making Phase 1 fit on T4.

## Data Sources

Source files are real copies in `../raw-data/`. See the project plan for overlap analysis and file selection rationale. 1,508 unique evaluation records across 11 source files.
