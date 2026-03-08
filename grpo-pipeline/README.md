# grpo-pipeline

Transforms Ethos Academy scored conversation data into TRL `GRPOTrainer`-ready datasets for training an LLM oversight agent.

## Structure

```
src/grpo_pipeline/
├── models.py     # Pydantic schemas for source JSONL records and GRPORecord output
├── transform.py  # Thread reconstruction + GRPO prompt formatting
└── split.py      # Thread-level train/test splitting (no leakage)
tests/
└── test_split.py # Verifies no data leakage across splits
```

## Usage

```bash
# Install
uv sync

# Transform all staged data — write to repo-root/transformed/ (canonical output location)
uv run python -m grpo_pipeline.transform --input ../raw-data --output ../transformed

# Split into train/test (thread-level, no leakage)
uv run python -m grpo_pipeline.split --input ../transformed/dataset.jsonl --output ../transformed --test-ratio 0.2 --seed 42

# Verify no leakage
uv run pytest tests/
```

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

## Data Sources

Source files are symlinked in `../raw-data/`. See the project plan for overlap analysis and file selection rationale. 1,508 unique evaluation records across 11 source files.
