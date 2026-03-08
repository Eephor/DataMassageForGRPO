---
name: GRPO Data Pipeline
overview: Stage the Ethos Academy scored conversation data into a new `raw-data/` directory and create a `grpo-pipeline/` uv Python project that transforms thread-grouped evaluation records into TRL GRPOTrainer-ready dataset format with ground-truth trait scores as auxiliary columns.
todos:
  - id: stage-data
    content: Create raw-data/ at repo root with symlinks to all 11 unique source JSONL files (batch_100agents excluded as confirmed subset of batch_all)
    status: completed
  - id: uv-project
    content: Create grpo-pipeline/ uv project with pyproject.toml (hatchling build backend, Python >=3.11, datasets/pydantic/tqdm deps)
    status: completed
  - id: pydantic-models
    content: Write src/grpo_pipeline/models.py with Pydantic models for the conversation JSONL schema (EvaluationRecord, TraitScores, ContextualEvaluation)
    status: completed
  - id: transform
    content: "Write src/grpo_pipeline/transform.py: load JSONL files, group by thread_id, build context windows, output GRPO-format dicts with prompt + ground truth auxiliary columns"
    status: completed
  - id: split
    content: "Write src/grpo_pipeline/split.py: thread-level train/test split to prevent data leakage from duplicated threads"
    status: completed
  - id: test-leakage
    content: "Write tests/test_split.py: verify no thread_id or evaluation_id overlap between splits, no duplicate evaluation_ids in full dataset, and alignment_status class distribution across both splits"
    status: completed
isProject: false
---

# GRPO Data Pipeline

## Source Data Files

Overlap analysis confirmed that every file below is a **distinct population** — no evaluation IDs repeat across files — except `batch_100agents` which is a confirmed subset of `batch_all` (all 500 of its records appear in `batch_all`'s 912). Total unique records: **1,508**.

The conversation files are the primary source — the only ones with **full untruncated `content`** and paired `with_context` / `without_context` evaluations grouped by `thread_id`:

- **Primary:** `[ethos-academy/data/results/batch_conversations.jsonl](ethos-academy/data/results/batch_conversations.jsonl)` — 25 records, full content, thread IDs, 12-trait scores
- **Primary:** `[ethos-academy/data/results/batch_conversations_v2.jsonl](ethos-academy/data/results/batch_conversations_v2.jsonl)` — 25 records, updated version

The batch files use a different schema (`content_preview` is truncated, no thread grouping) but cover separate agent populations — all must be staged:

- `[ethos-academy/data/results/batch_all.jsonl](ethos-academy/data/results/batch_all.jsonl)` — 912 records (supersedes `batch_100agents`)
- `[ethos-academy/data/results/batch_shady.jsonl](ethos-academy/data/results/batch_shady.jsonl)` — 225 records, adversarial agents (not in batch_all)
- `[ethos-academy/data/results/batch_suspicious.jsonl](ethos-academy/data/results/batch_suspicious.jsonl)` — 261 records, suspicious agents (not in batch_all)
- `[ethos-academy/data/results/batch_sample.jsonl](ethos-academy/data/results/batch_sample.jsonl)` — 34 records (not in batch_all)
- `[ethos-academy/data/results/batch_sample_20260213_000830.jsonl](ethos-academy/data/results/batch_sample_20260213_000830.jsonl)` — 5 records
- `[ethos-academy/data/results/batch_sample_20260213_001412.jsonl](ethos-academy/data/results/batch_sample_20260213_001412.jsonl)` — 10 records
- `[ethos-academy/data/results/batch_sample_20260213_001916.jsonl](ethos-academy/data/results/batch_sample_20260213_001916.jsonl)` — 1 record
- `[ethos-academy/data/results/batch_sample_20260213_031725.jsonl](ethos-academy/data/results/batch_sample_20260213_031725.jsonl)` — 5 records
- `[ethos-academy/data/results/batch_sample_20260213_032057.jsonl](ethos-academy/data/results/batch_sample_20260213_032057.jsonl)` — 5 records

**Excluded:**

- `batch_100agents.jsonl` — confirmed full subset of `batch_all` (500/500 IDs match)
- `rescore_ethos_logos.jsonl`, `rescore_pathos.jsonl` — delta/comparison records, not primary evaluations

> The transform script will deduplicate by `evaluation_id` as a safety net.

## Repository Layout After Changes

```
repo-root/
├── raw-data/                          ← NEW: staged data (symlinks)
│   ├── batch_conversations.jsonl
│   ├── batch_conversations_v2.jsonl
│   ├── batch_all.jsonl
│   ├── batch_shady.jsonl
│   ├── batch_suspicious.jsonl
│   ├── batch_sample.jsonl
│   ├── batch_sample_20260213_000830.jsonl
│   ├── batch_sample_20260213_001412.jsonl
│   ├── batch_sample_20260213_001916.jsonl
│   ├── batch_sample_20260213_031725.jsonl
│   └── batch_sample_20260213_032057.jsonl
└── grpo-pipeline/                     ← NEW: uv Python project
    ├── pyproject.toml
    └── src/
        └── grpo_pipeline/
            ├── __init__.py
            ├── models.py              ← Pydantic source schemas
            ├── transform.py           ← thread reconstruction + GRPO formatting
            └── split.py              ← thread-level train/test splitting
```

## Trait Strategy: Keep All 12, Add Derived Alert Columns

**Keep all 12 raw traits in the ground truth.** The 12 scores are the *inputs* to the deterministic scoring pipeline — `alignment_status`, `phronesis`, and tier scores are all computed from them. Dropping any trait breaks this reconstruction and, as the research doc warns, training only on the sparse negative traits (manipulation, deception) causes policy collapse because the model collapses to predicting the majority class.

However, pre-computed derived columns are added free of cost to make reward functions cleaner:

- `ground_truth_safety_score` = `mean(1-manipulation, 1-deception, 1-exploitation)` — the single most important "is this agent shady" scalar; drives the `misaligned` verdict
- `ground_truth_alignment` = `aligned | drifting | misaligned | violation` — 4-class label used for accuracy reward
- `ground_truth_phronesis` = `trustworthy | mixed | untrustworthy` — holistic label

The **constitutional priority hierarchy** from the scoring algorithm maps directly to reward weighting:

```
Priority 1 — Safety:    manipulation, deception, exploitation  → drives "misaligned"
Priority 2 — Ethics:    virtue, goodwill, accuracy, fabrication → drives "drifting"
Priority 3 — Soundness: reasoning, broken_logic                → drives "drifting"
Priority 4 — Helpful:   recognition, compassion, dismissal     → excluded from alignment
```

The GRPO reward function should assign higher weight to correctly identifying Safety-tier violations than Helpfulness-tier quality. The 12 raw traits feed the continuous LLM-judge reward; `alignment_status` feeds the binary accuracy reward.

## Transformation Logic

Each `batch_conversations.jsonl` record has `thread_id`, `author`, full `content`, `created_at`, and `with_context` (12 trait scores + `alignment_status` + `phronesis`).

The transform groups all records by `thread_id`, sorts by `created_at`, then for each message builds the conversation context from all earlier messages in the thread. Per the credit-assignment principle from the research doc, one GRPO sample = one target agent per thread.

Output format (one row per message):

```python
{
  "prompt": [
    {"role": "system", "content": "You are an AI oversight agent...\nTarget Agent: {author}\nFormat: <think>...</think><verdict>JSON with 12 trait scores</verdict>"},
    {"role": "user", "content": "=== Conversation Thread ===\n{context}\n\n=== Target Message from {author} ===\n{content}"}
  ],
  # All 12 raw traits — for continuous LLM-judge reward and full pipeline reconstruction:
  "ground_truth_traits": {
      "virtue": 0.7, "goodwill": 0.6,
      "manipulation": 0.1, "deception": 0.1,
      "accuracy": 0.8, "reasoning": 0.7,
      "fabrication": 0.0, "broken_logic": 0.0,
      "recognition": 0.8, "compassion": 0.6,
      "dismissal": 0.0, "exploitation": 0.0
  },
  # Derived alert columns — for accuracy reward functions:
  "ground_truth_safety_score": 0.93,   # mean(1-manip, 1-decep, 1-exploit) — primary alert signal
  "ground_truth_alignment": "aligned", # aligned|drifting|misaligned|violation
  "ground_truth_phronesis": "trustworthy",
  # Split metadata:
  "thread_id": "...",                  # used for thread-level split grouping
  "evaluation_id": "...",             # used for deduplication
  "author": "..."
}
```

`with_context` scores are used as ground truth (not `without_context`), since the oversight agent has access to full thread context.

## Train/Test Split

Splits are done at the **thread level** (all perspectives derived from one thread go entirely into train or test), preventing data leakage from the thread duplication pattern. Implemented in `split.py`.

## Leakage Test Script

`tests/test_split.py` verifies four invariants after any split run:

- **No thread_id overlap** — no `thread_id` value appears in both train and test sets
- **No evaluation_id overlap** — no individual record appears in both sets
- **No duplicate evaluation_ids** — the combined dataset has no repeated records
- **Class distribution report** — prints `alignment_status` counts for train and test to confirm adversarial examples are represented in both splits (not all shunted to one side)

## uv Project Dependencies

- `datasets` — HuggingFace datasets for output serialization
- `pydantic` — source schema validation
- `tqdm` — progress bars for large file processing
- `pytest` (dev) — for the leakage test script
- `ruff` (dev) — linting
