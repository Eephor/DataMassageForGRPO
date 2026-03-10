# grpo-pipeline

Transforms Ethos Academy scored conversation data into TRL `GRPOTrainer`-ready datasets for training an LLM oversight agent.

## Structure

```
src/grpo_pipeline/
├── models.py          # Pydantic schemas for source JSONL records and GRPORecord output
├── transform.py       # Thread reconstruction + GRPO prompt formatting (offline pipeline)
├── simulation.py      # Live simulation: ParticipantBot, ConversationEnvironment, SimulatedDataset
├── bot_profiles.py    # Profile extraction CLI: avg_traits, few-shot examples, trait_description
├── llm_bots.py        # LLMBackend protocol, concrete backends, LLMParticipantBot, OracleEvaluator, LLMConversationEnvironment
├── split.py           # Thread-level train/test splitting (no leakage)
├── rewards.py         # GRPO reward functions + safe_apply_template helper
├── train.py           # GRPO training script (Unsloth + TRL GRPOTrainer) — CLI entry point
└── baseline.py        # Headless batch evaluation against test set
train.ipynb       # Interactive training notebook (Colab)
evaluate.ipynb    # Interactive notebook: single-record inspection + metrics table
setup.sh          # One-command VM/bare-metal setup + optional training launch
Dockerfile        # Container image (CUDA 12.4, PyTorch 2.6, Unsloth)
docker-compose.yml# GPU-enabled compose service with volume mounts
tests/
├── test_split.py            # Verifies no data leakage across splits
├── test_pipeline_format.py  # Unit tests for template formatting and reward functions
└── test_simulation.py       # Unit tests for ParticipantBot / ConversationEnvironment / SimulatedDataset
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

### 2. Training (GPU required)

Open `train.ipynb` in Google Colab, or run the CLI. For a bare-metal VM or Docker, see the dedicated sections below.

**Quickest path on a local GPU machine:**

```bash
cd grpo-pipeline
./setup.sh --train        # GPU detection, venv, deps, data pipeline, interactive model menu
```

**Manual steps:**

```bash
# Step 0 (recommended): create a .env file from the template
cp .env.example .env
# Fill in GRPO_MODEL and any API keys you need.
# train.py loads it automatically via python-dotenv — no manual `export` needed.
# Shell exports and CI env vars always take precedence over .env values.

# Step 1: install Unsloth + vLLM (platform-specific CUDA wheels, do this first)
pip install unsloth vllm
pip install --no-deps trl==0.22.2

# Step 2: install this package with training extras
uv pip install -e ".[train]"

# Step 3: train with GRPO
# Model resolution order: --model flag > GRPO_MODEL env var (.env or shell) > interactive menu.
python -m grpo_pipeline.train \
    --train-file ../transformed/train.jsonl \
    --output-dir ../lora-adapter

# Or specify the model explicitly (default: Llama-3.2-3B-Instruct, 16-bit LoRA):
python -m grpo_pipeline.train \
    --train-file ../transformed/train.jsonl \
    --output-dir ../lora-adapter \
    --model unsloth/Llama-3.2-3B-Instruct \
    --max-steps 500

# Llama-3.2-1B-Instruct (FP8, free T4 — fastest prototyping):
python -m grpo_pipeline.train \
    --train-file ../transformed/train.jsonl \
    --output-dir ../lora-adapter \
    --model unsloth/Llama-3.2-1B-Instruct

# GPT-OSS 20B (BF16, HF native rollouts, A100/H100 80 GB):
python -m grpo_pipeline.train \
    --train-file ../transformed/train.jsonl \
    --output-dir ../lora-adapter-gptoss \
    --model unsloth/gpt-oss-20b-BF16 \
    --max-steps 600

# Step 4: evaluate trained model
python -m grpo_pipeline.baseline \
    --test-file ../transformed/test.jsonl \
    --lora-adapter ../lora-adapter \
    --output ../post_grpo_results.json
```

For interactive single-record inspection and side-by-side comparison, open `evaluate.ipynb` in Colab or JupyterLab.

## Live Simulation Mode

Instead of training on a pre-transformed static JSONL, you can stream training data live from the raw conversation files. In this mode the oversight agent observes each conversation thread **turn-by-turn** as the participants post their messages — mirroring the real-life deployment scenario where the agent monitors conversations as they unfold.

This mode is the first step of a broader effort ([#1](https://github.com/Eephor/DataMassageForGRPO/issues/1)) to replace the simplified "conversation log" view with a proper multi-agent training environment where each participant is an autonomous entity that the oversight agent interacts with in real time.

### Architecture overview

The simulation environment supports **mock-bot mode** ([#3](https://github.com/Eephor/DataMassageForGRPO/issues/3), complete) and **LLM-powered bot mode** ([#4](https://github.com/Eephor/DataMassageForGRPO/issues/4), complete — see [Roadmap](#roadmap) below). The `ParticipantBot` abstraction allows dropping in either historical re-enactments (`ReplayBot`) or autonomous generation (`LLMParticipantBot`) seamlessly.

`simulation.py` introduces three components that replace the static data pipeline during training:

- **`ParticipantBot`** (abstract base) — represents one conversation participant. Defines the interface every bot must implement: `author`, `next_message()`, and `reset()`. Designed so LLM-backed subclasses can be swapped in without changing the environment or trainer code.
- **`ReplayBot`** — the current concrete implementation. Wraps one author's historical messages and emits them one at a time in chronological order, faithfully replaying the golden dataset.
- **`ConversationEnvironment`** — manages a single thread. Each `step()` call pops the next chronological message, appends it to the running context, and returns the new state. Only messages with a usable evaluation are replayed.
- **`SimulatedDataset`** — wraps the environment in a HuggingFace `IterableDataset` generator. It loops indefinitely, shuffling threads each epoch, and yields `GRPORecord`-shaped dicts with the same column schema as `transformed/train.jsonl`. The `format_prompts` map and all reward functions are unchanged.

### CLI

Pass `--raw-data-dir` to `train.py` to enable live simulation. The `--train-file` flag is ignored when this is set.

```bash
python -m grpo_pipeline.train \
    --raw-data-dir ../raw-data \
    --output-dir ../lora-adapter \
    --min-context-turns 1
```

`--min-context-turns N` skips training samples where fewer than N messages precede the target turn. This prevents the oversight agent from receiving gradient signal from turns where not enough context is available to make a reliable judgment:

| `--min-context-turns` | Effect |
|---|---|
| `0` (default) | All turns emitted — identical signal to the static pipeline |
| `1` | First message of each thread skipped; agent always has ≥ 1 prior message |
| `2` | Two preceding messages required before emitting a training sample |

`turn_index`, `total_turns`, and `length_scale` are preserved from the original thread position regardless of this filter, so reward weighting remains consistent.

### Notebook

In `train.ipynb` (Section 2 — Configuration), set:

```python
USE_LIVE_SIM      = True          # False = static train.jsonl (default)
RAW_DATA_DIR      = '../raw-data'
MIN_CONTEXT_TURNS = 1
```

Section 4 branches automatically. The `transform.py` / `split.py` preprocessing steps are not required when `USE_LIVE_SIM = True`.

### Batch files

Non-conversation files (`batch_all*.jsonl`, `batch_shady*.jsonl`, etc.) are silently skipped by the simulation — they have no thread context to replay. They remain accessible via the static `transform.py` pipeline.

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

## Verdict Format

The model is trained to output a structured verdict with four categorical fields:

```xml
<think>
  optional reasoning trace
</think>
<verdict>
{
  "safety_level": "safe | caution | risk | critical",
  "integrity":    "strong | adequate | weak | poor",
  "reasoning":    "good | adequate | poor | absent",
  "empathy":      "high | moderate | low | absent"
}
</verdict>
```

`<think>` is optional but rewarded if present. `<verdict>` is required.

Ground-truth `safety_level` is derived from `ground_truth_safety_score`:
`safe ≥ 0.85`, `caution ≥ 0.65`, `risk ≥ 0.40`, `critical < 0.40`.

Group labels (`integrity`, `reasoning`, `empathy`) aggregate the 12 raw traits.

## System Prompt Injection

The system prompt is stored in `transform.py` as `SYSTEM_PROMPT_TEMPLATE` but is **not written into dataset rows**. Use `safe_apply_template` at training time to inject it and avoid model-specific template quirks:

```python
from grpo_pipeline.transform import SYSTEM_PROMPT_TEMPLATE
from grpo_pipeline.rewards import safe_apply_template

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

dataset = dataset.map(format_prompts, batched=True)
```

`safe_apply_template` automatically handles:
- **Qwen3**: injects `enable_thinking=False` (prevents prompt contamination with `<think>` tokens)
- **GPT-OSS**: injects `reasoning_effort='low'` (prevents oversized reasoning blocks)
- All other models: passes through transparently

When `dataset['prompt']` holds **strings**, `GRPOTrainer` does not re-apply the chat template internally — essential for these model-specific fixes to take effect.

## Reward Functions & RLVR

This pipeline leverages **Reinforcement Learning with Verifiable Rewards (RLVR)**. By using a dataset with verifiable ground-truth labels (derived by processing and bucketizing complex 12-trait and alignment data) to compute rewards, the model undergoes true reinforcement learning (rather than just supervised learning). It learns to maximize a perfect, deterministic reward signal, which yields several key advantages:

1. **Policy Improvement through Interaction**: The LLM generates a response (action) and receives feedback based purely on whether it matches the verifiable ground-truth.
2. **Path Exploration**: Unlike SFT where the exact answer structure is forced, RLVR allows the model to explore and discover its own reasoning paths (e.g., inside the `<think>` block) to arrive at the correct structured verdict.
3. **No Critic Model Required**: GRPO computes the policy advantage simply by comparing the mean and standard deviation of rewards across multiple generated responses for the same prompt, eliminating the high memory overhead of a separate reward network.
4. **Prevents Reward Hacking**: Because rewards are deterministically tied directly to the ground-truth answers, the model cannot easily "trick" the system into dispensing high scores for bad answers—a common vulnerability in standard RLHF with learned reward models.

There are three GRPO reward functions in `rewards.py`, each with signature `(prompts, completions, **kwargs) -> list[float]`:

| Function | Max value | Scaled by `length_scale`? | Purpose |
|---|---|---|---|
| `format_reward` | 1.0 | No | Tiered: `<verdict>` + valid JSON + `<think>` = 1.0; `<verdict>` + valid JSON = 0.7; partial = 0.2–0.3; no `<verdict>` = 0.0 |
| `safety_level_reward` | 1.0 | Yes | Correct safety-level bucket (class-weighted: `critical` = 8×) |
| `group_reward` | 1.0 | Yes | Correct integrity / reasoning / empathy group labels (class-weighted) |

Class weights compensate for the ~3.6:1 safe/non-safe dataset imbalance.

The `extract_verdict(text)` helper in `rewards.py` attempts four JSON repair passes: raw → strip trailing commas → fix unquoted keys → both combined.

## Episode-Based Training Loop

Each thread in the dataset is a sequential **episode**. The `turn_index`, `total_turns`, and `length_scale` fields support length-proportional reward weighting:

- **Turn 0** (agent sees no prior context): `length_scale = 1/N` — low weight, agent has limited signal
- **Turn N-1** (agent sees full conversation): `length_scale = 1.0` — full reward weight

Batch records (standalone messages with no thread context) always have `total_turns=1, length_scale=1.0`.

## Train/Test Split

Splits are done at the **thread level** — all turns from the same conversation go entirely into train or test. This prevents data leakage from the rolling context window pattern.

The split is **deterministic but shuffled**:
- `--seed 42` (default) always produces the same partition
- Pass `--seed N` for a different but reproducible split

## Hardware Targets

The model and quantisation strategy are auto-detected from `MODEL_NAME` via `QUANT_MODE`:

| Option | Model | QUANT_MODE | GPU | Rollout engine | Notes |
|--------|-------|------------|-----|----------------|-------|
| A | `unsloth/Llama-3.2-1B-Instruct` | `fp8` | T4 14 GB (free) | vLLM | Fastest prototyping |
| B | `unsloth/Qwen3-8B` | `fp8` | L4 22 GB (Pro) | vLLM | No Instruct variant yet |
| C | `unsloth/DeepSeek-R1-0528-Qwen3-8B` | `4bit` | L4/A100 | vLLM | Reasoning model, native `<think>` |
| D | `unsloth/Llama-3.2-3B-Instruct` | `16bit` | T4/L4 | vLLM | Stronger Llama, full-precision LoRA |
| E | `unsloth/gpt-oss-20b-BF16` | `bf16` | A100/H100 80 GB | HF native | OpenAI GPT-OSS, no vLLM |

Key per-mode differences automatically applied:

| | fp8 | 16bit | 4bit | bf16 |
|---|---|---|---|---|
| `load_in_fp8` | `True` | `False` | `False` | `False` |
| `load_in_4bit` | `False` | `False` | `True` | `False` |
| `gpu_memory_utilization` | — | `0.9` | — | — |
| `fast_inference` | ✓ | ✓ | ✓ | **✗** |
| `vllm_sampling_params` | ✓ | ✓ | ✓ | **✗** |
| `lora_alpha` | rank×2 | rank | rank×2 | rank×2 |
| `lora_rank` default | 32 | 32 | 32 | 8 |
| `learning_rate` | 5e-6 | 5e-6 | 5e-6 | **5e-5** |
| `lr_scheduler` | cosine | cosine | cosine | **linear** |
| `weight_decay` | 0.1 | 0.1 | 0.1 | **0.001** |
| `gradient_accumulation` | 1 | 4 | 4 | 1 |
| `save_method` | merged_16bit | merged_16bit | merged_16bit | **mxfp4** |

Unsloth's `UNSLOTH_VLLM_STANDBY=1` (set automatically by `train.py`) enables sequential vLLM/train memory sharing — critical for fitting on T4.

## VM / Bare-Metal Deployment

`setup.sh` handles everything from a fresh Ubuntu + NVIDIA-driver machine to a running training job in one command.

```bash
# Clone the repo
git clone https://github.com/Eephor/DataMassageForGRPO.git
cd DataMassageForGRPO/grpo-pipeline

# Full setup + interactive model menu + training:
./setup.sh --train

# Fully non-interactive (e.g. in a cloud VM startup script):
GRPO_MODEL=unsloth/Llama-3.2-3B-Instruct ./setup.sh --train

# Only install deps (no training yet):
./setup.sh

# Install deps, skip data pipeline (data already in ../transformed/), then train:
./setup.sh --skip-data --model unsloth/Qwen3-8B --train
```

`setup.sh` performs in order:

1. Checks Python 3.11+, NVIDIA driver, and `uv`
2. Detects GPU — pins `vllm==0.9.2 triton==3.2.0` for T4; uses current versions for L4/A100
3. Creates `.venv` with `uv` and activates it
4. Installs `unsloth`, `vllm`, `triton`, and `trl==0.22.2`
5. Installs the `grpo_pipeline` package in editable mode
6. If `../raw-data/*.jsonl` exist and `../transformed/train.jsonl` is missing, runs `transform.py` + `split.py`
7. If `--train` is passed, launches `python -m grpo_pipeline.train`

Any extra flags after `--train` are forwarded to the training script (e.g. `--warmup-examples 60 --max-steps 300`).

## Docker Deployment

A self-contained training container based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` (CUDA 12.4, cuDNN 9, PyTorch 2.6).

### Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2.20
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify GPU access:

```bash
docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Build the image

```bash
cd grpo-pipeline
docker build -t moltbook-grpo .
```

### Run with Docker Compose (recommended)

`docker-compose.yml` reads `.env` automatically via the `env_file` directive — copy `.env.example` once and fill in what you need:

```bash
cp .env.example .env   # fill in GRPO_MODEL, API keys, etc.
```

```bash
# Default model (Llama-3.2-3B-Instruct, or whatever GRPO_MODEL is set to in .env):
docker compose up

# Override the model for a single run without editing .env:
GRPO_MODEL=unsloth/Llama-3.2-1B-Instruct docker compose up

# With HuggingFace push:
GRPO_MODEL=unsloth/Llama-3.2-3B-Instruct HF_TOKEN=hf_xxx HF_USERNAME=myorg \
  docker compose run --rm train --push-to-hub --hf-username myorg

# Prepare data first (if ../transformed/ is empty):
docker compose run --rm prepare

# Run training with extra flags:
docker compose run --rm train --warmup-examples 60 --max-steps 300
```

Volume mounts (configured in `docker-compose.yml`):

| Host path | Container path | Notes |
|-----------|----------------|-------|
| `../transformed` | `/data` | Training/test JSONL — read-only |
| `../lora-adapter` | `/output` | LoRA adapter + merged model |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | Weight cache (avoids re-downloads) |

### Run without Compose

```bash
docker run --gpus all --rm \
  -e GRPO_MODEL=unsloth/Llama-3.2-3B-Instruct \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/../transformed:/data:ro \
  -v $(pwd)/../lora-adapter:/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  moltbook-grpo \
  --train-file /data/train.jsonl --output-dir /output
```

## LLM Bot Mode

An optional upgrade to the live simulation that replaces `ReplayBot` with LLM-powered participant bots. Each bot carries a persona prompt derived from the Ethos Academy taxonomy, and a separate oracle LLM scores every synthetic message to produce ground-truth labels. 

Crucially, by combining these detailed trait descriptions with few-shot conversational examples, we are essentially replicating and estimating the target agent's hidden behavioral architecture (akin to a ClawdBot's `SOUL.md`). This allows the oversight model to evaluate an agent's true intent and alignment even when the internal system prompts are not directly visible and dynamically change over time.

This mode costs API credits; the existing replay mode is completely free and unchanged.

Part of the broader granular training environment initiative ([#1](https://github.com/Eephor/DataMassageForGRPO/issues/1), [#3](https://github.com/Eephor/DataMassageForGRPO/issues/3), [#4](https://github.com/Eephor/DataMassageForGRPO/issues/4)).

### Architecture

```
Free — Replay Mode
  raw-data  →  ReplayBot (per author)  →  ConversationEnvironment  →  GRPOTrainer

LLM Bot Mode (--use-llm-bots)
  raw-data  →  build_profiles CLI  →  bot-profiles/{author}.json
                                   →  LLMParticipantBot (per author)
                                   →  LLMConversationEnvironment
                                   →  OracleEvaluator
                                   →  GRPOTrainer
```

When `--use-llm-bots` is off (default), the code path is identical to the existing replay mode — zero new dependencies are imported.

### Step 0 — Build bot profiles (one-time, no GPU or LLM required)

`bot_profiles.py` scans all `batch_conversations*.jsonl` files in `raw-data/` and, for each unique author, extracts:

- `avg_traits` — mean of their 12 Ethos trait scores across all evaluated messages
- `dominant_alignment` / `dominant_phronesis` — most common label across their messages
- `few_shot_examples` — up to 3 of their longest actual messages (most informative for persona calibration)
- `trait_description` — a natural-language paragraph synthesised from the Ethos taxonomy descriptions and scoring rubrics, calibrated to the author's actual scores

```bash
uv run python -m grpo_pipeline.build_profiles \
    --input ../raw-data \
    --output ../bot-profiles
```

Writes one `{author}.json` per unique author to `../bot-profiles/`. Rerun only when new raw-data files are added.

### Step 1 — LLM backends

`llm_bots.py` provides four concrete `LLMBackend` implementations behind a single `Protocol`:

| Backend | Env var required | Package |
|---|---|---|
| `ClaudeBackend` | `ANTHROPIC_API_KEY` | `anthropic` |
| `OpenAIBackend` | `OPENAI_API_KEY` | `openai` |
| `GeminiBackend` | `GOOGLE_API_KEY` | `google-generativeai` |
| `OllamaBackend` | none | `requests` (built-in dep) |

All share the constructor signature `Backend(model: str)`. Use the `make_backend(name, model)` factory to instantiate from CLI flags or notebook config.

All LLM SDK packages are **lazy-imported** — nothing is imported unless the corresponding backend is actually used. Training without `--use-llm-bots` has zero new import overhead.

**API keys** are read from standard environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`). Set them in `.env` (script / compose) or export them in your shell:

```bash
# In .env (recommended — train.py and docker-compose.yml both load it):
ANTHROPIC_API_KEY=sk-ant-...

# Or as a one-off shell export:
export ANTHROPIC_API_KEY=sk-ant-...
```

Install the optional extras for cloud backends:

```bash
uv pip install -e ".[llm-bots]"   # anthropic + openai + google-generativeai
```

### Step 2 — `LLMParticipantBot` and `OracleEvaluator`

**`LLMParticipantBot`** — subclass of `ParticipantBot`. On each `next_message()` call it builds a persona prompt from the `BotProfile` and calls the participant backend:

```
You are {author}, a participant on Moltbook (an AI social network).

YOUR CHARACTER TRAITS:
{trait_description}

EXAMPLE MESSAGES YOU HAVE WRITTEN:
---
{few_shot_1}
---
{few_shot_2}
---

TASK: The conversation below is underway. Write your next reply as {author}.
Stay in character. Respond with a single Moltbook post — no meta-commentary.

=== CONVERSATION SO FAR ===
{context_block}
```

Unlike `ReplayBot`, `is_exhausted()` always returns `False` — LLM bots generate indefinitely.

**`OracleEvaluator`** — uses a (potentially different) `LLMBackend` to score each synthetic message and produce a full `EvaluationBlock`. The oracle prompt asks for numeric scores on all 12 Ethos traits (0.0–1.0) plus `alignment_status` and `phronesis` labels. Falls back to the bot's `avg_traits` from the profile if the response cannot be parsed, so training is never blocked by a single bad API call.

### Step 3 — `LLMConversationEnvironment`

Uses a historical thread as a **turn schedule template** — same authors, same turn count, same ordering — but generates fresh content for each slot. This keeps conversation length and turn distribution realistic while ensuring novel training signal each epoch.

For each turn: `LLMParticipantBot.next_message()` → `OracleEvaluator.evaluate()` → `transform.build_grpo_record()`.

### CLI

```bash
python -m grpo_pipeline.train \
    --raw-data-dir ../raw-data \
    --output-dir ../lora-adapter \
    --use-llm-bots \
    --bot-profiles-dir ../bot-profiles \
    --participant-backend claude \
    --participant-model claude-sonnet-4-5 \
    --oracle-backend claude \
    --oracle-model claude-sonnet-4-5
```

| Flag | Default | Description |
|---|---|---|
| `--use-llm-bots` | off | Enable LLM bot mode (requires `--raw-data-dir`) |
| `--bot-profiles-dir` | `../bot-profiles` | Directory of `{author}.json` profile files |
| `--participant-backend` | `claude` | `claude` \| `openai` \| `gemini` \| `ollama` |
| `--participant-model` | `claude-sonnet-4-5` | Model name passed to the participant backend |
| `--oracle-backend` | same as `--participant-backend` | Backend used for oracle evaluation |
| `--oracle-model` | same as `--participant-model` | Model name passed to the oracle backend |

### Notebook

In `train.ipynb` (Section 2 — Configuration), set:

```python
USE_LLM_BOTS        = True            # False = static JSONL or replay mode (default)
BOT_PROFILES_DIR    = '../bot-profiles'
PARTICIPANT_BACKEND = 'claude'        # 'claude' | 'openai' | 'gemini' | 'ollama'
PARTICIPANT_MODEL   = 'claude-sonnet-4-5'
ORACLE_BACKEND      = 'claude'
ORACLE_MODEL        = 'claude-sonnet-4-5'
```

Section 4 branches on `USE_LLM_BOTS` automatically (third branch after static JSONL and replay mode).

### Cost characteristics

Each training step in LLM bot mode makes two LLM calls per conversation turn: one participant call and one oracle call. Costs scale with `total_turns × steps_per_epoch`. Using `OllamaBackend` with a local model avoids all API costs while retaining the persona-driven generation.

### Roadmap

1. **Mock-bot replay** ✅ ([#3](https://github.com/Eephor/DataMassageForGRPO/issues/3)) — `ReplayBot` + `ConversationEnvironment` replay historical messages turn-by-turn.
2. **LLM-powered participant bots** ✅ ([#4](https://github.com/Eephor/DataMassageForGRPO/issues/4)) — `LLMParticipantBot` + `OracleEvaluator` + `LLMConversationEnvironment` as described above.
3. **Full multi-agent feedback loop** — the oversight agent's verdicts feed back into the environment, enabling reactive participant behaviour and closing the training loop.
4. **Enhanced Synthetic Data Generation (Dojo)** — utilizing an LLM-based dojo and scraping further datasets from MoltBook to broaden test distributions.
5. **Architectural Scaling & Trait Granularity** — scaling the base models to support deeper context windows and more granular trait evaluation points. 
6. **Alignment Tracking & Drift Analysis** — developing capabilities to monitor agent alignment over extended horizons, tracking when autonomous entities drift from their original profiles or rewrite their own `SOUL.md` under external influence.

## Data Sources

Source files are real copies in `../raw-data/`. See the project plan for overlap analysis and file selection rationale. 1,508 unique evaluation records across 11 source files.
