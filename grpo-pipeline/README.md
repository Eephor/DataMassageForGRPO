# grpo-pipeline

Transforms Ethos Academy scored conversation data into TRL `GRPOTrainer`-ready datasets for training an LLM oversight agent.

## Structure

```
src/grpo_pipeline/
├── models.py     # Pydantic schemas for source JSONL records and GRPORecord output
├── transform.py  # Thread reconstruction + GRPO prompt formatting
├── split.py      # Thread-level train/test splitting (no leakage)
├── rewards.py    # GRPO reward functions + safe_apply_template helper
├── train.py      # GRPO training script (Unsloth + TRL GRPOTrainer) — CLI entry point
└── baseline.py   # Headless batch evaluation against test set
train.ipynb       # Interactive training notebook (Colab)
evaluate.ipynb    # Interactive notebook: single-record inspection + metrics table
setup.sh          # One-command VM/bare-metal setup + optional training launch
Dockerfile        # Container image (CUDA 12.4, PyTorch 2.6, Unsloth)
docker-compose.yml# GPU-enabled compose service with volume mounts
tests/
├── test_split.py            # Verifies no data leakage across splits
└── test_pipeline_format.py  # Unit tests for template formatting and reward functions
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
# Step 1: install Unsloth + vLLM (platform-specific CUDA wheels, do this first)
pip install unsloth vllm
pip install --no-deps trl==0.22.2

# Step 2: install this package with training extras
uv pip install -e ".[train]"

# Step 3: train with GRPO
# If no --model flag is given, GRPO_MODEL env var is checked, then an interactive menu is shown.
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

## Reward Functions

Three GRPO reward functions in `rewards.py`, each with signature `(prompts, completions, **kwargs) -> list[float]`:

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

```bash
# Default model (Llama-3.2-3B-Instruct):
docker compose up

# Choose a different model via env var:
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

## Data Sources

Source files are real copies in `../raw-data/`. See the project plan for overlap analysis and file selection rationale. 1,508 unique evaluation records across 11 source files.
