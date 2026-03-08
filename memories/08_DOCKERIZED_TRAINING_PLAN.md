---
name: VM and Docker Training
overview: Bring `train.py` to full parity with the notebook (SFT warmup, per-mode defaults, HF push), add interactive/env-var model selection, write a `setup.sh` for bare-metal VM deployment, and a `Dockerfile` + `docker-compose.yml` for containerised GPU training.
todos:
  - id: train-py-parity
    content: "train.py: add model selection (env var + interactive menu), per-mode auto-defaults, SFT warmup flag, HF push flag"
    status: completed
  - id: setup-sh
    content: "Create grpo-pipeline/setup.sh: GPU detection, uv venv, version-pinned installs, optional transform+split+train launch"
    status: completed
  - id: dockerfile
    content: Create grpo-pipeline/Dockerfile with pytorch/cuda base, layered dep install, volume-mount entrypoint
    status: completed
  - id: compose
    content: Create grpo-pipeline/docker-compose.yml with nvidia runtime, env vars, volume mounts
    status: completed
  - id: readme-update
    content: Update grpo-pipeline/README.md with VM and Docker usage sections
    status: completed
isProject: false
---

# VM and Docker Training Deployment Plan

## Scope

Three deliverables:

1. `grpo-pipeline/src/grpo_pipeline/train.py` — bring to notebook parity
2. `grpo-pipeline/setup.sh` — one-command VM setup + optional launch
3. `grpo-pipeline/Dockerfile` + `grpo-pipeline/docker-compose.yml`

---

## 1. `train.py` Gaps to Fill

### 1a. Interactive / env-var model selection

When neither `--model` nor `GRPO_MODEL` env var is set and stdin is a TTY, show a numbered menu:

```
Select model:
  1) unsloth/Llama-3.2-1B-Instruct     [fp8  – T4 14 GB free]
  2) unsloth/Qwen3-8B                  [fp8  – L4 22 GB]
  3) unsloth/DeepSeek-R1-0528-Qwen3-8B [4bit – L4/A100]
  4) unsloth/Llama-3.2-3B-Instruct     [16bit– T4/L4]  ← default
  5) unsloth/gpt-oss-20b-BF16          [bf16 – A100/H100 80 GB]
Enter 1-5 [4]:
```

Resolution order: `--model` flag → `GRPO_MODEL` env var → interactive menu → default.

### 1b. Per-mode auto-defaults

Currently `train.py` has fixed defaults (lora_rank=32, batch_size=4, num_generations=4, max_steps=350) regardless of model. Match the notebook:


| mode  | lora_rank | batch_size | num_generations | max_steps |
| ----- | --------- | ---------- | --------------- | --------- |
| fp8   | 32        | 4          | 4               | 500       |
| 16bit | 32        | 4          | 4               | 500       |
| 4bit  | 32        | 4          | 4               | 500       |
| bf16  | **8**     | **1**      | **2**           | **600**   |


These become the effective defaults when the user does not pass the corresponding flag. A CLI flag always wins. Logic lives in the body of `main()` after `quant_mode` is determined.

### 1c. SFT format warmup

Add `--warmup-examples N` (default `0` = skip). Port the full notebook implementation:

- `build_gold_completion(record)` generates a `<think>/<verdict>` from ground truth
- Balanced sampling across the four safety buckets (`per_bucket = N // 4`)
- `SFTTrainer` with `lr=2e-4`, 3 epochs, `max_seq_length = max_seq_length // 2`
- Source: `[train.ipynb](grpo-pipeline/train.ipynb)` cells 18–19

### 1d. HuggingFace push

Add flags `--push-to-hub` (bool, default False) and `--hf-username TEXT` (default empty). Reads `HF_TOKEN` env var. After saving locally, calls `model.push_to_hub_merged(...)` with the same `MODEL_NAME_STUB` derivation already used in the notebook.

---

## 2. `setup.sh` (VM / bare-metal)

Location: `grpo-pipeline/setup.sh`

```
Usage: ./setup.sh [--model MODEL_NAME] [--train]
```

Steps:

1. Check for Python 3.11+, NVIDIA driver, `uv`
2. Detect GPU via `nvidia-smi` (T4 → `vllm==0.9.2 triton==3.2.0`; else current vllm)
3. `uv venv .venv && source .venv/bin/activate`
4. `pip install unsloth vllm[cuda]` (version-pinned by GPU)
5. `pip install --no-deps trl==0.22.2`
6. `uv pip install -e ".[train]"`
7. Optionally run `python -m grpo_pipeline.transform` + `split` if `../raw-data/` exists
8. If `--train` passed, launch `python -m grpo_pipeline.train` (with `--model` forwarded if given, else env var / menu kicks in)

---

## 3. `Dockerfile` + `docker-compose.yml`

### Dockerfile

Location: `grpo-pipeline/Dockerfile`

```
Base: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

(includes CUDA 12.4, cuDNN 9, Python 3.11, PyTorch 2.6 — matches Unsloth requirements)

Layers (ordered for cache efficiency):

1. System deps (`git`, `curl`, `build-essential`)
2. `pip install unsloth vllm` + `trl==0.22.2` — expensive layer, cache-stable
3. `COPY` project source + `pip install -e ".[train]"`
4. `ENV UNSLOTH_VLLM_STANDBY=1`
5. `WORKDIR /workspace`
6. `ENTRYPOINT ["python", "-m", "grpo_pipeline.train"]`

Data and output are volume-mounted at runtime — no weights baked in.

### docker-compose.yml

```yaml
services:
  train:
    build: .
    runtime: nvidia
    environment:
      - GRPO_MODEL=${GRPO_MODEL:-unsloth/Llama-3.2-3B-Instruct}
      - HF_TOKEN=${HF_TOKEN:-}
      - UNSLOTH_VLLM_STANDBY=1
    volumes:
      - ../transformed:/data:ro        # training data (read-only)
      - ../lora-adapter:/output        # LoRA + merged model output
      - ~/.cache/huggingface:/root/.cache/huggingface  # model weight cache
    command: >
      --train-file /data/train.jsonl
      --output-dir /output
```

Run with: `GRPO_MODEL=unsloth/Llama-3.2-1B-Instruct docker compose up`

---

## Files Changed


| File                                                                                   | Action                                                 |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `[grpo-pipeline/src/grpo_pipeline/train.py](grpo-pipeline/src/grpo_pipeline/train.py)` | Add model menu, per-mode defaults, SFT warmup, HF push |
| `grpo-pipeline/setup.sh`                                                               | Create (new)                                           |
| `grpo-pipeline/Dockerfile`                                                             | Create (new)                                           |
| `grpo-pipeline/docker-compose.yml`                                                     | Create (new)                                           |
| `[grpo-pipeline/README.md](grpo-pipeline/README.md)`                                   | Add VM + Docker usage sections                         |
