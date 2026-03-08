# Moltbook Oversight Agent — GRPO training container
#
# Base: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
#   - CUDA 12.4, cuDNN 9, Python 3.11, PyTorch 2.6 — matches Unsloth requirements.
#
# Volume mounts (see docker-compose.yml):
#   /data     — training data (train.jsonl, test.jsonl) — read-only at runtime
#   /output   — LoRA adapter + merged model are written here
#   /root/.cache/huggingface — model weight cache (shared with host to avoid re-downloads)
#
# Build:
#   docker build -t moltbook-grpo .
#
# Run without compose:
#   docker run --gpus all --rm \
#     -e GRPO_MODEL=unsloth/Llama-3.2-3B-Instruct \
#     -e HF_TOKEN=$HF_TOKEN \
#     -v $(pwd)/../transformed:/data:ro \
#     -v $(pwd)/../lora-adapter:/output \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     moltbook-grpo \
#     --train-file /data/train.jsonl --output-dir /output

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# ── Layer 1: system packages ──────────────────────────────────────────────────
# Kept minimal — only what is needed to build / download python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: heavy ML dependencies (cache-stable layer) ───────────────────────
# Install Unsloth, vLLM, and a pinned TRL separately so Docker can cache this
# expensive layer as long as these version constraints do not change.
#
# Note: vllm is installed WITHOUT the [cuda] extra here because the base image
# already ships the correct CUDA runtime libraries.  Unsloth auto-detects
# the GPU and selects the right kernels at load time.
RUN pip install --no-cache-dir \
        unsloth \
        vllm \
        triton \
    && pip install --no-cache-dir --no-deps trl==0.22.2

# ── Layer 3: project source ───────────────────────────────────────────────────
# Copy the package source tree (changes more frequently than ML deps → later layer).
WORKDIR /workspace
COPY . /workspace/

# Install the grpo_pipeline package in editable mode so local source changes
# are picked up when the container mounts a source overlay at runtime.
RUN pip install --no-cache-dir -e ".[train]"

# ── Layer 4: runtime configuration ───────────────────────────────────────────
# UNSLOTH_VLLM_STANDBY=1 enables sequential memory sharing so the vLLM
# inference engine and the training backward pass never compete for GPU VRAM.
ENV UNSLOTH_VLLM_STANDBY=1

# Default data / output paths (overridden by -v mounts in docker-compose.yml)
ENV GRPO_TRAIN_FILE=/data/train.jsonl
ENV GRPO_OUTPUT_DIR=/output

# Entrypoint — any flags passed to `docker run` are appended.
# docker-compose.yml sets GRPO_MODEL, HF_TOKEN, and mounts /data and /output.
ENTRYPOINT ["python", "-m", "grpo_pipeline.train"]
CMD ["--train-file", "/data/train.jsonl", "--output-dir", "/output"]
