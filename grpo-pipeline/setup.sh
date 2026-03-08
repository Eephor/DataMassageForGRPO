#!/usr/bin/env bash
# setup.sh — one-command setup for the Moltbook GRPO training pipeline on a bare-metal VM.
#
# Usage:
#   ./setup.sh                           # install only
#   ./setup.sh --train                   # install + run training (interactive model menu)
#   ./setup.sh --model unsloth/Llama-3.2-3B-Instruct --train
#   ./setup.sh --skip-data --train       # skip transform/split if data already staged
#   GRPO_MODEL=unsloth/Qwen3-8B ./setup.sh --train
#
# Environment variables honoured:
#   GRPO_MODEL      model identifier (skips interactive menu)
#   HF_TOKEN        HuggingFace token (for --push-to-hub inside training)
#   HF_USERNAME     HuggingFace username (for --push-to-hub)

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} WARNING: $*"; }
die()   { echo -e "${RED}[setup]${NC} ERROR: $*" >&2; exit 1; }

# ── Argument parsing ──────────────────────────────────────────────────────────
TRAIN=false
SKIP_DATA=false
MODEL_ARG=""
TRAIN_EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)        TRAIN=true; shift ;;
        --skip-data)    SKIP_DATA=true; shift ;;
        --model)        MODEL_ARG="$2"; shift 2 ;;
        --model=*)      MODEL_ARG="${1#*=}"; shift ;;
        --warmup-examples|--max-steps|--output-dir|--train-file|--lora-rank|--kl-coef)
                        TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS $1 $2"; shift 2 ;;
        --push-to-hub|--no-push-to-hub)
                        TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS $1"; shift ;;
        --hf-username)  TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS $1 $2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--train] [--skip-data] [--model MODEL_NAME] [training flags...]"
            echo ""
            echo "Options:"
            echo "  --train              Run training after setup"
            echo "  --skip-data          Skip transform + split (data already in ../transformed/)"
            echo "  --model MODEL_NAME   Model to train (overrides GRPO_MODEL env var)"
            echo ""
            echo "Any additional flags are forwarded to 'python -m grpo_pipeline.train'."
            echo ""
            echo "Supported models:"
            echo "  unsloth/Llama-3.2-1B-Instruct      fp8  - T4 14 GB (free)"
            echo "  unsloth/Qwen3-8B                   fp8  - L4 22 GB"
            echo "  unsloth/DeepSeek-R1-0528-Qwen3-8B  4bit - L4/A100"
            echo "  unsloth/Llama-3.2-3B-Instruct      16bit- T4/L4  [default]"
            echo "  unsloth/gpt-oss-20b-BF16            bf16 - A100/H100 80 GB"
            exit 0 ;;
        *)
            TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS $1"; shift ;;
    esac
done

# Script lives in grpo-pipeline/; resolve repo root relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_DIR="$SCRIPT_DIR"

info "Repository root: $REPO_ROOT"
info "Pipeline dir:    $PIPELINE_DIR"

# ── 1. Prerequisites check ────────────────────────────────────────────────────
info "Checking prerequisites ..."

# Python 3.11+
if ! command -v python3 &>/dev/null; then
    die "python3 not found. Install Python 3.11+ and retry."
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 || ("$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11) ]]; then
    die "Python 3.11+ required (found $PY_VERSION). Install a newer Python and retry."
fi
ok "Python $PY_VERSION"

# NVIDIA driver
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Install the NVIDIA driver and retry."
fi
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
ok "GPU: ${GPU_INFO:-unknown}"

# uv
if ! command -v uv &>/dev/null; then
    info "uv not found — installing via pip ..."
    pip install --quiet uv || die "Failed to install uv."
fi
ok "uv $(uv --version 2>&1 | head -1)"

# ── 2. Virtual environment ────────────────────────────────────────────────────
VENV_DIR="$PIPELINE_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at $VENV_DIR ..."
    uv venv "$VENV_DIR" --python python3
else
    info "Virtual environment already exists at $VENV_DIR."
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
ok "Activated venv: $VIRTUAL_ENV"

# ── 3. Detect GPU to pick correct vllm/triton versions ───────────────────────
# T4 (compute capability 7.5) needs vllm==0.9.2 — newer vllm dropped its CUDA kernels.
# L4 / A100 / H100 can use the current vllm release.
SMI_OUTPUT=$(nvidia-smi 2>/dev/null || true)
IS_T4=false
if echo "$SMI_OUTPUT" | grep -qi "Tesla T4"; then
    IS_T4=true
fi

if $IS_T4; then
    VLLM_PIN="vllm==0.9.2"
    TRITON_PIN="triton==3.2.0"
    info "T4 GPU detected → pinning vllm=0.9.2 triton=3.2.0"
else
    VLLM_PIN="vllm"
    TRITON_PIN="triton"
    info "Non-T4 GPU detected → using current vllm/triton"
fi

# ── 4. Install Unsloth, vLLM, TRL ────────────────────────────────────────────
info "Installing Unsloth + vLLM (this may take several minutes) ..."
pip install --quiet unsloth "$VLLM_PIN" "$TRITON_PIN" || \
    die "Failed to install unsloth/vllm/triton. Check your CUDA version and internet connection."

info "Pinning TRL to 0.22.2 ..."
pip install --quiet --no-deps trl==0.22.2 || \
    die "Failed to install trl==0.22.2."

# ── 5. Install grpo-pipeline package ─────────────────────────────────────────
info "Installing grpo-pipeline package ..."
cd "$PIPELINE_DIR"
uv pip install --quiet -e ".[train]" || \
    die "Failed to install grpo-pipeline. Check pyproject.toml and try again."
ok "grpo-pipeline installed."

# ── 6. Optional data pipeline ────────────────────────────────────────────────
RAW_DATA_DIR="$REPO_ROOT/raw-data"
TRANSFORMED_DIR="$REPO_ROOT/transformed"
TRAIN_FILE="$TRANSFORMED_DIR/train.jsonl"

if ! $SKIP_DATA; then
    if [[ -d "$RAW_DATA_DIR" ]] && compgen -G "$RAW_DATA_DIR/*.jsonl" &>/dev/null; then
        if [[ -f "$TRAIN_FILE" ]]; then
            info "Transformed data already exists at $TRANSFORMED_DIR — skipping transform/split."
            info "(Pass --skip-data to suppress this check, or delete $TRANSFORMED_DIR to re-run.)"
        else
            info "Running data transformation ..."
            python3 -m grpo_pipeline.transform \
                --input "$RAW_DATA_DIR" \
                --output "$TRANSFORMED_DIR"

            info "Running train/test split ..."
            python3 -m grpo_pipeline.split \
                --input "$TRANSFORMED_DIR/dataset.jsonl" \
                --output "$TRANSFORMED_DIR" \
                --test-ratio 0.2 \
                --seed 42

            ok "Data pipeline complete: $TRANSFORMED_DIR"
        fi
    else
        warn "No JSONL files found in $RAW_DATA_DIR — skipping data pipeline."
        warn "Place your raw data in $RAW_DATA_DIR and re-run without --skip-data."
    fi
else
    info "Skipping data pipeline (--skip-data)."
fi

# ── 7. Optional training launch ───────────────────────────────────────────────
if $TRAIN; then
    if [[ ! -f "$TRAIN_FILE" ]]; then
        die "Training data not found at $TRAIN_FILE. Run setup.sh without --skip-data first."
    fi

    TRAIN_CMD="python3 -m grpo_pipeline.train --train-file $TRAIN_FILE --output-dir $REPO_ROOT/lora-adapter"

    if [[ -n "$MODEL_ARG" ]]; then
        TRAIN_CMD="$TRAIN_CMD --model $MODEL_ARG"
    fi

    # Append any extra forwarded flags
    if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
        TRAIN_CMD="$TRAIN_CMD $TRAIN_EXTRA_ARGS"
    fi

    info "Launching training:"
    info "  $TRAIN_CMD"
    echo ""
    eval "$TRAIN_CMD"
else
    echo ""
    ok "Setup complete!"
    echo ""
    echo "  To start training, run one of:"
    echo ""
    echo "  # Interactive model menu:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python -m grpo_pipeline.train --train-file $TRAIN_FILE --output-dir $REPO_ROOT/lora-adapter"
    echo ""
    echo "  # Or specify the model directly:"
    echo "  GRPO_MODEL=unsloth/Llama-3.2-3B-Instruct python -m grpo_pipeline.train \\"
    echo "      --train-file $TRAIN_FILE --output-dir $REPO_ROOT/lora-adapter"
    echo ""
    echo "  # Or re-run setup with --train:"
    echo "  ./setup.sh --train"
fi
