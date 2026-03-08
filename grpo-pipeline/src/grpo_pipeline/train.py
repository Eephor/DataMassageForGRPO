"""GRPO training script for the Moltbook oversight agent.

Trains a causal LM to evaluate conversation-thread participants using
Group Relative Policy Optimization (GRPO) with a simplified 4-field categorical verdict.

The reward signal has three components (see rewards.py for details):
  1. format_reward        — structured <think>/<verdict> output
  2. safety_level_reward  — correct safety-level bucket (class-weighted)
  3. group_reward         — correct integrity / reasoning / empathy group labels (class-weighted)

Rewards 2 and 3 are scaled by `length_scale` = (turn_index + 1) / total_turns,
so early-turn judgments with little context contribute less to the gradient.
Class weights compensate for the ~3.6:1 safe/non-safe dataset imbalance.

Supported models and quantisation modes
----------------------------------------
The script detects the appropriate loading strategy from the model name:

  "fp8"   — unsloth/Llama-3.2-1B-Instruct, unsloth/Qwen3-8B
              load_in_fp8=True, fast_inference=True (vLLM)
              lora_alpha = lora_rank * 2

  "16bit" — unsloth/Llama-3.2-3B-Instruct  (full-precision 16-bit LoRA)
              load_in_fp8=False, load_in_4bit=False, gpu_memory_utilization=0.9
              lora_alpha = lora_rank  (per Unsloth 3B notebook)
              gradient_accumulation_steps = 4, fast_inference=True (vLLM)

  "4bit"  — unsloth/DeepSeek-R1-0528-Qwen3-8B  (BitsAndBytes 4-bit)
              load_in_fp8=False, load_in_4bit=True
              lora_alpha = lora_rank * 2
              gradient_accumulation_steps = 4, fast_inference=True (vLLM)

  "bf16"  — unsloth/gpt-oss-20b-BF16  (OpenAI GPT-OSS, BF16, HF native rollouts)
              load_in_fp8=False, load_in_4bit=False, NO fast_inference
              vllm_sampling_params omitted from GRPOConfig
              lora_alpha = lora_rank * 2, lora_rank default = 8
              learning_rate = 5e-5, lr_scheduler = linear, weight_decay = 0.001
              safe_apply_template auto-injects reasoning_effort='low'
              save_method = 'mxfp4' (OpenAI native precision)

Prerequisites
-------------
    # Install Unsloth + vLLM (platform-specific, do this before uv sync)
    pip install unsloth vllm
    pip install --no-deps trl==0.22.2
    # Then install this package
    uv pip install -e ".[train]"

Usage
-----
    python -m grpo_pipeline.train \\
        --train-file ../transformed/train.jsonl \\
        --output-dir ../lora-adapter \\
        --max-steps 500 \\
        --kl-coef 0.1

    # Llama 3.2 3B (16-bit LoRA, T4/L4):
    python -m grpo_pipeline.train \\
        --train-file ../transformed/train.jsonl \\
        --output-dir ../lora-adapter-3b \\
        --model unsloth/Llama-3.2-3B-Instruct \\
        --max-steps 500

    # DeepSeek-R1 (4-bit BnB, L4/A100):
    python -m grpo_pipeline.train \\
        --train-file ../transformed/train.jsonl \\
        --output-dir ../lora-adapter-deepseek \\
        --model unsloth/DeepSeek-R1-0528-Qwen3-8B \\
        --max-steps 500

    # GPT-OSS 20B (BF16, HF native rollouts, A100/H100 80 GB):
    python -m grpo_pipeline.train \\
        --train-file ../transformed/train.jsonl \\
        --output-dir ../lora-adapter-gptoss \\
        --model unsloth/gpt-oss-20b-BF16 \\
        --max-steps 600
"""

from __future__ import annotations

import os

# MUST be set before importing unsloth / vllm — enables sequential memory sharing
# so the vLLM inference engine and the training backward pass never compete for
# GPU memory at the same time (critical for running on T4 16 GB).
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(pretty_exceptions_enable=False)

# ---------------------------------------------------------------------------
# Lazy imports — unsloth / trl are optional heavy dependencies
# ---------------------------------------------------------------------------


def _import_training_deps():
    """Import ML training dependencies. Raises ImportError with a helpful message if missing."""
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel

        return FastLanguageModel, GRPOConfig, GRPOTrainer, Dataset
    except ImportError as exc:
        raise ImportError(
            "Training dependencies are not installed. Run:\n"
            "  pip install unsloth vllm\n"
            "  pip install --no-deps trl==0.22.2\n"
            "  uv pip install -e '.[train]'"
        ) from exc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_train_dataset(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------


def _get_system_prompt_template() -> str:
    """Import the canonical SYSTEM_PROMPT_TEMPLATE from transform.py."""
    from grpo_pipeline.transform import SYSTEM_PROMPT_TEMPLATE  # noqa: PLC0415

    return SYSTEM_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Main training command
# ---------------------------------------------------------------------------


@app.command()
def main(
    train_file: Annotated[
        Path,
        typer.Option("--train-file", "-i", help="Path to train.jsonl produced by split.py."),
    ] = Path("../transformed/train.jsonl"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to save the LoRA adapter."),
    ] = Path("../lora-adapter"),
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Unsloth model identifier. "
                "FP8 (T4 14 GB): unsloth/Llama-3.2-1B-Instruct or unsloth/Qwen3-8B. "
                "16-bit LoRA (T4/L4): unsloth/Llama-3.2-3B-Instruct. "
                "4-bit BnB (L4/A100): unsloth/DeepSeek-R1-0528-Qwen3-8B. "
                "BF16 HF native (A100/H100): unsloth/gpt-oss-20b-BF16. "
                "Quantisation mode is inferred automatically from the model name."
            ),
        ),
    ] = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length: Annotated[
        int,
        typer.Option("--max-seq-length", help="Maximum total sequence length (prompt + completion)."),
    ] = 2048,
    max_completion_length: Annotated[
        int,
        typer.Option("--max-completion-length", help="Maximum tokens the model may generate per sample."),
    ] = 768,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Total GRPO training steps. Set to -1 for full epoch."),
    ] = 350,
    num_generations: Annotated[
        int,
        typer.Option(
            "--num-generations",
            help="Number of completions to sample per prompt (GRPO group size). Reduce if OOM.",
        ),
    ] = 4,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Per-device training batch size."),
    ] = 4,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="AdamW learning rate."),
    ] = 5e-6,
    lora_rank: Annotated[
        int,
        typer.Option("--lora-rank", help="LoRA rank. Higher = more capacity, more VRAM."),
    ] = 32,
    kl_coef: Annotated[
        float,
        typer.Option(
            "--kl-coef",
            help=(
                "KL divergence penalty coefficient (beta in GRPOConfig). "
                "Default 0.1 suppresses the large KL spikes seen with the ~0.04 TRL default "
                "when class-weighted rewards are high (critical class weight = 8.0)."
            ),
        ),
    ] = 0.1,
    save_steps: Annotated[
        int,
        typer.Option("--save-steps", help="Save a checkpoint every N steps."),
    ] = 100,
    report_to: Annotated[
        str,
        typer.Option("--report-to", help="Logging backend: 'none', 'wandb', or 'tensorboard'."),
    ] = "none",
) -> None:
    """Train the Moltbook oversight agent with GRPO on evaluation data."""

    FastLanguageModel, GRPOConfig, GRPOTrainer, Dataset = _import_training_deps()

    # Import reward functions after confirming deps are available
    from grpo_pipeline.rewards import (  # noqa: PLC0415
        format_reward,
        group_reward,
        safe_apply_template,
        safety_level_reward,
    )

    # ------------------------------------------------------------------
    # 1. Detect quantisation / inference mode from model name
    # ------------------------------------------------------------------
    _name = model.lower()
    if "gpt-oss" in _name or "gpt_oss" in _name:
        quant_mode = "bf16"
    elif "deepseek" in _name or "-r1" in _name:
        quant_mode = "4bit"
    elif "llama" in _name and "3b" in _name:
        quant_mode = "16bit"
    else:
        quant_mode = "fp8"

    # lora_alpha: 16-bit LoRA uses rank (per Unsloth 3B notebook); others use rank×2
    lora_alpha = lora_rank if quant_mode == "16bit" else lora_rank * 2
    # gradient_accumulation_steps: 4 for large/16-bit/4-bit models, 1 for FP8 and bf16
    grad_accum = 4 if quant_mode in ("16bit", "4bit") else 1
    # learning_rate / scheduler / weight_decay differ for GPT-OSS 20B
    if quant_mode == "bf16":
        effective_lr = 5e-5
        lr_scheduler = "linear"
        weight_decay = 0.001
    else:
        effective_lr = learning_rate
        lr_scheduler = "cosine"
        weight_decay = 0.1

    # ------------------------------------------------------------------
    # 2. Load model + tokenizer
    # ------------------------------------------------------------------
    typer.echo(f"Loading model: {model}  [{quant_mode}]")
    typer.echo(f"  max_seq_length={max_seq_length}, UNSLOTH_VLLM_STANDBY={os.environ.get('UNSLOTH_VLLM_STANDBY')}")

    _load_kwargs: dict = dict(
        model_name=model,
        max_seq_length=max_seq_length,
    )
    if quant_mode == "fp8":
        _load_kwargs.update(load_in_fp8=True, load_in_4bit=False,
                            fast_inference=True, max_lora_rank=lora_rank)
    elif quant_mode == "16bit":
        _load_kwargs.update(load_in_fp8=False, load_in_4bit=False,
                            gpu_memory_utilization=0.9,
                            fast_inference=True, max_lora_rank=lora_rank)
    elif quant_mode == "4bit":
        _load_kwargs.update(load_in_fp8=False, load_in_4bit=True,
                            fast_inference=True, max_lora_rank=lora_rank)
    else:  # "bf16" — HF native generation, no vLLM
        _load_kwargs.update(load_in_fp8=False, load_in_4bit=False)

    loaded_model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)

    # ------------------------------------------------------------------
    # 3. Apply LoRA adapter
    # ------------------------------------------------------------------
    typer.echo(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha}) ...")
    loaded_model = FastLanguageModel.get_peft_model(
        loaded_model,
        r=lora_rank,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        # lora_dropout and bias intentionally omitted — Unsloth optimised defaults apply
    )

    # ------------------------------------------------------------------
    # 4. Load and prepare dataset
    # ------------------------------------------------------------------
    typer.echo(f"Loading training data from {train_file} ...")
    raw_records = load_train_dataset(train_file)
    typer.echo(f"  Loaded {len(raw_records)} records.")

    dataset = Dataset.from_list(raw_records)

    # Pre-format prompts as strings using safe_apply_template so that:
    # (a) Qwen3 thinking-mode injection is suppressed (enable_thinking=False)
    # (b) GRPOTrainer receives strings and skips its internal apply_chat_template,
    #     ensuring our formatting is not overridden and completions arrive as strings
    #     (compatible with _completion_text in rewards.py)
    template = _get_system_prompt_template()

    def format_prompts(batch: dict) -> dict:
        return {"prompt": [
            safe_apply_template(
                tokenizer,
                [{"role": "system", "content": template.format(author=a)}] + p,
                tokenize=False,
                add_generation_prompt=True,
            )
            for a, p in zip(batch["author"], batch["prompt"])
        ]}

    dataset = dataset.map(format_prompts, batched=True, desc="Formatting prompts")

    typer.echo(f"  Dataset ready: {len(dataset)} records (pre-formatted strings).")

    # ------------------------------------------------------------------
    # 5. Configure GRPO training
    # ------------------------------------------------------------------
    max_prompt_length = max_seq_length - max_completion_length

    # vLLM sampling params are only valid when fast_inference=True (not bf16 mode)
    _grpo_extra: dict = {}
    if quant_mode != "bf16":
        from vllm import SamplingParams  # noqa: PLC0415
        _grpo_extra["vllm_sampling_params"] = SamplingParams(
            min_p=0.1,
            top_p=1.0,
            top_k=-1,
            seed=42,
            stop=[tokenizer.eos_token],
            include_stop_str_in_output=True,
        )

    training_args = GRPOConfig(
        **_grpo_extra,
        temperature=1.0,
        learning_rate=effective_lr,
        beta=kl_coef,              # KL penalty
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type=lr_scheduler,
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps if max_steps > 0 else None,
        num_train_epochs=1 if max_steps <= 0 else None,
        max_grad_norm=1.0,
        save_steps=save_steps,
        output_dir=str(output_dir),
        report_to=report_to,
    )

    # ------------------------------------------------------------------
    # 6. Create trainer and train
    # ------------------------------------------------------------------
    typer.echo("Creating GRPOTrainer ...")
    typer.echo("  Reward functions: format_reward, safety_level_reward, group_reward")
    typer.echo(f"  num_generations={num_generations}, batch_size={batch_size}, grad_accum={grad_accum}")
    typer.echo(f"  max_prompt_length={max_prompt_length}, max_completion_length={max_completion_length}")
    typer.echo(f"  lr={effective_lr}, lr_scheduler={lr_scheduler}, wd={weight_decay}")
    typer.echo(f"  vllm_sampling={'yes' if quant_mode != 'bf16' else 'no (bf16 HF native)'}")

    trainer = GRPOTrainer(
        model=loaded_model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,          # 1. structural format check (not length-scaled)
            safety_level_reward,    # 2. correct safety-level bucket (class-weighted, length-scaled)
            group_reward,           # 3. correct integrity/reasoning/empathy labels (class-weighted, length-scaled)
        ],
        args=training_args,
        train_dataset=dataset,
    )

    typer.echo("Starting GRPO training ...")
    typer.echo("  Tip: watch for 'reward' column in logs to confirm reward functions are firing.")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save LoRA adapter and merged 16-bit model
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir.parent / (output_dir.name + "-merged-16bit")

    typer.echo(f"Saving LoRA adapter to {output_dir} ...")
    loaded_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # GPT-OSS 20B supports mxfp4 (OpenAI native micro-float precision, ~4-bit).
    # All other models are saved as merged_16bit (bfloat16 base + LoRA baked in).
    save_method = "mxfp4" if quant_mode == "bf16" else "merged_16bit"
    typer.echo(f"Saving merged model [{save_method}] to {merged_dir} ...")
    loaded_model.save_pretrained_merged(str(merged_dir), tokenizer, save_method=save_method)

    typer.echo(f"\nDone.")
    typer.echo(f"  LoRA adapter:      {output_dir}")
    typer.echo(f"  Merged 16-bit:     {merged_dir}")
    typer.echo("To evaluate, run:")
    typer.echo(f"  python -m grpo_pipeline.baseline --test-file ../transformed/test.jsonl --lora-adapter {output_dir}")


if __name__ == "__main__":
    app()
