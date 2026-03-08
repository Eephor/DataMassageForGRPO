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

Hardware targets
----------------
Phase 1 — prototype (default model):
    unsloth/Llama-3.2-1B-Instruct
    Free Colab T4 (16 GB) with UNSLOTH_VLLM_STANDBY=1
    max_seq_length=2048, num_generations=4, batch_size=4

Phase 2 — scale up (pass --model):
    unsloth/Qwen3-8B-Instruct
    Colab Pro L4 (22 GB) or RTX 3090/4090 (24 GB)
    Same script, same reward functions, no code changes

Key alignment with Unsloth FP8 GRPO notebook
--------------------------------------------
- fast_inference=True + max_lora_rank + load_in_fp8=True passed to from_pretrained
- lora_alpha = lora_rank * 2  (Unsloth 2x convergence recommendation)
- lora_dropout / bias omitted (Unsloth applies its own optimised defaults)
- random_state = 3407 (Unsloth canonical seed)
- Saves both LoRA adapter and merged 16-bit model

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
        --max-steps 350 \\
        --kl-coef 0.1

    # Scale to 8B:
    python -m grpo_pipeline.train \\
        --train-file ../transformed/train.jsonl \\
        --output-dir ../lora-adapter-8b \\
        --model unsloth/Qwen3-8B-Instruct \\
        --max-steps 500
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
                "Phase 1 default: unsloth/Llama-3.2-1B-Instruct (fits T4 16 GB). "
                "Phase 2: unsloth/Qwen3-8B-Instruct (needs L4/A100 22+ GB)."
            ),
        ),
    ] = "unsloth/Llama-3.2-1B-Instruct",
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
    # 1. Load model + tokenizer
    # ------------------------------------------------------------------
    typer.echo(f"Loading model: {model}")
    typer.echo(f"  max_seq_length={max_seq_length}, UNSLOTH_VLLM_STANDBY={os.environ.get('UNSLOTH_VLLM_STANDBY')}")

    loaded_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,      # FP8, not 4-bit
        fast_inference=True,     # enable vLLM rollout engine for GRPO
        max_lora_rank=lora_rank, # pre-allocate LoRA memory at load time
        load_in_fp8=True,        # Float8 quantisation — halves VRAM vs bfloat16
    )

    # ------------------------------------------------------------------
    # 2. Apply LoRA adapter
    # ------------------------------------------------------------------
    typer.echo(f"Applying LoRA (rank={lora_rank}, alpha={lora_rank * 2}) ...")
    loaded_model = FastLanguageModel.get_peft_model(
        loaded_model,
        r=lora_rank,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,          # 2x = faster convergence (Unsloth recommendation)
        use_gradient_checkpointing="unsloth",
        random_state=3407,                 # Unsloth canonical seed
        # lora_dropout and bias intentionally omitted — Unsloth optimised defaults apply
    )

    # ------------------------------------------------------------------
    # 3. Load and prepare dataset
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
    # 4. Configure GRPO training
    # ------------------------------------------------------------------
    max_prompt_length = max_seq_length - max_completion_length

    from vllm import SamplingParams  # noqa: PLC0415

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=42,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=learning_rate,
        beta=kl_coef,              # KL penalty; default 0.1 prevents large spikes
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps if max_steps > 0 else None,
        num_train_epochs=1 if max_steps <= 0 else None,
        save_steps=save_steps,
        output_dir=str(output_dir),
        report_to=report_to,
    )

    # ------------------------------------------------------------------
    # 5. Create trainer and train
    # ------------------------------------------------------------------
    typer.echo("Creating GRPOTrainer ...")
    typer.echo("  Reward functions: format_reward, safety_level_reward, group_reward")
    typer.echo(f"  num_generations={num_generations}, batch_size={batch_size}")
    typer.echo(f"  max_prompt_length={max_prompt_length}, max_completion_length={max_completion_length}")

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
    # 6. Save LoRA adapter and merged 16-bit model
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir.parent / (output_dir.name + "-merged-16bit")

    typer.echo(f"Saving LoRA adapter to {output_dir} ...")
    loaded_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    typer.echo(f"Saving merged 16-bit model to {merged_dir} ...")
    loaded_model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    typer.echo(f"\nDone.")
    typer.echo(f"  LoRA adapter:      {output_dir}")
    typer.echo(f"  Merged 16-bit:     {merged_dir}")
    typer.echo("To evaluate, run:")
    typer.echo(f"  python -m grpo_pipeline.baseline --test-file ../transformed/test.jsonl --lora-adapter {output_dir}")


if __name__ == "__main__":
    app()
