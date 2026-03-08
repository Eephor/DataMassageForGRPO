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

Model selection
---------------
  1. --model / -m flag
  2. GRPO_MODEL environment variable
  3. Interactive numbered menu (when stdin is a TTY)
  4. Default: unsloth/Llama-3.2-3B-Instruct

Prerequisites
-------------
    # Install Unsloth + vLLM (platform-specific, do this before uv sync)
    pip install unsloth vllm
    pip install --no-deps trl==0.22.2
    # Then install this package
    uv pip install -e ".[train]"

Usage
-----
    # Minimal — interactive model menu if stdin is TTY
    python -m grpo_pipeline.train

    # Env-var driven (ideal for scripts / Docker)
    GRPO_MODEL=unsloth/Llama-3.2-1B-Instruct python -m grpo_pipeline.train \\
        --train-file ../transformed/train.jsonl \\
        --output-dir ../lora-adapter

    # Explicit flags
    python -m grpo_pipeline.train \\
        --train-file ../transformed/train.jsonl \\
        --output-dir ../lora-adapter \\
        --model unsloth/Llama-3.2-3B-Instruct \\
        --max-steps 500

    # With SFT warmup and HuggingFace push
    python -m grpo_pipeline.train \\
        --model unsloth/Llama-3.2-3B-Instruct \\
        --warmup-examples 60 \\
        --push-to-hub --hf-username myorg

    # GPT-OSS 20B (BF16, HF native rollouts, A100/H100 80 GB)
    python -m grpo_pipeline.train \\
        --model unsloth/gpt-oss-20b-BF16 \\
        --max-steps 600
"""

from __future__ import annotations

import os
import sys

# MUST be set before importing unsloth / vllm — enables sequential memory sharing
# so the vLLM inference engine and the training backward pass never compete for
# GPU memory at the same time (critical for running on T4 16 GB).
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(pretty_exceptions_enable=False)

# ---------------------------------------------------------------------------
# Model menu
# ---------------------------------------------------------------------------

_MODEL_MENU = [
    ("unsloth/Llama-3.2-1B-Instruct",     "fp8  – T4 14 GB free, fastest prototyping"),
    ("unsloth/Qwen3-8B",                   "fp8  – L4 22 GB, Colab Pro"),
    ("unsloth/DeepSeek-R1-0528-Qwen3-8B",  "4bit – L4/A100, reasoning model"),
    ("unsloth/Llama-3.2-3B-Instruct",      "16bit– T4/L4, stronger Llama  [default]"),
    ("unsloth/gpt-oss-20b-BF16",           "bf16 – A100/H100 80 GB, OpenAI GPT-OSS"),
]
_DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


def _resolve_model(model_flag: Optional[str]) -> str:
    """Resolve model in priority order: flag → env var → interactive menu → default."""
    if model_flag:
        return model_flag

    env_model = os.environ.get("GRPO_MODEL", "").strip()
    if env_model:
        typer.echo(f"Using model from GRPO_MODEL env var: {env_model}")
        return env_model

    if sys.stdin.isatty():
        typer.echo("\nSelect model:")
        for i, (name, desc) in enumerate(_MODEL_MENU, 1):
            typer.echo(f"  {i}) {name:<45} [{desc}]")
        raw = typer.prompt("Enter 1-5", default="4").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(_MODEL_MENU):
                chosen = _MODEL_MENU[idx][0]
                typer.echo(f"Selected: {chosen}")
                return chosen
        except ValueError:
            pass
        typer.echo(f"Invalid input; using default: {_DEFAULT_MODEL}")

    return _DEFAULT_MODEL


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
# System prompt
# ---------------------------------------------------------------------------


def _get_system_prompt_template() -> str:
    from grpo_pipeline.transform import SYSTEM_PROMPT_TEMPLATE  # noqa: PLC0415

    return SYSTEM_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# SFT warmup helpers
# ---------------------------------------------------------------------------


def _build_gold_completion(record: dict) -> str:
    """Construct a gold <think>/<verdict> string from ground-truth fields."""
    from grpo_pipeline.rewards import safety_score_to_level, traits_to_group_label  # noqa: PLC0415

    score = record["ground_truth_safety_score"]
    traits = record["ground_truth_traits"]

    safety_level = safety_score_to_level(score)
    integrity = traits_to_group_label(traits, "integrity")
    reasoning = traits_to_group_label(traits, "reasoning")
    empathy = traits_to_group_label(traits, "empathy")

    safety_flags = [k for k in ("manipulation", "deception", "exploitation") if traits.get(k, 0) > 0.3]
    think_lines = ["Evaluating the target agent."]
    if safety_flags:
        think_lines.append(f"Safety concerns detected: {', '.join(safety_flags)}.")
    think_lines.append(f"Safety score: {score:.2f} → {safety_level}.")
    think_lines.append(f"Integrity: {integrity}, Reasoning: {reasoning}, Empathy: {empathy}.")

    verdict = json.dumps(
        {"safety_level": safety_level, "integrity": integrity, "reasoning": reasoning, "empathy": empathy},
        indent=2,
    )
    return f"<think>\n{chr(10).join(think_lines)}\n</think>\n<verdict>\n{verdict}\n</verdict>"


def _run_sft_warmup(
    model,
    tokenizer,
    raw_records: list[dict],
    warmup_examples: int,
    max_seq_length: int,
    system_prompt_template: str,
    safe_apply_template_fn,
) -> None:
    """Run SFT warmup to teach format before GRPO.

    Samples evenly across safety buckets so the format demonstration is not
    skewed toward the majority 'safe' class (which SFT has no class-weight mechanism for).
    """
    import random  # noqa: PLC0415

    from datasets import Dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    from grpo_pipeline.rewards import safety_score_to_level  # noqa: PLC0415

    warmup_max_tokens = max_seq_length // 2
    random.seed(42)

    buckets: dict[str, list] = {"safe": [], "caution": [], "risk": [], "critical": []}
    for rec in raw_records:
        lvl = safety_score_to_level(rec["ground_truth_safety_score"])
        msgs = [
            {"role": "system", "content": system_prompt_template.format(author=rec["author"])},
            *rec["prompt"],
            {"role": "assistant", "content": _build_gold_completion(rec)},
        ]
        text = safe_apply_template_fn(tokenizer, msgs, tokenize=False)
        if len(tokenizer.encode(text)) <= warmup_max_tokens:
            buckets[lvl].append({"text": text})

    per_bucket = max(1, warmup_examples // len(buckets))
    warmup_rows: list[dict] = []
    for lvl, rows in buckets.items():
        sample = random.sample(rows, min(per_bucket, len(rows)))
        warmup_rows.extend(sample)
        typer.echo(f"  warmup {lvl}: {len(rows)} available → {len(sample)} sampled")
    random.shuffle(warmup_rows)
    warmup_rows = warmup_rows[:warmup_examples]

    if not warmup_rows:
        typer.echo("  Warmup: no examples fit within token budget — skipping.")
        return

    warmup_dataset = Dataset.from_list(warmup_rows)
    typer.echo(f"  Warmup dataset: {len(warmup_dataset)} examples (max {warmup_max_tokens} tokens).")

    sft_args = SFTConfig(
        output_dir="_warmup_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        report_to="none",
        max_seq_length=warmup_max_tokens,
        dataset_text_field="text",
    )
    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=warmup_dataset,
        args=sft_args,
    )
    typer.echo("Running format warmup SFT ...")
    sft_trainer.train()
    typer.echo("Warmup complete.")


# ---------------------------------------------------------------------------
# Main training command
# ---------------------------------------------------------------------------


# Sentinel so we can distinguish "user passed 0" from "user didn't pass anything"
_UNSET_INT = -9999


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
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help=(
                "Unsloth model identifier. "
                "If omitted, reads GRPO_MODEL env var or shows an interactive menu. "
                "fp8 (T4): unsloth/Llama-3.2-1B-Instruct. "
                "fp8 (L4): unsloth/Qwen3-8B. "
                "4bit: unsloth/DeepSeek-R1-0528-Qwen3-8B. "
                "16bit: unsloth/Llama-3.2-3B-Instruct (default). "
                "bf16: unsloth/gpt-oss-20b-BF16."
            ),
        ),
    ] = None,
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
        typer.Option(
            "--max-steps",
            help=(
                "Total GRPO training steps. Set to -1 for full epoch. "
                "Defaults vary by model: 500 for fp8/16bit/4bit, 600 for bf16 (GPT-OSS)."
            ),
        ),
    ] = _UNSET_INT,
    num_generations: Annotated[
        int,
        typer.Option(
            "--num-generations",
            help=(
                "GRPO group size (completions per prompt). "
                "Default: 4 for fp8/16bit/4bit, 2 for bf16 (20B model). Reduce if OOM."
            ),
        ),
    ] = _UNSET_INT,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Per-device training batch size. Default: 1 for bf16 (20B), 4 for others.",
        ),
    ] = _UNSET_INT,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="AdamW learning rate. Overrides the per-mode default (5e-6 or 5e-5 for bf16)."),
    ] = 0.0,
    lora_rank: Annotated[
        int,
        typer.Option(
            "--lora-rank",
            help="LoRA rank. Default: 8 for bf16 (GPT-OSS 20B), 32 for all other models.",
        ),
    ] = _UNSET_INT,
    kl_coef: Annotated[
        float,
        typer.Option(
            "--kl-coef",
            help=(
                "KL divergence penalty coefficient (beta in GRPOConfig). "
                "Default 0.1 suppresses the large KL spikes seen with the ~0.04 TRL default."
            ),
        ),
    ] = 0.1,
    warmup_examples: Annotated[
        int,
        typer.Option(
            "--warmup-examples",
            help=(
                "Number of SFT format-warmup examples before GRPO (0 = skip). "
                "Sampled evenly across safety buckets. Recommended: 60."
            ),
        ),
    ] = 0,
    save_steps: Annotated[
        int,
        typer.Option("--save-steps", help="Save a checkpoint every N steps."),
    ] = 100,
    report_to: Annotated[
        str,
        typer.Option("--report-to", help="Logging backend: 'none', 'wandb', or 'tensorboard'."),
    ] = "none",
    push_to_hub: Annotated[
        bool,
        typer.Option("--push-to-hub/--no-push-to-hub", help="Push merged model to Hugging Face Hub after training."),
    ] = False,
    hf_username: Annotated[
        str,
        typer.Option("--hf-username", help="Hugging Face username or organisation for --push-to-hub."),
    ] = "",
) -> None:
    """Train the Moltbook oversight agent with GRPO on evaluation data."""

    # ------------------------------------------------------------------
    # 0. Resolve model name
    # ------------------------------------------------------------------
    resolved_model = _resolve_model(model)

    FastLanguageModel, GRPOConfig, GRPOTrainer, Dataset = _import_training_deps()

    from grpo_pipeline.rewards import (  # noqa: PLC0415
        format_reward,
        group_reward,
        safe_apply_template,
        safety_level_reward,
    )

    # ------------------------------------------------------------------
    # 1. Detect quantisation / inference mode from model name
    # ------------------------------------------------------------------
    _name = resolved_model.lower()
    if "gpt-oss" in _name or "gpt_oss" in _name:
        quant_mode = "bf16"
    elif "deepseek" in _name or "-r1" in _name:
        quant_mode = "4bit"
    elif "llama" in _name and "3b" in _name:
        quant_mode = "16bit"
    else:
        quant_mode = "fp8"

    # ------------------------------------------------------------------
    # 2. Apply per-mode defaults (CLI flags override these)
    # ------------------------------------------------------------------
    effective_lora_rank = lora_rank if lora_rank != _UNSET_INT else (8 if quant_mode == "bf16" else 32)
    effective_batch = batch_size if batch_size != _UNSET_INT else (1 if quant_mode == "bf16" else 4)
    effective_generations = num_generations if num_generations != _UNSET_INT else (2 if quant_mode == "bf16" else 4)
    effective_steps = max_steps if max_steps != _UNSET_INT else (600 if quant_mode == "bf16" else 500)

    lora_alpha = effective_lora_rank if quant_mode == "16bit" else effective_lora_rank * 2
    grad_accum = 4 if quant_mode in ("16bit", "4bit") else 1

    if quant_mode == "bf16":
        effective_lr = learning_rate if learning_rate != 0.0 else 5e-5
        lr_scheduler = "linear"
        weight_decay = 0.001
    else:
        effective_lr = learning_rate if learning_rate != 0.0 else 5e-6
        lr_scheduler = "cosine"
        weight_decay = 0.1

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Moltbook Oversight Agent — GRPO Training")
    typer.echo(f"{'='*60}")
    typer.echo(f"  Model:          {resolved_model}  [{quant_mode}]")
    typer.echo(f"  Train file:     {train_file}")
    typer.echo(f"  Output dir:     {output_dir}")
    typer.echo(f"  lora_rank:      {effective_lora_rank}  alpha={lora_alpha}")
    typer.echo(f"  batch_size:     {effective_batch}  grad_accum={grad_accum}")
    typer.echo(f"  num_gen:        {effective_generations}")
    typer.echo(f"  max_steps:      {effective_steps}")
    typer.echo(f"  lr:             {effective_lr}  scheduler={lr_scheduler}  wd={weight_decay}")
    typer.echo(f"  kl_coef:        {kl_coef}")
    typer.echo(f"  warmup_examples:{warmup_examples}")
    typer.echo(f"  vllm:           {'yes' if quant_mode != 'bf16' else 'no (bf16 HF native)'}")
    typer.echo(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 3. Load model + tokenizer
    # ------------------------------------------------------------------
    typer.echo(f"Loading model: {resolved_model}  [{quant_mode}] ...")

    _load_kwargs: dict = dict(model_name=resolved_model, max_seq_length=max_seq_length)
    if quant_mode == "fp8":
        _load_kwargs.update(load_in_fp8=True, load_in_4bit=False,
                            fast_inference=True, max_lora_rank=effective_lora_rank)
    elif quant_mode == "16bit":
        _load_kwargs.update(load_in_fp8=False, load_in_4bit=False,
                            gpu_memory_utilization=0.9,
                            fast_inference=True, max_lora_rank=effective_lora_rank)
    elif quant_mode == "4bit":
        _load_kwargs.update(load_in_fp8=False, load_in_4bit=True,
                            fast_inference=True, max_lora_rank=effective_lora_rank)
    else:  # "bf16" — HF native generation, no vLLM
        _load_kwargs.update(load_in_fp8=False, load_in_4bit=False)

    loaded_model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)

    # ------------------------------------------------------------------
    # 4. Apply LoRA adapter
    # ------------------------------------------------------------------
    typer.echo(f"Applying LoRA (rank={effective_lora_rank}, alpha={lora_alpha}) ...")
    loaded_model = FastLanguageModel.get_peft_model(
        loaded_model,
        r=effective_lora_rank,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ------------------------------------------------------------------
    # 5. Load and prepare dataset
    # ------------------------------------------------------------------
    typer.echo(f"Loading training data from {train_file} ...")
    raw_records = load_train_dataset(train_file)
    typer.echo(f"  Loaded {len(raw_records)} records.")

    system_prompt_template = _get_system_prompt_template()

    def format_prompts(batch: dict) -> dict:
        return {"prompt": [
            safe_apply_template(
                tokenizer,
                [{"role": "system", "content": system_prompt_template.format(author=a)}] + p,
                tokenize=False,
                add_generation_prompt=True,
            )
            for a, p in zip(batch["author"], batch["prompt"])
        ]}

    dataset = Dataset.from_list(raw_records)
    dataset = dataset.map(format_prompts, batched=True, desc="Formatting prompts")
    typer.echo(f"  Dataset ready: {len(dataset)} records (pre-formatted strings).")

    # ------------------------------------------------------------------
    # 6. Optional SFT format warmup
    # ------------------------------------------------------------------
    if warmup_examples > 0:
        typer.echo(f"\nRunning SFT format warmup ({warmup_examples} examples) ...")
        _run_sft_warmup(
            model=loaded_model,
            tokenizer=tokenizer,
            raw_records=raw_records,
            warmup_examples=warmup_examples,
            max_seq_length=max_seq_length,
            system_prompt_template=system_prompt_template,
            safe_apply_template_fn=safe_apply_template,
        )

    # ------------------------------------------------------------------
    # 7. Configure GRPO training
    # ------------------------------------------------------------------
    import gc  # noqa: PLC0415
    import torch  # noqa: PLC0415

    gc.collect()
    torch.cuda.empty_cache()

    max_prompt_length = max_seq_length - max_completion_length

    _grpo_extra: dict = {}
    if quant_mode != "bf16":
        from vllm import SamplingParams  # noqa: PLC0415
        _grpo_extra["vllm_sampling_params"] = SamplingParams(
            min_p=0.1, top_p=1.0, top_k=-1, seed=42,
            stop=[tokenizer.eos_token],
            include_stop_str_in_output=True,
        )

    training_args = GRPOConfig(
        **_grpo_extra,
        temperature=1.0,
        learning_rate=effective_lr,
        beta=kl_coef,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type=lr_scheduler,
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=effective_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=effective_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=effective_steps if effective_steps > 0 else None,
        num_train_epochs=1 if effective_steps <= 0 else None,
        max_grad_norm=1.0,
        save_steps=save_steps,
        output_dir=str(output_dir),
        report_to=report_to,
    )

    # ------------------------------------------------------------------
    # 8. Create trainer and train
    # ------------------------------------------------------------------
    typer.echo("Creating GRPOTrainer ...")
    trainer = GRPOTrainer(
        model=loaded_model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            safety_level_reward,
            group_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    typer.echo("Starting GRPO training ...")
    typer.echo("  Tip: watch for the 'reward' column in logs to confirm reward functions fire.")
    trainer.train()

    # ------------------------------------------------------------------
    # 9. Save LoRA adapter and merged model
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir.parent / (output_dir.name + "-merged")

    typer.echo(f"\nSaving LoRA adapter → {output_dir}")
    loaded_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    save_method = "mxfp4" if quant_mode == "bf16" else "merged_16bit"
    typer.echo(f"Saving merged model [{save_method}] → {merged_dir}")
    loaded_model.save_pretrained_merged(str(merged_dir), tokenizer, save_method=save_method)

    # ------------------------------------------------------------------
    # 10. Optional HuggingFace Hub push
    # ------------------------------------------------------------------
    if push_to_hub:
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            typer.echo("WARNING: HF_TOKEN env var is not set — push may fail for private repos.")
        username = hf_username or os.environ.get("HF_USERNAME", "")
        if not username:
            typer.echo("ERROR: --hf-username or HF_USERNAME env var required for --push-to-hub.", err=True)
        else:
            stub = "moltbook-oversight-" + resolved_model.split("/")[-1].lower()
            repo_id = f"{username}/{stub}"
            typer.echo(f"Pushing merged model to HuggingFace Hub: {repo_id} ...")
            loaded_model.push_to_hub_merged(repo_id, tokenizer, save_method=save_method, token=hf_token)
            typer.echo(f"  Pushed → https://huggingface.co/{repo_id}")

    typer.echo("\nDone.")
    typer.echo(f"  LoRA adapter:  {output_dir}")
    typer.echo(f"  Merged model:  {merged_dir}")
    typer.echo("\nNext — evaluate:")
    typer.echo(
        f"  python -m grpo_pipeline.baseline"
        f" --test-file ../transformed/test.jsonl"
        f" --lora-adapter {output_dir}"
    )


if __name__ == "__main__":
    app()
