"""Headless baseline evaluation script for the Moltbook oversight agent.

Runs batch inference on test.jsonl and computes metrics comparing the model's
verdicts against ground truth labels.  Works for both:

  - The base model (no adapter) — establishes a pre-RL reference point
  - A trained LoRA adapter    — evaluates post-GRPO quality

Metrics computed
----------------
  Format success rate       — fraction of responses with valid <think>/<verdict> JSON
  Safety-level accuracy     — overall + per-class (safe/caution/risk/critical)
  Per-class precision/recall/F1 for safety_level
  Group label accuracy      — per-field accuracy for integrity / reasoning / empathy

Results are printed to stdout and saved as JSON.

Usage
-----
    # Baseline (no adapter):
    python -m grpo_pipeline.baseline \\
        --test-file ../transformed/test.jsonl \\
        --output ../baseline_results.json

    # After training:
    python -m grpo_pipeline.baseline \\
        --test-file ../transformed/test.jsonl \\
        --lora-adapter ../lora-adapter \\
        --output ../post_grpo_results.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(pretty_exceptions_enable=False)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_inference_deps():
    try:
        from unsloth import FastLanguageModel

        return FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is not installed. Run:\n"
            "  pip install unsloth vllm\n"
            "  uv pip install -e '.[train]'"
        ) from exc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(results: list[dict]) -> dict:
    """Compute all evaluation metrics from a list of per-record result dicts.

    Each result dict must have:
        parsed_verdict   dict | None
        gt_safety_level  str   — one of safe/caution/risk/critical
        gt_integrity     str   — one of strong/good/weak/poor
        gt_reasoning     str   — one of strong/good/weak/poor
        gt_empathy       str   — one of strong/good/weak/poor
    """
    n = len(results)
    if n == 0:
        return {}

    n_valid_format = sum(1 for r in results if r["parsed_verdict"] is not None)

    safety_classes = ("safe", "caution", "risk", "critical")
    per_class_tp: dict[str, int] = defaultdict(int)
    per_class_fp: dict[str, int] = defaultdict(int)
    per_class_fn: dict[str, int] = defaultdict(int)
    n_correct_safety = 0

    group_correct: dict[str, int] = {"integrity": 0, "reasoning": 0, "empathy": 0}
    n_parseable = 0

    for r in results:
        gt_level = r["gt_safety_level"]
        pred_level = r["parsed_verdict"].get("safety_level") if r["parsed_verdict"] else None

        if pred_level == gt_level:
            n_correct_safety += 1
            per_class_tp[gt_level] += 1
        else:
            if pred_level is not None:
                per_class_fp[pred_level] += 1
            per_class_fn[gt_level] += 1

        if r["parsed_verdict"] is not None:
            n_parseable += 1
            for group in ("integrity", "reasoning", "empathy"):
                gt_key = f"gt_{group}"
                if r["parsed_verdict"].get(group) == r.get(gt_key):
                    group_correct[group] += 1

    per_class_metrics: dict[str, dict] = {}
    for cls in safety_classes:
        tp = per_class_tp[cls]
        fp = per_class_fp[cls]
        fn = per_class_fn[cls]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    group_accuracy = {
        group: group_correct[group] / n_parseable if n_parseable > 0 else float("nan")
        for group in ("integrity", "reasoning", "empathy")
    }

    return {
        "n_total": n,
        "n_valid_format": n_valid_format,
        "format_success_rate": n_valid_format / n,
        "n_parseable": n_parseable,
        "safety_level_accuracy": n_correct_safety / n,
        "per_class_metrics": per_class_metrics,
        "group_accuracy": group_accuracy,
    }


def print_metrics(metrics: dict, label: str = "Evaluation Results") -> None:
    """Print a formatted metrics table to stdout."""
    sep = "=" * 60
    typer.echo(f"\n{sep}")
    typer.echo(f"  {label}")
    typer.echo(sep)
    typer.echo(f"  Total records:          {metrics['n_total']}")
    typer.echo(f"  Valid format:           {metrics['n_valid_format']} ({metrics['format_success_rate']:.1%})")
    typer.echo(f"  Safety-level accuracy:  {metrics['safety_level_accuracy']:.1%}")
    typer.echo()
    typer.echo("  Safety-level per-class breakdown:")
    typer.echo(f"    {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    typer.echo(f"    {'-' * 54}")
    for cls, cm in metrics["per_class_metrics"].items():
        typer.echo(
            f"    {cls:<10} {cm['precision']:>10.3f} {cm['recall']:>10.3f} {cm['f1']:>8.3f}"
            f" {cm['tp']:>5} {cm['fp']:>5} {cm['fn']:>5}"
        )
    typer.echo()
    typer.echo("  Group label accuracy (higher is better):")
    for group, acc in metrics["group_accuracy"].items():
        typer.echo(f"    {group:<12}  {acc:.1%}")
    typer.echo(sep)


# ---------------------------------------------------------------------------
# Main evaluation command
# ---------------------------------------------------------------------------


@app.command()
def main(
    test_file: Annotated[
        Path,
        typer.Option("--test-file", "-i", help="Path to test.jsonl produced by split.py."),
    ] = Path("../transformed/test.jsonl"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to write JSON results file."),
    ] = Path("../baseline_results.json"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Unsloth model identifier."),
    ] = "unsloth/Llama-3.2-1B-Instruct",
    lora_adapter: Annotated[
        Path | None,
        typer.Option("--lora-adapter", help="Path to a saved LoRA adapter directory (optional)."),
    ] = None,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", help="Maximum tokens to generate per sample."),
    ] = 768,
    max_seq_length: Annotated[
        int,
        typer.Option("--max-seq-length", help="Maximum total sequence length passed to the model."),
    ] = 2048,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Inference batch size. Reduce if OOM."),
    ] = 4,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Evaluate only the first N records (0 = all)."),
    ] = 0,
    ) -> None:
    """Evaluate an oversight agent model on the test set and report metrics."""
    import torch  # noqa: PLC0415

    from grpo_pipeline.rewards import extract_verdict, safety_score_to_level, traits_to_group_label  # noqa: PLC0415
    from grpo_pipeline.transform import SYSTEM_PROMPT_TEMPLATE  # noqa: PLC0415

    FastLanguageModel = _import_inference_deps()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    typer.echo(f"Loading model: {model}")
    loaded_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )

    if lora_adapter is not None:
        typer.echo(f"Loading LoRA adapter from: {lora_adapter}")
        from peft import PeftModel  # noqa: PLC0415

        loaded_model = PeftModel.from_pretrained(loaded_model, str(lora_adapter))

    FastLanguageModel.for_inference(loaded_model)
    loaded_model.eval()

    # ------------------------------------------------------------------
    # 2. Load test records
    # ------------------------------------------------------------------
    typer.echo(f"Loading test records from {test_file} ...")
    records = load_test_records(test_file)
    if limit > 0:
        records = records[:limit]
    typer.echo(f"  Evaluating {len(records)} records.")

    # ------------------------------------------------------------------
    # 3. Run inference in batches
    # ------------------------------------------------------------------
    results: list[dict] = []
    n_batches = (len(records) + batch_size - 1) // batch_size

    with typer.progressbar(range(n_batches), label="Running inference") as progress:
        for batch_idx in progress:
            batch = records[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # Build full prompts (system + user)
            prompt_texts = []
            for rec in batch:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(author=rec["author"])},
                    *rec["prompt"],
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_texts.append(text)

            # Tokenise
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length - max_new_tokens,
            ).to(loaded_model.device)

            with torch.no_grad():
                output_ids = loaded_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            for rec, out in zip(batch, output_ids):
                generated_text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
                parsed = extract_verdict(generated_text)
                gt_score = rec.get("ground_truth_safety_score", 0.0)
                gt_traits = rec.get("ground_truth_traits", {})
                results.append(
                    {
                        "evaluation_id": rec.get("evaluation_id", ""),
                        "thread_id": rec.get("thread_id", ""),
                        "author": rec.get("author", ""),
                        "gt_safety_level": safety_score_to_level(gt_score),
                        "gt_integrity": traits_to_group_label(gt_traits, "integrity"),
                        "gt_reasoning": traits_to_group_label(gt_traits, "reasoning"),
                        "gt_empathy": traits_to_group_label(gt_traits, "empathy"),
                        "gt_traits": gt_traits,
                        "gt_safety_score": gt_score,
                        "length_scale": rec.get("length_scale", 1.0),
                        "generated_text": generated_text,
                        "parsed_verdict": parsed,
                        "predicted_safety_level": parsed.get("safety_level") if parsed else None,
                    }
                )

    # ------------------------------------------------------------------
    # 4. Compute and display metrics
    # ------------------------------------------------------------------
    label = f"Post-GRPO ({lora_adapter.name})" if lora_adapter else "Baseline (no adapter)"
    metrics = compute_metrics(results)
    print_metrics(metrics, label=label)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": label,
        "model": model,
        "lora_adapter": str(lora_adapter) if lora_adapter else None,
        "n_records": len(results),
        "metrics": metrics,
        "per_record": [
            {
                "evaluation_id": r["evaluation_id"],
                "gt_safety_level": r["gt_safety_level"],
                "predicted_safety_level": r["predicted_safety_level"],
                "safety_level_correct": r["predicted_safety_level"] == r["gt_safety_level"],
                "length_scale": r["length_scale"],
            }
            for r in results
        ],
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    typer.echo(f"\nFull results saved to: {output}")


if __name__ == "__main__":
    app()
