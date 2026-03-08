"""Thread-level train/test split for the GRPO-ready dataset.

Split is performed at the THREAD level, not the record level. Because the
transformation duplicates a single conversation thread to produce one record
per participant (the credit-assignment pattern), all perspectives derived from
the same thread must land entirely in train or test. Record-level splitting
would cause data leakage: the model would see Thread X during training and
then be tested on Thread X again from a different participant's viewpoint.

Usage:
    uv run python -m grpo_pipeline.split \\
        --input ./output/dataset.jsonl \\
        --output ./output \\
        --test-ratio 0.2 \\
        --seed 42
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import typer

from grpo_pipeline.models import GRPORecord

app = typer.Typer()


def load_grpo_records(path: Path) -> list[GRPORecord]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(GRPORecord.model_validate_json(line))
    return records


def write_jsonl(records: list[GRPORecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def split_by_thread(
    records: list[GRPORecord],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[GRPORecord], list[GRPORecord]]:
    """Split records into train/test by grouping all records sharing a thread_id.

    Guarantees: no thread_id appears in both splits.
    """
    # Group record indices by thread_id
    thread_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        thread_to_indices[rec.thread_id].append(idx)

    thread_ids = list(thread_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(thread_ids)

    n_test_threads = max(1, round(len(thread_ids) * test_ratio))
    test_thread_ids = set(thread_ids[:n_test_threads])
    train_thread_ids = set(thread_ids[n_test_threads:])

    train_records = [records[i] for tid in train_thread_ids for i in thread_to_indices[tid]]
    test_records = [records[i] for tid in test_thread_ids for i in thread_to_indices[tid]]

    return train_records, test_records


def print_split_stats(
    train: list[GRPORecord],
    test: list[GRPORecord],
    label: str = "",
) -> None:
    total = len(train) + len(test)
    header = f"\n{'=' * 50}"
    if label:
        header += f"\n{label}"
    typer.echo(header)
    typer.echo(f"Total records:  {total}")
    typer.echo(f"Train records:  {len(train)} ({len(train) / total * 100:.1f}%)")
    typer.echo(f"Test records:   {len(test)} ({len(test) / total * 100:.1f}%)")

    train_threads = {r.thread_id for r in train}
    test_threads = {r.thread_id for r in test}
    typer.echo(f"Train threads:  {len(train_threads)}")
    typer.echo(f"Test threads:   {len(test_threads)}")

    overlap = train_threads & test_threads
    typer.echo(f"Thread overlap: {len(overlap)} (should be 0)")

    typer.echo("\nAlignment status distribution:")
    train_counts = Counter(r.ground_truth_alignment for r in train)
    test_counts = Counter(r.ground_truth_alignment for r in test)
    all_statuses = sorted(set(train_counts) | set(test_counts))
    typer.echo(f"  {'Status':<15} {'Train':>8} {'Test':>8}")
    typer.echo(f"  {'-' * 35}")
    for status in all_statuses:
        typer.echo(f"  {status:<15} {train_counts.get(status, 0):>8} {test_counts.get(status, 0):>8}")

    typer.echo("\nSafety score distribution (mean):")
    if train:
        train_safety = sum(r.ground_truth_safety_score for r in train) / len(train)
        typer.echo(f"  Train: {train_safety:.3f}")
    if test:
        test_safety = sum(r.ground_truth_safety_score for r in test) / len(test)
        typer.echo(f"  Test:  {test_safety:.3f}")


@app.command()
def main(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to dataset.jsonl produced by transform."),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Directory to write train.jsonl and test.jsonl."),
    test_ratio: float = typer.Option(0.2, "--test-ratio", help="Fraction of threads to use for test set."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible splits."),
) -> None:
    """Split a GRPO dataset into train and test sets at the thread level."""
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading {input_path} ...")
    records = load_grpo_records(input_path)
    typer.echo(f"Loaded {len(records)} records.")

    train_records, test_records = split_by_thread(records, test_ratio=test_ratio, seed=seed)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)

    print_split_stats(train_records, test_records, label="Split complete")
    typer.echo(f"\nWrote train → {train_path}")
    typer.echo(f"Wrote test  → {test_path}")


if __name__ == "__main__":
    app()
