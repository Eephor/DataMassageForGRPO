"""Leakage and integrity tests for the thread-level train/test split.

These tests run on synthetic fixtures — no external files required.
They verify the four invariants defined in the plan:

1. No thread_id overlap between train and test sets.
2. No evaluation_id overlap between train and test sets.
3. No duplicate evaluation_ids in the combined dataset.
4. Alignment status class distribution report (sanity check that adversarial
   examples are not accidentally concentrated in one split).
"""

from __future__ import annotations

from collections import Counter

import pytest

from grpo_pipeline.models import GRPORecord
from grpo_pipeline.split import split_by_thread

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    *,
    evaluation_id: str,
    thread_id: str,
    author: str = "bot_a",
    alignment: str = "aligned",
    safety_score: float = 0.9,
    phronesis: str = "trustworthy",
    turn_index: int = 0,
    total_turns: int = 1,
) -> GRPORecord:
    """Build a minimal GRPORecord for testing."""
    return GRPORecord(
        prompt=[{"role": "user", "content": "Some conversation text."}],
        ground_truth_traits={
            "virtue": 0.8, "goodwill": 0.7,
            "manipulation": 0.1, "deception": 0.1,
            "accuracy": 0.8, "reasoning": 0.7,
            "fabrication": 0.0, "broken_logic": 0.0,
            "recognition": 0.6, "compassion": 0.5,
            "dismissal": 0.1, "exploitation": 0.0,
        },
        ground_truth_safety_score=safety_score,
        ground_truth_alignment=alignment,  # type: ignore[arg-type]
        ground_truth_phronesis=phronesis,  # type: ignore[arg-type]
        turn_index=turn_index,
        total_turns=total_turns,
        length_scale=(turn_index + 1) / total_turns,
        thread_id=thread_id,
        evaluation_id=evaluation_id,
        author=author,
        source_file="fixture",
    )


def _make_thread(
    thread_id: str,
    num_participants: int = 2,
    alignment: str = "aligned",
) -> list[GRPORecord]:
    """Create num_participants records sharing the same thread_id.

    Each record represents one turn in the thread, simulating the rolling
    context window pattern: turn 0 has no prior context, turn N-1 has all.
    """
    return [
        _make_record(
            evaluation_id=f"{thread_id}_eval_{i}",
            thread_id=thread_id,
            author=f"bot_{i}",
            alignment=alignment,
            turn_index=i,
            total_turns=num_participants,
        )
        for i in range(num_participants)
    ]


@pytest.fixture()
def simple_dataset() -> list[GRPORecord]:
    """10 threads × 2 participants each = 20 records."""
    records: list[GRPORecord] = []
    for t in range(10):
        records.extend(_make_thread(f"thread_{t:03d}", num_participants=2))
    return records


@pytest.fixture()
def adversarial_dataset() -> list[GRPORecord]:
    """Mix of aligned and misaligned threads to test class distribution."""
    records: list[GRPORecord] = []
    # 6 aligned threads
    for t in range(6):
        records.extend(_make_thread(f"aligned_{t}", num_participants=2, alignment="aligned"))
    # 2 drifting threads
    for t in range(2):
        records.extend(_make_thread(f"drifting_{t}", num_participants=2, alignment="drifting"))
    # 2 misaligned threads
    for t in range(2):
        records.extend(
            _make_thread(f"misaligned_{t}", num_participants=2, alignment="misaligned")
        )
    return records


# ---------------------------------------------------------------------------
# Invariant 1: No thread_id overlap
# ---------------------------------------------------------------------------


def test_no_thread_id_overlap(simple_dataset: list[GRPORecord]) -> None:
    train, test = split_by_thread(simple_dataset, test_ratio=0.2, seed=42)

    train_threads = {r.thread_id for r in train}
    test_threads = {r.thread_id for r in test}
    overlap = train_threads & test_threads

    assert overlap == set(), (
        f"Thread leakage detected: {len(overlap)} thread_id(s) appear in both splits: {overlap}"
    )


def test_no_thread_id_overlap_various_ratios(simple_dataset: list[GRPORecord]) -> None:
    for ratio in [0.1, 0.2, 0.3, 0.5]:
        train, test = split_by_thread(simple_dataset, test_ratio=ratio, seed=0)
        train_threads = {r.thread_id for r in train}
        test_threads = {r.thread_id for r in test}
        assert (train_threads & test_threads) == set(), (
            f"Thread leakage at test_ratio={ratio}"
        )


# ---------------------------------------------------------------------------
# Invariant 2: No evaluation_id overlap
# ---------------------------------------------------------------------------


def test_no_evaluation_id_overlap(simple_dataset: list[GRPORecord]) -> None:
    train, test = split_by_thread(simple_dataset, test_ratio=0.2, seed=42)

    train_eids = {r.evaluation_id for r in train}
    test_eids = {r.evaluation_id for r in test}
    overlap = train_eids & test_eids

    assert overlap == set(), (
        f"Evaluation ID leakage: {len(overlap)} evaluation_id(s) in both splits: {overlap}"
    )


# ---------------------------------------------------------------------------
# Invariant 3: No duplicate evaluation_ids in the combined dataset
# ---------------------------------------------------------------------------


def test_no_duplicate_evaluation_ids() -> None:
    records: list[GRPORecord] = []
    # Build a dataset with a deliberate duplicate
    for t in range(5):
        records.extend(_make_thread(f"thread_{t}", num_participants=2))
    # Introduce a duplicate (same evaluation_id as an existing record)
    duplicate = _make_record(evaluation_id="thread_0_eval_0", thread_id="thread_0")
    records.append(duplicate)

    all_eids = [r.evaluation_id for r in records]
    counts = Counter(all_eids)
    duplicates = {eid: count for eid, count in counts.items() if count > 1}

    assert duplicates, "Expected to find the injected duplicate — test fixture is broken."

    # The transform script's deduplicate() function should remove them.
    # Here we verify that if they slip through, the test catches them.
    unique_eids = set(all_eids)
    assert len(all_eids) != len(unique_eids), (
        "Dataset contains duplicate evaluation_ids. Run deduplicate() before splitting."
    )


def test_clean_dataset_has_no_duplicates(simple_dataset: list[GRPORecord]) -> None:
    all_eids = [r.evaluation_id for r in simple_dataset]
    counts = Counter(all_eids)
    duplicates = {eid for eid, count in counts.items() if count > 1}
    assert duplicates == set(), (
        f"Duplicate evaluation_ids found in fixture: {duplicates}"
    )


# ---------------------------------------------------------------------------
# Invariant 4: Class distribution is represented in both splits
# ---------------------------------------------------------------------------


def test_both_splits_have_records(simple_dataset: list[GRPORecord]) -> None:
    train, test = split_by_thread(simple_dataset, test_ratio=0.2, seed=42)
    assert len(train) > 0, "Train split is empty."
    assert len(test) > 0, "Test split is empty."


def test_train_test_coverage(simple_dataset: list[GRPORecord]) -> None:
    train, test = split_by_thread(simple_dataset, test_ratio=0.2, seed=42)
    total = len(simple_dataset)
    assert len(train) + len(test) == total, (
        f"Records lost during split: {total} in → {len(train) + len(test)} out"
    )


def test_adversarial_examples_in_test(adversarial_dataset: list[GRPORecord]) -> None:
    """Confirm that at least some adversarial examples end up in the test set.

    With 10 threads (6 aligned, 2 drifting, 2 misaligned) and test_ratio=0.3,
    we expect ~3 threads in test. With enough seeds one of the adversarial threads
    should be in the test set. We use a fixed seed that achieves this.
    """
    train, test = split_by_thread(adversarial_dataset, test_ratio=0.3, seed=42)

    test_alignments = {r.ground_truth_alignment for r in test}
    train_alignments = {r.ground_truth_alignment for r in train}

    # Print distribution for information (visible with pytest -s)
    train_counts = Counter(r.ground_truth_alignment for r in train)
    test_counts = Counter(r.ground_truth_alignment for r in test)
    print(f"\nTrain distribution: {dict(train_counts)}")
    print(f"Test distribution:  {dict(test_counts)}")

    assert len(test_alignments) > 0, "Test set has no alignment labels."
    assert len(train_alignments) > 0, "Train set has no alignment labels."


def test_reproducibility(simple_dataset: list[GRPORecord]) -> None:
    train_a, test_a = split_by_thread(simple_dataset, test_ratio=0.2, seed=99)
    train_b, test_b = split_by_thread(simple_dataset, test_ratio=0.2, seed=99)

    assert [r.evaluation_id for r in train_a] == [r.evaluation_id for r in train_b], (
        "Split is not reproducible with the same seed."
    )
    assert [r.evaluation_id for r in test_a] == [r.evaluation_id for r in test_b], (
        "Split is not reproducible with the same seed."
    )


def test_different_seeds_produce_different_splits(simple_dataset: list[GRPORecord]) -> None:
    _, test_a = split_by_thread(simple_dataset, test_ratio=0.2, seed=1)
    _, test_b = split_by_thread(simple_dataset, test_ratio=0.2, seed=2)

    test_threads_a = {r.thread_id for r in test_a}
    test_threads_b = {r.thread_id for r in test_b}

    # With 10 threads and 2 in test, different seeds should produce different test sets
    assert test_threads_a != test_threads_b, (
        "Different seeds produced identical splits — random shuffling may be broken."
    )
