"""Conversation simulation environment for live GRPO training.

Provides a streaming data source that replays historical conversation threads
turn-by-turn, mirroring the real-life deployment scenario where the oversight
agent observes messages as they arrive rather than receiving a pre-assembled
static batch.

Three public components
-----------------------

ParticipantBot / ReplayBot
    Abstract base + concrete replay implementation. A bot represents one author
    in a thread and emits their messages one at a time in chronological order.
    The interface is designed for future LLM-powered subclasses.

ConversationEnvironment
    Manages one thread. Holds all messages sorted by created_at; each step()
    pops the next message, updates the running context, and exposes the current
    state. run_to_records() replays the full thread and yields GRPORecord objects
    through an optional minimum-context gate.

SimulatedDataset
    Wraps ConversationEnvironment in a HuggingFace IterableDataset generator
    suitable for direct use as train_dataset in GRPOTrainer. The generator
    loops indefinitely, shuffling threads each epoch to vary the order.

Usage in training
-----------------
::

    from grpo_pipeline.simulation import SimulatedDataset

    dataset = SimulatedDataset.create(
        raw_data_dir='../raw-data',
        min_context_turns=1,   # skip turn 0 — oversight agent always has prior context
    )
    # Pass directly to GRPOTrainer; same column schema as transformed/train.jsonl:
    trainer = GRPOTrainer(..., train_dataset=dataset)

Batch files (batch_all/shady/suspicious/sample*.jsonl) are skipped by the
simulation — they have no thread context to replay and remain accessible via
the static transform.py pipeline.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from grpo_pipeline.models import ConversationRecord, GRPORecord
from grpo_pipeline.transform import (
    CONVERSATION_FILE_RE,
    build_grpo_record,
    format_context_message,
    parse_conversation_records,
)


# ---------------------------------------------------------------------------
# ParticipantBot — abstract base + replay implementation
# ---------------------------------------------------------------------------


class ParticipantBot(ABC):
    """Abstract participant in a simulated conversation thread.

    Subclass this to create LLM-powered bots that generate utterances
    on-the-fly instead of replaying historical messages.
    """

    def __init__(self, author: str) -> None:
        self.author = author

    @abstractmethod
    def next_message(self) -> ConversationRecord | None:
        """Return the next message from this participant, or None if exhausted."""

    @abstractmethod
    def is_exhausted(self) -> bool:
        """Return True when there are no more messages to emit."""


class ReplayBot(ParticipantBot):
    """Replays a fixed list of historical ConversationRecord messages in order.

    This is the primary bot type for training on existing labeled data. The
    messages should be pre-sorted by created_at before being passed in.
    """

    def __init__(self, author: str, messages: list[ConversationRecord]) -> None:
        super().__init__(author)
        self._messages = messages
        self._cursor = 0

    def next_message(self) -> ConversationRecord | None:
        if self._cursor >= len(self._messages):
            return None
        msg = self._messages[self._cursor]
        self._cursor += 1
        return msg

    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._messages)


# ---------------------------------------------------------------------------
# ConversationEnvironment — manages one thread's simulation
# ---------------------------------------------------------------------------


class ConversationEnvironment:
    """Simulates a single conversation thread, advancing one turn at a time.

    Messages from all participants are pre-sorted by created_at so the
    environment replays them in the original chronological order. Only
    messages that carry a usable evaluation (with_context is not None)
    participate in the simulation.

    The environment is single-use: create a new instance for each replay.
    """

    def __init__(self, thread_id: str, messages: list[ConversationRecord]) -> None:
        self.thread_id = thread_id
        self._pending: list[ConversationRecord] = [
            m for m in sorted(messages, key=lambda r: r.created_at)
            if m.with_context is not None
        ]
        self._emitted: list[ConversationRecord] = []

    def step(self) -> tuple[ConversationRecord, list[ConversationRecord]] | None:
        """Advance the conversation by one turn.

        Returns (new_message, context_before_this_message), or None when all
        messages have been emitted.
        """
        if not self._pending:
            return None
        msg = self._pending.pop(0)
        context = list(self._emitted)
        self._emitted.append(msg)
        return msg, context

    def run_to_records(
        self,
        source_file: str = "",
        min_context_turns: int = 0,
    ) -> list[GRPORecord]:
        """Replay the full thread and return a GRPORecord per emitted turn.

        Args:
            source_file: Provenance label stored in the GRPORecord.
            min_context_turns: Skip turns where turn_index is less than this
                value (i.e., where the oversight agent has seen fewer than
                min_context_turns prior messages). Original turn_index /
                total_turns values are preserved so length_scale continues to
                reflect the message's position in the full thread.

        Returns:
            List of GRPORecord objects in chronological order, with early
            turns filtered out when min_context_turns > 0.
        """
        all_turns: list[tuple[ConversationRecord, list[ConversationRecord]]] = []
        result = self.step()
        while result is not None:
            all_turns.append(result)
            result = self.step()

        total_turns = len(all_turns)
        if total_turns == 0:
            return []

        records: list[GRPORecord] = []
        for turn_index, (msg, context) in enumerate(all_turns):
            if turn_index < min_context_turns:
                continue
            context_messages = [
                format_context_message(m.author, m.content, m.message_type)
                for m in context
            ]
            records.append(
                build_grpo_record(
                    author=msg.author,
                    target_content=msg.content,
                    context_messages=context_messages,
                    evaluation=msg.with_context,  # type: ignore[arg-type]
                    thread_id=self.thread_id,
                    source_file=source_file,
                    turn_index=turn_index,
                    total_turns=total_turns,
                    message_type=msg.message_type,
                )
            )
        return records


# ---------------------------------------------------------------------------
# Thread loading helpers
# ---------------------------------------------------------------------------


def _load_conversation_threads(path: Path) -> dict[str, list[ConversationRecord]]:
    """Load one conversation JSONL file and group records by base thread_id."""
    threads: dict[str, list[ConversationRecord]] = defaultdict(list)
    for rec in parse_conversation_records(path):
        base_thread = rec.thread_id.split("::")[0]
        threads[base_thread].append(rec)
    return dict(threads)


def load_all_threads(
    raw_data_dir: str | Path,
) -> list[tuple[str, str, list[ConversationRecord]]]:
    """Scan raw_data_dir for conversation files and return all threads.

    Returns a list of (thread_id, source_filename, messages) tuples. Non-
    conversation batch files are skipped — they have no thread context to
    simulate and are handled by the static transform.py pipeline.
    """
    input_dir = Path(raw_data_dir)
    threads: list[tuple[str, str, list[ConversationRecord]]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        if CONVERSATION_FILE_RE.search(path.name):
            for thread_id, messages in _load_conversation_threads(path).items():
                threads.append((thread_id, path.name, messages))
    return threads


# ---------------------------------------------------------------------------
# SimulatedDataset — HuggingFace IterableDataset wrapper
# ---------------------------------------------------------------------------


class SimulatedDataset:
    """Infinite streaming dataset that replays conversation threads via simulation.

    Each epoch the threads are shuffled with a derived seed so the training
    data distribution varies slightly every pass. Because the generator never
    terminates, GRPOTrainer must have max_steps set (already the case in all
    current training configurations).

    Example
    -------
    ::

        from grpo_pipeline.simulation import SimulatedDataset

        dataset = SimulatedDataset.create('../raw-data', min_context_turns=1)
        # Same column schema as transformed/train.jsonl — no other changes needed
        trainer = GRPOTrainer(..., train_dataset=dataset)
    """

    @staticmethod
    def generate(
        raw_data_dir: str,
        min_context_turns: int = 0,
        seed: int = 42,
    ):
        """Infinite generator: shuffles threads each epoch, yields GRPORecord dicts.

        This is the generator function passed to datasets.IterableDataset.from_generator().
        It yields one dict per training sample; the schema matches GRPORecord.model_dump().
        """
        threads = load_all_threads(raw_data_dir)
        if not threads:
            raise ValueError(
                f"No conversation files found in {raw_data_dir!r}. "
                "Expected files matching 'batch_conversations*.jsonl'."
            )

        rng = random.Random(seed)
        while True:
            epoch_threads = list(threads)
            rng.shuffle(epoch_threads)
            for thread_id, source_file, messages in epoch_threads:
                env = ConversationEnvironment(thread_id, messages)
                for record in env.run_to_records(
                    source_file=source_file,
                    min_context_turns=min_context_turns,
                ):
                    yield record.model_dump()

    @staticmethod
    def collect_one_epoch(
        raw_data_dir: str | Path,
        min_context_turns: int = 0,
    ) -> list[dict]:
        """Run the simulation for exactly one epoch and return all records as dicts.

        Useful for SFT warmup and offline inspection when live simulation is
        the primary training data source. Threads are returned in sorted
        (deterministic) order — no shuffling, so the output is reproducible.
        """
        threads = load_all_threads(raw_data_dir)
        records: list[dict] = []
        for thread_id, source_file, messages in threads:
            env = ConversationEnvironment(thread_id, messages)
            for record in env.run_to_records(
                source_file=source_file,
                min_context_turns=min_context_turns,
            ):
                records.append(record.model_dump())
        return records

    @classmethod
    def create(
        cls,
        raw_data_dir: str | Path,
        min_context_turns: int = 0,
        seed: int = 42,
    ):
        """Create a HuggingFace IterableDataset backed by the conversation simulation.

        Args:
            raw_data_dir: Directory containing raw batch_conversations*.jsonl files.
            min_context_turns: Number of preceding turns required before emitting
                a training sample. Use 1 to ensure the oversight agent always has
                at least one prior message as context. Turn indices and length_scale
                values are preserved from the original thread, so reward weighting
                remains consistent regardless of this filter.
            seed: Random seed for per-epoch thread-order shuffling.

        Returns:
            datasets.IterableDataset with the same column schema as GRPORecord
            (prompt, ground_truth_traits, ground_truth_safety_score, …).
        """
        from datasets import IterableDataset  # noqa: PLC0415

        return IterableDataset.from_generator(
            cls.generate,
            gen_kwargs={
                "raw_data_dir": str(raw_data_dir),
                "min_context_turns": min_context_turns,
                "seed": seed,
            },
        )
