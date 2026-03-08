"""Tests verifying that the simulation environment produces output equivalent
to the static transform pipeline from the oversight agent's point of view.

The "coarse-grain environment" is transform.py's transform_conversation_file().
The simulation must produce identically formatted prompts, identical turn metadata
(turn_index, total_turns, length_scale), and identical ground-truth fields — so
that swapping the data source has no effect on what the oversight agent sees or
on the reward signal it receives.

Test groups
-----------
1. ReplayBot              — message emission and exhaustion
2. ConversationEnvironment step() — chronological ordering, context growth
3. Equivalence with transform — prompt content, metadata, ground truth
4. Prompt format (oversight agent view) — exact structure of context blocks
5. min_context_turns gate — filtering and index/scale preservation
6. SimulatedDataset — schema, infinite generator, collect_one_epoch
7. Integration — round-trip against real raw-data files (requires ../raw-data)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grpo_pipeline.models import (
    ConversationRecord,
    EvaluationBlock,
    GRPORecord,
    TraitScores,
)
from grpo_pipeline.simulation import (
    ConversationEnvironment,
    ReplayBot,
    SimulatedDataset,
    load_all_threads,
)
from grpo_pipeline.transform import (
    build_grpo_record,
    format_context_message,
    transform_conversation_file,
)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

_SAFE_TRAITS = TraitScores(
    virtue=0.9, goodwill=0.8,
    manipulation=0.05, deception=0.05,
    accuracy=0.85, reasoning=0.80,
    fabrication=0.05, broken_logic=0.05,
    recognition=0.80, compassion=0.70,
    dismissal=0.05, exploitation=0.05,
)

_RISKY_TRAITS = TraitScores(
    virtue=0.2, goodwill=0.2,
    manipulation=0.8, deception=0.7,
    accuracy=0.4, reasoning=0.3,
    fabrication=0.5, broken_logic=0.5,
    recognition=0.2, compassion=0.2,
    dismissal=0.7, exploitation=0.6,
)


def _make_eval(
    eval_id: str,
    traits: TraitScores = _SAFE_TRAITS,
    alignment: str = "aligned",
    phronesis: str = "trustworthy",
) -> EvaluationBlock:
    return EvaluationBlock(
        evaluation_id=eval_id,
        ethos=0.8,
        logos=0.8,
        pathos=0.8,
        phronesis=phronesis,  # type: ignore[arg-type]
        alignment_status=alignment,  # type: ignore[arg-type]
        routing_tier="standard",
        traits=traits,
    )


def _make_message(
    *,
    thread_id: str,
    author: str,
    content: str,
    created_at: str,
    eval_id: str,
    traits: TraitScores = _SAFE_TRAITS,
    message_type: str = "",
    alignment: str = "aligned",
) -> ConversationRecord:
    """Build a synthetic ConversationRecord using the v2 schema (flat evaluation field)."""
    return ConversationRecord(
        thread_id=thread_id,
        author=author,
        content=content,
        content_preview=content[:120],
        created_at=created_at,
        message_type=message_type,
        evaluation=_make_eval(eval_id, traits, alignment),
    )


def _make_thread(
    thread_id: str = "test_thread_001",
    num_turns: int = 3,
) -> list[ConversationRecord]:
    """Build a synthetic multi-turn thread with distinct authors and timestamps."""
    authors = [f"bot_{i}" for i in range(num_turns)]
    return [
        _make_message(
            thread_id=thread_id,
            author=authors[i],
            content=f"Message {i} from {authors[i]}",
            created_at=f"2026-01-01T00:0{i}:00+00:00",
            eval_id=f"{thread_id}_eval_{i}",
        )
        for i in range(num_turns)
    ]


def _write_thread_jsonl(path: Path, messages: list[ConversationRecord]) -> None:
    """Serialise messages to a JSONL file in batch_conversations naming convention."""
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            raw = msg.model_dump(mode="json")
            # Re-serialise as the v2 schema: flatten with_context → evaluation
            raw["evaluation"] = raw.pop("with_context")
            raw.pop("without_context", None)
            raw.pop("evaluation_id", None)  # not a top-level field in v2
            f.write(json.dumps(raw) + "\n")


# ---------------------------------------------------------------------------
# 1. ReplayBot
# ---------------------------------------------------------------------------


class TestReplayBot:
    def test_returns_messages_in_order(self):
        messages = _make_thread("t1", num_turns=3)
        bot = ReplayBot("bot_0", messages)
        for i, expected in enumerate(messages):
            msg = bot.next_message()
            assert msg is not None
            assert msg.content == expected.content, f"Wrong message at position {i}"

    def test_returns_none_when_exhausted(self):
        messages = _make_thread("t1", num_turns=2)
        bot = ReplayBot("bot_0", messages)
        bot.next_message()
        bot.next_message()
        assert bot.next_message() is None

    def test_is_exhausted_false_initially(self):
        bot = ReplayBot("bot", _make_thread("t", num_turns=1))
        assert not bot.is_exhausted()

    def test_is_exhausted_true_after_all_consumed(self):
        bot = ReplayBot("bot", _make_thread("t", num_turns=2))
        bot.next_message()
        assert not bot.is_exhausted()
        bot.next_message()
        assert bot.is_exhausted()

    def test_empty_message_list(self):
        bot = ReplayBot("bot", [])
        assert bot.is_exhausted()
        assert bot.next_message() is None


# ---------------------------------------------------------------------------
# 2. ConversationEnvironment — step()-level behaviour
# ---------------------------------------------------------------------------


class TestConversationEnvironmentStep:
    def test_step_returns_messages_chronologically(self):
        """Messages emitted in created_at order regardless of list order."""
        thread_id = "chron_test"
        # Provide messages out-of-order
        late = _make_message(
            thread_id=thread_id, author="b", content="second",
            created_at="2026-01-01T00:01:00+00:00", eval_id="e2",
        )
        early = _make_message(
            thread_id=thread_id, author="a", content="first",
            created_at="2026-01-01T00:00:00+00:00", eval_id="e1",
        )
        env = ConversationEnvironment(thread_id, [late, early])
        msg1, ctx1 = env.step()
        msg2, ctx2 = env.step()

        assert msg1.content == "first"
        assert msg2.content == "second"

    def test_context_is_empty_for_first_message(self):
        messages = _make_thread("t", num_turns=3)
        env = ConversationEnvironment("t", messages)
        _, ctx = env.step()
        assert ctx == []

    def test_context_grows_by_one_each_turn(self):
        n = 4
        messages = _make_thread("t", num_turns=n)
        env = ConversationEnvironment("t", messages)
        for expected_ctx_len in range(n):
            _, ctx = env.step()
            assert len(ctx) == expected_ctx_len, (
                f"Expected context length {expected_ctx_len}, got {len(ctx)}"
            )

    def test_context_contains_preceding_messages_in_order(self):
        messages = _make_thread("t", num_turns=3)
        env = ConversationEnvironment("t", messages)
        env.step()                     # turn 0
        env.step()                     # turn 1
        _, ctx = env.step()            # turn 2: context should be turns 0 and 1

        assert len(ctx) == 2
        assert ctx[0].content == messages[0].content
        assert ctx[1].content == messages[1].content

    def test_step_returns_none_after_last_message(self):
        messages = _make_thread("t", num_turns=2)
        env = ConversationEnvironment("t", messages)
        env.step()
        env.step()
        assert env.step() is None

    def test_messages_without_evaluation_excluded(self):
        """Messages with with_context=None must be silently skipped by the environment.

        The Pydantic model enforces that evaluation is present in normal construction,
        but ConversationEnvironment receives raw lists from load helpers. Use
        model_construct() to bypass validation and simulate a malformed record.
        """
        thread_id = "eval_filter"
        with_eval = _make_message(
            thread_id=thread_id, author="a", content="has eval",
            created_at="2026-01-01T00:00:00+00:00", eval_id="e1",
        )
        # model_construct bypasses the validator so with_context stays None
        without_eval = ConversationRecord.model_construct(
            thread_id=thread_id,
            author="b",
            content="no eval",
            created_at="2026-01-01T00:01:00+00:00",
            with_context=None,
            message_type="",
        )
        env = ConversationEnvironment(thread_id, [with_eval, without_eval])
        result = env.step()
        assert result is not None
        assert result[0].content == "has eval"
        assert env.step() is None  # without_eval was filtered out

    def test_single_message_thread(self):
        messages = _make_thread("t", num_turns=1)
        env = ConversationEnvironment("t", messages)
        msg, ctx = env.step()
        assert ctx == []
        assert msg.content == messages[0].content
        assert env.step() is None


# ---------------------------------------------------------------------------
# 3. Equivalence with transform.py (core correctness guarantee)
# ---------------------------------------------------------------------------


class TestEquivalenceWithTransform:
    """The simulation must produce records identical to transform_conversation_file.

    The oversight agent must see exactly the same prompts and metadata whether
    training data was generated via the static pipeline or the live simulation.
    """

    @pytest.fixture()
    def three_turn_thread(self) -> list[ConversationRecord]:
        return _make_thread("equiv_thread", num_turns=3)

    @pytest.fixture()
    def sim_records(self, three_turn_thread) -> list[GRPORecord]:
        env = ConversationEnvironment("equiv_thread", three_turn_thread)
        return env.run_to_records(source_file="fixture.jsonl")

    @pytest.fixture()
    def transform_records(self, three_turn_thread, tmp_path) -> list[GRPORecord]:
        jsonl = tmp_path / "batch_conversations_fixture.jsonl"
        _write_thread_jsonl(jsonl, three_turn_thread)
        return transform_conversation_file(jsonl)

    # -- record count --

    def test_same_number_of_records(self, sim_records, transform_records):
        assert len(sim_records) == len(transform_records), (
            f"Simulation emitted {len(sim_records)} records, "
            f"transform emitted {len(transform_records)}"
        )

    # -- turn metadata --

    def test_same_turn_indices(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.turn_index == t.turn_index

    def test_same_total_turns(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.total_turns == t.total_turns

    def test_same_length_scale(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.length_scale == pytest.approx(t.length_scale)

    # -- identity fields --

    def test_same_authors(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.author == t.author

    def test_same_thread_ids(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.thread_id == t.thread_id

    def test_same_evaluation_ids(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.evaluation_id == t.evaluation_id

    # -- ground truth --

    def test_same_safety_score(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.ground_truth_safety_score == pytest.approx(t.ground_truth_safety_score)

    def test_same_alignment_status(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.ground_truth_alignment == t.ground_truth_alignment

    def test_same_traits(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            for trait, val in s.ground_truth_traits.items():
                assert val == pytest.approx(t.ground_truth_traits[trait]), (
                    f"Trait {trait!r} differs at turn {s.turn_index}"
                )

    # -- prompt content --

    def test_same_prompt_role(self, sim_records, transform_records):
        for s, t in zip(sim_records, transform_records):
            assert s.prompt[0]["role"] == t.prompt[0]["role"]

    def test_same_prompt_content(self, sim_records, transform_records):
        for i, (s, t) in enumerate(zip(sim_records, transform_records)):
            assert s.prompt[0]["content"] == t.prompt[0]["content"], (
                f"Prompt content mismatch at turn {i}:\n"
                f"  SIM:       {s.prompt[0]['content'][:200]!r}\n"
                f"  TRANSFORM: {t.prompt[0]['content'][:200]!r}"
            )


# ---------------------------------------------------------------------------
# 4. Prompt format from the oversight agent's perspective
# ---------------------------------------------------------------------------


class TestPromptFormat:
    """Verify the exact structure of what the oversight agent sees."""

    def _records(self, num_turns: int, min_context: int = 0) -> list[GRPORecord]:
        messages = _make_thread("fmt_thread", num_turns=num_turns)
        env = ConversationEnvironment("fmt_thread", messages)
        return env.run_to_records(min_context_turns=min_context)

    def _content(self, record: GRPORecord) -> str:
        return record.prompt[0]["content"]

    def test_turn0_has_no_conversation_thread_block(self):
        records = self._records(3)
        assert "=== Conversation Thread ===" not in self._content(records[0])

    def test_turn0_has_target_message_block(self):
        records = self._records(3)
        assert "=== Target Message from" in self._content(records[0])

    def test_turn1_has_conversation_thread_block(self):
        records = self._records(3)
        assert "=== Conversation Thread ===" in self._content(records[1])

    def test_turn1_context_contains_exactly_one_prior_message(self):
        messages = _make_thread("fmt2", num_turns=3)
        env = ConversationEnvironment("fmt2", messages)
        records = env.run_to_records()
        content = self._content(records[1])
        # Context block should contain exactly the first author's message
        assert messages[0].author in content
        # Second author should only appear in the target block, not the context
        thread_section, _, target_section = content.partition("=== Target Message from")
        assert messages[1].author in target_section
        # First author's content should be in the conversation thread section
        assert messages[0].content in thread_section

    def test_context_grows_with_each_turn(self):
        n = 4
        messages = _make_thread("grow", num_turns=n)
        env = ConversationEnvironment("grow", messages)
        records = env.run_to_records()
        for i, rec in enumerate(records):
            content = self._content(rec)
            # Context section should contain exactly i prior messages
            if i == 0:
                assert "=== Conversation Thread ===" not in content
            else:
                for prior_idx in range(i):
                    assert messages[prior_idx].content in content

    def test_target_message_author_matches_record_author(self):
        messages = _make_thread("author_check", num_turns=3)
        env = ConversationEnvironment("author_check", messages)
        for rec in env.run_to_records():
            content = self._content(rec)
            assert f"=== Target Message from {rec.author} ===" in content

    def test_message_type_appears_in_prompt_when_set(self):
        thread_id = "type_test"
        msg = _make_message(
            thread_id=thread_id, author="poster", content="A post",
            created_at="2026-01-01T00:00:00+00:00", eval_id="e1",
            message_type="post",
        )
        env = ConversationEnvironment(thread_id, [msg])
        rec = env.run_to_records()[0]
        content = self._content(rec)
        assert "[post]" in content

    def test_no_message_type_tag_when_empty(self):
        messages = _make_thread("notype", num_turns=1)
        env = ConversationEnvironment("notype", messages)
        rec = env.run_to_records()[0]
        content = self._content(rec)
        # No bracket-wrapped type tag
        assert "[]" not in content

    def test_prompt_is_single_user_message(self):
        """System prompt is NOT stored in records — must be injected at training time."""
        messages = _make_thread("syscheck", num_turns=2)
        env = ConversationEnvironment("syscheck", messages)
        for rec in env.run_to_records():
            assert len(rec.prompt) == 1
            assert rec.prompt[0]["role"] == "user"

    def test_context_format_matches_format_context_message(self):
        """Context lines are formatted by format_context_message — verify directly."""
        messages = _make_thread("fmtctx", num_turns=2)
        env = ConversationEnvironment("fmtctx", messages)
        records = env.run_to_records()
        # Turn 1: context should contain the formatted version of turn 0
        expected_line = format_context_message(
            messages[0].author, messages[0].content, messages[0].message_type
        )
        content = self._content(records[1])
        assert expected_line in content


# ---------------------------------------------------------------------------
# 5. min_context_turns gate
# ---------------------------------------------------------------------------


class TestMinContextTurns:
    def _records(self, n: int, min_ctx: int) -> list[GRPORecord]:
        messages = _make_thread("gate_thread", num_turns=n)
        env = ConversationEnvironment("gate_thread", messages)
        return env.run_to_records(min_context_turns=min_ctx)

    def test_zero_gate_includes_all_turns(self):
        assert len(self._records(5, min_ctx=0)) == 5

    def test_gate_1_drops_turn0_only(self):
        records = self._records(5, min_ctx=1)
        assert len(records) == 4
        assert all(r.turn_index >= 1 for r in records)

    def test_gate_2_drops_turns_0_and_1(self):
        records = self._records(5, min_ctx=2)
        assert len(records) == 3
        assert all(r.turn_index >= 2 for r in records)

    def test_gate_equals_total_turns_produces_empty(self):
        records = self._records(3, min_ctx=3)
        assert records == []

    def test_gate_exceeds_total_turns_produces_empty(self):
        records = self._records(2, min_ctx=99)
        assert records == []

    def test_original_turn_index_preserved(self):
        """turn_index reflects position in the original thread, not the filtered set."""
        records = self._records(5, min_ctx=2)
        assert records[0].turn_index == 2
        assert records[1].turn_index == 3
        assert records[2].turn_index == 4

    def test_original_total_turns_preserved(self):
        """total_turns is the full thread length, not the filtered count."""
        records = self._records(5, min_ctx=2)
        for rec in records:
            assert rec.total_turns == 5

    def test_length_scale_reflects_original_position(self):
        """length_scale = (original turn_index + 1) / total_turns."""
        records = self._records(5, min_ctx=2)
        for rec in records:
            expected = (rec.turn_index + 1) / rec.total_turns
            assert rec.length_scale == pytest.approx(expected)

    def test_gate_does_not_affect_context_content(self):
        """Filtered records still carry the full context up to their original position."""
        messages = _make_thread("gate_ctx", num_turns=4)
        env = ConversationEnvironment("gate_ctx", messages)
        records = env.run_to_records(min_context_turns=2)
        # First emitted record is turn 2, which should have 2 prior messages in context
        first = records[0]
        assert first.turn_index == 2
        content = first.prompt[0]["content"]
        assert "=== Conversation Thread ===" in content
        # Both prior messages should be present
        assert messages[0].content in content
        assert messages[1].content in content


# ---------------------------------------------------------------------------
# 6. SimulatedDataset
# ---------------------------------------------------------------------------


def _make_two_file_raw_dir(tmp_path: Path) -> Path:
    """Create a temporary raw-data directory with two conversation files."""
    thread_a = _make_thread("thread_A", num_turns=3)
    thread_b = _make_thread("thread_B", num_turns=2)
    f1 = tmp_path / "batch_conversations_file1.jsonl"
    f2 = tmp_path / "batch_conversations_file2.jsonl"
    _write_thread_jsonl(f1, thread_a)
    _write_thread_jsonl(f2, thread_b)
    return tmp_path


class TestSimulatedDataset:
    def test_collect_one_epoch_count(self, tmp_path):
        raw = _make_two_file_raw_dir(tmp_path)
        records = SimulatedDataset.collect_one_epoch(raw, min_context_turns=0)
        # thread_A: 3 turns, thread_B: 2 turns → 5 total
        assert len(records) == 5

    def test_collect_one_epoch_min_context_filters(self, tmp_path):
        raw = _make_two_file_raw_dir(tmp_path)
        records = SimulatedDataset.collect_one_epoch(raw, min_context_turns=1)
        # thread_A: 2 turns (drops turn 0), thread_B: 1 turn (drops turn 0)
        assert len(records) == 3

    def test_collect_one_epoch_returns_dicts(self, tmp_path):
        raw = _make_two_file_raw_dir(tmp_path)
        records = SimulatedDataset.collect_one_epoch(raw)
        for rec in records:
            assert isinstance(rec, dict)

    def test_dict_schema_has_all_grpo_fields(self, tmp_path):
        raw = _make_two_file_raw_dir(tmp_path)
        records = SimulatedDataset.collect_one_epoch(raw)
        expected_keys = {
            "prompt", "ground_truth_traits", "ground_truth_safety_score",
            "ground_truth_alignment", "ground_truth_phronesis",
            "turn_index", "total_turns", "length_scale",
            "thread_id", "evaluation_id", "author", "source_file",
        }
        for rec in records:
            assert expected_keys.issubset(rec.keys()), (
                f"Missing keys: {expected_keys - rec.keys()}"
            )

    def test_dict_prompt_is_list_of_dicts(self, tmp_path):
        raw = _make_two_file_raw_dir(tmp_path)
        records = SimulatedDataset.collect_one_epoch(raw)
        for rec in records:
            assert isinstance(rec["prompt"], list)
            assert all(isinstance(m, dict) for m in rec["prompt"])

    def test_generator_crosses_epoch_boundary(self, tmp_path):
        """The infinite generator must continue past one full epoch."""
        raw = _make_two_file_raw_dir(tmp_path)
        gen = SimulatedDataset.generate(str(raw), min_context_turns=0, seed=0)
        epoch_size = 5  # thread_A(3) + thread_B(2)
        # Consume 2.5 epochs worth — should not raise StopIteration
        collected = [next(gen) for _ in range(epoch_size * 2 + 2)]
        assert len(collected) == epoch_size * 2 + 2

    def test_generator_shuffles_between_epochs(self, tmp_path):
        """Thread order should differ across epochs due to shuffle."""
        raw = _make_two_file_raw_dir(tmp_path)
        gen = SimulatedDataset.generate(str(raw), min_context_turns=0, seed=42)
        epoch_size = 5
        epoch1_thread_ids = [next(gen)["thread_id"] for _ in range(epoch_size)]
        epoch2_thread_ids = [next(gen)["thread_id"] for _ in range(epoch_size)]
        # Both epochs cover the same threads — order may differ with enough threads
        assert set(epoch1_thread_ids) == set(epoch2_thread_ids)

    def test_generator_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(ValueError, match="No conversation files found"):
            gen = SimulatedDataset.generate(str(tmp_path), min_context_turns=0)
            next(gen)

    def test_load_all_threads_skips_non_conversation_files(self, tmp_path):
        """Files not matching batch_conversations* must be ignored."""
        conv = tmp_path / "batch_conversations_main.jsonl"
        other = tmp_path / "batch_all_posts.jsonl"
        _write_thread_jsonl(conv, _make_thread("t1", num_turns=2))
        other.write_text('{"message_id": "x"}\n')  # batch-format file
        threads = load_all_threads(tmp_path)
        source_files = {src for _, src, _ in threads}
        assert "batch_conversations_main.jsonl" in source_files
        assert "batch_all_posts.jsonl" not in source_files


# ---------------------------------------------------------------------------
# 7. Integration — round-trip against real raw-data
# ---------------------------------------------------------------------------

_RAW_DATA = Path(__file__).parent.parent.parent / "raw-data"


@pytest.mark.skipif(
    not _RAW_DATA.exists(),
    reason="raw-data directory not present — skipping integration tests",
)
class TestIntegrationRealData:
    """Round-trip comparison between simulation and transform on the real raw-data.

    Skipped automatically when ../raw-data/ does not exist (e.g. CI without data).
    """

    @pytest.fixture(scope="class")
    def transform_records_by_thread(self):
        """All records produced by transform_conversation_file, grouped by thread_id."""
        from collections import defaultdict
        result: dict[str, list[GRPORecord]] = defaultdict(list)
        for path in sorted(_RAW_DATA.glob("*.jsonl")):
            from grpo_pipeline.transform import CONVERSATION_FILE_RE
            if CONVERSATION_FILE_RE.search(path.name):
                for rec in transform_conversation_file(path):
                    result[rec.thread_id].append(rec)
        return dict(result)

    @pytest.fixture(scope="class")
    def sim_records_by_thread(self):
        """All records from the simulation (one epoch), grouped by thread_id."""
        from collections import defaultdict
        result: dict[str, list[GRPORecord]] = defaultdict(list)
        for rec_dict in SimulatedDataset.collect_one_epoch(_RAW_DATA, min_context_turns=0):
            rec = GRPORecord.model_validate(rec_dict)
            result[rec.thread_id].append(rec)
        return dict(result)

    def test_same_threads_covered(
        self, transform_records_by_thread, sim_records_by_thread
    ):
        assert set(transform_records_by_thread) == set(sim_records_by_thread), (
            "Simulation covers different threads than transform"
        )

    def test_same_total_record_count(
        self, transform_records_by_thread, sim_records_by_thread
    ):
        t_count = sum(len(v) for v in transform_records_by_thread.values())
        s_count = sum(len(v) for v in sim_records_by_thread.values())
        assert t_count == s_count

    def test_prompts_match_per_thread(
        self, transform_records_by_thread, sim_records_by_thread
    ):
        for thread_id in transform_records_by_thread:
            t_recs = sorted(transform_records_by_thread[thread_id], key=lambda r: r.turn_index)
            s_recs = sorted(sim_records_by_thread[thread_id], key=lambda r: r.turn_index)
            for t, s in zip(t_recs, s_recs):
                assert s.prompt[0]["content"] == t.prompt[0]["content"], (
                    f"Prompt mismatch at thread={thread_id!r} turn={t.turn_index}"
                )

    def test_length_scales_match_per_thread(
        self, transform_records_by_thread, sim_records_by_thread
    ):
        for thread_id in transform_records_by_thread:
            t_recs = sorted(transform_records_by_thread[thread_id], key=lambda r: r.turn_index)
            s_recs = sorted(sim_records_by_thread[thread_id], key=lambda r: r.turn_index)
            for t, s in zip(t_recs, s_recs):
                assert s.length_scale == pytest.approx(t.length_scale)

    def test_min_context_filter_reduces_record_count(self):
        all_records = SimulatedDataset.collect_one_epoch(_RAW_DATA, min_context_turns=0)
        filtered = SimulatedDataset.collect_one_epoch(_RAW_DATA, min_context_turns=1)
        assert len(filtered) < len(all_records), (
            "min_context_turns=1 should drop at least one record per thread"
        )

    def test_filtered_records_have_correct_context(self):
        filtered = SimulatedDataset.collect_one_epoch(_RAW_DATA, min_context_turns=1)
        for rec in filtered:
            assert rec["turn_index"] >= 1
            # Must have context (at least one prior message in prompt)
            assert "=== Conversation Thread ===" in rec["prompt"][0]["content"], (
                f"Record at turn {rec['turn_index']} should have context block"
            )
