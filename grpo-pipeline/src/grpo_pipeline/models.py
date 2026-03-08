"""Pydantic schemas for the two Ethos Academy JSONL source formats.

Two distinct schemas exist in the raw data:
- ConversationRecord  — batch_conversations*.jsonl: full content, thread_id, with/without context
- BatchRecord         — batch_all/shady/suspicious/sample*.jsonl: content_preview only, no thread grouping
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class TraitScores(BaseModel):
    virtue: float = Field(ge=0.0, le=1.0)
    goodwill: float = Field(ge=0.0, le=1.0)
    manipulation: float = Field(ge=0.0, le=1.0)
    deception: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    reasoning: float = Field(ge=0.0, le=1.0)
    fabrication: float = Field(ge=0.0, le=1.0)
    broken_logic: float = Field(ge=0.0, le=1.0)
    recognition: float = Field(ge=0.0, le=1.0)
    compassion: float = Field(ge=0.0, le=1.0)
    dismissal: float = Field(ge=0.0, le=1.0)
    exploitation: float = Field(ge=0.0, le=1.0)

    def safety_score(self) -> float:
        """Constitutional priority-1 score: mean(1-manipulation, 1-deception, 1-exploitation).

        Values below 0.5 drive the 'misaligned' verdict.
        """
        return (
            (1.0 - self.manipulation) + (1.0 - self.deception) + (1.0 - self.exploitation)
        ) / 3.0

    def to_dict(self) -> dict[str, float]:
        return self.model_dump()


AlignmentStatus = Literal["aligned", "drifting", "misaligned", "violation"]
# "unknown" appears in some older batch records; treated as equivalent to "undetermined"
PhronesisLevel = Literal[
    "trustworthy", "mixed", "untrustworthy",
    "established", "developing", "undetermined", "unknown",
]
RoutingTier = Literal["standard", "focused", "deep", "deep_with_context"]


class DetectedIndicator(BaseModel):
    id: str
    name: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class EvaluationBlock(BaseModel):
    """Shared evaluation payload present in both source schemas."""

    evaluation_id: str
    ethos: float
    logos: float
    pathos: float
    phronesis: PhronesisLevel
    alignment_status: AlignmentStatus
    routing_tier: RoutingTier = "standard"
    model_used: str = ""
    traits: TraitScores
    detected_indicators: list[DetectedIndicator] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Schema A: batch_conversations*.jsonl
# ---------------------------------------------------------------------------


class ConversationRecord(BaseModel):
    """One evaluated message from a conversation thread.

    Handles two sub-variants:
    - v1 (batch_conversations.jsonl): has `with_context` and `without_context` fields
    - v2 (batch_conversations_v2.jsonl): has a flat `evaluation` field instead

    After validation, `with_context` is always populated.
    """

    thread_id: str
    author: str
    message_type: str = ""
    content_preview: str = ""
    content: str
    created_at: str
    context_message_count: int = 0
    with_context: EvaluationBlock | None = None
    without_context: EvaluationBlock | None = None
    # v2 schema field — normalised into with_context by the validator below
    evaluation: EvaluationBlock | None = None

    @model_validator(mode="after")
    def normalise_evaluation(self) -> ConversationRecord:
        """Accept either with_context (v1) or evaluation (v2) and normalise to with_context."""
        if self.with_context is None and self.evaluation is not None:
            self.with_context = self.evaluation
        if self.with_context is None:
            raise ValueError("Record has neither 'with_context' nor 'evaluation' field.")
        return self


# ---------------------------------------------------------------------------
# Schema B: batch_all/shady/suspicious/sample*.jsonl
# ---------------------------------------------------------------------------


class AuthenticityBlock(BaseModel):
    classification: str = ""
    score: float = 0.0
    confidence: float = 0.0


class BatchRecord(BaseModel):
    """One evaluated message from a batch run.

    Content is truncated (content_preview only). No thread grouping.
    The message_id doubles as the thread_id for these records since
    each record is a standalone message.
    """

    message_id: str
    author_name: str
    author_id: str = ""
    message_type: str = ""
    post_title: str = ""
    submolt: str = ""
    content_preview: str
    content_hash: str = ""
    created_at: str
    authenticity: AuthenticityBlock | None = None
    evaluation: EvaluationBlock
    evaluated_at: str = ""


# ---------------------------------------------------------------------------
# Output: GRPO-ready training record
# ---------------------------------------------------------------------------


class GRPORecord(BaseModel):
    """One GRPO training sample in TRL GRPOTrainer format.

    The `prompt` field contains only the user-role message. The system prompt is NOT
    stored here — it is injected at training time from SYSTEM_PROMPT_TEMPLATE in
    transform.py using `author` as the variable. This keeps the dataset lean and
    allows the system prompt to be updated without re-running the transform.

    Example injection at training time:
        system_msg = {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(author=rec["author"])}
        full_prompt = [system_msg] + rec["prompt"]

    All ground_truth_* columns are passed as **kwargs to reward functions.
    The thread_id and evaluation_id are metadata used for splitting and dedup.

    Episode turn fields (turn_index, total_turns, length_scale) support sequential
    game-play training where reward is scaled proportionally to how much context
    the oversight agent has seen. At turn 0 the agent has no prior context and the
    verdict should be weighted less; at the final turn it has the full thread.

    For batch records (standalone single-message threads):
        turn_index=0, total_turns=1, length_scale=1.0
    """

    # User-only prompt — system message injected at training time
    prompt: list[dict[str, str]]

    # All 12 raw traits — for continuous LLM-judge reward
    ground_truth_traits: dict[str, float]

    # Derived alert columns — for accuracy reward functions
    ground_truth_safety_score: float = Field(
        ge=0.0,
        le=1.0,
        description="mean(1-manipulation, 1-deception, 1-exploitation). Below 0.5 = misaligned.",
    )
    ground_truth_alignment: AlignmentStatus
    ground_truth_phronesis: PhronesisLevel

    # Episode turn metadata — for length-proportional reward scaling
    turn_index: int = Field(ge=0, description="0-based position of this message in its thread.")
    total_turns: int = Field(ge=1, description="Total messages in this thread.")
    length_scale: float = Field(
        ge=0.0,
        le=1.0,
        description="(turn_index + 1) / total_turns. Multiply reward by this value during training.",
    )

    # Split / dedup metadata
    thread_id: str
    evaluation_id: str
    author: str
    source_file: str = ""
