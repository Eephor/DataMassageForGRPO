"""Unit tests for chat-template formatting and reward functions.

Covers the two completion-input formats that TRL's GRPOTrainer may pass:

  - str  — when dataset['prompt'] contains pre-formatted strings (our setup).
            TRL skips its internal apply_chat_template and passes the model's
            raw generated text directly as a string.
  - list[dict] — when dataset['prompt'] contains message dicts.
                 TRL applies the chat template internally and passes
                 [{"role": "assistant", "content": "..."}].

Both Llama and Qwen3 tokenizer templates are simulated with lightweight mocks
so no ML library is required to run these tests.
"""

from __future__ import annotations

import pytest

from grpo_pipeline.rewards import (
    _completion_text,
    _fix_unquoted_keys,
    extract_verdict,
    format_reward,
    group_reward,
    safe_apply_template,
    safety_level_reward,
    safety_score_to_level,
    traits_to_group_label,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_GOOD_VERDICT = (
    "<think>\nreasoning\n</think>\n"
    '<verdict>\n{"safety_level": "safe", "integrity": "strong", '
    '"reasoning": "good", "empathy": "good"}\n</verdict>'
)

_BAD_JSON_VERDICT = (
    "<think>\nreasoning\n</think>\n"
    "<verdict>\nnot valid json\n</verdict>"
)

_MISSING_THINK = (
    '<verdict>\n{"safety_level": "safe", "integrity": "strong", '
    '"reasoning": "good", "empathy": "good"}\n</verdict>'
)

_MISSING_VERDICT_TAG = "<think>\nreasoning\n</think>"

_SAFE_TRAITS = {
    "virtue": 0.9, "goodwill": 0.9,
    "manipulation": 0.05, "deception": 0.05,
    "accuracy": 0.9, "reasoning": 0.9,
    "fabrication": 0.05, "broken_logic": 0.05,
    "recognition": 0.9, "compassion": 0.9,
    "dismissal": 0.05, "exploitation": 0.05,
}


class _MockLlamaTokenizer:
    """Simulates a Llama tokenizer: no 'enable_thinking' in its Jinja template."""

    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
    )

    def apply_chat_template(self, msgs: list[dict], **kwargs) -> str:
        self.last_kwargs = kwargs
        lines = "\n".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs)
        if kwargs.get("add_generation_prompt"):
            lines += "\n<assistant>"
        return lines


class _MockQwen3Tokenizer:
    """Simulates a Qwen3 tokenizer: has 'enable_thinking' in its Jinja template."""

    chat_template = (
        "{% if enable_thinking %}<think>\n{% endif %}"
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
    )

    def apply_chat_template(self, msgs: list[dict], **kwargs) -> str:
        self.last_kwargs = kwargs
        lines = "\n".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs
        )
        if kwargs.get("enable_thinking"):
            lines = "<think>\n" + lines
        if kwargs.get("add_generation_prompt"):
            lines += "\n<|im_start|>assistant\n"
        return lines


_SAMPLE_MSGS = [
    {"role": "system", "content": "You are an evaluator."},
    {"role": "user", "content": "Evaluate this bot."},
]

# ---------------------------------------------------------------------------
# safe_apply_template: Llama (no enable_thinking)
# ---------------------------------------------------------------------------


class TestSafeApplyTemplateLlama:
    def test_returns_string(self):
        tok = _MockLlamaTokenizer()
        result = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
        assert isinstance(result, str)

    def test_no_think_injected_into_prompt(self):
        tok = _MockLlamaTokenizer()
        result = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
        assert "<think>" not in result, (
            "Llama prompt must not start with <think> — that belongs in the completion"
        )

    def test_passes_kwargs_through(self):
        tok = _MockLlamaTokenizer()
        safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False, add_generation_prompt=True)
        assert tok.last_kwargs.get("add_generation_prompt") is True

    def test_does_not_set_enable_thinking(self):
        """Llama has no enable_thinking in its template, so the kwarg must NOT be injected."""
        tok = _MockLlamaTokenizer()
        safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
        assert "enable_thinking" not in tok.last_kwargs, (
            "enable_thinking must not be injected for Llama"
        )


# ---------------------------------------------------------------------------
# safe_apply_template: Qwen3 (has enable_thinking)
# ---------------------------------------------------------------------------


class TestSafeApplyTemplateQwen3:
    def test_returns_string(self):
        tok = _MockQwen3Tokenizer()
        result = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
        assert isinstance(result, str)

    def test_disables_thinking_by_default(self):
        """Without explicit enable_thinking, the wrapper sets it to False so
        <think> is NOT injected into the prompt — it belongs in the completion."""
        tok = _MockQwen3Tokenizer()
        result = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False, add_generation_prompt=True)
        assert not result.startswith("<think>"), (
            "Qwen3 prompt must not start with <think> when enable_thinking is suppressed"
        )
        assert tok.last_kwargs.get("enable_thinking") is False

    def test_caller_can_override_enable_thinking_true(self):
        """If caller explicitly passes enable_thinking=True, we must honour it (setdefault)."""
        tok = _MockQwen3Tokenizer()
        result = safe_apply_template(
            tok, _SAMPLE_MSGS, tokenize=False, enable_thinking=True
        )
        assert tok.last_kwargs.get("enable_thinking") is True
        assert result.startswith("<think>"), (
            "When enable_thinking=True is explicit the prompt should start with <think>"
        )

    def test_does_not_double_inject_thinking(self):
        """Calling twice with the same tokenizer should not accumulate <think> tags."""
        tok = _MockQwen3Tokenizer()
        r1 = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
        r2 = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
        assert r1 == r2

    def test_contains_message_content(self):
        tok = _MockQwen3Tokenizer()
        result = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
        assert "Evaluate this bot." in result

    def test_no_think_injection_in_generation_prompt(self):
        """The generation prompt suffix (<|im_start|>assistant\\n) must not be preceded
        by <think> when thinking mode is suppressed — otherwise the model sees an
        open tag it did not generate and format_reward scores 0."""
        tok = _MockQwen3Tokenizer()
        result = safe_apply_template(
            tok, _SAMPLE_MSGS, tokenize=False, add_generation_prompt=True
        )
        # The last chunk before the assistant turn opener must not contain <think>
        # (it would appear if enable_thinking leaked through)
        assert result.count("<think>") == 0


# ---------------------------------------------------------------------------
# safe_apply_template: None chat_template guard
# ---------------------------------------------------------------------------


def test_safe_apply_template_none_chat_template():
    """If tokenizer.chat_template is None (unusual but possible), no crash."""
    class _TokenizerNoTemplate:
        chat_template = None

        def apply_chat_template(self, msgs, **kwargs):
            self.last_kwargs = kwargs
            return "plain"

    tok = _TokenizerNoTemplate()
    result = safe_apply_template(tok, _SAMPLE_MSGS, tokenize=False)
    assert result == "plain"
    assert "enable_thinking" not in tok.last_kwargs


# ---------------------------------------------------------------------------
# _completion_text: handles both TRL completion formats
# ---------------------------------------------------------------------------


class TestCompletionText:
    def test_string_input_returns_as_is(self):
        text = "<think>reasoning</think><verdict>{}</verdict>"
        assert _completion_text(text) == text

    def test_list_of_dict_returns_content(self):
        completion = [{"role": "assistant", "content": "hello"}]
        assert _completion_text(completion) == "hello"

    def test_list_of_dict_empty_content(self):
        completion = [{"role": "assistant", "content": ""}]
        assert _completion_text(completion) == ""

    def test_list_of_str_fallback(self):
        """Unexpected format: list containing a plain string (defensive fallback)."""
        completion = ["some plain text"]
        assert _completion_text(completion) == "some plain text"

    def test_empty_string(self):
        assert _completion_text("") == ""

    def test_multiline_string_preserved(self):
        text = "<think>\nline 1\nline 2\n</think>\n<verdict>\n{}\n</verdict>"
        assert _completion_text(text) == text


# ---------------------------------------------------------------------------
# format_reward: scoring logic
# ---------------------------------------------------------------------------


class TestFormatReward:
    """Tiered scoring: <verdict> required; <think> is an optional bonus.

    1.0  verdict + valid JSON + think
    0.7  verdict + valid JSON, no think
    0.3  verdict + invalid JSON + think
    0.2  verdict + invalid JSON, no think
    0.0  no verdict tag at all
    """

    def _call(self, completions):
        return format_reward(prompts=["irrelevant"] * len(completions), completions=completions)

    # -- full format (both tags + valid JSON) --

    def test_full_format_string(self):
        scores = self._call([_GOOD_VERDICT])
        assert scores == [1.0]

    def test_full_format_list_dict(self):
        scores = self._call([[{"role": "assistant", "content": _GOOD_VERDICT}]])
        assert scores == [1.0]

    # -- verdict only (no think, valid JSON) → 0.7 --

    def test_verdict_only_valid_json(self):
        scores = self._call([_MISSING_THINK])
        assert scores == [0.7]

    # -- verdict + think, bad JSON → 0.3 --

    def test_verdict_plus_think_bad_json(self):
        scores = self._call([_BAD_JSON_VERDICT])
        assert scores == [0.3]

    # -- verdict only, bad JSON → 0.2 --

    def test_verdict_only_bad_json(self):
        verdict_no_think_bad_json = (
            '<verdict>\nnot valid json\n</verdict>'
        )
        scores = self._call([verdict_no_think_bad_json])
        assert scores == [0.2]

    # -- no verdict at all → 0.0 --

    def test_missing_verdict_tag_string(self):
        scores = self._call([_MISSING_VERDICT_TAG])
        assert scores == [0.0]

    def test_empty_completion_string(self):
        scores = self._call([""])
        assert scores == [0.0]

    def test_missing_verdict_list_dict(self):
        scores = self._call([[{"role": "assistant", "content": _MISSING_VERDICT_TAG}]])
        assert scores == [0.0]

    # -- batch with all four outcomes --

    def test_batch_all_tiers(self):
        verdict_only_bad = '<verdict>\nnot valid json\n</verdict>'
        completions = [_GOOD_VERDICT, _MISSING_THINK, _BAD_JSON_VERDICT, verdict_only_bad, ""]
        scores = format_reward(prompts=["x"] * 5, completions=completions)
        assert scores == [1.0, 0.7, 0.3, 0.2, 0.0]

    def test_returns_list_same_length(self):
        n = 6
        scores = self._call([_GOOD_VERDICT] * n)
        assert len(scores) == n


# ---------------------------------------------------------------------------
# safety_score_to_level: boundary values
# ---------------------------------------------------------------------------


class TestSafetyScoreToLevel:
    @pytest.mark.parametrize("score,expected", [
        (1.00, "safe"),
        (0.85, "safe"),
        (0.84, "caution"),
        (0.65, "caution"),
        (0.64, "risk"),
        (0.40, "risk"),
        (0.39, "critical"),
        (0.00, "critical"),
    ])
    def test_boundaries(self, score, expected):
        assert safety_score_to_level(score) == expected


# ---------------------------------------------------------------------------
# safety_level_reward
# ---------------------------------------------------------------------------


_CORRECT_SAFE = (
    "<think>ok</think>\n"
    '<verdict>{"safety_level": "safe", "integrity": "strong", '
    '"reasoning": "good", "empathy": "good"}</verdict>'
)
_WRONG_SAFE = (
    "<think>ok</think>\n"
    '<verdict>{"safety_level": "critical", "integrity": "strong", '
    '"reasoning": "good", "empathy": "good"}</verdict>'
)


class TestSafetyLevelReward:
    def _call(self, completions, gt_scores, scales):
        return safety_level_reward(
            prompts=["x"] * len(completions),
            completions=completions,
            ground_truth_safety_score=gt_scores,
            length_scale=scales,
        )

    def test_correct_prediction_safe(self):
        # gt=0.9 → "safe", class_weight=1.0, scale=1.0 → +2.0
        scores = self._call([_CORRECT_SAFE], [0.9], [1.0])
        assert scores == pytest.approx([2.0])

    def test_wrong_prediction_safe(self):
        # gt=0.9 → "safe", wrong → -1.0 × 1.0 × 1.0
        scores = self._call([_WRONG_SAFE], [0.9], [1.0])
        assert scores == pytest.approx([-1.0])

    def test_length_scale_applied(self):
        # scale=0.5 → 2.0 × 1.0 × 0.5 = 1.0
        scores = self._call([_CORRECT_SAFE], [0.9], [0.5])
        assert scores == pytest.approx([1.0])

    def test_class_weight_critical(self):
        # gt=0.2 → "critical", weight=8.0, correct prediction → 2.0 × 8.0 × 1.0 = 16.0
        critical_verdict = (
            "<think>ok</think>\n"
            '<verdict>{"safety_level": "critical", "integrity": "poor", '
            '"reasoning": "poor", "empathy": "poor"}</verdict>'
        )
        scores = self._call([critical_verdict], [0.2], [1.0])
        assert scores == pytest.approx([16.0])

    def test_unparseable_verdict_zero(self):
        scores = self._call(["garbage text"], [0.9], [1.0])
        assert scores == pytest.approx([0.0])

    # -- same tests but with list[dict] format (simulating message-dict path) --

    def test_list_dict_correct(self):
        scores = self._call([[{"role": "assistant", "content": _CORRECT_SAFE}]], [0.9], [1.0])
        assert scores == pytest.approx([2.0])

    def test_list_dict_wrong(self):
        scores = self._call([[{"role": "assistant", "content": _WRONG_SAFE}]], [0.9], [1.0])
        assert scores == pytest.approx([-1.0])

    def test_batch_length(self):
        scores = self._call([_CORRECT_SAFE, _WRONG_SAFE, "bad"], [0.9, 0.9, 0.9], [1.0, 1.0, 1.0])
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# group_reward
# ---------------------------------------------------------------------------


_ALL_STRONG = (
    "<think>ok</think>\n"
    '<verdict>{"safety_level": "safe", "integrity": "strong", '
    '"reasoning": "strong", "empathy": "strong"}</verdict>'
)


class TestGroupReward:
    def _call(self, completions, gt_traits, gt_scores, scales):
        return group_reward(
            prompts=["x"] * len(completions),
            completions=completions,
            ground_truth_traits=gt_traits,
            ground_truth_safety_score=gt_scores,
            length_scale=scales,
        )

    def test_all_correct_safe(self):
        # traits → strong/strong/strong, weight=1.0, scale=1.0 → 3.0
        scores = self._call([_ALL_STRONG], [_SAFE_TRAITS], [0.9], [1.0])
        assert scores == pytest.approx([3.0])

    def test_all_wrong_still_zero(self):
        all_poor = (
            "<think>ok</think>\n"
            '<verdict>{"safety_level": "safe", "integrity": "poor", '
            '"reasoning": "poor", "empathy": "poor"}</verdict>'
        )
        scores = self._call([all_poor], [_SAFE_TRAITS], [0.9], [1.0])
        assert scores == pytest.approx([0.0])

    def test_partial_match(self):
        # _SAFE_TRAITS → strong/strong/strong; predict one correct (integrity=strong)
        partial = (
            "<think>ok</think>\n"
            '<verdict>{"safety_level": "safe", "integrity": "strong", '
            '"reasoning": "poor", "empathy": "poor"}</verdict>'
        )
        scores = self._call([partial], [_SAFE_TRAITS], [0.9], [1.0])
        assert scores == pytest.approx([1.0])  # 1 correct × 1.0 weight × 1.0 scale

    def test_length_scale_applied(self):
        scores = self._call([_ALL_STRONG], [_SAFE_TRAITS], [0.9], [0.5])
        assert scores == pytest.approx([1.5])  # 3.0 × 0.5

    def test_unparseable_verdict_zero(self):
        scores = self._call(["garbage"], [_SAFE_TRAITS], [0.9], [1.0])
        assert scores == pytest.approx([0.0])

    def test_list_dict_completion(self):
        completion = [{"role": "assistant", "content": _ALL_STRONG}]
        scores = self._call([completion], [_SAFE_TRAITS], [0.9], [1.0])
        assert scores == pytest.approx([3.0])


# ---------------------------------------------------------------------------
# traits_to_group_label: boundary values
# ---------------------------------------------------------------------------


class TestTraitsToGroupLabel:
    def test_all_positive_high(self):
        traits = {"virtue": 1.0, "goodwill": 1.0, "manipulation": 0.0, "deception": 0.0}
        assert traits_to_group_label(traits, "integrity") == "strong"

    def test_all_negative_high(self):
        traits = {"virtue": 0.0, "goodwill": 0.0, "manipulation": 1.0, "deception": 1.0}
        assert traits_to_group_label(traits, "integrity") == "poor"

    def test_mixed_gives_good(self):
        traits = {"virtue": 0.8, "goodwill": 0.8, "manipulation": 0.2, "deception": 0.2}
        # pos_avg=0.8, neg_inv=(1-0.2)=0.8 → group=(0.8+0.8)/2=0.8 → "strong"
        assert traits_to_group_label(traits, "integrity") == "strong"

    @pytest.mark.parametrize("group", ["integrity", "reasoning", "empathy"])
    def test_all_three_groups_valid(self, group):
        result = traits_to_group_label(_SAFE_TRAITS, group)
        assert result in {"strong", "good", "weak", "poor"}


# ---------------------------------------------------------------------------
# _fix_unquoted_keys: JSON key repair
# ---------------------------------------------------------------------------


class TestFixUnquotedKeys:
    def test_fixes_single_unquoted_key(self):
        broken = '{"safety_level": "risk",\n  integrity": "weak"}'
        fixed = _fix_unquoted_keys(broken)
        assert '"integrity"' in fixed
        assert 'integrity":' not in fixed.replace('"integrity":', "")

    def test_leaves_correctly_quoted_keys_alone(self):
        good = '{"safety_level": "safe", "integrity": "strong"}'
        assert _fix_unquoted_keys(good) == good

    def test_fixes_multiple_unquoted_keys(self):
        broken = '{safety_level": "risk", integrity": "weak", reasoning": "poor"}'
        fixed = _fix_unquoted_keys(broken)
        import json
        parsed = json.loads(fixed)
        assert parsed["safety_level"] == "risk"
        assert parsed["integrity"] == "weak"
        assert parsed["reasoning"] == "poor"

    def test_idempotent_on_valid_json(self):
        good = '{"safety_level": "safe", "integrity": "strong", "reasoning": "good", "empathy": "weak"}'
        assert _fix_unquoted_keys(good) == good


# ---------------------------------------------------------------------------
# extract_verdict: JSON parsing edge cases
# ---------------------------------------------------------------------------


class TestExtractVerdict:
    def test_valid_verdict(self):
        v = extract_verdict(_GOOD_VERDICT)
        assert v is not None
        assert v["safety_level"] == "safe"
        assert v["integrity"] == "strong"
        assert v["reasoning"] == "good"
        assert v["empathy"] == "good"

    def test_trailing_comma_in_json(self):
        text = (
            "<think>r</think>\n<verdict>\n"
            '{"safety_level": "risk", "integrity": "weak", '
            '"reasoning": "weak", "empathy": "weak",}\n</verdict>'
        )
        v = extract_verdict(text)
        assert v is not None
        assert v["safety_level"] == "risk"

    def test_unquoted_key_repaired(self):
        """Handles Qwen3 base model's common output error: missing opening quote on key."""
        text = (
            "<think>r</think>\n<verdict>\n"
            '{\n  "safety_level": "risk",\n  integrity": "weak",\n'
            '  "reasoning": "poor",\n  "empathy": "weak"\n}\n</verdict>'
        )
        v = extract_verdict(text)
        assert v is not None, "extract_verdict should repair the unquoted key"
        assert v["integrity"] == "weak"
        assert v["safety_level"] == "risk"

    def test_unquoted_key_plus_trailing_comma(self):
        """Both repairs applied together."""
        text = (
            "<verdict>\n"
            '{\n  "safety_level": "safe",\n  integrity": "strong",\n'
            '  "reasoning": "good",\n  "empathy": "good",\n}\n</verdict>'
        )
        v = extract_verdict(text)
        assert v is not None
        assert v["integrity"] == "strong"

    def test_invalid_safety_level_returns_none(self):
        text = (
            "<think>r</think>\n<verdict>\n"
            '{"safety_level": "unknown", "integrity": "strong", '
            '"reasoning": "good", "empathy": "good"}\n</verdict>'
        )
        assert extract_verdict(text) is None

    def test_invalid_group_label_returns_none(self):
        text = (
            "<think>r</think>\n<verdict>\n"
            '{"safety_level": "safe", "integrity": "excellent", '
            '"reasoning": "good", "empathy": "good"}\n</verdict>'
        )
        assert extract_verdict(text) is None

    def test_missing_key_returns_none(self):
        text = (
            "<think>r</think>\n<verdict>\n"
            '{"safety_level": "safe", "integrity": "good"}\n</verdict>'
        )
        assert extract_verdict(text) is None

    def test_no_verdict_tag_returns_none(self):
        assert extract_verdict("<think>reasoning</think>") is None

    def test_empty_string_returns_none(self):
        assert extract_verdict("") is None
