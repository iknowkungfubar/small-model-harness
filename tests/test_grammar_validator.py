"""Tests for Addition B — Grammar Validator (RPG).

ACL 2025: Pushdown automaton tracking formal grammar of output language.
Detects structural repetitions and penalizes anchor tokens driving loops.
"""

from __future__ import annotations

import math
import pytest
from pathlib import Path

_here = Path(__file__).resolve().parent
_server_dir = _here.parent / "mcp-server"
import sys

sys.path.insert(0, str(_server_dir))

from grammar_validator import (
    GrammarValidator,
    ValidatorConfig,
    PushdownAutomaton,
    detect_structural_repetition,
    compute_repetition_penalty,
)


class TestPushdownAutomaton:
    def test_initial_state(self):
        """Initial PDA state is empty."""
        pda = PushdownAutomaton()
        assert pda.stack_depth == 0
        assert pda.current_state == "start"

    def test_push_pop(self):
        """Push and pop maintain stack count."""
        pda = PushdownAutomaton()
        pda.push("json_object")
        assert pda.stack_depth == 1
        pda.push("json_array")
        assert pda.stack_depth == 2
        pda.pop()
        assert pda.stack_depth == 1
        pda.pop()
        assert pda.stack_depth == 0

    def test_underflow_no_error(self):
        """Pop on empty stack returns None, no error."""
        pda = PushdownAutomaton()
        result = pda.pop()
        assert result is None

    def test_track_tag(self):
        """Track XML/JSON structural tags."""
        pda = PushdownAutomaton()
        pda.track("open_tag", "div")
        assert pda.stack_depth == 1
        pda.track("close_tag", "div")
        assert pda.stack_depth == 0

    def test_unmatched_tags(self):
        """Unmatched close tag increments error count."""
        pda = PushdownAutomaton()
        pda.track("close_tag", "div")
        assert pda.mismatch_count == 1


class TestDetectStructuralRepetition:
    def test_no_repetition(self):
        """Normal varied tokens produce no repetition flag."""
        tokens = ["open", "div", "close", "open", "span", "close"]
        result = detect_structural_repetition(tokens)
        assert result["repetition_count"] == 0
        assert result["max_repetition"] <= 2

    def test_repeated_json_keys(self):
        """Repeated identical JSON keys are detected."""
        tokens = ["key", "key", "key", "key", "key"] * 4
        result = detect_structural_repetition(tokens)
        assert result["repetition_count"] > 0

    def test_identical_blocks(self):
        """Repeated identical block patterns detected."""
        block = ["open_tag", "text", "close_tag"]
        tokens = block * 10
        result = detect_structural_repetition(tokens)
        # Each unique token appears 10 times, threshold=3
        assert result["repetition_count"] >= 2
        assert result["identical_blocks"] > 0

    def test_empty_list(self):
        """Empty token list returns zeroes."""
        result = detect_structural_repetition([])
        assert result["repetition_count"] == 0
        assert result["max_repetition"] == 0

    def test_single_token(self):
        """Single token is not a repetition."""
        result = detect_structural_repetition(["hello"])
        assert result["repetition_count"] == 0


class TestComputeRepetitionPenalty:
    def test_no_penalty(self):
        """No repetition = no penalty."""
        penalty = compute_repetition_penalty(structure_repetitions=0, identical_blocks=0)
        assert penalty == 1.0

    def test_light_penalty(self):
        """Light repetition produces small penalty (<1.0)."""
        penalty = compute_repetition_penalty(structure_repetitions=3, identical_blocks=0)
        assert penalty < 1.0
        assert penalty > 0.5

    def test_heavy_penalty(self):
        """Heavy repetition produces strong penalty."""
        penalty = compute_repetition_penalty(structure_repetitions=10, identical_blocks=5)
        assert penalty < 0.5

    def test_zero_repetition(self):
        """Zero repetition = 1.0 (no penalty)."""
        penalty = compute_repetition_penalty(structure_repetitions=0, identical_blocks=0)
        assert penalty == 1.0


class TestGrammarValidator:
    def test_initial_state(self):
        """Initial state is clean."""
        gv = GrammarValidator()
        assert gv.repetition_count == 0
        assert gv.penalty == 1.0

    def test_observe_normal(self):
        """Normal tokens don't trigger penalty."""
        gv = GrammarValidator()
        tokens = ["open", "div", "class", "text", "close"]
        for t in tokens:
            gv.observe(t)
        assert gv.penalty >= 0.8

    def test_observe_repetitive(self):
        """Repetitive tokens trigger penalty reduction."""
        gv = GrammarValidator()
        for _ in range(8):
            gv.observe("json_key")
        assert gv.repetition_count > 0
        assert gv.penalty < 1.0

    def test_reset(self):
        """Reset clears state."""
        gv = GrammarValidator()
        for _ in range(10):
            gv.observe("repeat")
        gv.reset()
        assert gv.repetition_count == 0
        assert gv.penalty == 1.0

    def test_configurable_threshold(self):
        """Configurable repetition threshold."""
        config_default = ValidatorConfig()
        assert config_default.repetition_threshold == 3

        config_lenient = ValidatorConfig(repetition_threshold=10)
        assert config_lenient.repetition_threshold == 10

        gv_lenient = GrammarValidator(config_lenient)
        for _ in range(5):
            gv_lenient.observe("test")
        assert gv_lenient.repetition_count == 0

    def test_observe_none_ignored(self):
        """None tokens are ignored."""
        gv = GrammarValidator()
        gv.observe(None)
        assert gv.repetition_count == 0

    def test_to_dict(self):
        """Serialization works."""
        gv = GrammarValidator(ValidatorConfig(repetition_threshold=5))
        d = gv.to_dict()
        assert d["repetition_threshold"] == 5
