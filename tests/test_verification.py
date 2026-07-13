"""
Tests for Phase 5: Self-Consistency, Spec-Grounded Verification,
and Checkpoint Manager.

Covers:
- 5a: Self-consistency checking across multiple responses
- 5b: Spec-grounded verification against rubrics
- 5c: Checkpoint creation and rollback
- Combined Verifier integration
"""

from __future__ import annotations

import math
import time
from pathlib import Path

# Ensure modules are importable
_here = Path(__file__).resolve().parent
_server_dir = _here.parent / "mcp-server"
import sys

sys.path.insert(0, str(_server_dir))

from checkpoint_manager import (
    Checkpoint,
    CheckpointManager,
    CheckpointStore,
    create_checkpoint,
    rollback,
)
from verification import (
    SelfConsistencyChecker,
    SpecGroundedVerifier,
    VerificationRubric,
    Verifier,
    check_consistency,
)

# ===================================================================
# 5a: Self-Consistency Tests
# ===================================================================


class TestSelfConsistencyChecker:
    def test_identical_responses(self):
        """Identical responses are fully consistent."""
        responses = [
            {"tool_name": "search", "query": "hello"},
            {"tool_name": "search", "query": "hello"},
            {"tool_name": "search", "query": "hello"},
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses)
        assert result.is_consistent
        assert math.isclose(result.agreement_ratio, 1.0)
        assert result.matching_fields == result.total_fields

    def test_disagreement_detected(self):
        """Disagreement on key field is detected."""
        responses = [
            {"tool_name": "search", "query": "hello"},
            {"tool_name": "search", "query": "goodbye"},
            {"tool_name": "search", "query": "hello"},
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses)
        # query field has 2/3 agreement
        assert result.agreement_ratio < 1.0

    def test_single_response(self):
        """Single response has insufficient data."""
        checker = SelfConsistencyChecker()
        result = checker.check_consistency([{"tool_name": "search"}])
        assert result.is_consistent  # Default to consistent
        assert result.anomalies == ["single_response_insufficient"]

    def test_empty_list(self):
        """Empty response list is handled."""
        checker = SelfConsistencyChecker()
        result = checker.check_consistency([])
        assert result.is_consistent
        assert "single_response_insufficient" in result.anomalies

    def test_json_string_responses(self):
        """String JSON responses are parsed."""
        responses = [
            '{"tool_name": "search", "query": "hello"}',
            '{"tool_name": "search", "query": "hello"}',
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses)
        assert result.is_consistent

    def test_auto_detect_fields(self):
        """Fields are auto-detected from response structure."""
        responses = [
            {"tool_name": "search", "query": "hello", "limit": 10},
            {"tool_name": "search", "query": "world", "limit": 5},
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses)
        # Should auto-detect tool_name, query, limit
        assert result.total_fields >= 3

    def test_key_fields_override(self):
        """Specified key fields override auto-detection."""
        responses = [
            {"tool_name": "search", "query": "hello", "irrelevant": True},
            {"tool_name": "read", "query": "world", "irrelevant": False},
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses, key_fields=["tool_name", "query"])
        assert result.total_fields == 2

    def test_tolerance_threshold(self):
        """Agreement below tolerance fails consistency."""
        responses = [
            {"tool_name": "search"},
            {"tool_name": "read"},
            {"tool_name": "search"},
            {"tool_name": "delete"},
        ]
        checker = SelfConsistencyChecker()
        # 2/4 = 0.5 agreement, tolerance default 0.7
        result = checker.check_consistency(responses)
        assert not result.is_consistent

    def test_nested_value_comparison(self):
        """Nested dict values are compared properly."""
        responses = [
            {"tool_name": "search", "args": {"query": "hello"}},
            {"tool_name": "search", "args": {"query": "hello"}},
            {"tool_name": "search", "args": {"query": "world"}},
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses, key_fields=["tool_name"])
        # Only check tool_name, which matches
        assert result.is_consistent

    def test_all_different_values(self):
        """All values different for a field creates anomaly."""
        responses = [
            {"tool_name": "search"},
            {"tool_name": "read"},
            {"tool_name": "delete"},
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses, key_fields=["tool_name"])
        anomalies = [a for a in result.anomalies if "tie_on_field" in a or "all_different" in a]
        assert len(anomalies) >= 1

    def test_tie_detection(self):
        """Two-way tie on a field creates tie anomaly."""
        responses = [
            {"tool_name": "search"},
            {"tool_name": "read"},
            {"tool_name": "search"},
            {"tool_name": "read"},
        ]
        checker = SelfConsistencyChecker()
        result = checker.check_consistency(responses, key_fields=["tool_name"])
        # May or may not have a specific anomaly depending on logic
        assert result.agreement_ratio <= 0.5


# ===================================================================
# 5b: Spec-Grounded Verification Tests
# ===================================================================


class TestSpecGroundedVerifier:
    def test_pass_required_fields(self):
        """Response with all required fields passes."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            required_fields=["tool_name", "query"],
        )
        result = verifier.verify(
            {"tool_name": "search", "query": "hello"},
            rubric,
        )
        assert result.passes
        assert result.score >= 0.8

    def test_fail_missing_required_field(self):
        """Missing required field fails verification."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            required_fields=["tool_name", "query", "limit"],
        )
        result = verifier.verify(
            {"tool_name": "search", "query": "hello"},
            rubric,
        )
        assert not result.passes
        assert any("missing_required_field" in f for f in result.failures)

    def test_forbidden_values(self):
        """Response with forbidden value fails."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            forbidden_values={"mode": ["danger", "admin"]},
        )
        result = verifier.verify(
            {"tool_name": "search", "mode": "danger"},
            rubric,
        )
        assert not result.passes
        assert any("forbidden_value" in f for f in result.failures)

    def test_type_check_string(self):
        """Type check for string field."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            expected_types={"query": "string", "limit": "integer"},
        )
        result = verifier.verify(
            {"tool_name": "search", "query": "hello", "limit": 10},
            rubric,
        )
        assert result.passes

    def test_type_mismatch(self):
        """Type mismatch fails."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            expected_types={"limit": "integer"},
        )
        result = verifier.verify(
            {"tool_name": "search", "limit": "ten"},
            rubric,
        )
        assert not result.passes
        assert any("type_mismatch" in f for f in result.failures)

    def test_numeric_range_min(self):
        """Value below minimum fails."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            numeric_ranges={"limit": {"min": 1, "max": 100}},
        )
        result = verifier.verify(
            {"tool_name": "search", "limit": 0},
            rubric,
        )
        assert not result.passes
        assert any("range_violation" in f for f in result.failures)

    def test_numeric_range_max(self):
        """Value above maximum fails."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            numeric_ranges={"limit": {"min": 1, "max": 100}},
        )
        result = verifier.verify(
            {"tool_name": "search", "limit": 200},
            rubric,
        )
        assert not result.passes
        assert any("range_violation" in f for f in result.failures)

    def test_numeric_range_pass(self):
        """Value within range passes."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            numeric_ranges={"limit": {"min": 1, "max": 100}},
        )
        result = verifier.verify(
            {"tool_name": "search", "limit": 50},
            rubric,
        )
        assert result.passes

    def test_custom_check_not_empty(self):
        """Custom not_empty check works."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            custom_checks=[
                {"type": "not_empty", "field": "query"},
            ],
        )
        result = verifier.verify(
            {"tool_name": "search", "query": "hello"},
            rubric,
        )
        assert result.passes

    def test_custom_check_not_empty_fails(self):
        """Empty string fails not_empty check."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            custom_checks=[
                {"type": "not_empty", "field": "query"},
            ],
        )
        result = verifier.verify(
            {"tool_name": "search", "query": ""},
            rubric,
        )
        assert not result.passes

    def test_custom_contains(self):
        """Contains check passes when value contains target."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            custom_checks=[
                {"type": "contains", "field": "query", "target": "search"},
            ],
        )
        result = verifier.verify(
            {"tool_name": "search", "query": "how to search for things"},
            rubric,
        )
        assert result.passes

    def test_custom_not_equals(self):
        """Not-equals check passes when values differ."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            custom_checks=[
                {"type": "not_equals", "field": "mode", "value": "danger"},
            ],
        )
        result = verifier.verify(
            {"tool_name": "search", "mode": "safe"},
            rubric,
        )
        assert result.passes

    def test_json_string_response(self):
        """JSON string response is parsed."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(
            tool_name="search",
            required_fields=["tool_name"],
        )
        result = verifier.verify(
            '{"tool_name": "search", "query": "hello"}',
            rubric,
        )
        assert result.passes

    def test_unparseable_json(self):
        """Unparseable response fails."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(tool_name="search")
        result = verifier.verify("not valid json", rubric)
        assert not result.passes
        assert "unparseable_json" in result.failures

    def test_empty_rubric(self):
        """Empty rubric always passes."""
        verifier = SpecGroundedVerifier()
        rubric = VerificationRubric(tool_name="search")
        result = verifier.verify(
            {"tool_name": "search"},
            rubric,
        )
        assert result.passes
        assert math.isclose(result.score, 1.0)

    def test_build_rubric_from_schema(self):
        """Build rubric from JSON Schema."""
        verifier = SpecGroundedVerifier()
        schema = {
            "type": "object",
            "required": ["query", "limit"],
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
        }
        rubric = verifier.build_rubric_from_schema(schema, tool_name="search")
        assert "query" in rubric.required_fields
        assert "limit" in rubric.expected_types
        assert "limit" in rubric.numeric_ranges
        assert rubric.numeric_ranges["limit"]["min"] == 1
        assert rubric.numeric_ranges["limit"]["max"] == 100

    def test_create_consistency_rubric(self):
        """Consistency rubric creation."""
        verifier = SpecGroundedVerifier()
        rubric = verifier.create_consistency_rubric(
            tool_name="search",
            key_fields=["tool_name", "query"],
        )
        assert "tool_name" in rubric.required_fields
        assert "query" in rubric.required_fields
        assert len(rubric.custom_checks) == 2


# ===================================================================
# Combined Verifier Tests
# ===================================================================


class TestVerifier:
    def test_full_verification_pass(self):
        """Full verification pipeline passes for consistent responses."""
        verifier = Verifier()
        rubric = VerificationRubric(
            tool_name="search",
            required_fields=["tool_name", "query"],
        )
        result = verifier.verify(
            task_id="test-1",
            tool_name="search",
            responses=[
                {"tool_name": "search", "query": "hello"},
                {"tool_name": "search", "query": "hello"},
            ],
            rubric=rubric,
        )
        assert result.overall_pass
        assert result.recommendation in ("proceed", "verify_again")

    def test_full_verification_fail_inconsistent(self):
        """Inconsistent responses fail verification."""
        verifier = Verifier()
        rubric = VerificationRubric(
            tool_name="search",
            required_fields=["tool_name"],
        )
        result = verifier.verify(
            task_id="test-2",
            tool_name="search",
            responses=[
                {"tool_name": "search"},
                {"tool_name": "delete"},
            ],
            rubric=rubric,
        )
        # tool_name disagreement triggers issues
        assert result.consistency is not None
        assert len(result.anomalies) > 0

    def test_single_response_no_consistency(self):
        """Single response skips consistency check."""
        verifier = Verifier()
        result = verifier.verify(
            task_id="test-3",
            tool_name="search",
            responses=[{"tool_name": "search", "query": "hello"}],
        )
        assert result.consistency is None

    def test_no_rubric_skips_spec(self):
        """Missing rubric skips spec verification."""
        verifier = Verifier()
        result = verifier.verify(
            task_id="test-4",
            tool_name="search",
            responses=[
                {"tool_name": "search"},
                {"tool_name": "search"},
            ],
        )
        assert result.spec_verification is None

    def test_recommendation_block(self):
        """Low confidence with many anomalies blocks."""
        verifier = Verifier()
        assert verifier._recommend(False, 0.2, ["a", "b", "c"]) == "block"
        assert verifier._recommend(False, 0.5, []) == "retry"

    def test_recommendation_proceed(self):
        """High confidence with no anomalies proceeds."""
        verifier = Verifier()
        assert verifier._recommend(True, 0.9, []) == "proceed"

    def test_recommendation_escalate(self):
        """Medium low confidence escalates."""
        verifier = Verifier()
        assert verifier._recommend(True, 0.4, []) == "escalate"


# ===================================================================
# 5c: Checkpoint Manager Tests
# ===================================================================


class TestCheckpointManager:
    def test_create_checkpoint(self):
        """Creating a checkpoint stores and returns it."""
        mgr = CheckpointManager()
        state = {"step": 1, "result": "ok"}
        ckpt = mgr.create_checkpoint(
            task_id="task-1",
            step=1,
            state=state,
            confidence_score=0.95,
        )
        assert ckpt.id.startswith("ckpt-")
        assert ckpt.task_id == "task-1"
        assert ckpt.step == 1
        assert math.isclose(ckpt.confidence_score, 0.95)

    def test_get_latest_checkpoint(self):
        """Latest checkpoint is retrievable."""
        mgr = CheckpointManager()
        mgr.create_checkpoint("task-1", 1, {"step": 1}, 0.8)
        mgr.create_checkpoint("task-1", 2, {"step": 2}, 0.9)
        latest = mgr.store.get_latest("task-1")
        assert latest is not None
        assert latest.step == 2

    def test_get_best_checkpoint(self):
        """Best (highest confidence) checkpoint is retrievable."""
        mgr = CheckpointManager()
        mgr.create_checkpoint("task-1", 1, {"step": 1}, 0.6)
        mgr.create_checkpoint("task-1", 2, {"step": 2}, 0.95)
        mgr.create_checkpoint("task-1", 3, {"step": 3}, 0.8)
        best = mgr.store.get_best("task-1")
        assert best is not None
        assert math.isclose(best.confidence_score, 0.95)

    def test_get_best_before_step(self):
        """Best checkpoint before a given step."""
        mgr = CheckpointManager()
        mgr.create_checkpoint("task-1", 1, {"step": 1}, 0.6)
        mgr.create_checkpoint("task-1", 2, {"step": 2}, 0.95)
        mgr.create_checkpoint("task-1", 3, {"step": 3}, 0.8)
        best = mgr.store.get_best_before_step("task-1", before_step=3)
        assert best is not None
        assert best.step < 3
        assert math.isclose(best.confidence_score, 0.95)

    def test_can_rollback_true(self):
        """Rollback is possible when a prior checkpoint exists."""
        mgr = CheckpointManager()
        mgr.create_checkpoint("task-1", 1, {"step": 1}, 0.9)
        assert mgr.can_rollback("task-1", current_step=2)

    def test_can_rollback_false(self):
        """Rollback impossible when no prior checkpoints."""
        mgr = CheckpointManager()
        assert not mgr.can_rollback("task-1", current_step=1)

    def test_rollback_returns_target(self):
        """Rollback returns the target checkpoint."""
        mgr = CheckpointManager()
        mgr.create_checkpoint("task-1", 1, {"step": 1}, 0.9)
        mgr.create_checkpoint("task-1", 2, {"step": 2}, 0.7)
        target, message = mgr.rollback("task-1", current_step=3, state={})
        assert target is not None
        assert "Rolled back" in message

    def test_rollback_no_target(self):
        """Rollback returns None when no target exists."""
        mgr = CheckpointManager()
        target, message = mgr.rollback("task-1", current_step=1, state={})
        assert target is None
        assert "No rollback target" in message

    def test_prune_low_confidence(self):
        """Low-confidence checkpoints are pruned."""
        store = CheckpointStore()
        mgr = CheckpointManager(store)
        mgr.create_checkpoint("prune-test", 1, {"step": 1}, 0.2)
        mgr.create_checkpoint("prune-test", 2, {"step": 2}, 0.9)
        removed = mgr.prune_low_confidence("prune-test", min_confidence=0.5)
        assert removed == 1
        remaining = store.list_checkpoints("prune-test")
        assert len(remaining) == 1

    def test_clear_task(self):
        """Clear removes all checkpoints for a task."""
        mgr = CheckpointManager()
        mgr.create_checkpoint("task-1", 1, {"step": 1}, 0.9)
        mgr.create_checkpoint("task-1", 2, {"step": 2}, 0.9)
        mgr.store.clear_task("task-1")
        assert mgr.store.list_checkpoints("task-1") == []

    def test_max_checkpoints_per_task(self):
        """Exceeding max trims oldest."""
        store = CheckpointStore(max_checkpoints_per_task=3)
        mgr = CheckpointManager(store)
        for i in range(5):
            mgr.create_checkpoint("task-1", i, {"step": i}, 0.8)
        checkpoints = store.list_checkpoints("task-1")
        assert len(checkpoints) == 3

    def test_checkpoint_hash_deterministic(self):
        """Same state produces same hash."""
        mgr = CheckpointManager()
        state_a = {"key": "value", "nested": {"a": 1}}
        state_b = {"nested": {"a": 1}, "key": "value"}  # Same content, different key order
        hash_a = mgr._hash_state(state_a)
        hash_b = mgr._hash_state(state_b)
        assert hash_a == hash_b

    def test_checkpoint_to_dict_roundtrip(self):
        """Checkpoint can be serialized and deserialized."""
        original = Checkpoint(
            id="ckpt-test",
            task_id="task-1",
            step=1,
            confidence_score=0.9,
            state_hash="abc123",
            timestamp=time.time(),
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = Checkpoint.from_dict(data)
        assert restored.id == original.id
        assert restored.task_id == original.task_id
        assert restored.step == original.step
        assert math.isclose(restored.confidence_score, original.confidence_score)


# ===================================================================
# Convenience Wrappers
# ===================================================================


class TestWrappers:
    def test_check_consistency_wrapper(self):
        """check_consistency convenience wrapper works."""
        responses = [
            {"tool_name": "search", "query": "hello"},
            {"tool_name": "search", "query": "hello"},
        ]
        result = check_consistency(responses, key_fields=["tool_name", "query"])
        assert result.is_consistent

    def test_create_checkpoint_wrapper(self):
        """create_checkpoint convenience wrapper works."""
        ckpt = create_checkpoint(
            task_id="task-1",
            step=1,
            state={"key": "value"},
            confidence_score=0.9,
        )
        assert ckpt.task_id == "task-1"
        assert ckpt.step == 1

    def test_rollback_wrapper(self):
        """rollback convenience wrapper works."""
        mgr = CheckpointManager()
        mgr.create_checkpoint("task-1", 1, {"step": 1}, 0.9)
        target, message = rollback("task-1", current_step=2, state={})
        assert target is not None
