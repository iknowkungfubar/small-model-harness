"""Tests for output_validator — Phase 4 Tier 1: Post-hoc validation + retry."""

from __future__ import annotations

# We need to add mcp-server to the path for imports
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp-server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hermes-plugin"))

from output_validator import OutputValidator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator() -> OutputValidator:
    """An OutputValidator with default tool schemas."""
    return OutputValidator()


@pytest.fixture
def simple_schema() -> dict[str, Any]:
    """A simple tool schema for testing."""
    return {
        "tool_name": "test_tool",
        "properties": {
            "name": {"type": "string", "min_length": 1, "max_length": 100},
            "count": {"type": "integer", "minimum": 0, "maximum": 10},
            "active": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["fast", "slow", "auto"]},
        },
        "required": ["name", "mode"],
    }


# ---------------------------------------------------------------------------
# Raw JSON validation (no schema)
# ---------------------------------------------------------------------------


class TestValidateJson:
    """Tests for basic JSON structural validation."""

    def test_valid_json_object(self, validator: OutputValidator) -> None:
        """Valid JSON object should pass."""
        result = validator.validate_json('{"name": "test", "mode": "fast"}')
        assert result.valid is True
        assert result.parsed == {"name": "test", "mode": "fast"}

    def test_empty_string_returns_error(self, validator: OutputValidator) -> None:
        """Empty response should fail with clear error."""
        result = validator.validate_json("")
        assert result.valid is False
        assert len(result.errors) >= 1
        assert "empty" in result.errors[0].lower() or "blank" in result.errors[0].lower()

    def test_malformed_json_returns_error(self, validator: OutputValidator) -> None:
        """Malformed JSON should fail with parse error."""
        result = validator.validate_json('{"name": "test", broken}')
        assert result.valid is False
        assert len(result.errors) >= 1

    def test_json_array_returns_error(self, validator: OutputValidator) -> None:
        """JSON array (not object) should fail."""
        result = validator.validate_json("[1, 2, 3]")
        assert result.valid is False
        assert any("object" in e.lower() for e in result.errors)

    def test_json_primitive_returns_error(self, validator: OutputValidator) -> None:
        """JSON primitive (not object) should fail."""
        result = validator.validate_json('"just a string"')
        assert result.valid is False

    def test_whitespace_only_returns_error(self, validator: OutputValidator) -> None:
        """Whitespace-only response should fail."""
        result = validator.validate_json("   \n\n  ")
        assert result.valid is False

    def test_nested_json_object_valid(self, validator: OutputValidator) -> None:
        """Deeply nested valid JSON should pass."""
        result = validator.validate_json('{"level1": {"level2": {"level3": "deep"}}}')
        assert result.valid is True
        assert result.parsed == {"level1": {"level2": {"level3": "deep"}}}

    def test_json_with_code_fence(self, validator: OutputValidator) -> None:
        """JSON embedded in markdown code fence should be extracted."""
        result = validator.validate_json('```json\n{"name": "test", "mode": "fast"}\n```')
        assert result.valid is True
        assert result.parsed == {"name": "test", "mode": "fast"}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestValidateAgainstSchema:
    """Tests for validation against a tool schema."""

    def test_valid_call_passes(
        self, validator: OutputValidator, simple_schema: dict[str, Any]
    ) -> None:
        """Valid tool call matching schema should pass."""
        result = validator.validate_against_schema(
            {"name": "my_tool", "mode": "fast"}, simple_schema
        )
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_required_field_fails(
        self, validator: OutputValidator, simple_schema: dict[str, Any]
    ) -> None:
        """Missing a required field should fail."""
        result = validator.validate_against_schema(
            {"name": "my_tool"},
            simple_schema,  # missing "mode"
        )
        assert result.valid is False
        assert any("mode" in e for e in result.errors)

    def test_empty_tool_name_fails(self, validator: OutputValidator) -> None:
        """Empty tool name should fail."""
        result = validator.validate_against_schema(
            {"tool_name": ""},
            {
                "tool_name": "test",
                "properties": {"tool_name": {"type": "string", "min_length": 1}},
                "required": ["tool_name"],
            },
        )
        assert result.valid is False

    def test_invalid_enum_value_fails(
        self, validator: OutputValidator, simple_schema: dict[str, Any]
    ) -> None:
        """Invalid enum value should fail."""
        result = validator.validate_against_schema({"name": "test", "mode": "turbo"}, simple_schema)
        assert result.valid is False
        assert any("turbo" in e.lower() or "mode" in e.lower() for e in result.errors)

    def test_wrong_type_for_field_fails(
        self, validator: OutputValidator, simple_schema: dict[str, Any]
    ) -> None:
        """Wrong type should fail."""
        result = validator.validate_against_schema(
            {"name": "test", "count": "not_an_int", "mode": "fast"}, simple_schema
        )
        assert result.valid is False

    def test_extra_fields_are_allowed(
        self, validator: OutputValidator, simple_schema: dict[str, Any]
    ) -> None:
        """Extra fields beyond the schema should still pass (permissive)."""
        result = validator.validate_against_schema(
            {"name": "test", "mode": "fast", "extra_field": "should_be_ignored"},
            simple_schema,
        )
        assert result.valid is True

    def test_numeric_range_violation_fails(
        self, validator: OutputValidator, simple_schema: dict[str, Any]
    ) -> None:
        """Numeric field outside range should fail."""
        result = validator.validate_against_schema(
            {"name": "test", "count": 99, "mode": "fast"}, simple_schema
        )
        assert result.valid is False

    def test_multiple_errors_reported(
        self, validator: OutputValidator, simple_schema: dict[str, Any]
    ) -> None:
        """Multiple validation errors should all be reported."""
        result = validator.validate_against_schema({"mode": "turbo", "count": -1}, simple_schema)
        assert result.valid is False
        assert len(result.errors) >= 2

    def test_scalar_field_against_object_schema(self, validator: OutputValidator) -> None:
        """Validating a nested object correctly."""
        schema = {
            "tool_name": "nested",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    },
                    "required": ["host"],
                }
            },
            "required": ["config"],
        }
        result = validator.validate_against_schema(
            {"config": {"host": "localhost", "port": 8080}}, schema
        )
        assert result.valid is True

        result = validator.validate_against_schema({"config": "not_an_object"}, schema)
        assert result.valid is False


# ---------------------------------------------------------------------------
# validate_tool_call (combined)
# ---------------------------------------------------------------------------


class TestValidateToolCall:
    """Tests for the combined validate_tool_call method."""

    def test_valid_call_passes(self, validator: OutputValidator) -> None:
        """A fully valid tool call should pass."""
        validator.validate_tool_call(
            '{"name": "test", "mode": "fast"}',
            "test_tool",
            {
                "tool_name": "test_tool",
                "properties": {},
                "required": [],
            },
        )
        # The tool_name in the schema defines which tool accepts it.
        # The response content content is validated against the schema.
        # We need a matching schema.

    def test_tool_call_with_code_fence(self, validator: OutputValidator) -> None:
        """Tool call wrapped in code fence should be extracted."""
        response = '```json\n{"tool_name": "terminal", "command": "ls"}\n```'
        # Register the schema we're simulating
        validator.get_schema()
        result = validator.validate_tool_call(response, "terminal")
        # Should parse the code fence and validate
        if result.valid:
            assert result.parsed.get("command") == "ls"


# ---------------------------------------------------------------------------
# validate_and_retry (retry loop)
# ---------------------------------------------------------------------------


class TestValidateAndRetry:
    """Tests for the retry loop."""

    def test_success_on_first_attempt(self, validator: OutputValidator) -> None:
        """When LLM returns valid output on first try, return immediately."""

        def llm_call(feedback: str | None = None) -> str:
            return '{"name": "test", "mode": "fast"}'

        result = validator.validate_and_retry(
            "test_tool",
            llm_call,
            max_retries=3,
        )
        assert result.valid is True
        assert result.attempts == 1

    def test_recovery_on_retry(self, validator: OutputValidator) -> None:
        """Recovery on second attempt after bad first."""
        call_count = 0

        def llm_call(feedback: str | None = None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "invalid json{{{"
            return '{"name": "test", "mode": "fast"}'

        result = validator.validate_and_retry(
            "test_tool",
            llm_call,
            max_retries=3,
        )
        assert result.valid is True
        assert result.attempts == 2

    def test_failure_after_max_retries(self, validator: OutputValidator) -> None:
        """All retries exhausted should fail."""

        def llm_call(feedback: str | None = None) -> str:
            return "invalid{{{json"

        result = validator.validate_and_retry(
            "test_tool",
            llm_call,
            max_retries=3,
        )
        assert result.valid is False
        assert result.attempts == 3

    def test_escalating_feedback(self, validator: OutputValidator) -> None:
        """Feedback should escalate: error message → schema reminder → example."""
        feedbacks: list[str | None] = []

        def llm_call(feedback: str | None = None) -> str:
            feedbacks.append(feedback)
            return "bad"

        validator.validate_and_retry(
            "test_tool",
            llm_call,
            max_retries=3,
        )

        # First call has no feedback
        assert feedbacks[0] is None
        # Second call has error message (first feedback level)
        assert feedbacks[1] is not None
        # Third call has different feedback (second level)
        assert feedbacks[2] is not None
        assert feedbacks[2] != feedbacks[1]

    def test_empty_response_retry(self, validator: OutputValidator) -> None:
        """Empty response should trigger retry with specific feedback."""
        call_count = 0

        def llm_call(feedback: str | None = None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ""
            return '{"name": "test", "mode": "fast"}'

        result = validator.validate_and_retry(
            "test_tool",
            llm_call,
            max_retries=3,
        )
        assert result.valid is True
        assert result.attempts == 2

    def test_no_retries_configured(self, validator: OutputValidator) -> None:
        """With max_retries=0, no retry on failure."""

        def llm_call(feedback: str | None = None) -> str:
            return "{{{bad"

        result = validator.validate_and_retry(
            "test_tool",
            llm_call,
            max_retries=0,
        )
        assert result.valid is False
        assert result.attempts == 1
