"""E2E integration tests for the full small-model-harness validation pipeline.

Tests the complete flow: raw output → JSON parsing → schema validation →
retry with feedback → constrained decode (Tier 2) → hybrid fallback (Tier 2→1).

All tests use mock LLM calls — no real backend needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hermes-plugin"))

from constrained_decode import (
    BackendDetector,
    ConstrainedDecoder,
    HybridValidator,
    XGrammarCompiler,
)
from output_validator import OutputValidator

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def output_validator() -> OutputValidator:
    return OutputValidator()


@pytest.fixture
def xgrammar() -> XGrammarCompiler:
    return XGrammarCompiler()


@pytest.fixture
def terminal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string", "min_length": 1},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
            "workdir": {"type": "string"},
        },
        "required": ["command"],
    }


# =========================================================================
# E2E Test 1: Full validation pipeline — valid output
# =========================================================================


class TestFullPipeline:
    """End-to-end validation of a complete tool call."""

    def test_valid_terminal_call_through_full_pipeline(
        self,
        output_validator: OutputValidator,
        terminal_schema: dict[str, Any],
    ) -> None:
        """A valid terminal command should pass all stages."""
        raw_output = '{"command": "ls -la", "timeout": 30}'

        # Stage 1: Parse JSON
        parsed = json.loads(raw_output)
        assert isinstance(parsed, dict)
        assert "command" in parsed

        # Stage 2: Validate against schema
        result = output_validator.validate_against_schema(parsed, terminal_schema)
        assert result.valid is True
        assert len(result.errors) == 0

        # Stage 3: Full tool call validation (raw response = just the args)
        tc_result = output_validator.validate_tool_call(json.dumps(parsed), "terminal")
        assert tc_result.valid is True

    def test_invalid_output_rejected_by_pipeline(
        self,
        output_validator: OutputValidator,
        terminal_schema: dict[str, Any],
    ) -> None:
        """Malformed output should be caught at every stage."""
        raw_output = '{"command": ""}'  # Empty command (min_length violation)

        # Stage 1: JSON is valid
        parsed = json.loads(raw_output)
        assert isinstance(parsed, dict)

        # Stage 2: Schema validation catches it
        result = output_validator.validate_against_schema(parsed, terminal_schema)
        assert result.valid is False
        assert any("command" in e.lower() for e in result.errors)

    def test_retry_recovers_from_malformed_output(
        self,
        output_validator: OutputValidator,
    ) -> None:
        """validate_and_retry should recover from initial malformed output."""
        call_count: int = 0

        def llm_call(feedback: str | None = None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not json at all"
            return '{"command": "ls -la", "timeout": 30}'

        result = output_validator.validate_and_retry(
            tool_name="terminal",
            llm_call=llm_call,
            max_retries=3,
        )
        assert result.valid is True
        assert call_count == 2
        assert result.attempts == 2


# =========================================================================
# E2E Test 2: XGrammar → constrained decode → parsed output
# =========================================================================


class TestDecodePipeline:
    """End-to-end constrained decode with grammar."""

    def test_compile_then_decode_then_validate(
        self,
        xgrammar: XGrammarCompiler,
        terminal_schema: dict[str, Any],
    ) -> None:
        """Full Tier 2 pipeline: compile grammar → decode → validate."""
        grammar = xgrammar.compile(terminal_schema)
        assert grammar is not None
        assert "command" in grammar
        assert "timeout" in grammar

        decoder = ConstrainedDecoder(backend_type="vllm", grammar=grammar)
        assert decoder.available is True

        def llm_call(prompt: str) -> str:
            return '{"command": "ls -la", "timeout": 30}'

        result = decoder.generate("run command", llm_call)
        assert isinstance(result, dict)
        assert result.get("command") == "ls -la"

    def test_lm_studio_passthrough_no_grammar(
        self,
        terminal_schema: dict[str, Any],
    ) -> None:
        """LM Studio backend should skip grammar and pass through."""
        decoder = ConstrainedDecoder(backend_type="lm_studio")
        assert decoder.available is False

        def llm_call(prompt: str) -> str:
            return '{"command": "ps aux"}'

        result = decoder.generate("run command", llm_call)
        assert result == '{"command": "ps aux"}'

    def test_backend_detection_grammar_support(
        self,
    ) -> None:
        """Backend detector should correctly identify grammar support."""
        assert BackendDetector.supports_grammar("vllm") is True
        assert BackendDetector.supports_grammar("llamacpp") is True
        assert BackendDetector.supports_grammar("lm_studio") is False
        assert BackendDetector.supports_grammar(None) is False

    def test_vllm_decode_tool_call(
        self,
        xgrammar: XGrammarCompiler,
    ) -> None:
        """Full tool call schema should compile and produce valid output."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "enum": ["terminal", "read_file"]},
                "arguments": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
            "required": ["tool_name", "arguments"],
        }
        grammar = xgrammar.compile(schema)
        assert grammar is not None
        assert "terminal" in grammar

        decoder = ConstrainedDecoder(backend_type="vllm", grammar=grammar)

        def llm_call(prompt: str) -> str:
            return '{"tool_name": "terminal", "arguments": {"command": "ls"}}'

        result = decoder.generate("", llm_call)
        assert isinstance(result, dict)
        assert result.get("tool_name") == "terminal"


# =========================================================================
# E2E Test 3: Hybrid Tier 2 → Tier 1 fallback
# =========================================================================


class TestHybridPipeline:
    """Hybrid validation with graceful fallback."""

    def test_tier2_success_path(
        self,
    ) -> None:
        """Tier 2 constrained decode produces valid output."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        }
        hybrid = HybridValidator(backend_type="vllm", schema=schema)
        assert hybrid.tier2_available is True

        def llm_call(feedback: str | None = None) -> str:
            return '{"cmd": "ls"}'

        result = hybrid.validate("test", llm_call)
        assert result.valid is True
        assert result.attempts == 1

    def test_tier2_fallback_to_tier1_on_schema_mismatch(
        self,
        output_validator: OutputValidator,
    ) -> None:
        """When Tier 2 output doesn't match schema, fall back to Tier 1."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        hybrid = HybridValidator(
            backend_type="vllm",
            schema=schema,
            output_validator=output_validator,
        )

        call_count: int = 0

        def llm_call(feedback: str | None = None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Tier 2 output — wrong field name
                return '{"cmd": "no_name_field"}'
            # Tier 1 retry with feedback — correct output
            return '{"name": "corrected"}'

        result = hybrid.validate("test", llm_call)
        assert result.valid is True
        assert call_count >= 1

    def test_lm_studio_uses_tier1_only(
        self,
        output_validator: OutputValidator,
    ) -> None:
        """LM Studio should skip Tier 2 entirely."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        hybrid = HybridValidator(
            backend_type="lm_studio",
            schema=schema,
            output_validator=output_validator,
        )
        assert hybrid.tier2_available is False

        def llm_call(feedback: str | None = None) -> str:
            return '{"x": 42}'

        result = hybrid.validate("test", llm_call)
        assert result.valid is True


# =========================================================================
# E2E Test 4: Plugin integration (module-level)
# =========================================================================


class TestPluginIntegration:
    """Verify that the plugin __init__.py has the Phase 4 wiring."""

    PLUGIN_INIT = Path(__file__).parent.parent / "hermes-plugin" / "__init__.py"

    def test_output_validator_imported_in_plugin(self) -> None:
        """__init__.py should import output_validator."""
        init_text = self.PLUGIN_INIT.read_text()
        assert "output_validator" in init_text
        assert "_HAS_OUTPUT_VALIDATOR" in init_text

    def test_output_validator_initialized_in_ensure(self) -> None:
        """_ensure_initialized should create OutputValidator instance."""
        init_text = self.PLUGIN_INIT.read_text()
        assert "OutputValidator" in init_text
        assert "_output_validator" in init_text

    def test_output_validator_in_component_list(self) -> None:
        """register() should include output_validator in components."""
        init_text = self.PLUGIN_INIT.read_text()
        assert "output_validator(phase4)" in init_text
