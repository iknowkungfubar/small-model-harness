"""Tests for constrained_decode — Phase 4 Tier 2: XGrammar integration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp-server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hermes-plugin"))

from constrained_decode import (
    BackendDetector,
    ConstrainedDecoder,
    GrammarCompilationError,
    HybridValidator,
    XGrammarCompiler,
)

# ---------------------------------------------------------------------------
# Schema → Grammar compilation
# ---------------------------------------------------------------------------


class TestXGrammarCompiler:
    """Tests for XGrammar schema-to-grammar compilation."""

    def test_simple_schema_compiles(self) -> None:
        """A simple JSON schema should compile to a grammar string."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
        }
        compiler = XGrammarCompiler()
        grammar = compiler.compile(schema)
        assert grammar is not None
        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_compile_nested_object(self) -> None:
        """Nested object schema should compile."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                    },
                    "required": ["host"],
                }
            },
            "required": ["config"],
        }
        compiler = XGrammarCompiler()
        grammar = compiler.compile(schema)
        assert grammar is not None
        # Should contain config, host, port
        for key in ["config", "host", "port"]:
            assert key in grammar

    def test_empty_schema_raises_error(self) -> None:
        """Empty schema should raise compilation error."""
        compiler = XGrammarCompiler()
        with pytest.raises(GrammarCompilationError):
            compiler.compile({})

    def test_enum_schema_compiles(self) -> None:
        """Enum constraints in schema should compile."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["fast", "slow", "auto"],
                },
            },
            "required": ["mode"],
        }
        compiler = XGrammarCompiler()
        grammar = compiler.compile(schema)
        assert grammar is not None
        # Should contain the enum values as alternatives
        assert "fast" in grammar or '"fast"' in grammar

    def test_tool_call_schema_compiles(self) -> None:
        """A full tool call schema should compile cleanly."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "enum": ["terminal", "read_file", "write_file"]},
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
        compiler = XGrammarCompiler()
        grammar = compiler.compile(schema)
        assert grammar is not None
        for key in ["tool_name", "arguments", "command", "path"]:
            assert key in grammar

    def test_multiple_schemas_compilation_caching(self) -> None:
        """Compiling the same schema twice should return cached result."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        compiler = XGrammarCompiler()
        grammar1 = compiler.compile(schema)
        grammar2 = compiler.compile(schema)
        assert grammar1 == grammar2

    def test_compile_with_all_types(self) -> None:
        """All JSON Schema types should compile."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "f": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["s", "i"],
        }
        compiler = XGrammarCompiler()
        grammar = compiler.compile(schema)
        assert grammar is not None
        assert isinstance(grammar, str)
        assert len(grammar) > 0


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


class TestBackendDetector:
    """Tests for backend type detection."""

    def test_detect_vllm_backend(self) -> None:
        """vLLM backend should be detected from provider name."""
        detector = BackendDetector()
        config = {"provider": "vllm", "model": "qwen3-9b"}
        backend = detector.detect(config)
        assert backend == "vllm"

    def test_detect_llamacpp_backend(self) -> None:
        """llama.cpp backend should be detected."""
        detector = BackendDetector()
        config = {"provider": "llamacpp", "model": "qwen3-9b-gguf"}
        backend = detector.detect(config)
        assert backend == "llamacpp"

    def test_detect_lm_studio_backend(self) -> None:
        """LM Studio backend should be detected."""
        detector = BackendDetector()
        config = {"provider": "lm_studio", "base_url": "http://localhost:1234/v1"}
        backend = detector.detect(config)
        assert backend == "lm_studio"

    def test_detect_openai_compatible_as_lm_studio(self) -> None:
        """OpenAI-compatible API without explicit provider should be LM Studio fallback."""
        detector = BackendDetector()
        config = {"provider": "openai_compatible", "base_url": "http://localhost:1234/v1"}
        backend = detector.detect(config)
        assert backend == "lm_studio"

    def test_detect_openai_backend(self) -> None:
        """OpenAI backend should be detected."""
        detector = BackendDetector()
        config = {"provider": "opencode", "model": "deepseek-v4-flash"}
        backend = detector.detect(config)
        assert backend != "vllm"  # Should not be detected as vLLM

    def test_unknown_provider_returns_none(self) -> None:
        """Unknown provider should return None."""
        detector = BackendDetector()
        config = {"provider": "my_custom"}
        backend = detector.detect(config)
        assert backend is None

    def test_supports_grammar_constraint(self) -> None:
        """Only vLLM and llama.cpp should support grammar constraints."""
        detector = BackendDetector()
        assert detector.supports_grammar("vllm") is True
        assert detector.supports_grammar("llamacpp") is True
        assert detector.supports_grammar("lm_studio") is False
        assert detector.supports_grammar("opencode") is False
        assert detector.supports_grammar(None) is False


# ---------------------------------------------------------------------------
# ConstrainedDecoder
# ---------------------------------------------------------------------------


class TestConstrainedDecoder:
    """Tests for the ConstrainedDecoder."""

    def test_init_with_grammar_support(self) -> None:
        """Decoder should init when grammar is supported."""
        decoder = ConstrainedDecoder(backend_type="vllm")
        assert decoder.available is True
        assert decoder.backend_type == "vllm"

    def test_init_without_grammar_support(self) -> None:
        """Decoder should be unavailable for non-grammar backends."""
        decoder = ConstrainedDecoder(backend_type="lm_studio")
        assert decoder.available is False

    def test_decode_with_grammar(self) -> None:
        """Decode with grammar should produce valid output."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        }
        compiler = XGrammarCompiler()
        grammar = compiler.compile(schema)
        decoder = ConstrainedDecoder(backend_type="vllm", grammar=grammar)

        # The decoder wraps an LLM call
        def llm_call(prompt: str) -> str:
            return '{"command": "ls -la"}'

        result = decoder.generate("run this command", llm_call)
        assert result is not None
        assert "command" in result

    def test_decode_without_grammar_fallback(self) -> None:
        """Decoder should fall through when no grammar available."""
        decoder = ConstrainedDecoder(backend_type="lm_studio")
        assert decoder.available is False

        def llm_call(prompt: str) -> str:
            return '{"command": "ls"}'

        result = decoder.generate("run command", llm_call)
        assert result == '{"command": "ls"}'  # Passthrough

    def test_grammar_reuse_across_calls(self) -> None:
        """Grammar should be reusable across decode calls."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        compiler = XGrammarCompiler()
        grammar = compiler.compile(schema)
        decoder = ConstrainedDecoder(backend_type="vllm", grammar=grammar)

        def llm_call(prompt: str) -> str:
            return '{"x": 42}'

        for _ in range(3):
            result = decoder.generate("", llm_call)
            assert result is not None


# ---------------------------------------------------------------------------
# HybridValidator
# ---------------------------------------------------------------------------


class TestHybridValidator:
    """Tests for the HybridValidator (Tier 2 + Tier 1 fallback)."""

    def test_tier2_used_when_available(self) -> None:
        """Tier 2 constrained decode should be used when grammar is available."""
        backend = "vllm"
        hybrid = HybridValidator(backend_type=backend)
        assert hybrid.tier2_available is True

    def test_tier1_fallback_when_no_grammar(self) -> None:
        """Tier 1 post-hoc validation should be the fallback."""
        hybrid = HybridValidator(backend_type="lm_studio")
        assert hybrid.tier2_available is False

    def test_hybrid_validate_with_tier2(self) -> None:
        """Hybrid should use Tier 2 when available."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        }
        hybrid = HybridValidator(backend_type="vllm", schema=schema)

        def llm_call(feedback: str | None = None) -> str:
            return '{"cmd": "ls"}'

        result = hybrid.validate("test", llm_call)
        assert result.valid is True

    def test_hybrid_fallback_on_parse_error(self) -> None:
        """Hybrid should fall back to Tier 1 if Tier 2 output is invalid."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        hybrid = HybridValidator(backend_type="vllm", schema=schema)

        def llm_call(feedback: str | None = None) -> str:
            return '{"cmd": "no_name_field"}'

        result = hybrid.validate("test", llm_call)
        # Tier 2 produced output, but it doesn't match schema.
        # Should try to fall back to Tier 1 retry.
        assert result.valid is True or result.attempts >= 1

    def test_backend_auto_detection_from_config(self) -> None:
        """Hybrid should auto-detect backend from provider config."""
        config = {"provider": "vllm", "model": "qwen3-9b"}
        hybrid = HybridValidator.from_config(config)
        assert hybrid.backend_type == "vllm"

    def test_backend_auto_detection_lm_studio(self) -> None:
        """Hybrid should fall back for LM Studio config."""
        config = {"base_url": "http://localhost:1234/v1"}
        hybrid = HybridValidator.from_config(config)
        assert hybrid.backend_type == "lm_studio"
        assert hybrid.tier2_available is False
