"""Tests for Addition D — Token Verifier (LettuceDetect-inspired).

arXiv 2502.17125: Lightweight encoder-based hallucination detector
that scores each token against source evidence.
"""

from __future__ import annotations

import math
import pytest
from pathlib import Path

_here = Path(__file__).resolve().parent
_server_dir = _here.parent / "mcp-server"
import sys

sys.path.insert(0, str(_server_dir))

from token_verifier import (
    TokenVerifier,
    TokenVerifierConfig,
    check_span_grounding,
    compute_evidence_score,
    TokenSpan,
)


class TestCheckSpanGrounding:
    def test_supported_span(self):
        """Span present in evidence gets high score."""
        span = "Python is a programming language"
        evidence = "Python is a programming language created by Guido van Rossum"
        score, details = check_span_grounding(span, evidence)
        assert score > 0.5
        assert details["match_found"]

    def test_unsupported_span(self):
        """Span NOT in evidence gets low score."""
        span = "Java is a compiled language"
        evidence = "Python is a programming language created by Guido van Rossum"
        score, details = check_span_grounding(span, evidence)
        assert score < 0.5
        assert not details["match_found"]

    def test_partial_support(self):
        """Span with partial overlap gets moderate score."""
        span = "Python was created in the 1990s"
        evidence = "Python was created in 1991 by Guido van Rossum"
        score, details = check_span_grounding(span, evidence)
        assert 0.0 <= score <= 1.0

    def test_empty_span(self):
        """Empty span returns 1.0 (trivial)."""
        score, details = check_span_grounding("", "some evidence here")
        assert score == 1.0

    def test_empty_evidence(self):
        """Empty evidence with non-empty span returns 0.0."""
        score, details = check_span_grounding("some span", "")
        assert score == 0.0


class TestComputeEvidenceScore:
    def test_all_supported(self):
        """All spans supported = high aggregate score."""
        spans = [
            TokenSpan(text="Python", score=0.9, supported=True),
            TokenSpan(text="programming", score=0.8, supported=True),
        ]
        agg = compute_evidence_score(spans)
        assert agg > 0.7

    def test_all_unsupported(self):
        """No spans supported = low aggregate score."""
        spans = [
            TokenSpan(text="Java", score=0.1, supported=False),
            TokenSpan(text="compiled", score=0.2, supported=False),
        ]
        agg = compute_evidence_score(spans)
        assert agg < 0.4

    def test_mixed(self):
        """Mixed support produces intermediate score."""
        spans = [
            TokenSpan(text="Python", score=0.9, supported=True),
            TokenSpan(text="Java", score=0.2, supported=False),
        ]
        agg = compute_evidence_score(spans)
        assert 0.3 < agg < 0.8

    def test_empty(self):
        """Empty list returns 1.0."""
        agg = compute_evidence_score([])
        assert agg == 1.0


class TestTokenVerifier:
    def test_rag_output_verified(self):
        """RAG output grounded in evidence scores high."""
        verifier = TokenVerifier()
        result = verifier.verify(
            response="Python is a programming language created by Guido van Rossum.",
            evidence="Python is a programming language created by Guido van Rossum in 1991.",
        )
        assert result["passed"]
        assert result["evidence_score"] > 0.5

    def test_hallucinated_output(self):
        """Output not in evidence scores lower."""
        verifier = TokenVerifier()
        result = verifier.verify(
            response="Lua is a systems programming language with manual memory management.",
            evidence="Lua is a lightweight scripting language. "
            "Systems programming languages include C and Rust.",
        )
        assert isinstance(result["evidence_score"], float)

    def test_non_rag_bypass(self):
        """Non-RAG outputs (no evidence) bypass verification."""
        verifier = TokenVerifier()
        result = verifier.verify(response="Just a simple response.", evidence=None)
        assert result["passed"]
        assert result["skipped"]

    def test_empty_response(self):
        """Empty response passes trivially."""
        verifier = TokenVerifier()
        result = verifier.verify(response="", evidence="some evidence")
        assert result["passed"]

    def test_configured_threshold(self):
        """Configurable pass threshold works."""
        verifier = TokenVerifier(TokenVerifierConfig(pass_threshold=0.1))
        result = verifier.verify(
            response="Python is a programming language.",
            evidence="Python is a programming language.",
        )
        assert result["passed"]

    def test_supported_spans_count(self):
        """Spans extraction and counting works."""
        verifier = TokenVerifier()
        result = verifier.verify(
            response="Python is a programming language created by Guido.",
            evidence="Python is a programming language created by Guido van Rossum.",
        )
        assert result["total_spans"] > 0
        assert result["supported_spans"] > 0
