"""Addition D — Token Verifier (LettuceDetect-inspired).

arXiv 2502.17125: Lightweight encoder-based hallucination detector
that scores each token or span against source evidence. Designed for
RAG outputs where the generated text should be grounded in provided
evidence documents.

This module implements:
1. Span-level grounding check: substring matching + similarity heuristics.
2. Aggregate evidence scoring across multiple spans.
3. A TokenVerifier class that wraps the verification logic with
   configurable thresholds and non-RAG bypass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenSpan:
    """A token or phrase span with associated grounding score.

    Attributes:
        text: The span text.
        score: Grounding score (0.0-1.0).
        supported: Whether this span is supported by evidence.

    """

    text: str
    score: float = 0.0
    supported: bool = False


@dataclass
class TokenVerifierConfig:
    """Configuration for the token verifier.

    Attributes:
        pass_threshold: Minimum aggregate evidence score to pass.
        span_min_length: Minimum character length for a span to be checked.
        max_spans: Maximum number of spans to extract.
        overlap_threshold: Fractional overlap needed to consider a span supported.

    """

    pass_threshold: float = 0.5
    span_min_length: int = 5
    max_spans: int = 10
    overlap_threshold: float = 0.3


def extract_spans(text: str, min_length: int = 5, max_spans: int = 10) -> list[str]:
    """Extract candidate spans from text for grounding verification.

    Splits on sentence boundaries and selects substantive spans.

    Args:
        text: The text to extract spans from.
        min_length: Minimum character length for a span.
        max_spans: Maximum number of spans.

    Returns:
        List of span strings.

    """
    if not text:
        return []

    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    spans = [p.strip() for p in parts if len(p.strip()) >= min_length]

    if not spans:
        # Fall back to phrase-level split
        spans = [p.strip() for p in re.split(r"[,;:]", text) if len(p.strip()) >= min_length]

    return spans[:max_spans]


STOP_WORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "by",
    "with",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "not",
    "no",
    "nor",
    "so",
    "as",
    "if",
    "than",
    "then",
    "also",
    "very",
    "just",
    "about",
}


def check_span_grounding(span: str, evidence: str) -> tuple[float, dict[str, Any]]:
    """Check if a span is grounded in the provided evidence.

    Uses multiple heuristics:
    1. Exact substring match.
    2. Word overlap fraction (Jaccard-like, stop-word filtered).
    3. Content word overlap.

    Args:
        span: The span text to check.
        evidence: The evidence text to check against.

    Returns:
        Tuple of (score, details_dict).

    """
    if not span:
        return 1.0, {"match_found": True, "method": "empty_span"}
    if not evidence:
        return 0.0, {"match_found": False, "method": "no_evidence"}

    span_lower = span.lower().strip()
    evidence_lower = evidence.lower().strip()

    # 1. Exact substring match
    if span_lower in evidence_lower:
        return 1.0, {"match_found": True, "method": "exact_substring"}

    # 2. Word overlap with stop word filtering
    raw_span_words = set(re.findall(r"\w+", span_lower))
    raw_evidence_words = set(re.findall(r"\w+", evidence_lower))

    span_words = raw_span_words - STOP_WORDS
    evidence_words = raw_evidence_words - STOP_WORDS

    if not span_words:
        # All stop words: rely on content-length words
        content_words = {w for w in raw_span_words if len(w) > 3}
        if not content_words:
            return 0.5, {"match_found": False, "method": "all_common_words"}
        span_words = content_words

    jaccard = len(span_words & evidence_words) / len(span_words)

    # 3. Content word overlap (words longer than 3 chars)
    content_words = {w for w in span_words if len(w) > 3}
    content_overlap = 0.0
    if content_words:
        content_overlap = len(content_words & evidence_words) / len(content_words)

    combined = max(jaccard, content_overlap)
    match_found = combined >= 0.5

    return combined, {
        "match_found": match_found,
        "method": "word_overlap",
        "jaccard": jaccard,
        "content_overlap": content_overlap,
    }


def compute_evidence_score(spans: list[TokenSpan]) -> float:
    """Compute aggregate evidence score from a list of verified spans.

    Uses weighted average: supported spans contribute their score,
    unsupported spans penalize.

    Args:
        spans: List of verified TokenSpan objects.

    Returns:
        Aggregate score between 0.0 and 1.0.

    """
    if not spans:
        return 1.0

    total = 0.0
    for s in spans:
        if s.supported:
            total += s.score
        else:
            total += (1.0 - s.score) * 0.3  # penalty for unsupported

    return max(0.0, min(1.0, total / len(spans)))


@dataclass
class TokenVerifier:
    """Lightweight token/span verifier for RAG outputs.

    Usage:
        verifier = TokenVerifier()
        result = verifier.verify(
            response="Python is a programming language created by Guido.",
            evidence="Python is a programming language created by Guido van Rossum.",
        )
        if result["passed"]:
            print("Output is grounded in evidence")
    """

    config: TokenVerifierConfig = field(default_factory=TokenVerifierConfig)
    _verification_count: int = 0

    def verify(
        self,
        response: str,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        """Verify a response against evidence.

        Args:
            response: The generated response text.
            evidence: Evidence text to ground against. None or empty bypasses.

        Returns:
            Dict with keys:
                - passed (bool): whether the check passes
                - evidence_score (float): aggregate evidence score
                - skipped (bool): true if verification was skipped (no evidence)
                - total_spans (int): number of spans checked
                - supported_spans (int): number of supported spans
                - spans (list): per-span details

        """
        # Bypass for non-RAG outputs
        if not evidence:
            return {
                "passed": True,
                "evidence_score": 1.0,
                "skipped": True,
                "total_spans": 0,
                "supported_spans": 0,
                "spans": [],
            }

        if not response:
            return {
                "passed": True,
                "evidence_score": 1.0,
                "skipped": False,
                "total_spans": 0,
                "supported_spans": 0,
                "spans": [],
            }

        # Extract and verify spans
        raw_spans = extract_spans(response, self.config.span_min_length, self.config.max_spans)
        verified_spans: list[TokenSpan] = []

        for span_text in raw_spans:
            score, details = check_span_grounding(span_text, evidence)
            ts = TokenSpan(
                text=span_text,
                score=score,
                supported=details.get("match_found", False),
            )
            verified_spans.append(ts)

        evidence_score = compute_evidence_score(verified_spans)

        self._verification_count += 1

        return {
            "passed": evidence_score >= self.config.pass_threshold,
            "evidence_score": evidence_score,
            "skipped": False,
            "total_spans": len(verified_spans),
            "supported_spans": sum(1 for s in verified_spans if s.supported),
            "spans": [
                {"text": s.text[:80], "score": round(s.score, 3), "supported": s.supported}
                for s in verified_spans
            ],
        }
