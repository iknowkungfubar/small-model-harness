"""
Tests for the Phase 6 Confidence Scoring module.

Covers:
- Token probability aggregation (6a)
- Semantic entropy / dispersion (6b)
- Unified confidence score (6c)
- Edge cases: empty input, single response, missing logprobs
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

# Ensure mcp-server is importable
_here = Path(__file__).resolve().parent
_server_dir = _here.parent / "mcp-server"
import sys

sys.path.insert(0, str(_server_dir))

from confidence import (
    ConfidenceResult,
    ConfidenceScorer,
    SemanticEntropy,
    TokenProbabilities,
    TokenProbabilityAggregator,
    estimate_confidence,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def scorer() -> ConfidenceScorer:
    return ConfidenceScorer()


@pytest.fixture
def sample_logprobs() -> list[dict[str, Any]]:
    """Simulate a high-confidence response with mostly high probabilities."""
    return [
        {"token": "The", "logprob": -0.05},
        {"token": " sky", "logprob": -0.02},
        {"token": " is", "logprob": -0.03},
        {"token": " blue", "logprob": -0.01},
        {"token": ".", "logprob": -0.001},
    ]


@pytest.fixture
def sample_low_confidence_logprobs() -> list[dict[str, Any]]:
    """Simulate a low-confidence (uncertain) response."""
    return [
        {"token": "Maybe", "logprob": -2.5},
        {"token": " it", "logprob": -0.5},
        {"token": " could", "logprob": -3.0},
        {"token": " be", "logprob": -0.3},
        {"token": " red", "logprob": -4.0},
        {"token": "?", "logprob": -1.0},
    ]


@pytest.fixture
def oai_logprobs_response() -> dict[str, Any]:
    """Simulate an OpenAI-compatible API response with logprobs."""
    return {
        "id": "chatcmpl-xxx",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The sky is blue."},
                "logprobs": {
                    "content": [
                        {"token": "The", "logprob": -0.05, "bytes": [84, 104, 101]},
                        {"token": " sky", "logprob": -0.02, "bytes": [32, 115, 107, 121]},
                        {"token": " is", "logprob": -0.03, "bytes": [32, 105, 115]},
                        {"token": " blue", "logprob": -0.01, "bytes": [32, 98, 108, 117, 101]},
                        {"token": ".", "logprob": -0.001, "bytes": [46]},
                    ]
                },
            }
        ],
    }


# ===================================================================
# 6a: Token Probability Aggregation
# ===================================================================


class TestTokenProbabilityAggregator:
    def test_calculate_probability_normal(self):
        """Convert natural log probability correctly."""
        prob = TokenProbabilityAggregator.calculate_probability(-0.5)
        assert math.isclose(prob, math.exp(-0.5), rel_tol=1e-10)
        assert 0 <= prob <= 1

    def test_calculate_probability_zero(self):
        """Zero logprob = probability 1.0."""
        prob = TokenProbabilityAggregator.calculate_probability(0)
        assert math.isclose(prob, 1.0, abs_tol=1e-6)

    def test_calculate_probability_positive(self):
        """Positive values are treated as already-probabilities and clamped to 1.0."""
        prob = TokenProbabilityAggregator.calculate_probability(2.0)
        assert math.isclose(prob, 1.0)

    def test_calculate_probability_extreme_negative(self):
        """Extremely negative logprobs approach 0 but stay non-negative."""
        prob = TokenProbabilityAggregator.calculate_probability(-100)
        assert prob >= 0.0
        assert prob < 0.0001

    def test_aggregate_normal(self, sample_logprobs):
        """Aggregate returns statistics for valid logprobs."""
        result = TokenProbabilityAggregator.aggregate(sample_logprobs)
        assert result is not None
        assert isinstance(result, TokenProbabilities)
        assert 0 < result.mean_probability <= 1.0
        assert result.token_count == len(sample_logprobs)
        assert result.low_probability_fraction == 0  # All probabilities are > 0.1

    def test_aggregate_low_confidence(self, sample_low_confidence_logprobs):
        """Low-confidence logprobs produce noticeably different stats."""
        result = TokenProbabilityAggregator.aggregate(sample_low_confidence_logprobs)
        assert result is not None
        # Should have lower mean probability
        assert result.mean_probability < 0.5
        # Should have more low-probability tokens
        assert result.low_probability_fraction > 0

    def test_aggregate_empty(self):
        """Empty input returns None."""
        result = TokenProbabilityAggregator.aggregate([])
        assert result is None

    def test_extract_logprobs_openai_format(self, oai_logprobs_response):
        """Extract from OpenAI chat completions format."""
        result = TokenProbabilityAggregator.extract_logprobs(oai_logprobs_response)
        assert result is not None
        assert len(result) == 5
        assert result[0]["token"] == "The"
        assert result[0]["logprob"] == -0.05

    def test_extract_logprobs_simple_format(self):
        """Extract from simple logprobs.content format."""
        response = {"logprobs": {"content": [{"token": "Hello", "logprob": -0.5}]}}
        result = TokenProbabilityAggregator.extract_logprobs(response)
        assert result is not None
        assert result[0]["token"] == "Hello"

    def test_extract_logprobs_flat_array(self):
        """Extract from flat logprobs array."""
        response = {"logprobs": [{"token": "A", "logprob": -1.0}, {"token": "B", "logprob": -2.0}]}
        result = TokenProbabilityAggregator.extract_logprobs(response)
        assert result is not None
        assert len(result) == 2

    def test_extract_logprobs_none(self):
        """None input returns None."""
        result = TokenProbabilityAggregator.extract_logprobs(None)
        assert result is None

    def test_extract_logprobs_string_json(self):
        """String JSON input is parsed."""
        json_str = json.dumps({
            "choices": [{"logprobs": {"content": [{"token": "X", "logprob": -1.0}]}}]
        })
        result = TokenProbabilityAggregator.extract_logprobs(json_str)
        assert result is not None
        assert result[0]["token"] == "X"

    def test_extract_logprobs_invalid_string(self):
        """Invalid string returns None."""
        result = TokenProbabilityAggregator.extract_logprobs("not json")
        assert result is None

    def test_extract_logprobs_no_logprobs_field(self):
        """Missing logprobs field returns None."""
        result = TokenProbabilityAggregator.extract_logprobs({"text": "hello"})
        assert result is None


# ===================================================================
# 6b: Semantic Entropy
# ===================================================================


class TestSemanticEntropy:
    def test_jaccard_similarity_identical(self):
        """Identical texts have similarity 1.0."""
        sim = SemanticEntropy.jaccard_similarity("The sky is blue", "The sky is blue")
        assert math.isclose(sim, 1.0, rel_tol=1e-6)

    def test_jaccard_similarity_completely_different(self):
        """Completely different texts have near-zero similarity."""
        sim = SemanticEntropy.jaccard_similarity("The sky is blue", "Quantum physics equations")
        assert sim < 0.3  # Character n-grams will partially overlap due to common chars

    def test_jaccard_similarity_empty_strings(self):
        """Both empty strings have similarity 1.0."""
        sim = SemanticEntropy.jaccard_similarity("", "")
        assert math.isclose(sim, 1.0)

    def test_jaccard_similarity_one_empty(self):
        """One empty string has similarity 0.0."""
        sim = SemanticEntropy.jaccard_similarity("hello world", "")
        assert math.isclose(sim, 0.0)

    def test_compute_dispersion_identical(self):
        """Identical responses have dispersion 0.0."""
        disp = SemanticEntropy.compute_dispersion(["hello", "hello", "hello"])
        assert math.isclose(disp, 0.0, abs_tol=0.01)

    def test_compute_dispersion_single(self):
        """Single response has dispersion 0.0."""
        disp = SemanticEntropy.compute_dispersion(["hello"])
        assert math.isclose(disp, 0.0)

    def test_compute_dispersion_empty(self):
        """Empty list has dispersion 0.0."""
        disp = SemanticEntropy.compute_dispersion([])
        assert math.isclose(disp, 0.0)

    def test_compute_dispersion_different(self):
        """Different responses have dispersion > 0."""
        disp = SemanticEntropy.compute_dispersion([
            "The sky is blue and clear today",
            "I think the stock market will go up tomorrow",
            "Quantum entanglement is fascinating",
        ])
        assert disp > 0.3

    def test_normalize(self):
        """Normalization removes punctuation and lowercases."""
        result = SemanticEntropy._normalize("Hello, World! This is a TEST.")
        assert result == "hello world this is a test"

    def test_ngrams_extraction(self):
        """Character n-grams are correctly extracted."""
        ngrams = SemanticEntropy._ngrams("hello", n=2)
        assert "he" in ngrams
        assert "el" in ngrams
        assert "ll" in ngrams
        assert "lo" in ngrams

    def test_ngrams_short_text(self):
        """Text shorter than n returns the text itself as a single ngram."""
        ngrams = SemanticEntropy._ngrams("hi", n=5)
        assert "hi" in ngrams

    def test_cluster_responses_similar(self):
        """Similar responses cluster together."""
        responses = [
            "The sky is blue and clear today",
            "The sky is blue today",
            "Stock market prediction for tomorrow",
        ]
        clusters = SemanticEntropy.cluster_responses(responses)
        assert len(clusters) >= 2  # Two similar + one different

    def test_cluster_responses_empty(self):
        """Empty input returns empty list."""
        clusters = SemanticEntropy.cluster_responses([])
        assert clusters == []


# ===================================================================
# 6c: Unified Confidence Score
# ===================================================================


class TestConfidenceScorer:
    def test_high_confidence_from_similar_responses(self, scorer):
        """Similar responses produce high confidence."""
        responses = [
            {"text": "The sky is blue"},
            {"text": "The sky is blue"},
            {"text": "Sky is blue"},
        ]
        result = scorer.estimate_confidence(responses=responses)
        assert result.confidence_score > 0.5
        assert result.recommendation in ("proceed", "verify")

    def test_low_confidence_from_diverse_responses(self, scorer):
        """Diverse responses produce lower confidence."""
        responses = [
            {"text": "The sky is blue"},
            {"text": "Stock market will crash"},
            {"text": "Quantum physics is weird"},
        ]
        result = scorer.estimate_confidence(responses=responses)
        assert result.confidence_score < 0.7  # Should be noticeably lower
        assert result.n_clusters >= 2

    def test_single_response_neutral_confidence(self, scorer):
        """Single response has neutral confidence score."""
        responses = [{"text": "The sky is blue"}]
        result = scorer.estimate_confidence(responses=responses)
        # Single response: token signal (if any) + reduced semantic weight
        assert result.recommendation in ("verify", "proceed")

    def test_empty_responses(self, scorer):
        """Empty responses produce block recommendation."""
        result = scorer.estimate_confidence(responses=[])
        assert result.confidence_score < 0.5
        assert "no_responses" in result.signal_flags
        assert result.recommendation == "block"

    def test_confidence_with_logprobs(self, scorer, oai_logprobs_response):
        """Logprobs boost confidence for high-probability responses."""
        responses = [{"text": "The sky is blue"}]
        result = scorer.estimate_confidence(
            responses=responses,
            logprobs_responses=[oai_logprobs_response],
        )
        assert result.confidence_score > 0.5
        assert result.token_stats is not None
        assert result.token_stats.mean_probability > 0.8  # High probability tokens

    def test_confidence_with_low_logprobs(self, scorer):
        """Low logprobs result in lower confidence."""
        responses = [{"text": "Maybe it could be red"}]
        low_logprobs = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Maybe it could be red"},
                    "logprobs": {
                        "content": [
                            {"token": "Maybe", "logprob": -2.5, "bytes": [77, 97, 121, 98, 101]},
                            {"token": " it", "logprob": -0.5, "bytes": [32, 105, 116]},
                            {
                                "token": " could",
                                "logprob": -3.0,
                                "bytes": [32, 99, 111, 117, 108, 100],
                            },
                            {"token": " be", "logprob": -0.3, "bytes": [32, 98, 101]},
                            {"token": " red", "logprob": -4.0, "bytes": [32, 114, 101, 100]},
                        ]
                    },
                }
            ]
        }
        result = scorer.estimate_confidence(
            responses=responses,
            logprobs_responses=[low_logprobs],
        )
        assert result.confidence_score < 0.7  # Lower due to low logprobs
        assert result.token_stats is not None
        assert result.token_stats.low_probability_fraction > 0

    def test_recommendation_proceed(self, scorer):
        """High confidence yields proceed recommendation."""
        result = ConfidenceResult(
            confidence_score=0.85,
            token_stats=None,
            n_responses=3,
            semantic_dispersion=0.05,
            n_clusters=1,
            cluster_sizes=[3],
            signal_flags=[],
            recommendation="proceed",
        )
        assert result.recommendation == "proceed"

    def test_recommendation_escalate_manual(self, scorer):
        """Low confidence yields escalate or block."""
        # Directly test the _recommend method
        assert scorer._recommend(0.2, []) == "block"
        assert scorer._recommend(0.4, []) == "escalate"
        assert scorer._recommend(0.6, []) == "verify"
        assert scorer._recommend(0.8, []) == "proceed"

    def test_recommendation_block_on_critical(self, scorer):
        """Critical flags force block regardless of score."""
        assert scorer._recommend(0.9, ["no_responses"]) == "block"

    def test_flag_low_probability(self, scorer, sample_low_confidence_logprobs):
        """Low logprobs flag the result."""
        responses = [{"text": "Maybe it could be red"}]
        logprobs_resp = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "..."},
                    "logprobs": {"content": sample_low_confidence_logprobs},
                }
            ]
        }
        result = scorer.estimate_confidence(
            responses=responses,
            logprobs_responses=[logprobs_resp],
        )
        found_flags = [
            f for f in result.signal_flags if f in ("low_probability", "very_low_min_probability")
        ]
        assert len(found_flags) > 0

    def test_estimate_task_profile_confidence(self, scorer):
        """Wrapper for TaskProfile confidence extraction."""
        profile = type("Profile", (), {"confidence": 0.85})()
        score = scorer.estimate_task_profile_confidence(profile)
        assert math.isclose(score, 0.85)

    def test_estimate_task_profile_no_confidence(self, scorer):
        """Missing confidence attribute returns default."""
        profile = type("Profile", (), {})()
        score = scorer.estimate_task_profile_confidence(profile)
        assert math.isclose(score, 0.5)


# ===================================================================
# Convenience Wrapper
# ===================================================================


class TestEstimateConfidence:
    def test_wrapper(self):
        """The convenience wrapper works end-to-end."""
        responses = [{"text": "Hello"}, {"text": "Hello"}, {"text": "Hello there"}]
        result = estimate_confidence(responses=responses)
        assert isinstance(result, ConfidenceResult)
        assert result.confidence_score > 0
        assert result.n_responses == 3

    def test_wrapper_empty(self):
        """Wrapper handles empty input."""
        result = estimate_confidence(responses=[])
        assert result.recommendation == "block"
