"""Confidence Scoring & Semantic Uncertainty — Phase 6

Detects when the model is uncertain or confabulating by combining
token-level probability signals, semantic entropy, and structural
consistency into a unified confidence score (0.0–1.0).

Three components:
  6a: Token Probability Aggregation — mean/min/entropy per response
  6b: Semantic Entropy (lightweight) — dispersion across N samples
  6c: Unified Confidence Score — merged signal for routing decisions

Usage:
    from confidence import ConfidenceScorer
    scorer = ConfidenceScorer()
    score = scorer.estimate_confidence(responses=[...])
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class TokenProbabilities:
    """Per-response token probability statistics."""

    mean_probability: float  # Average token probability (0.0–1.0)
    min_probability: float  # Minimum token probability (0.0–1.0)
    entropy: float  # Per-token entropy (bits)
    token_count: int
    low_probability_fraction: float  # Fraction of tokens below 0.1 probability


@dataclass
class SemanticCluster:
    """A cluster of semantically similar responses."""

    indices: list[int]
    centroid: str  # Representative text of the cluster
    size: int


@dataclass
class ConfidenceResult:
    """Complete confidence assessment for one or more responses."""

    confidence_score: float  # 0.0–1.0 unified score
    token_stats: TokenProbabilities | None
    n_responses: int
    semantic_dispersion: float  # 0.0 = all identical, 1.0 = fully dispersed
    n_clusters: int
    cluster_sizes: list[int]
    signal_flags: list[str]  # e.g., ["low_probability", "high_dispersion"]
    recommendation: str  # "proceed" | "verify" | "escalate" | "block"


# ---------------------------------------------------------------------------
# Component 6a: Token Probability Aggregation
# ---------------------------------------------------------------------------


class TokenProbabilityAggregator:
    """Aggregate per-token log probabilities into response-level statistics.

    Supports the LM Studio API (and OpenAI-compatible) logprobs format.

    Expected logprobs format:
        {
            "content": [
                {"token": "Hello", "logprob": -0.5, "bytes": [72, ...]},
                ...
            ]
        }
    or the simpler:
        [{"token": "Hello", "logprob": -0.5}, ...]
    """

    LOW_PROBABILITY_THRESHOLD = 0.1  # Tokens below this are "low confidence"

    @staticmethod
    def extract_logprobs(response: dict[str, Any] | str | None) -> list[dict[str, Any]] | None:
        """Extract token-level logprobs from an LLM API response.

        Handles multiple response formats:
        - OpenAI-compatible: response.choices[0].logprobs.content[]
        - Simplified: response.logprobs.content[]
        - Full token objects: {"token": ..., "logprob": ...}

        Returns None if no logprobs are available.
        """
        if response is None:
            return None

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                return None

        if not isinstance(response, dict):
            return None

        # Try multiple paths to find logprobs
        logprobs_data = None

        # Path 1: OpenAI chat completions format
        try:
            logprobs_data = response["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError):
            pass

        # Path 2: Direct logprobs.content
        if logprobs_data is None:
            try:
                logprobs_data = response["logprobs"]["content"]
            except (KeyError, TypeError):
                pass

        # Path 3: Flat logprobs array
        if logprobs_data is None:
            try:
                logprobs_data = response.get("logprobs")
                if isinstance(logprobs_data, list):
                    pass  # Use as-is
                else:
                    logprobs_data = None
            except (KeyError, TypeError):
                pass

        if logprobs_data is None or not isinstance(logprobs_data, list):
            return None

        # Normalize: ensure each entry has token/logprob keys
        normalized: list[dict[str, Any]] = []
        for entry in logprobs_data:
            if isinstance(entry, dict):
                norm = {
                    "token": entry.get("token", ""),
                    "logprob": entry.get("logprob", entry.get("probability", 0.0)),
                    "probability": entry.get("probability", entry.get("logprob", 0.0)),
                }
                normalized.append(norm)
            elif isinstance(entry, (int, float)):
                normalized.append({"token": "", "logprob": entry, "probability": math.exp(entry)})

        return normalized or None

    @staticmethod
    def calculate_probability(logprob: float) -> float:
        """Convert log probability to probability (0.0–1.0).

        LM Studio / OpenAI logprobs use natural log scale.
        A logprob of 0 means p=1.0 (certain token).
        Positive logprobs are clamped to 1.0.
        """
        if logprob < 0:
            prob = math.exp(logprob)
        elif logprob == 0:
            prob = 1.0
        else:
            # Positive shouldn't happen from standard APIs, clamp safely
            prob = min(logprob, 1.0)
        return max(0.0, min(prob, 1.0))

    @staticmethod
    def aggregate(logprobs: list[dict[str, Any]]) -> TokenProbabilities | None:
        """Aggregate token-level logprobs into response statistics.

        Args:
            logprobs: List of per-token logprob objects.

        Returns:
            TokenProbabilities with mean/min/entropy, or None if empty.

        """
        if not logprobs:
            return None

        probabilities: list[float] = []
        token_count = 0
        low_prob_count = 0
        total_entropy = 0.0

        for entry in logprobs:
            logprob = entry.get("logprob", 0.0)
            prob = TokenProbabilityAggregator.calculate_probability(logprob)
            probabilities.append(prob)
            token_count += 1

            if prob < TokenProbabilityAggregator.LOW_PROBABILITY_THRESHOLD:
                low_prob_count += 1

            # Entropy: -p * log2(p) for each token, but since we have single-token
            # probability we compute expected entropy from the probability distribution
            # If the model assigned probability p to this token, the entropy is
            # -p*log2(p) - (1-p)*log2(1-p) (binary entropy)
            if 0 < prob < 1:
                eps = 1e-10
                p = max(eps, min(prob, 1 - eps))
                token_entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
                total_entropy += token_entropy

        if token_count == 0:
            return None

        mean_prob = sum(probabilities) / token_count
        min_prob = min(probabilities) if probabilities else 0.0
        mean_entropy = total_entropy / token_count
        low_fraction = low_prob_count / token_count

        return TokenProbabilities(
            mean_probability=mean_prob,
            min_probability=min_prob,
            entropy=mean_entropy,
            token_count=token_count,
            low_probability_fraction=low_fraction,
        )


# ---------------------------------------------------------------------------
# Component 6b: Semantic Entropy (lightweight)
# ---------------------------------------------------------------------------


class SemanticEntropy:
    """Lightweight semantic dispersion measurement.

    Uses character n-gram overlap as a proxy for semantic similarity
    (no embedding model dependency). For production use, replace with
    sentence-transformers or similar embedding-based clustering.
    """

    # Character n-gram size for similarity
    NGRAM_N = 3
    # Similarity threshold for clustering (0.0–1.0)
    CLUSTER_THRESHOLD = 0.6

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        # Remove punctuation and normalize whitespace
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _ngrams(text: str, n: int = NGRAM_N) -> set[str]:
        """Extract character n-grams from text."""
        text = SemanticEntropy._normalize(text)
        if len(text) < n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    @staticmethod
    def jaccard_similarity(a: str, b: str) -> float:
        """Compute Jaccard similarity between character n-gram sets.

        Returns 0.0 (completely different) to 1.0 (identical n-grams).
        """
        ngrams_a = SemanticEntropy._ngrams(a)
        ngrams_b = SemanticEntropy._ngrams(b)

        if not ngrams_a and not ngrams_b:
            return 1.0
        if not ngrams_a or not ngrams_b:
            return 0.0

        intersection = ngrams_a & ngrams_b
        union = ngrams_a | ngrams_b
        return len(intersection) / len(union)

    @staticmethod
    def cluster_responses(responses: list[str]) -> list[SemanticCluster]:
        """Cluster responses by semantic similarity.

        Uses a simple greedy clustering approach: for each response,
        find the best cluster match above threshold, or create a new cluster.
        """
        if not responses:
            return []

        clusters: list[SemanticCluster] = []
        assigned: set[int] = set()

        for i, response in enumerate(responses):
            if i in assigned:
                continue

            # Start a new cluster with this response
            cluster_indices = [i]
            assigned.add(i)

            for j in range(i + 1, len(responses)):
                if j in assigned:
                    continue

                similarity = SemanticEntropy.jaccard_similarity(response, responses[j])
                if similarity >= SemanticEntropy.CLUSTER_THRESHOLD:
                    cluster_indices.append(j)
                    assigned.add(j)

            clusters.append(
                SemanticCluster(
                    indices=cluster_indices,
                    centroid=response[:200],
                    size=len(cluster_indices),
                )
            )

        return clusters

    @staticmethod
    def compute_dispersion(responses: list[str]) -> float:
        """Compute semantic dispersion (0.0 = identical, 1.0 = fully dispersed).

        Uses pairwise Jaccard similarity and returns 1 - mean_similarity.
        """
        if len(responses) <= 1:
            return 0.0

        total_similarity = 0.0
        pairs = 0

        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                total_similarity += SemanticEntropy.jaccard_similarity(responses[i], responses[j])
                pairs += 1

        mean_similarity = total_similarity / pairs if pairs > 0 else 1.0
        return 1.0 - mean_similarity


# ---------------------------------------------------------------------------
# Component 6c: Unified Confidence Score
# ---------------------------------------------------------------------------


class ConfidenceScorer:
    """Merge multiple signals into a unified confidence score.

    Combines:
    - Token probability signal (when logprobs available)
    - Semantic entropy / dispersion (when multiple samples)
    - Structural consistency (when available)

    Returns a 0.0–1.0 score with routing recommendation.
    """

    # Thresholds
    HIGH_CONFIDENCE = 0.7  # Proceed normally
    MEDIUM_CONFIDENCE = 0.5  # Run verification
    LOW_CONFIDENCE = 0.3  # Escalate or block

    # Weights for each signal (when available)
    WEIGHT_TOKEN_PROB = 0.4
    WEIGHT_SEMANTIC = 0.4
    WEIGHT_STRUCTURAL = 0.2

    def estimate_confidence(
        self,
        responses: list[dict[str, Any] | str | None],
        logprobs_responses: list[dict[str, Any] | None] | None = None,
    ) -> ConfidenceResult:
        """Estimate a unified confidence score from one or more responses.

        Args:
            responses: List of raw LLM response strings or dicts.
            logprobs_responses: Optional list of full API responses containing
                logprobs data. If provided, token probability signals are used.

        Returns:
            ConfidenceResult with score, flags, and recommendation.

        """
        signal_flags: list[str] = []
        token_stats: TokenProbabilities | None = None
        token_score = 0.5  # Default (neutral)
        semantic_score = 0.5  # Default (neutral)

        # --- Token Probability Signal (6a) ---
        if logprobs_responses and len(logprobs_responses) > 0:
            for logprob_response in logprobs_responses:
                logprobs = TokenProbabilityAggregator.extract_logprobs(logprob_response)
                if logprobs:
                    stats = TokenProbabilityAggregator.aggregate(logprobs)
                    if stats:
                        token_stats = stats
                        # Score from mean probability
                        token_score = stats.mean_probability

                        if stats.low_probability_fraction > 0.3:
                            signal_flags.append("low_probability")
                        if stats.min_probability < 0.01:
                            signal_flags.append("very_low_min_probability")
                        break  # Use first available logprobs

        # --- Semantic Entropy Signal (6b) ---
        response_texts: list[str] = []
        for r in responses:
            if r is None:
                continue
            if isinstance(r, dict):
                # Try to extract text content
                text = r.get("text", r.get("content", r.get("response", json.dumps(r))))
            else:
                text = str(r)
            response_texts.append(text)

        n_responses = len(response_texts)
        semantic_dispersion = 0.0
        n_clusters = 1
        cluster_sizes: list[int] = [n_responses]

        if n_responses >= 2:
            # Compute dispersion
            semantic_dispersion = SemanticEntropy.compute_dispersion(response_texts)

            # Cluster
            clusters = SemanticEntropy.cluster_responses(response_texts)
            n_clusters = len(clusters)
            cluster_sizes = [c.size for c in clusters]

            # Score: low dispersion = high confidence
            semantic_score = 1.0 - semantic_dispersion

            if semantic_dispersion > 0.3:
                signal_flags.append("high_dispersion")
            if n_clusters >= 3:
                signal_flags.append("many_clusters")
        elif n_responses == 1:
            # Single response: neutral semantic score
            semantic_score = 0.5

        # --- Merge Signals ---
        weight_sum = 0.0
        weighted_score = 0.0

        if token_stats is not None:
            weighted_score += token_score * self.WEIGHT_TOKEN_PROB
            weight_sum += self.WEIGHT_TOKEN_PROB

        if n_responses >= 2:
            weighted_score += semantic_score * self.WEIGHT_SEMANTIC
            weight_sum += self.WEIGHT_SEMANTIC
        elif n_responses == 1:
            # Single response reduces weight
            weighted_score += semantic_score * (self.WEIGHT_SEMANTIC * 0.5)
            weight_sum += self.WEIGHT_SEMANTIC * 0.5

        if n_responses == 0:
            # No signals available
            weighted_score = 0.3
            weight_sum = 1.0
            signal_flags.append("no_responses")

        # If no signals were available, use default
        if weight_sum == 0.0:
            weighted_score = 0.5
            weight_sum = 1.0

        confidence_score = max(0.0, min(1.0, weighted_score / weight_sum))

        # --- Recommendation ---
        recommendation = self._recommend(confidence_score, signal_flags)

        return ConfidenceResult(
            confidence_score=confidence_score,
            token_stats=token_stats,
            n_responses=n_responses,
            semantic_dispersion=semantic_dispersion,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            signal_flags=signal_flags,
            recommendation=recommendation,
        )

    def _recommend(self, confidence_score: float, flags: list[str]) -> str:
        """Generate a routing recommendation based on confidence score and flags."""
        # Hard block on critical flags
        if "no_responses" in flags:
            return "block"

        # Score-based recommendation
        if confidence_score >= self.HIGH_CONFIDENCE:
            return "proceed"
        if confidence_score >= self.MEDIUM_CONFIDENCE:
            return "verify"
        if confidence_score >= self.LOW_CONFIDENCE:
            return "escalate"
        return "block"

    def estimate_task_profile_confidence(
        self,
        profile_task_profile: Any,  # Accept TaskProfile or similar
    ) -> float:
        """Estimate confidence from a TaskProfile's own signal (rule-based confidence).

        This is a wrapper that extracts the existing task-profile confidence
        and scales it with the same semantics as the unified scorer.
        """
        # If the object has a 'confidence' attribute
        if hasattr(profile_task_profile, "confidence"):
            base = getattr(profile_task_profile, "confidence", 0.5)
            if isinstance(base, (int, float)):
                return max(0.0, min(1.0, float(base)))
        return 0.5


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------


def estimate_confidence(
    responses: list[dict[str, Any] | str | None],
    logprobs_responses: list[dict[str, Any] | None] | None = None,
) -> ConfidenceResult:
    """One-shot confidence estimation wrapper.

    Args:
        responses: List of raw response texts or dicts.
        logprobs_responses: Optional full API responses with logprobs.

    Returns:
        ConfidenceResult with unified score and recommendation.

    """
    scorer = ConfidenceScorer()
    return scorer.estimate_confidence(
        responses=responses,
        logprobs_responses=logprobs_responses,
    )
