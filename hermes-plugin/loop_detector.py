"""Loop Detector — doom loop pattern detection for small models.

Uses a 4-signal ensemble to detect repetitive loop behaviors that
disproportionately affect small (<12B) models. Based on Liquid AI's
Antidoom (FTPO) findings (Jul 7 2026): Qwen3.5-4B doom loop rate
22.9% under greedy sampling. Trigger tokens extracted from the
Antidoom research.

Detection signals:
  1. N-gram overlap (trigram, sliding window of 8)
  2. Tool call diversity ratio
  3. Content stagnation (output similarity to recent calls)
  4. Trigger token frequency (the, So, Alternatively, Wait, But)
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Research-backed constants
# ---------------------------------------------------------------------------

# Trigger tokens identified in Antidoom (FTPO) research — tokens that
# correlate with model entering doom loops under greedy sampling.
# Qwen3.5-4B exhibited 22.9% loop rate on hard problems with these tokens.
TRIGGER_TOKENS: set[str] = {
    "the",
    "So",
    "Alternatively",
    "Wait",
    "But",
    "However",
    "Meanwhile",
    "Additionally",
    "Furthermore",
    "Thus",
    "Therefore",
    "Consequently",
    "Nevertheless",
}

# Pattern for repeated short token sequences (hallucination loop signature)
# e.g. "I'm sorry, but I can't help with that" repeated with variations
REPETITIVE_APOLOGY_PATTERN = re.compile(
    r"((I'?m\s+sorry|cannot|can'?t|couldn'?t)\s+.*?){2,}",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

LoopPattern = Literal[
    "tool_slam",  # Same tool + same args repeatedly
    "token_grind",  # Output n-gram overlap > threshold
    "stuck_retry",  # Same error returned repeatedly
    "hallucination",  # Plausible but wrong output repeated
    "apology_loop",  # "I'm sorry, I can't..." repeated
    "trigger_spiral",  # High frequency of trigger tokens
    "none",  # No loop detected
]


@dataclass
class CallRecord:
    """Record of a single tool call for loop detection."""

    tool_name: str
    args: dict | None = None
    error: str | None = None
    output_snippet: str | None = None


@dataclass
class LoopScore:
    """Ensemble loop detection score."""

    overall: float  # 0.0-1.0, 1.0 = almost certainly looping
    ngram_overlap: float  # 0.0-1.0
    diversity_ratio: float  # 0.0-1.0, lower = less diverse
    content_stagnation: float  # 0.0-1.0
    trigger_frequency: float  # 0.0-1.0
    pattern: LoopPattern = "none"

    def should_block(self, threshold: float = 0.75) -> bool:
        return self.overall >= threshold


# ---------------------------------------------------------------------------
# Loop Detector
# ---------------------------------------------------------------------------


class LoopDetector:
    """4-signal ensemble doom loop detector.

    Maintains a sliding window of recent tool calls and evaluates the
    loop probability on each new call. Designed to catch the failure
    modes that small models (<12B) are most susceptible to.

    Research basis:
    - Liquid AI Antidoom (FTPO) Jul 7 2026: 22.9% loop rate Qwen3.5-4B
    - Trigger tokens from Antidoom dataset
    - N-gram overlap detection: 4-gram with sliding window of 8
    """

    def __init__(self, window: int = 8, ngram_n: int = 4):
        self.window = window
        self.ngram_n = ngram_n
        self._calls: deque[CallRecord] = deque(maxlen=window)

    def record(self, call: CallRecord) -> None:
        """Record a tool call in the sliding window."""
        self._calls.append(call)
        logger.debug("LoopDetector: recorded %s (window now %d)", call.tool_name, len(self._calls))

    def score(self) -> LoopScore:
        """Run ensemble scoring against the current window.

        Returns a LoopScore with individual signal components and
        the weighted overall score. The overall score uses weights
        tuned to catch the specific failure modes of small models.
        """
        calls = list(self._calls)
        if len(calls) < 2:
            return LoopScore(
                overall=0.0,
                ngram_overlap=0.0,
                diversity_ratio=1.0,
                content_stagnation=0.0,
                trigger_frequency=0.0,
            )

        signals = {
            "ngram_overlap": self._score_ngram_overlap(calls),
            "diversity_ratio": self._score_diversity(calls),
            "content_stagnation": self._score_stagnation(calls),
            "trigger_frequency": self._score_triggers(calls),
        }

        # Weighted ensemble — weights tuned for small model failure modes
        weights = {
            "ngram_overlap": 0.35,  # Strongest signal for tool loops
            "diversity_ratio": 0.25,  # Low diversity = stuck behavior
            "content_stagnation": 0.25,  # Same output = hallucination loop
            "trigger_frequency": 0.15,  # Trigger tokens = early warning
        }

        overall = sum(signals[k] * weights[k] for k in weights)

        # Classify pattern
        pattern = self._classify_pattern(calls, signals)

        return LoopScore(
            overall=round(overall, 4),
            ngram_overlap=signals["ngram_overlap"],
            diversity_ratio=signals["diversity_ratio"],
            content_stagnation=signals["content_stagnation"],
            trigger_frequency=signals["trigger_frequency"],
            pattern=pattern,
        )

    def _score_ngram_overlap(self, calls: list[CallRecord]) -> float:
        """Score based on N-gram overlap of tool name sequences.

        A tool_slam loop has very high ngram overlap — the same
        sequence of tool calls repeats identically.
        """
        if len(calls) < self.ngram_n:
            return 0.0

        # Build ngrams of tool names
        ngrams: list[tuple] = []
        for i in range(len(calls) - self.ngram_n + 1):
            ngram = tuple(calls[i + j].tool_name for j in range(self.ngram_n))
            ngrams.append(ngram)

        if len(ngrams) < 2:
            return 0.0

        # Count unique ngrams vs total
        unique_count = len(set(ngrams))
        total_count = len(ngrams)

        if total_count <= 1:
            return 0.0

        # Overlap ratio: 1.0 = every ngram identical, 0.0 = all unique
        overlap = 1.0 - (unique_count / total_count)

        return min(overlap, 1.0)

    def _score_diversity(self, calls: list[CallRecord]) -> float:
        """Score based on tool call diversity.

        Low diversity (same tool called repeatedly) = stuck in a loop.
        Returns 0.0 (highly diverse) to 1.0 (same tool always).
        """
        if not calls:
            return 0.0

        unique_tools = len({c.tool_name for c in calls})
        total_calls = len(calls)

        # Diversity ratio: 1.0 = all calls are the same tool
        if unique_tools <= 1:
            return 1.0

        # Normalize: if every call is a different tool, score ~0
        # More tools than calls isn't possible, so max diversity is min(total, unique)
        max_diversity = min(total_calls, unique_tools)
        diversity = max_diversity / total_calls if total_calls > 0 else 1.0

        # Invert: we want 1.0 = NOT diverse
        return round(1.0 - diversity, 4)

    def _score_stagnation(self, calls: list[CallRecord]) -> float:
        """Score based on content stagnation — output similarity.

        Models in hallucination loops produce very similar or identical
        output across consecutive calls. Measures approximate similarity
        of recent output snippets.
        """
        with_output = [c for c in calls if c.output_snippet]
        if len(with_output) < 2:
            return 0.0

        # Compare consecutive outputs for stagnation
        similar_count = 0
        total_pairs = len(with_output) - 1

        for i in range(total_pairs):
            if self._outputs_similar(
                with_output[i].output_snippet or "",
                with_output[i + 1].output_snippet or "",
            ):
                similar_count += 1

        return similar_count / total_pairs if total_pairs > 0 else 0.0

    def _score_triggers(self, calls: list[CallRecord]) -> float:
        """Score based on trigger token frequency.

        The Antidoom research identified specific tokens that correlate
        with model entering doom loops. High frequency = high risk.
        """
        total_text = ""
        for c in calls:
            if c.output_snippet:
                total_text += " " + c.output_snippet
            if c.error:
                total_text += " " + c.error

        if not total_text.strip():
            return 0.0

        words = total_text.lower().split()
        if not words:
            return 0.0

        trigger_count = sum(1 for w in words if w in TRIGGER_TOKENS)
        trigger_ratio = trigger_count / len(words)

        # Scale: at ~20% trigger tokens, score is 1.0. At 5% it's ~0.25.
        # This gives gradual warning before critical levels.
        score = min(trigger_ratio * 5.0, 1.0)
        return round(score, 4)

    def _classify_pattern(
        self,
        calls: list[CallRecord],
        signals: dict[str, float],
    ) -> LoopPattern:
        """Classify the type of loop pattern detected."""
        # Tool slam: same tool + similar args repeatedly
        if signals["ngram_overlap"] > 0.7 and signals["diversity_ratio"] > 0.6:
            return "tool_slam"

        # Stuck retry: consecutive errors on same tool
        if self._detect_stuck_retry(calls):
            return "stuck_retry"

        # Apology loop: repetitive apology patterns
        for c in calls:
            if c.output_snippet and REPETITIVE_APOLOGY_PATTERN.search(c.output_snippet):
                return "apology_loop"

        # Token grind: very high ngram overlap but diverse tools
        if signals["ngram_overlap"] > 0.8:
            return "token_grind"

        # Trigger spiral: high trigger tokens but other signals moderate
        if signals["trigger_frequency"] > 0.5:
            return "trigger_spiral"

        # Hallucination loop: content stagnation + moderate diversity
        if signals["content_stagnation"] > 0.6 and signals["diversity_ratio"] < 0.5:
            return "hallucination"

        return "none"

    def _detect_stuck_retry(self, calls: list[CallRecord]) -> bool:
        """Detect repeated error on the same tool."""
        if len(calls) < 3:
            return False
        # Last 3 calls: same tool, same error
        last3 = calls[-3:]
        if all(c.error for c in last3):
            tools = {c.tool_name for c in last3}
            if len(tools) <= 1:
                return True
        return False

    @staticmethod
    def _outputs_similar(a: str, b: str) -> bool:
        """Check if two output snippets are approximately similar.

        Uses a simple character-level overlap heuristic. Returns True
        if the normalized outputs are >70% similar.
        """
        if not a or not b:
            return False

        # Normalize: strip whitespace, lowercase
        a_norm = " ".join(a.lower().split())
        b_norm = " ".join(b.lower().split())

        if not a_norm or not b_norm:
            return False

        # Quick check: if one contains the other, they're similar
        if a_norm in b_norm or b_norm in a_norm:
            return True

        # Token overlap ratio
        a_tokens = set(a_norm.split())
        b_tokens = set(b_norm.split())
        if not a_tokens or not b_tokens:
            return False

        intersection = a_tokens & b_tokens
        union = a_tokens | b_tokens
        jaccard = len(intersection) / len(union) if union else 0

        return jaccard > 0.7

    def reset(self) -> None:
        """Clear the call window."""
        self._calls.clear()
        logger.debug("LoopDetector: reset")
