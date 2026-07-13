"""Addition B — Grammar Validator (RPG / Repetition Grammar Penalization).

ACL 2025: Pushdown automaton tracking formal grammar of output language
to detect structural repetitions. Identifies when the model enters a loop
of identical JSON keys, repeated XML blocks, or recursive identical
structures and computes a penalty for the anchor tokens.

This module implements:
1. A simple pushdown automaton for tracking structural nesting.
2. Structural repetition detection using n-gram overlap on token sequences.
3. A penalty function mapping repetition severity to a scaling factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidatorConfig:
    """Configuration for the grammar validator.

    Attributes:
        repetition_threshold: Number of consecutive identical tokens before flagging.
        block_size: Size of n-gram window for block-level repetition detection.
        max_history: Maximum number of recent tokens to keep.
        penalty_scale: Scaling factor for penalty computation (0.0-1.0).

    """

    repetition_threshold: int = 3
    block_size: int = 4
    max_history: int = 100
    penalty_scale: float = 0.3


@dataclass
class PushdownAutomaton:
    """Simple pushdown automaton for tracking structural nesting.

    Tracks stack depth and structural mismatches (unclosed tags).
    """

    stack: list[str] = field(default_factory=list)
    mismatch_count: int = 0
    current_state: str = "start"

    @property
    def stack_depth(self) -> int:
        return len(self.stack)

    def push(self, symbol: str) -> None:
        """Push a symbol onto the stack."""
        self.stack.append(symbol)

    def pop(self) -> str | None:
        """Pop a symbol from the stack. Returns None on underflow."""
        if self.stack:
            return self.stack.pop()
        return None

    def track(self, token_type: str, value: str) -> None:
        """Track a structural token (open/close tag or structural marker).

        Args:
            token_type: One of 'open_tag', 'close_tag', 'open_bracket', 'close_bracket'.
            value: The tag or bracket value.

        """
        if token_type in ("open_tag", "open_bracket"):
            self.push(value)
        elif token_type in ("close_tag", "close_bracket"):
            if self.stack and self.stack[-1] == value:
                self.pop()
            else:
                self.mismatch_count += 1


def detect_structural_repetition(
    tokens: list[str],
    block_size: int = 4,
    threshold: int = 3,
) -> dict[str, Any]:
    """Detect structural repetition in a token sequence.

    Uses two mechanisms:
    1. N-gram overlap: identical consecutive n-gram blocks.
    2. Token-level frequency: tokens repeated above threshold.

    Args:
        tokens: Sequence of token strings.
        block_size: Size of n-gram window for block detection.
        threshold: Repetition threshold for token-level flagging.

    Returns:
        Dict with:
            - repetition_count (int): count of structural repetitions
            - max_repetition (int): maximum repeat count for any structure
            - identical_blocks (int): number of identical block sequences
            - block_repetitions (list): detailed block info

    """
    if not tokens:
        return {
            "repetition_count": 0,
            "max_repetition": 0,
            "identical_blocks": 0,
            "block_repetitions": [],
        }

    # Token-level repetition counting
    token_counts: dict[str, int] = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    repetition_count = sum(1 for c in token_counts.values() if c > threshold)
    max_repetition = max(token_counts.values()) if token_counts else 0

    # Block-level repetition detection (n-gram sliding window)
    blocks: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - block_size + 1):
        block = tuple(tokens[i : i + block_size])
        blocks[block] = blocks.get(block, 0) + 1

    identical_blocks = sum(1 for c in blocks.values() if c > 1)
    block_repetitions = [{"block": list(b), "count": c} for b, c in blocks.items() if c > 1]

    return {
        "repetition_count": repetition_count,
        "max_repetition": max_repetition,
        "identical_blocks": identical_blocks,
        "block_repetitions": block_repetitions[:10],  # cap output size
    }


def compute_repetition_penalty(
    structure_repetitions: int,
    identical_blocks: int,
    scale: float = 0.3,
) -> float:
    """Compute a penalty multiplier from repetition metrics.

    Returns 1.0 for no repetition (no penalty).
    Lower values indicate stronger penalty.

    Args:
        structure_repetitions: Count of repeated structural elements.
        identical_blocks: Count of repeated n-gram blocks.
        scale: Scaling factor (0.0-1.0).

    Returns:
        Penalty multiplier between 0.0 and 1.0.

    """
    if structure_repetitions == 0 and identical_blocks == 0:
        return 1.0

    raw = structure_repetitions * 0.1 + identical_blocks * 0.2
    return max(0.1, 1.0 - min(1.0, raw * scale))


@dataclass
class GrammarValidator:
    """Online grammar validator that detects structural repetition
    in token sequences and computes a penalty.

    Usage:
        gv = GrammarValidator()
        for token in generation_output:
            gv.observe(token)
        penalty = gv.penalty
    """

    config: ValidatorConfig = field(default_factory=ValidatorConfig)
    token_history: list[str] = field(default_factory=list)
    repetition_count: int = 0
    _penalty: float = 1.0

    @property
    def penalty(self) -> float:
        """Current penalty multiplier (1.0 = no penalty)."""
        return self._penalty

    def observe(self, token: str | None) -> None:
        """Record a generated token and update penalty.

        Args:
            token: Generated token string, or None (ignored).

        """
        if token is None:
            return

        self.token_history.append(str(token))
        if len(self.token_history) > self.config.max_history:
            self.token_history.pop(0)

        # Recompute penalty periodically
        if len(self.token_history) >= self.config.block_size:
            result = detect_structural_repetition(
                self.token_history,
                block_size=self.config.block_size,
                threshold=self.config.repetition_threshold,
            )
            self.repetition_count = result["repetition_count"]
            self._penalty = compute_repetition_penalty(
                result["repetition_count"],
                result["identical_blocks"],
                scale=self.config.penalty_scale,
            )

    def reset(self) -> None:
        """Clear all state and reset penalty to 1.0."""
        self.token_history.clear()
        self.repetition_count = 0
        self._penalty = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for diagnostics."""
        return {
            "repetition_threshold": self.config.repetition_threshold,
            "block_size": self.config.block_size,
            "max_history": self.config.max_history,
            "repetition_count": self.repetition_count,
            "penalty": self._penalty,
            "history_length": len(self.token_history),
        }
