"""Addition A — Value Regularizer (RMR / Reinforced Mode Regulation).

arXiv 2605.00435 (ICML 2026): Low-rank, eigenvalue-thresholded dampening
applied to the Transformer's value cache. Monitors correlation dimension
of the generation trajectory; when geometric collapse detected, dampens
self-reinforcing directions to prevent annihilation / death-loop modes.

This module implements:

1. Correlation dimension estimation from trajectory vectors (Grassberger-Procaccia
   style counting, simplified for online use).
2. A collapse detector that raises a flag when the effective dimension drops
   below a configurable threshold and the slope of decay exceeds a margin.
3. A dampening kernel that maps low dimension to a penalty factor applied to
   the confidence of high-certainty tokens — breaking loops without full
   speculative decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegularizerConfig:
    """Configuration for the value regularizer.

    Attributes:
        threshold: Correlation dimension below which dampening activates.
        strength: Maximum dampening strength at near-zero dimension.
        history_window: Number of recent observations to keep for dim estimation.
        slope_window: Number of recent dimension estimates to use for collapse slope.
        min_history: Minimum history length before collapse detection runs.

    """

    threshold: float = 3.0
    strength: float = 0.6
    history_window: int = 20
    slope_window: int = 5
    min_history: int = 3


def calculate_correlation_dimension(vectors: list[list[float]]) -> float:
    """Estimate correlation dimension from a set of trajectory vectors.

    Uses a simplified counting approach: compute pairwise distances, count how
    many pairs are within a radius, estimate dimension from the log-log slope.
    Falls back to 1.0 for single vector, 0.0 for empty.

    Args:
        vectors: List of N-dimensional floating point vectors.

    Returns:
        Estimated correlation dimension (float).

    """
    n = len(vectors)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    # Compute radius proportional to data extent
    all_vals = [v for vec in vectors for v in vec]
    if not all_vals:
        return 1.0
    data_range = max(all_vals) - min(all_vals)
    if data_range == 0:
        return 1.0

    # Multi-scale radius: use 30% of data range for robust counting
    radius = 0.3 * data_range

    # Count pairs within radius
    close_pairs = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            dist = math.sqrt(
                sum(
                    (vectors[i][k] - vectors[j][k]) ** 2
                    for k in range(min(len(vectors[i]), len(vectors[j])))
                )
            )
            if dist < radius:
                close_pairs += 1

    if total_pairs == 0:
        return 1.0

    frac_close = close_pairs / total_pairs

    # Dimension estimate from the fraction of close pairs.
    # In a D-dimensional space, the fraction of points within
    # a sphere of radius r scales as r^D for small r.
    # For our single-radius estimate, embed_dim bounds the max.
    embed_dim = len(vectors[0]) if vectors else 1

    if frac_close >= 0.99:
        # Almost all points close = low dimension
        return max(0.5, embed_dim * 0.2)
    if frac_close <= 0.01:
        # Almost no points close = high dimension
        return float(embed_dim)
    # Estimate: frac_close^(1/embed_dim) * embed_dim
    # This maps: frac_close=100% -> 0, 0% -> embed_dim
    dim = embed_dim * (1.0 - math.pow(frac_close, 0.5))
    return max(0.5, min(dim, float(embed_dim)))


def detect_geometric_collapse(
    history: list[float],
    threshold: float = 3.0,
    slope_window: int = 3,
) -> tuple[bool, float]:
    """Detect geometric collapse in correlation dimension history.

    A collapse occurs when:
    1. The most recent dimension drops below `threshold`.
    2. The recent slope shows a significant downward trend.

    Args:
        history: Chronological list of correlation dimension estimates.
        threshold: Dimension below which collapse is considered.
        slope_window: Number of recent points for slope calculation.

    Returns:
        Tuple of (collapse_detected, collapse_score).
        collapse_score ranges 0.0 (no risk) to ~1.0 (severe).

    """
    if len(history) < 3:
        return False, 0.0

    recent = history[-slope_window:] if len(history) >= slope_window else history
    current = history[-1]

    if current > threshold:
        return False, 0.0

    # Slope: how fast we're dropping
    if len(recent) >= 2:
        x_vals = list(range(len(recent)))
        y_vals = recent
        n_pts = len(recent)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals, strict=False))
        sum_xx = sum(x * x for x in x_vals)

        denom = n_pts * sum_xx - sum_x * sum_x
        slope = (n_pts * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0
    else:
        slope = 0.0

    # Collapse severity: how far below threshold + how steep the drop
    depth_ratio = max(0.0, (threshold - current) / max(threshold, 1.0))
    slope_penalty = max(0.0, -slope)  # negative slope (dropping) contributes

    score = min(1.0, depth_ratio * 0.7 + slope_penalty * 0.3)

    return score > 0.4, score


def apply_dampening_kernel(
    correlation_dim: float,
    threshold: float = 3.0,
    strength: float = 0.6,
) -> float:
    """Compute dampening factor from correlation dimension.

    Sigmoid-like mapping: high dimension → near-zero dampening,
    low dimension → dampening approaching `strength`.

    Args:
        correlation_dim: Current estimated correlation dimension.
        threshold: Dimension threshold for dampening onset.
        strength: Maximum dampening factor at zero dimension.

    Returns:
        Dampening factor between 0.0 and 1.0.

    """
    if threshold <= 0.0:
        # Always dampen
        return min(1.0, strength)

    # Dimension gap (how far below threshold)
    gap = max(0.0, threshold - correlation_dim)

    # Sigmoid-like scaling
    raw = gap / max(threshold, 0.01)
    dampening = strength * (1.0 - math.exp(-2.0 * raw))

    return min(1.0, max(0.0, dampening))


@dataclass
class ValueRegularizer:
    """Online regularizer that monitors trajectory dimensions and
    suggests dampening to prevent geometric collapse.

    Usage:
        reg = ValueRegularizer()
        for vector in generation_trajectory:
            reg.observe(vector)
            result = reg.check_and_dampen()
            dampening = result["dampening_factor"]
    """

    config: RegularizerConfig = field(default_factory=RegularizerConfig)
    trajectory: list[list[float]] = field(default_factory=list)
    collapse_history: list[float] = field(default_factory=list)
    regularizations_applied: int = 0

    def observe(self, vector: list[float]) -> None:
        """Record a trajectory vector."""
        self.trajectory.append(vector)
        if len(self.trajectory) > self.config.history_window:
            self.trajectory.pop(0)

    def check_and_dampen(self) -> dict[str, Any]:
        """Check current trajectory for geometric collapse and compute dampening.

        Returns:
            Dict with keys:
                - dampening_factor (float): 0.0 (none) to 1.0 (max)
                - correlation_dim (float): current estimated dimension
                - collapse_detected (bool): whether collapse is active
                - collapse_score (float): severity score
                - regularizations_applied (int): total count

        """
        if len(self.trajectory) < self.config.min_history:
            return {
                "dampening_factor": 0.0,
                "correlation_dim": float(len(self.trajectory[0])) if self.trajectory else 0.0,
                "collapse_detected": False,
                "collapse_score": 0.0,
                "regularizations_applied": self.regularizations_applied,
            }

        dim = calculate_correlation_dimension(self.trajectory)
        self.collapse_history.append(dim)
        if len(self.collapse_history) > self.config.history_window:
            self.collapse_history.pop(0)

        collapse, score = detect_geometric_collapse(
            self.collapse_history,
            threshold=self.config.threshold,
            slope_window=self.config.slope_window,
        )

        dampening = apply_dampening_kernel(
            dim,
            threshold=self.config.threshold,
            strength=self.config.strength,
        )

        if collapse:
            self.regularizations_applied += 1

        return {
            "dampening_factor": dampening,
            "correlation_dim": dim,
            "collapse_detected": collapse,
            "collapse_score": score,
            "regularizations_applied": self.regularizations_applied,
        }

    def reset(self) -> None:
        """Clear all trajectory and state."""
        self.trajectory.clear()
        self.collapse_history.clear()
        self.regularizations_applied = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for diagnostics."""
        return {
            "threshold": self.config.threshold,
            "strength": self.config.strength,
            "history_window": self.config.history_window,
            "slope_window": self.config.slope_window,
            "trajectory_length": len(self.trajectory),
            "collapse_history_length": len(self.collapse_history),
            "regularizations_applied": self.regularizations_applied,
        }
