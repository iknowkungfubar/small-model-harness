"""Tests for Addition A — Value Regularizer (RMR).

arXiv 2605.00435 (ICML 2026): Low-rank, eigenvalue-thresholded dampening
applied to the Transformer's value cache. Monitors correlation dimension
of the generation trajectory; when geometric collapse detected, dampens
self-reinforcing directions.
"""

from __future__ import annotations

import math
import pytest
from pathlib import Path

_here = Path(__file__).resolve().parent
_server_dir = _here.parent / "mcp-server"
import sys

sys.path.insert(0, str(_server_dir))

from value_regularizer import (
    ValueRegularizer,
    RegularizerConfig,
    calculate_correlation_dimension,
    detect_geometric_collapse,
    apply_dampening_kernel,
)


class TestCalculateCorrelationDimension:
    def test_high_dimension_random_vectors(self):
        """Random vectors produce reasonable correlation dimension."""
        vectors = [[i * 0.1 + j * 0.01 for j in range(8)] for i in range(20)]
        dim = calculate_correlation_dimension(vectors)
        assert 0.0 <= dim <= 12.0

    def test_low_dimension_collinear(self):
        """Collinear vectors produce low correlation dimension (~1)."""
        vectors = [[1.0 * i, 2.0 * i, 3.0 * i] for i in range(10)]
        dim = calculate_correlation_dimension(vectors)
        assert dim < 3.0

    def test_single_vector_returns_one(self):
        """Single vector returns dimension of 1."""
        dim = calculate_correlation_dimension([[1.0, 2.0, 3.0]])
        assert dim == 1.0

    def test_empty_returns_zero(self):
        """Empty list returns 0."""
        dim = calculate_correlation_dimension([])
        assert dim == 0.0

    def test_identical_vectors(self):
        """Identical vectors produce very low dimension."""
        vectors = [[1.0, 2.0, 3.0]] * 10
        dim = calculate_correlation_dimension(vectors)
        assert dim < 2.0


class TestDetectGeometricCollapse:
    def test_no_collapse_high_dim(self):
        """High and stable dimensions do not trigger collapse."""
        history = [6.0, 5.8, 6.2, 5.9, 6.1]
        collapse, score = detect_geometric_collapse(history, threshold=3.0)
        assert not collapse
        assert score < 0.5

    def test_collapse_detected(self):
        """Dropping below threshold triggers collapse."""
        history = [6.0, 5.0, 4.0, 3.0, 2.0]
        collapse, score = detect_geometric_collapse(history, threshold=3.0)
        assert collapse
        assert score > 0.5

    def test_short_history(self):
        """Short history (<3) does not trigger."""
        history = [6.0]
        collapse, _ = detect_geometric_collapse(history, threshold=3.0)
        assert not collapse

    def test_rapid_drop(self):
        """Rapid drop rate produces high collapse score."""
        history = [9.0, 7.0, 5.0, 3.0, 1.0]
        _, score = detect_geometric_collapse(history, threshold=2.0)
        assert score > 0.6


class TestApplyDampeningKernel:
    def test_high_dim_no_dampening(self):
        """High correlation dimension produces minimal dampening."""
        result = apply_dampening_kernel(correlation_dim=7.0, threshold=3.0, strength=0.5)
        assert 0.0 <= result <= 0.3

    def test_low_dim_strong_dampening(self):
        """Very low correlation dimension produces strong dampening."""
        result = apply_dampening_kernel(correlation_dim=1.5, threshold=3.0, strength=1.0)
        assert result > 0.5

    def test_zero_threshold_always_dampens(self):
        """Threshold of 0 means always dampen."""
        result = apply_dampening_kernel(correlation_dim=3.0, threshold=0.0, strength=1.0)
        assert result > 0.3


class TestValueRegularizer:
    def test_init_state(self):
        """Initial state is clean."""
        reg = ValueRegularizer()
        assert len(reg.collapse_history) == 0
        assert reg.regularizations_applied == 0

    def test_observe_and_check(self):
        """Push observations, check for collapse."""
        reg = ValueRegularizer()
        for i in range(5):
            reg.observe([math.sin(i), math.cos(i), math.sin(i * 2)])
        result = reg.check_and_dampen()
        assert isinstance(result, dict)
        assert "dampening_factor" in result
        assert "correlation_dim" in result
        assert "collapse_detected" in result

    def test_reset(self):
        """Reset clears state."""
        reg = ValueRegularizer()
        reg.observe([1.0, 2.0, 3.0])
        reg.regularizations_applied = 3
        reg.reset()
        assert len(reg.collapse_history) == 0
        assert reg.regularizations_applied == 0

    def test_to_dict(self):
        """Serialization roundtrip preserves config."""
        reg = ValueRegularizer(RegularizerConfig(threshold=2.5, strength=0.8))
        d = reg.to_dict()
        assert d["threshold"] == 2.5
        assert d["strength"] == 0.8
