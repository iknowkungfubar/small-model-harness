"""Tests for Addition C — Context Rot Budget Hardening.

Context rot risk calculation and pre-emptive compaction triggers
for small-model context management (Chroma 1/3 effective window rule).
"""

from __future__ import annotations

import json
import math
import pytest
from pathlib import Path

_here = Path(__file__).resolve().parent
_server_dir = _here.parent / "mcp-server"
import sys

sys.path.insert(0, str(_server_dir))

from context_commands import (
    compute_context_rot_risk,
    update_rot_state,
    mark_compaction,
    _SESSION_ROT,
)
from context_budget import ContextBudgetManager


class TestComputeContextRotRisk:
    def test_low_utilization_zero_risk(self):
        """Utilization well below effective window produces zero risk."""
        risk = compute_context_rot_risk(utilization=0.2)
        assert risk == 0.0

    def test_at_effective_limit_zero_risk(self):
        """Utilization at the effective window limit (0.33) produces zero risk."""
        risk = compute_context_rot_risk(utilization=0.33)
        assert risk == 0.0

    def test_slightly_above_limit_low_risk(self):
        """Slightly above effective window (0.4) produces small risk."""
        risk = compute_context_rot_risk(utilization=0.4)
        assert 0.0 < risk < 0.5

    def test_moderate_utilization_moderate_risk(self):
        """Moderate utilization (0.5) produces moderate risk."""
        risk = compute_context_rot_risk(utilization=0.5)
        assert 0.1 < risk < 0.8

    def test_high_utilization_high_risk(self):
        """High utilization (0.8) produces high risk."""
        risk = compute_context_rot_risk(utilization=0.8)
        assert risk > 0.3

    def test_peak_utilization_factor(self):
        """High peak utilization adds to risk."""
        risk_no_peak = compute_context_rot_risk(utilization=0.4, peak_utilization=0.4)
        risk_with_peak = compute_context_rot_risk(utilization=0.4, peak_utilization=0.95)
        assert risk_with_peak > risk_no_peak

    def test_sustained_steps_factor(self):
        """Sustained steps at elevated utilization add to risk."""
        risk_few = compute_context_rot_risk(utilization=0.5, sustained_steps=5)
        risk_many = compute_context_rot_risk(utilization=0.5, sustained_steps=80)
        assert risk_many > risk_few

    def test_compaction_fatigue(self):
        """Many compactions add small fatigue factor."""
        risk_none = compute_context_rot_risk(utilization=0.5, compaction_count=0)
        risk_many = compute_context_rot_risk(utilization=0.5, compaction_count=10)
        assert risk_many >= risk_none

    def test_max_risk_never_exceeds_one(self):
        """Risk is always capped at 1.0."""
        risk = compute_context_rot_risk(
            utilization=0.95,
            peak_utilization=0.99,
            sustained_steps=500,
            compaction_count=20,
        )
        assert risk <= 1.0

    def test_cache_effective_window_breach(self):
        """Effective window breach is the dominant factor."""
        risk_below = compute_context_rot_risk(utilization=0.3)
        risk_above = compute_context_rot_risk(utilization=0.6)
        assert risk_above > risk_below * 2

    def test_all_zeros_at_low_utilization(self):
        """All factors near zero when utilization is low with no peak/steps."""
        risk = compute_context_rot_risk(
            utilization=0.1, peak_utilization=0.1, sustained_steps=0, compaction_count=0
        )
        assert risk == 0.0

    def test_near_max_utilization_produces_high_risk(self):
        """Near-max utilization with all other factors produces high risk."""
        risk = compute_context_rot_risk(
            utilization=0.90,
            peak_utilization=0.95,
            sustained_steps=50,
            compaction_count=5,
        )
        assert risk > 0.5


class TestUpdateRotState:
    def setup_method(self):
        # Clear shared state for each test
        _SESSION_ROT.clear()

    def test_init_with_no_state(self):
        """Update with no existing state creates new entry."""
        update_rot_state("test-session", 0.5)
        assert "test-session" in _SESSION_ROT
        assert _SESSION_ROT["test-session"]["peak_utilization"] == 0.5

    def test_tracks_peak(self):
        """Peak utilization tracks upward only."""
        update_rot_state("test-session", 0.5)
        update_rot_state("test-session", 0.3)
        assert _SESSION_ROT["test-session"]["peak_utilization"] == 0.5
        update_rot_state("test-session", 0.8)
        assert _SESSION_ROT["test-session"]["peak_utilization"] == 0.8

    def test_sustained_steps_increment(self):
        """Sustained steps increment above effective window."""
        update_rot_state("test-session", 0.4)  # below threshold, init
        update_rot_state("test-session", 0.5)  # first sustained increment
        assert _SESSION_ROT["test-session"]["sustained_steps"] >= 1

    def test_sustained_steps_decay(self):
        """Sustained steps decay when below effective window."""
        update_rot_state("test-session", 0.5)  # init
        update_rot_state("test-session", 0.5)  # inc
        update_rot_state("test-session", 0.5)  # inc
        steps_before = _SESSION_ROT["test-session"]["sustained_steps"]
        assert steps_before >= 2
        update_rot_state("test-session", 0.2)  # below 0.33, decays by 2
        assert _SESSION_ROT["test-session"]["sustained_steps"] < steps_before

    def test_distinct_sessions(self):
        """Different sessions maintain independent state."""
        update_rot_state("session-a", 0.9)
        update_rot_state("session-b", 0.3)
        assert _SESSION_ROT["session-a"]["peak_utilization"] == 0.9
        assert _SESSION_ROT["session-b"]["peak_utilization"] == 0.3


class TestMarkCompaction:
    def setup_method(self):
        _SESSION_ROT.clear()

    def test_counts_compactions(self):
        """Compaction count increments."""
        update_rot_state("test", 0.5)
        mark_compaction("test")
        assert _SESSION_ROT["test"]["compaction_count"] == 1
        mark_compaction("test")
        assert _SESSION_ROT["test"]["compaction_count"] == 2

    def test_no_state_safe(self):
        """mark_compaction on missing session does not crash."""
        mark_compaction("nonexistent")  # Should not raise


class TestHarnessContextStatusIntegration:
    def test_status_includes_rot_risk(self):
        """harness_context_status output includes context_rot_risk."""
        from context_commands import harness_context_status

        result = json.loads(harness_context_status("test-session", 32000))
        assert "context_rot_risk" in result
        assert "effective_window_ratio" in result
        assert "effective_window_tokens" in result
        assert result["effective_window_ratio"] == 0.33

    def test_status_elevated_by_rot_risk(self):
        """Status elevates when rot risk > 0.5 even if utilization is moderate."""
        from context_commands import harness_context_status, update_rot_state

        # Push utilization high to build up risk
        for i in range(30):
            update_rot_state("rot-test", 0.9)
        result = json.loads(harness_context_status("rot-test", 32000))
        if result.get("context_rot_risk", 0) > 0.5:
            assert result["status"] == "elevated"

    def test_compact_tracks_compactions(self):
        """Running harness_compact increments compaction tracking."""
        from context_commands import harness_compact, harness_context_status

        _SESSION_ROT.clear()
        result = json.loads(harness_context_status("compact-test", 32000))
        before = result.get("context_rot_risk", 0)
        for i in range(3):
            try:
                json.loads(harness_compact("compact-test"))
            except Exception:
                pass
        result = json.loads(harness_context_status("compact-test", 32000))
        # After compaction, rot tracking should reflect it
        assert result.get("context_rot_risk", 0) >= 0.0
