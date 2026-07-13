"""Context Commands — MCP server tool implementations for the small-model-harness.

Provides:
- compact_context(): Sliding-window compaction engine
- harness_context_status(): Current budget state for a session
- harness_compact(): Trigger compaction for a session
- record_step(): Record a tool call step

These are the agent-visible interface layer, complementing the
plugin's automatic enforcement hooks.
"""

from __future__ import annotations

import json
import logging

# Import the plugin's budget manager for consistent budget tracking
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Add plugin directory to path so we can reuse ContextBudgetManager
_PLUGIN_DIR = Path(__file__).parent.parent.parent / "plugins" / "small-model-harness"
sys.path.insert(0, str(_PLUGIN_DIR.resolve()))

from context_budget import ContextBudgetManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session state (separate from plugin state — MCP server is its own process)
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, ContextBudgetManager] = {}
_SESSION_STEPS: dict[str, list[dict]] = {}
_SESSION_ROT: dict[str, dict[str, float | int]] = {}

# Effective window ratio for small models (Chroma 1/3 rule)
# Small models degrade well before the stated window is full
_EFFECTIVE_WINDOW_RATIO = 0.33


def _get_or_create_budget(session_id: str, stated_window: int = 32000) -> ContextBudgetManager:
    """Get or create a budget manager for a session."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = ContextBudgetManager(model_window=stated_window)
        _SESSION_STEPS[session_id] = []
        _SESSION_ROT[session_id] = {"peak_utilization": 0.0, "compaction_count": 0, "sustained_steps": 0}
    return _SESSIONS[session_id]


def record_step(
    session_id: str,
    tool_name: str,
    args: dict | None,
    result: Any,
) -> None:
    """Record a tool call step for a session.

    This is called by both the MCP server and can be called by the
    plugin to share state across processes (or used independently).
    """
    budget = _get_or_create_budget(session_id, 32000)
    budget.record_step(tool_name, args, result)

    # Keep a copy of each step for compaction
    step_record = {
        "tool": tool_name,
        "args": args or {},
        "result": str(result)[:200] if result is not None else "",
        "tokens": len(str(result or "")) // 4 + 250,  # Rough token estimate
    }
    _SESSION_STEPS.setdefault(session_id, []).append(step_record)


def compute_context_rot_risk(
    utilization: float,
    effective_window_ratio: float = _EFFECTIVE_WINDOW_RATIO,
    peak_utilization: float = 0.0,
    sustained_steps: int = 0,
    compaction_count: int = 0,
) -> float:
    """Compute context rot risk score (0.0-1.0).

    Context rot is the degradation in model output quality as the context
    window fills, particularly relevant for small models where the effective
    window is much smaller than the stated window.

    Factors:
    1. Effective window breach (Chroma 1/3 rule): how far past effective limit
    2. Peak utilization: how high budget has ever gone
    3. Sustained usage: number of steps at elevated utilization
    4. Compaction fatigue: diminishing returns from repeated compaction

    Args:
        utilization: Current budget utilization (0.0-1.0).
        effective_window_ratio: Ratio of stated window that's truly effective.
        peak_utilization: Highest utilization seen this session.
        sustained_steps: Number of steps recorded at elevated utilization.
        compaction_count: Number of compactions already performed.

    Returns:
        Risk score 0.0 (safe) to 1.0 (critical).
    """
    # 1. Effective window breach
    effective_limit = effective_window_ratio
    if utilization <= effective_limit:
        eff_factor = 0.0
    else:
        # Linear increase from 0 at effective_limit to 1.0 at 2x effective limit
        eff_factor = (utilization - effective_limit) / max(effective_limit, 0.01)
        eff_factor = min(1.0, eff_factor)

    # 2. Peak utilization penalty
    peak_factor = 0.0
    if peak_utilization > 0.8:
        peak_factor = (peak_utilization - 0.8) / 0.2  # 0.0 at 80%, 1.0 at 100%

    # 3. Sustained usage penalty
    sustained_factor = min(1.0, sustained_steps / 100.0)

    # 4. Compaction fatigue: each compaction provides less benefit
    # First few compactions help, beyond 5 there's diminishing returns
    compaction_factor = min(1.0, compaction_count / 10.0)

    # Combined: weighted sum
    risk = (
        eff_factor * 0.5 +         # Effective window breach: most important
        peak_factor * 0.2 +         # Peak utilization stress
        sustained_factor * 0.2 +    # Sustained high usage
        compaction_factor * 0.1     # Compaction fatigue
    )

    return round(min(1.0, risk), 4)


def update_rot_state(session_id: str, utilization: float) -> None:
    """Update rotation tracking state for a session."""
    rot = _SESSION_ROT.get(session_id)
    if rot is None:
        _SESSION_ROT[session_id] = {"peak_utilization": utilization, "compaction_count": 0, "sustained_steps": 0}
        return

    if utilization > rot["peak_utilization"]:
        rot["peak_utilization"] = utilization

    # Increment sustained steps when utilization is elevated
    if utilization > _EFFECTIVE_WINDOW_RATIO:
        rot["sustained_steps"] = rot["sustained_steps"] + 1  # type: ignore[operator]
    else:
        # Decay sustained steps when utilization drops
        rot["sustained_steps"] = max(0, rot["sustained_steps"] - 2)  # type: ignore[operator]


def mark_compaction(session_id: str) -> None:
    """Record that a compaction was performed."""
    rot = _SESSION_ROT.get(session_id)
    if rot is not None:
        rot["compaction_count"] = rot["compaction_count"] + 1  # type: ignore[operator]



def _generate_summary(steps: list[dict]) -> str:
    """Generate a structured bullet-point summary of old steps.

    Args:
        steps: List of step dicts to summarize.

    Returns:
        A compact structured string (~200-500 tokens worth of text).

    """
    if not steps:
        return "No steps to summarize."

    tool_counts = Counter(s["tool"] for s in steps)
    lines: list[str] = []

    lines.append(f"# Summarized Context ({len(steps)} steps)")
    lines.append("")

    # Tool frequency overview
    most_common = tool_counts.most_common(5)
    tools_summary = ", ".join(f"{tool} ({count}x)" for tool, count in most_common)
    lines.append(f"Tools used: {tools_summary}")
    lines.append("")

    # Key operations — group by tool for conciseness
    lines.append("Key operations:")
    for s in steps[:15]:  # Cap at 15 to keep summary tight
        tool = s.get("tool", "?")
        args_str = str(s.get("args", {}))[:80]
        result_str = str(s.get("result", ""))[:100]
        lines.append(f"- {tool}: {args_str} -> {result_str}")

    if len(steps) > 15:
        lines.append(f"- ... ({len(steps) - 15} more operations omitted)")

    return "\n".join(lines)


def compact_context(
    steps: list[dict] | None,
    model_window: int = 32000,
    sliding_window: int = 5,
) -> dict[str, Any]:
    """Compact a list of steps using a sliding-window strategy.

    Keeps the last N steps intact. Summarizes all older steps into a
    single compact structured summary.

    Args:
        steps: The list of step dicts to compact.
        model_window: The model's stated context window (for token estimates).
        sliding_window: Number of most recent steps to keep intact.

    Returns:
        Dict with:
            steps_before: Original step count
            steps_after: New step count (1 summary + intact)
            tokens_freed: Estimated tokens freed by compaction
            steps: The new compacted step list (or original if no compaction needed)

    """
    if not steps:
        return {
            "steps_before": 0,
            "steps_after": 0,
            "tokens_freed": 0,
            "steps": [],
            "compact_applied": False,
        }

    if len(steps) <= sliding_window + 1:
        # No benefit: summarizing 1 step doesn't reduce step count
        # and adds summary overhead. Only compact when >=2 steps
        # can be collapsed into 1 summary.
        return {
            "steps_before": len(steps),
            "steps_after": len(steps),
            "tokens_freed": 0,
            "steps": steps,
            "compact_applied": False,
        }

    # Split: old steps to summarize, recent steps to keep intact
    summarize_steps = steps[:-sliding_window]
    intact_steps = steps[-sliding_window:]

    # Generate compact summary
    summary_text = _generate_summary(summarize_steps)

    # Token accounting
    tokens_before = sum(s.get("tokens", 0) for s in summarize_steps)
    tokens_after = max(len(summary_text) // 4 + 100, 50)  # Rough estimate

    # Create summary step
    summary_step: dict = {
        "tool": "COMPACTION_SUMMARY",
        "args": {"summarized_count": len(summarize_steps)},
        "result": summary_text,
        "tokens": tokens_after,
    }

    new_steps = [summary_step] + intact_steps

    return {
        "steps_before": len(steps),
        "steps_after": len(new_steps),
        "tokens_freed": tokens_before - tokens_after,
        "steps": new_steps,
        "compact_applied": True,
    }


# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------


def harness_context_status(
    session_id: str = "default",
    stated_window: int = 32000,
) -> str:
    """Get context budget status for a session.

    This is the agent-facing tool that shows current budget state,
    utilization metrics, and compaction recommendations.

    Args:
        session_id: Session identifier (default: "default").
        stated_window: Model's stated context window in tokens.

    Returns:
        JSON string with budget status fields.

    """
    budget = _get_or_create_budget(session_id, stated_window)
    stats = budget.stats()
    stats["session_id"] = session_id
    stats["effective_window_ratio"] = _EFFECTIVE_WINDOW_RATIO
    stats["effective_window_tokens"] = int(stated_window * _EFFECTIVE_WINDOW_RATIO)

    # Context rot risk
    update_rot_state(session_id, budget.utilization)
    rot_state = _SESSION_ROT.get(session_id, {})
    rot_risk = compute_context_rot_risk(
        utilization=budget.utilization,
        peak_utilization=rot_state.get("peak_utilization", 0.0),  # type: ignore[arg-type]
        sustained_steps=int(rot_state.get("sustained_steps", 0)),  # type: ignore[arg-type]
        compaction_count=int(rot_state.get("compaction_count", 0)),  # type: ignore[arg-type]
    )
    stats["context_rot_risk"] = rot_risk

    # Human-readable status label with rot risk consideration
    if budget.utilization >= 0.9:
        stats["status"] = "critical"
        stats["recommendation"] = "Compact immediately before continuing"
    elif budget.utilization >= 0.8:
        stats["status"] = "warning"
        stats["recommendation"] = "Compaction recommended to free context space"
    elif rot_risk > 0.5:
        stats["status"] = "elevated"
        stats["recommendation"] = "Context rot risk elevated — consider compaction for quality"
    elif budget.utilization >= 0.65:
        stats["status"] = "elevated"
        stats["recommendation"] = "Monitor — budget is comfortably in use"
    else:
        stats["status"] = "normal"
        stats["recommendation"] = "Budget has ample room"

    return json.dumps(stats, indent=2)


def harness_compact(
    session_id: str = "default",
    sliding_window: int = 5,
) -> str:
    """Compact context for a session using sliding-window summarization.

    Keeps the last N (sliding_window) steps intact and summarizes
    all older steps into a compact structured overview. Frees token
    budget for continued work.

    Args:
        session_id: Session identifier (default: "default").
        sliding_window: Number of most recent steps to keep intact (default: 5).

    Returns:
        JSON string with compaction results (tokens freed, step counts).

    """
    if session_id not in _SESSION_STEPS:
        return json.dumps({
            "tokens_freed": 0,
            "steps_before": 0,
            "steps_after": 0,
            "note": "No steps recorded for this session",
            "steps": [],
        })

    steps = _SESSION_STEPS[session_id]
    result = compact_context(steps, sliding_window=sliding_window)

    # Update session state with compacted steps
    _SESSION_STEPS[session_id] = result["steps"]

    # Update budget to reflect freed tokens
    budget = _SESSIONS.get(session_id)
    if budget and result["tokens_freed"] > 0:
        budget.mark_compaction()
        mark_compaction(session_id)

    # Add session_id to result
    result["session_id"] = session_id
    result.pop("steps", None)  # Don't send full steps in MCP response

    return json.dumps(result, indent=2)
