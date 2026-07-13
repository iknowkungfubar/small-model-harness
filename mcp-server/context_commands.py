"""
Context Commands — MCP server tool implementations for the small-model-harness.

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
from collections import Counter
from typing import Any

# Import the plugin's budget manager for consistent budget tracking
import sys
from pathlib import Path

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


def _get_or_create_budget(session_id: str, stated_window: int = 32000) -> ContextBudgetManager:
    """Get or create a budget manager for a session."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = ContextBudgetManager(model_window=stated_window)
        _SESSION_STEPS[session_id] = []
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


# ---------------------------------------------------------------------------
# Compaction Engine
# ---------------------------------------------------------------------------


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

    # Human-readable status label
    if budget.utilization >= 0.9:
        stats["status"] = "critical"
        stats["recommendation"] = "Compact immediately before continuing"
    elif budget.utilization >= 0.8:
        stats["status"] = "warning"
        stats["recommendation"] = "Compaction recommended to free context space"
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

    # Add session_id to result
    result["session_id"] = session_id
    result.pop("steps", None)  # Don't send full steps in MCP response

    return json.dumps(result, indent=2)
