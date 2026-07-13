"""
Small-Model Harness — MCP Server

Provides tools for context budget management, task classification,
and model routing.

Tools:
  - harness_context_status: Query current budget state
  - harness_compact: Trigger context compaction
  - harness_classify_task: Classify task complexity
  - harness_route: Route task to model tier
  - harness_reset: Reset all harness state

Runs as a stdio MCP server for Hermes Agent integration.

Usage:
    python3 server.py
    # or from Hermes: hermes mcp add small-model-harness ...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the server can import sibling modules
_server_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_server_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("small-model-harness.mcp")

try:
    from fastmcp import FastMCP
except ImportError:
    logger.error("FastMCP not installed. Run: pip install fastmcp")
    sys.exit(1)

from context_commands import harness_context_status as _status_impl
from context_commands import harness_compact as _compact_impl
from routing_commands import harness_classify_task as _classify_impl
from routing_commands import harness_route as _route_impl

# ---------------------------------------------------------------------------
# FastMCP Application
# ---------------------------------------------------------------------------

app = FastMCP(
    "Small Model Harness",
    instructions="Context budget management and compaction for small local LLMs",
)


@app.tool(description="Get context budget status for a session")
def harness_context_status(
    session_id: str = "default",
    stated_window: int = 32000,
) -> str:
    """Query the current context budget state for a given session.

    Returns utilization metrics, effective capacity, token counts,
    compaction history, and a human-readable status recommendation.

    Args:
        session_id: Session identifier (default: "default"). Use the same
            ID across calls to maintain per-session budget tracking.
        stated_window: Model's stated context window in tokens. The
            effective capacity is calculated as stated_window / 3 per
            the Chroma/NVIDIA 1/3 effective window rule.
            Default: 32000 (Qwen3-9B class).

    Returns:
        JSON string with budget status fields including:
        - stated_window: The configured window
        - effective_capacity: stated_window / 3
        - used_tokens: Estimated tokens consumed
        - available_tokens: Remaining budget
        - utilization: Ratio (0.0-1.0)
        - steps: Number of tool call steps recorded
        - compactions: Number of compactions performed
        - status: "normal" | "elevated" | "warning" | "critical"
        - recommendation: Human-readable guidance
    """
    return _status_impl(session_id=session_id, stated_window=stated_window)


@app.tool(description="Compact context to free token budget")
def harness_compact(
    session_id: str = "default",
    sliding_window: int = 5,
) -> str:
    """Compact session context by summarizing old steps.

    Keeps the last N (sliding_window) steps intact and summarizes
    all older steps into a single compact overview. This frees token
    budget for continued work and reduces context pressure.

    Args:
        session_id: Session identifier (default: "default"). Must match
            the ID used in harness_context_status calls.
        sliding_window: Number of most recent steps to keep intact.
            Default: 5. Higher values preserve more detail but free
            fewer tokens.

    Returns:
        JSON string with compaction results:
        - tokens_freed: Estimated tokens reclaimed
        - steps_before: Original step count
        - steps_after: New step count (summary + intact)
        - session_id: The session that was compacted
    """
    return _compact_impl(session_id=session_id, sliding_window=sliding_window)


@app.tool(description="Classify task complexity and suggest model tier")
def harness_classify_task(
    task: str,
) -> str:
    """Classify a task's complexity and suggest the appropriate model tier.

    Uses rule-based heuristics (no LLM call) to estimate complexity,
    reasoning depth, tool count, and context needs. Returns a profile
    with tier recommendation, confidence score, and extracted features.

    Args:
        task: The task description or user request to classify.

    Returns:
        JSON string with the classification TaskProfile.
    """
    return _classify_impl(task=task)


@app.tool(description="Route a task to a model tier with cascade logic")
def harness_route(
    task: str,
    current_tier: str = "t2",
    failure_count: int = 0,
    available_tiers: str | None = None,
) -> str:
    """Route a task to the appropriate model tier.

    Classifies the task and produces a routing decision including
    cascade path, alternatives, and reasoning. Supports failure-based
    escalation — pass current_tier and failure_count to trigger
    automatic tier upgrades.

    Args:
        task: The task description to route.
        current_tier: Currently active tier (default: "t2").
        failure_count: Number of consecutive failures (default: 0).
        available_tiers: Comma-separated list, e.g. "t1,t2,t3,t4".
            Default: all tiers available.

    Returns:
        JSON string with routing decision.
    """
    return _route_impl(
        task=task,
        current_tier=current_tier,
        failure_count=failure_count,
        available_tiers=available_tiers,
    )


@app.tool(description="Reset all harness state for a session")
def harness_reset(
    session_id: str = "default",
) -> str:
    """Reset all budget tracking and step history for a session.

    Useful when starting a completely new task within the same
    conversation.

    Args:
        session_id: Session identifier (default: "default").

    Returns:
        JSON string confirming the reset.
    """
    from context_commands import _SESSIONS, _SESSION_STEPS

    _SESSIONS.pop(session_id, None)
    _SESSION_STEPS.pop(session_id, None)

    import json
    return json.dumps({
        "action": "reset",
        "session_id": session_id,
        "result": "ok",
    })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Small-Model Harness MCP server (stdio)")
    app.run(transport="stdio")
