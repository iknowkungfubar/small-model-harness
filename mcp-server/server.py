"""Small-Model Harness — MCP Server

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
_plugin_dir = _server_dir.parent / "hermes-plugin"
sys.path.insert(0, str(_server_dir))
sys.path.insert(0, str(_plugin_dir))

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

from confidence import estimate_confidence
from context_commands import harness_compact as _compact_impl
from context_commands import harness_context_status as _status_impl
from guardrails import check_input, check_output
from routing_commands import harness_classify_task as _classify_impl
from routing_commands import harness_route as _route_impl
from verification import check_consistency

_guardrails_available = True

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
    from context_commands import _SESSION_STEPS, _SESSIONS

    _SESSIONS.pop(session_id, None)
    _SESSION_STEPS.pop(session_id, None)

    import json

    return json.dumps({
        "action": "reset",
        "session_id": session_id,
        "result": "ok",
    })


@app.tool(
    description="Estimate confidence in model responses using token probabilities and semantic entropy"
)
def harness_estimate_confidence(
    responses_json: str,
    logprobs_json: str | None = None,
) -> str:
    """Estimate a unified confidence score (0.0–1.0) for one or more model responses.

    Combines token-level probability signals (when logprobs available)
    and semantic dispersion across multiple responses to produce a
    confidence score with routing recommendation.

    Args:
        responses_json: JSON array of response dicts or strings, e.g.
            [{"text": "response 1"}, {"text": "response 2"}]
        logprobs_json: Optional JSON array of full API response objects
            containing logprobs (OpenAI-compatible format).

    Returns:
        JSON string with confidence_score, recommendation,
        signal_flags, semantic_dispersion, and token_stats.

    """
    import json

    try:
        responses = json.loads(responses_json)
        if not isinstance(responses, list):
            return json.dumps({"error": "responses_json must be a JSON array"})
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Failed to parse responses_json: {e}"})

    logprobs_responses = None
    if logprobs_json:
        try:
            logprobs_responses = json.loads(logprobs_json)
        except (json.JSONDecodeError, TypeError) as e:
            return json.dumps({"error": f"Failed to parse logprobs_json: {e}"})

    try:
        result = estimate_confidence(
            responses=responses,
            logprobs_responses=logprobs_responses,
        )
        return json.dumps({
            "confidence_score": result.confidence_score,
            "recommendation": result.recommendation,
            "signal_flags": result.signal_flags,
            "semantic_dispersion": result.semantic_dispersion,
            "n_responses": result.n_responses,
            "n_clusters": result.n_clusters,
            "cluster_sizes": result.cluster_sizes,
            "has_token_stats": result.token_stats is not None,
            "token_mean_probability": result.token_stats.mean_probability
            if result.token_stats
            else None,
        })
    except Exception as e:
        return json.dumps({"error": f"Confidence estimation failed: {e}"})


@app.tool(description="Verify consistency across multiple model responses")
def harness_verify_consistency(
    responses_json: str,
    key_fields: str | None = None,
) -> str:
    """Check self-consistency across multiple responses.

    Compares specified (or auto-detected) fields across N responses
    and reports agreement ratio, anomalies, and field-level comparisons.

    Args:
        responses_json: JSON array of response dicts or strings.
        key_fields: Optional comma-separated list of fields to compare.
            Auto-detected from response structure if omitted.

    Returns:
        JSON string with is_consistent, agreement_ratio, anomalies,
        and field-level comparisons.

    """
    import json

    try:
        responses = json.loads(responses_json)
        if not isinstance(responses, list):
            return json.dumps({"error": "responses_json must be a JSON array"})
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Failed to parse responses_json: {e}"})

    field_list = key_fields.split(",") if key_fields else None

    try:
        result = check_consistency(responses=responses, key_fields=field_list)
        return json.dumps({
            "is_consistent": result.is_consistent,
            "confidence": result.confidence,
            "agreement_ratio": result.agreement_ratio,
            "total_fields": result.total_fields,
            "matching_fields": result.matching_fields,
            "anomalies": result.anomalies,
            "comparisons": [
                {
                    "field_name": c.field_name,
                    "all_match": c.all_match,
                    "agreement_ratio": c.agreement_ratio,
                }
                for c in result.comparisons
            ],
        })
    except Exception as e:
        return json.dumps({"error": f"Consistency check failed: {e}"})


@app.tool(description="Run input or output guardrails on text")
def harness_check_guardrails(
    mode: str,
    text: str,
) -> str:
    """Run guardrail checks on user input or model output.

    Supports input guardrails (injection detection, jailbreak, PII)
    and output guardrails (PII leakage, topic boundaries, argument
    validation).

    Args:
        mode: "input" for user input checks, "output" for model output checks.
        text: The text to check (plain string or JSON).

    Returns:
        JSON string with passed, score, flags, and recommendation.

    """
    import json

    try:
        if mode == "input":
            result = check_input(text)
        elif mode == "output":
            result = check_output(text)
        else:
            return json.dumps({"error": "mode must be 'input' or 'output'"})

        return json.dumps({
            "passed": result.passed,
            "score": result.score,
            "flags": result.flags,
            "recommendation": result.recommendation,
        })
    except Exception as e:
        return json.dumps({"error": f"Guardrail check failed: {e}"})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Small-Model Harness MCP server (stdio)")
    app.run(transport="stdio")
