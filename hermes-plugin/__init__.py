"""Small-Model Harness — Hermes Plugin.

Five-layer defensive harness for small local LLMs (1B-12B parameter).
Bridges the 5-13% reliability gap between local and frontier models.

Registered hooks:
  - pre_tool_call:  schema validation, loop detection, circuit breaker,
                     budget check, routing awareness
  - post_tool_call: track metrics, update loop detector, update budget,
                     track routing failures
  - pre_verify:     report state, suggest compaction, report routing tier

Research driving this implementation:
  - Liquid AI Antidoom (FTPO) Jul 7 2026: Qwen3.5-4B doom loop rate 22.9% -> 1%
  - Chroma Context Rot Jul 2025: ALL 18 models degrade with length, 1/3 rule
  - Ganglani Jun 2026: Qwen3-32B tool accuracy 87% vs GPT-4o 92%
  - Vectara HHEM + arXiv:2604.07035v1: Gemma 4 hallucination floor 0.51-0.55
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import sub-modules (graceful degradation if missing)
# ---------------------------------------------------------------------------

_HAS_VALIDATOR = False
_HAS_OUTPUT_VALIDATOR = False
_HAS_LOOP_DETECTOR = False
_HAS_CIRCUIT_BREAKER = False
_HAS_CONTEXT_BUDGET = False
_HAS_ROUTING = False

from . import circuit_breaker as _breaker_mod
from . import context_budget as _budget_mod
from . import loop_detector as _loop_mod
from . import output_validator as _output_validator_mod
from . import validator as _validator_mod

_HAS_VALIDATOR = True
_HAS_OUTPUT_VALIDATOR = True
_HAS_LOOP_DETECTOR = True
_HAS_CIRCUIT_BREAKER = True
_HAS_CONTEXT_BUDGET = True

# ---------------------------------------------------------------------------
# Global state (per-session)
# ---------------------------------------------------------------------------

# The harness maintains per-session state in memory. Because the Hermes
# plugin system runs in-process, this state persists across turns within
# a session and resets when the session ends.

_schema_validator = None
_output_validator = None
_loop_detector = None
_circuit_breaker = None
_context_budget = None
_task_classifier = None

# Current tier tracking for routing awareness
_current_tier = 2  # Default: T2 (4-8B)
_current_tier_label = "T2 (4-8B)"
_routing_failures = 0  # Consecutive routing failures

# Track whether this is the first call (for initial config logging)
_initialized = False

# Model tier map (from AGENTS.md hardware context)
MODEL_TIER_MAP: dict[str, int] = {
    "small": 1,  # <4B params
    "medium": 2,  # 4-8B params
    "large": 3,  # 9-12B params (Josh's LM Studio models)
    "cloud": 4,  # Cloud frontier
}

# Reverse lookup: tier -> suggested model size
TIER_LABELS: dict[int, str] = {
    1: "T1 (<4B) — small models",
    2: "T2 (4-8B) — medium models",
    3: "T3 (9-12B) — large local (Josh's LM Studio range)",
    4: "T4 — cloud frontier",
    5: "Human review required",
}


# ---------------------------------------------------------------------------
# Lazy initialization
# ---------------------------------------------------------------------------


def _ensure_initialized() -> None:
    """Initialize sub-components on first use."""
    global _schema_validator, _output_validator, _loop_detector, _circuit_breaker, _context_budget
    global _task_classifier, _HAS_ROUTING, _initialized

    if _initialized:
        return

    _initialized = True

    if _HAS_VALIDATOR:
        _schema_validator = _validator_mod.SchemaValidator()
        logger.info("Small-model harness: SchemaValidator initialized")

    if _HAS_OUTPUT_VALIDATOR:
        _output_validator = _output_validator_mod.OutputValidator()
        logger.info("Small-model harness: OutputValidator initialized (Phase 4)")

    if _HAS_LOOP_DETECTOR:
        _loop_detector = _loop_mod.LoopDetector()
        logger.info("Small-model harness: LoopDetector initialized")

    if _HAS_CIRCUIT_BREAKER:
        _circuit_breaker = _breaker_mod.CircuitBreaker()
        logger.info("Small-model harness: CircuitBreaker initialized")

    if _HAS_CONTEXT_BUDGET:
        _context_budget = _budget_mod.ContextBudgetManager(model_window=32000)
        logger.info("Small-model harness: ContextBudgetManager initialized (32K window)")

    # Initialize task classifier from MCP module
    try:
        _MCP_DIR = Path(__file__).parent.parent.parent / "mcp" / "small-model-harness"
        sys.path.insert(0, str(_MCP_DIR.resolve()))

        from routing_commands import classify_task as _classify_fn

        _task_classifier = _classify_fn
        _HAS_ROUTING = True
        logger.info("Small-model harness: TaskClassifier initialized (routing active)")
    except ImportError as e:
        logger.warning(
            "Small-model harness: TaskClassifier not available — %s. Routing will be disabled.",
            e,
        )
        _HAS_ROUTING = False
        _task_classifier = None


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------


def on_pre_tool_call(
    tool_name: str,
    args: dict | None,
    **kwargs,
) -> dict | None:
    """``pre_tool_call`` hook — validate, detect loops, check budget.

    All three safety checks run in sequence. Any check that fails
    returns a block decision with a specific error message.

    Returns:
      - ``None`` to allow the call (no issues)
      - ``{"action": "block", "message": "..."}`` to block
      - ``{"action": "approve", "message": "..."}`` to request human approval

    """
    _ensure_initialized()

    # Skip non-tool tools (Hermes internal)
    if tool_name in ("Bash", "terminal", "apply_patch", "Write", "Edit"):
        return None

    if args is None:
        args = {}

    # ------------------------------------------------------------------
    # 1. Schema validation (basic)
    # ------------------------------------------------------------------
    if _HAS_VALIDATOR and _schema_validator is not None:
        validation = _schema_validator.validate(tool_name, args)
        if not validation.valid:
            error_detail = "; ".join(validation.errors[:3])
            logger.warning(
                "Harness: schema validation failed for %s: %s",
                tool_name,
                error_detail,
            )
            return {
                "action": "block",
                "message": (
                    f"[Small-Model Harness] Schema validation failed for '{tool_name}': "
                    f"{error_detail}. "
                    f"The model produced malformed tool call arguments. "
                    f"Consider regenerating with corrected format."
                ),
            }

    # ------------------------------------------------------------------
    # 1b. Enhanced output validation (Phase 4 — richer error messages)
    # ------------------------------------------------------------------
    if _HAS_OUTPUT_VALIDATOR and _output_validator is not None:
        # Convert Hermes args format to a validate_tool_call dict
        tool_call_input = {"tool_name": tool_name, "arguments": dict(args or {})}
        o_result = _output_validator.validate_tool_call(
            json.dumps(tool_call_input),
            tool_name,
        )
        if not o_result.valid:
            error_detail = "; ".join(o_result.errors[:3])
            # Enhanced validation warnings are informative but don't block
            # if the basic validator already passed — the output is structurally
            # valid but may have type/range issues.
            logger.info(
                "Harness: enhanced validation note for %s: %s",
                tool_name,
                error_detail,
            )

    # ------------------------------------------------------------------
    # 2. Loop detection + circuit breaker
    # ------------------------------------------------------------------
    if _HAS_LOOP_DETECTOR and _loop_detector is not None:
        loop_score = _loop_detector.score()

        if _HAS_CIRCUIT_BREAKER and _circuit_breaker is not None:
            decision = _circuit_breaker.check(loop_score.overall)

            if decision.action == "block":
                logger.warning(
                    "Harness: circuit breaker BLOCKED %s — %s",
                    tool_name,
                    decision.message,
                )
                return {
                    "action": "block",
                    "message": (f"[Small-Model Harness] Circuit breaker: {decision.message}"),
                }

            if decision.action == "escalate_tier":
                logger.warning(
                    "Harness: circuit breaker escalated %s — T%d",
                    tool_name,
                    decision.target_tier or 0,
                )
                return {
                    "action": "approve",
                    "message": (
                        f"[Small-Model Harness] Loop detected — "
                        f"escalating to {TIER_LABELS.get(decision.target_tier or 4, 'higher tier')}. "
                        f"{decision.message}"
                    ),
                }

            if decision.action == "escalate_human":
                return {
                    "action": "approve",
                    "message": (
                        f"[Small-Model Harness] {decision.message} Manual intervention recommended."
                    ),
                }

    # ------------------------------------------------------------------
    # 3. Context budget check
    # ------------------------------------------------------------------
    if _HAS_CONTEXT_BUDGET and _context_budget is not None:
        budget_check = _context_budget.check_before_call(tool_name)
        if not budget_check.allowed:
            logger.warning(
                "Harness: budget check blocked %s — utilization %.0f%%",
                tool_name,
                budget_check.utilization * 100,
            )
            return {
                "action": "block",
                "message": (
                    f"[Small-Model Harness] Context budget exhausted "
                    f"({budget_check.utilization:.0%} of {budget_check.effective_capacity}). "
                    f"Compaction required before new tool calls."
                ),
            }

    # ------------------------------------------------------------------
    # 4. Routing awareness — suggest tier escalation if needed
    # ------------------------------------------------------------------
    if _HAS_ROUTING and _task_classifier is not None:
        tier_suggestion = _detect_tier_escalation(tool_name, args)
        if tier_suggestion is not None:
            logger.info(
                "Harness: routing suggests escalation — %s (tier=%d, failures=%d)",
                tier_suggestion,
                _current_tier,
                _routing_failures,
            )
            # Inform but don't block — the agent can choose to escalate
            return {
                "action": "continue",
                "message": (
                    f"[Small-Model Harness] Routing suggestion: {tier_suggestion}. "
                    f"Current tier: {_current_tier_label}. "
                    f"Consider routing to a higher tier model for this task."
                ),
            }

    # All checks passed
    return None


def _detect_tier_escalation(
    tool_name: str,
    args: dict | None,
) -> str | None:
    """Detect if the current tool call suggests we need a higher tier.

    Uses heuristics: tool type, argument complexity, error patterns.
    Returns a suggested tier label or None if current tier is fine.
    """
    global _routing_failures, _current_tier, _current_tier_label

    # Check tool type indicators of complexity
    planning_tools = {"plan", "write_plan", "design", "generate_design"}
    security_tools = {"security_scan", "vulnerability_check", "audit"}

    if tool_name in security_tools:
        # Security tools should run on T3+
        if _current_tier < 3:
            _routing_failures += 1
            return "T3 (9-12B) — security tools need higher tier"

    if tool_name in planning_tools:
        # Planning tools should run on T3+
        if _current_tier < 3:
            _routing_failures += 1
            return "T3 (9-12B) — planning tools need higher tier"

    # Check for large arguments (indicates complex task)
    if args:
        args_str = str(args)
        if len(args_str) > 2000 and _current_tier < 3:
            _routing_failures += 1
            return "T3 (9-12B) — large arguments suggest complex task"

    # Suggest tier change after 3+ routing failures
    if _routing_failures >= 3 and _current_tier < 4:
        target_tier = min(_current_tier + 1, 4)
        return f"T{target_tier} - multiple routing failures suggest tier escalation"

    return None


def on_post_tool_call(
    tool_name: str,
    args: dict | None = None,
    result: any | None = None,
    status: str | None = None,
    duration_ms: int = 0,
    **kwargs,
) -> None:
    """``post_tool_call`` hook — observer: update loop detector and budget.

    Cannot block the tool — only tracks metrics for future decisions.
    """
    _ensure_initialized()

    # Update loop detector with this call's record
    if _HAS_LOOP_DETECTOR and _loop_detector is not None:
        output_snippet = None
        if result is not None:
            output_str = str(result)
            output_snippet = output_str[:500]  # Keep first 500 chars

        call_record = _loop_mod.CallRecord(
            tool_name=tool_name,
            args=args,
            error=status if status and status != "ok" else None,
            output_snippet=output_snippet,
        )
        _loop_detector.record(call_record)

        # Run loop detection after recording
        loop_score = _loop_detector.score()
        if loop_score.overall >= 0.5:
            logger.info(
                "Harness: loop score %.2f after %s — pattern=%s",
                loop_score.overall,
                tool_name,
                loop_score.pattern,
            )

        # If loop detected, register circuit break
        if (
            _HAS_CIRCUIT_BREAKER
            and _circuit_breaker is not None
            and loop_score.overall >= _circuit_breaker.loop_threshold
        ):
            decision = _circuit_breaker.register_break(loop_score.overall)
            logger.warning(
                "Harness: LOOP DETECTED (%.2f, pattern=%s) — %s",
                loop_score.overall,
                loop_score.pattern,
                decision.message,
            )

    # Update context budget
    if _HAS_CONTEXT_BUDGET and _context_budget is not None:
        _context_budget.record_step(tool_name, args, result)

    # Log metrics at debug level
    logger.debug(
        "Harness post_tool_call: %s %s (%dms)",
        tool_name,
        status or "ok",
        duration_ms,
    )


def on_pre_verify(**kwargs) -> dict | None:
    """``pre_verify`` hook — report harness state at verification points.

    Injects context about current harness state when it's relevant.
    """
    _ensure_initialized()

    messages = []

    if _HAS_LOOP_DETECTOR and _loop_detector is not None:
        # Access the tracked call count through the LoopDetector's internal
        # state. The number of calls is tracked regardless of module type.
        try:
            calls_tracked = len(_loop_detector._calls)  # type: ignore[attr-defined]
        except AttributeError:
            calls_tracked = 0
        if calls_tracked > 0:
            score = _loop_detector.score()
            if score.overall > 0.3:
                messages.append(
                    f"Loop risk: {score.overall:.0%} "
                    f"(pattern={score.pattern}, {calls_tracked} calls tracked)"
                )

    if _HAS_CIRCUIT_BREAKER and _circuit_breaker is not None:
        stats = _circuit_breaker.stats()
        if stats["break_count"] > 0:
            messages.append(
                f"Circuit breaker: {stats['state']}, "
                f"{stats['break_count']} breaks, "
                f"tier={stats['tier']}"
            )

    if _HAS_CONTEXT_BUDGET and _context_budget is not None:
        usage = _context_budget.utilization
        if usage > 0.5:
            messages.append(
                f"Context budget: {usage:.0%} used "
                f"({_context_budget.get_available_tokens()} tokens remaining)"
            )
            if _context_budget.needs_compaction:
                messages.append("Compaction recommended.")

    # Report routing tier status
    if _HAS_ROUTING and _current_tier is not None:
        routing_msg = f"Model tier: {_current_tier_label}"
        if _routing_failures > 0:
            routing_msg += f" ({_routing_failures} routing failures detected)"
        messages.append(routing_msg)

    if messages:
        return {
            "action": "continue",
            "message": "[Small-Model Harness] " + " | ".join(messages),
        }

    return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin registration — called by Hermes plugin loader."""
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_verify", on_pre_verify)

    # Log registration with component availability
    components = []
    if _HAS_VALIDATOR:
        components.append("validator")
    if _HAS_OUTPUT_VALIDATOR:
        components.append("output_validator(phase4)")
    if _HAS_LOOP_DETECTOR:
        components.append("loop_detector")
    if _HAS_CIRCUIT_BREAKER:
        components.append("circuit_breaker")
    if _HAS_CONTEXT_BUDGET:
        components.append("context_budget")
    if _HAS_ROUTING:
        components.append("routing")

    logger.info(
        "Small-model harness registered (hooks: pre_tool_call + post_tool_call + pre_verify) "
        "| components: %s",
        ", ".join(components) if components else "NONE — all disabled",
    )
