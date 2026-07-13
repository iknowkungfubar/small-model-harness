# Small-Model Harness MCP Server

> MCP server providing analytical tools for the 5-layer harness.
> Companion to the small-model-harness Hermes plugin.

## Server Architecture

```python
"""MCP server for small-model-harness.

Provides tools for:
- L1: Task classification and model routing
- L2: Tool call validation
- L4: Loop detection and circuit breaker management
- L5: Context budget tracking and compaction
"""

from typing import Any
import json
import time
import hashlib

# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------

async def handle_classify_task(task: str, tools: list[str] = None) -> dict:
    """Classify a task's complexity and suggest optimal model tier."""
    
    if tools is None:
        tools = []
    
    score = 0
    reasoning_depth = "none"
    
    # Heuristic scoring
    debug_kw = {"debug", "why", "diagnose", "investigate", "root cause"}
    plan_kw = {"plan", "design", "architect", "strategy", "roadmap"}
    security_kw = {"security", "vulnerability", "exploit", "cve", "hardening"}
    
    task_lower = task.lower()
    
    if any(kw in task_lower for kw in debug_kw):
        score += 2
        reasoning_depth = "shallow"
    if any(kw in task_lower for kw in plan_kw):
        score += 3
        reasoning_depth = "deep"
    if any(kw in task_lower for kw in security_kw):
        score += 2  # don't auto-route to large model
    
    # Tool count
    if len(tools) > 5:
        score += 2
    elif len(tools) > 2:
        score += 1
    
    # Complexity bucket
    if score <= 2:
        complexity = "simple"
        tier = "t1"
    elif score <= 4:
        complexity = "moderate"
        tier = "t2"
    elif score <= 6:
        complexity = "complex"
        tier = "t3"
    else:
        complexity = "critical"
        tier = "t3"  # cascade from t3 upward
    
    return {
        "tier": tier,
        "complexity": complexity,
        "reasoning_depth": reasoning_depth,
        "tool_count": len(tools),
        "confidence": min(1.0, 0.5 + score * 0.08),
        "score": score,
    }


async def handle_route(classification: dict, available_models: dict = None) -> dict:
    """Route a classified task to the appropriate model tier."""
    
    if available_models is None:
        available_models = {"t1": False, "t2": True, "t3": True, "t4": True}
    
    tier = classification["tier"]
    
    # Fall back if tier unavailable
    if not available_models.get(tier, False):
        fallback_chain = {"t1": "t2", "t2": "t3", "t3": "t4"}
        while not available_models.get(tier, False) and tier in fallback_chain:
            tier = fallback_chain[tier]
    
    # Tier capabilities
    tier_config = {
        "t1": {"max_tools": 2, "max_steps": 10, "effective_context": 8192,
               "suitable": ["formatting", "extraction", "single_tool"]},
        "t2": {"max_tools": 5, "max_steps": 25, "effective_context": 16384,
               "suitable": ["multi_step", "moderate_reasoning", "code_gen"]},
        "t3": {"max_tools": 10, "max_steps": 50, "effective_context": 32768,
               "suitable": ["complex_reasoning", "bug_diagnosis", "planning"]},
        "t4": {"max_tools": 20, "max_steps": 100, "effective_context": 131072,
               "suitable": ["security", "high_stakes", "critical"]},
    }
    
    return {
        "assigned_tier": tier,
        "config": tier_config.get(tier, {}),
        "fallback_chain": ["t1", "t2", "t3", "t4"],
    }


async def handle_validate_call(tool_schema: dict, args: dict, 
                                session_history: list = None) -> dict:
    """Validate a tool call against its schema."""
    
    if session_history is None:
        session_history = []
    
    errors = []
    
    # Check required fields
    required = tool_schema.get("required", [])
    for field in required:
        if field not in args:
            errors.append({"field": field, "error": "required", 
                          "message": f"Missing required field '{field}'"})
    
    # Check field types
    properties = tool_schema.get("properties", {})
    type_map = {"string": str, "integer": int, "number": (int, float),
                "boolean": bool, "array": (list, tuple), "object": dict}
    
    for field, value in args.items():
        if field in properties:
            expected = properties[field].get("type")
            if expected and expected in type_map:
                if not isinstance(value, type_map[expected]):
                    errors.append({"field": field, "error": "type_mismatch",
                                  "expected": expected,
                                  "got": type(value).__name__})
    
    # Check for unknown fields
    if "additionalProperties" in tool_schema and not tool_schema["additionalProperties"]:
        defined = set(properties.keys())
        provided = set(args.keys())
        unknown = provided - defined
        if unknown:
            errors.append({"field": list(unknown)[0], "error": "unknown_field",
                          "message": f"Unknown fields: {unknown}"})
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "suggestions": _generate_suggestions(errors) if errors else [],
    }


async def handle_detect_loop(recent_calls: list) -> dict:
    """Score recent tool calls for loop patterns."""
    
    if len(recent_calls) < 4:
        return {"loop_score": 0.0, "pattern": None, "needs_history": True}
    
    # Signal 1: Token repetition
    outputs = [c.get("output", "")[:500] for c in recent_calls[-5:]]
    ngram_score = _ngram_overlap(outputs)
    
    # Signal 2: Tool diversity
    tools = [c.get("tool", "") for c in recent_calls[-8:]]
    unique_ratio = len(set(tools)) / max(len(tools), 1)
    diversity_score = 1.0 - unique_ratio
    
    # Signal 3: Content stagnation
    if len(recent_calls) >= 4:
        outputs_trimmed = [c.get("output", "")[:200] for c in recent_calls[-4:]]
        sim_score = _semantic_similarity(outputs_trimmed)
    else:
        sim_score = 0.0
    
    # Weighted ensemble
    loop_score = (ngram_score * 0.4 + diversity_score * 0.3 + sim_score * 0.3)
    
    pattern = None
    if loop_score > 0.8:
        pattern = _classify_pattern(recent_calls)
    
    return {
        "loop_score": round(loop_score, 3),
        "pattern": pattern,
        "signals": {
            "ngram_overlap": round(ngram_score, 3),
            "tool_diversity": round(diversity_score, 3),
            "content_stagnation": round(sim_score, 3),
        }
    }


async def handle_circuit_break(session_id: str, action: str = "check",
                                 loop_score: float = 0.0) -> dict:
    """Manage circuit breaker state for a session."""
    
    state = _get_breaker_state(session_id)
    
    if action == "check":
        return state
    
    elif action == "open":
        state["state"] = "open"
        state["last_break_time"] = time.time()
        state["break_count"] += 1
        state["break_times"].append(time.time())
        _save_breaker_state(session_id, state)
        return state
    
    elif action == "close":
        state["state"] = "closed"
        _save_breaker_state(session_id, state)
        return state
    
    elif action == "reset":
        cleared = {"state": "closed", "break_count": 0, 
                   "last_break_time": 0, "break_times": [],
                   "cooling_period": 60, "max_breaks": 5}
        _save_breaker_state(session_id, cleared)
        return cleared
    
    return state


async def handle_context_status(session_id: str) -> dict:
    """Show current context budget status."""
    
    budget = _get_budget(session_id)
    
    return {
        "session_id": session_id,
        "stated_window": budget["stated_window"],
        "effective_capacity": budget["effective_capacity"],
        "used_tokens": budget["used_tokens"],
        "utilization": f"{budget['used_tokens'] / max(budget['effective_capacity'], 1) * 100:.1f}%",
        "step_count": budget["step_count"],
        "compaction_count": budget["compaction_count"],
        "needs_compaction": budget["used_tokens"] > budget["effective_capacity"] * 0.9,
        "output_headroom": int(budget["effective_capacity"] * 0.1),
    }


async def handle_compact(session_id: str, steps: list = None) -> dict:
    """Trigger context compaction."""
    
    if steps is None:
        steps = []
    
    keep = 5
    if len(steps) <= keep:
        return {"compacted": False, "reason": "Not enough steps to compact"}
    
    summarize = steps[:-keep]
    tokens_before = sum(s.get("tokens", 500) for s in summarize)
    tokens_after = 300  # estimated summary tokens
    
    budget = _get_budget(session_id)
    budget["used_tokens"] -= (tokens_before - tokens_after)
    budget["compaction_count"] += 1
    budget["last_compaction_at"] = time.time()
    _save_budget(session_id, budget)
    
    return {
        "compacted": True,
        "steps_summarized": len(summarize),
        "steps_remaining": keep,
        "tokens_freed": tokens_before - tokens_after,
        "new_used_tokens": budget["used_tokens"],
        "new_utilization": f"{budget['used_tokens'] / max(budget['effective_capacity'], 1) * 100:.1f}%",
    }


async def handle_config(session_id: str, config_update: dict = None) -> dict:
    """Get or update harness configuration."""
    
    config = _get_config(session_id)
    
    if config_update:
        if "effective_capacity_ratio" in config_update:
            val = config_update["effective_capacity_ratio"]
            config["effective_capacity_ratio"] = max(0.1, min(1.0, val))
        if "compaction_threshold" in config_update:
            val = config_update["compaction_threshold"]
            config["compaction_threshold"] = max(0.5, min(1.0, val))
        if "sliding_window_steps" in config_update:
            val = config_update["sliding_window_steps"]
            config["sliding_window_steps"] = max(3, min(20, val))
        if "loop_threshold" in config_update:
            val = config_update["loop_threshold"]
            config["loop_threshold"] = max(0.5, min(1.0, val))
        if "cooling_period" in config_update:
            val = config_update["cooling_period"]
            config["cooling_period"] = max(10, min(300, val))
        
        _save_config(session_id, config)
    
    return {"config": config}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _ngram_overlap(texts: list[str], n: int = 4) -> float:
    """Jaccard similarity of n-gram sets between consecutive texts."""
    if len(texts) < 2:
        return 0.0
    
    def get_ngrams(text: str, n: int) -> set:
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    overlaps = []
    for i in range(1, len(texts)):
        prev_grams = get_ngrams(texts[i-1], n)
        curr_grams = get_ngrams(texts[i], n)
        if prev_grams and curr_grams:
            jaccard = len(prev_grams & curr_grams) / len(prev_grams | curr_grams)
            overlaps.append(jaccard)
    
    return sum(overlaps) / max(len(overlaps), 1)


def _semantic_similarity(texts: list[str]) -> float:
    """Token set overlap as approximate semantic similarity."""
    if len(texts) < 2:
        return 0.0
    
    def tokenize(text: str) -> set:
        return set(text.lower().split())
    
    similarities = []
    for i in range(1, len(texts)):
        t1 = tokenize(texts[i-1])
        t2 = tokenize(texts[i])
        if t1 and t2:
            jaccard = len(t1 & t2) / len(t1 | t2)
            similarities.append(jaccard)
    
    return sum(similarities) / max(len(similarities), 1)


def _classify_pattern(recent_calls: list) -> str:
    """Classify the loop pattern from recent calls."""
    last = recent_calls[-1] if recent_calls else {}
    last_tool = last.get("tool", "")
    last_args = last.get("args", {})
    last_error = last.get("error")
    
    # Tool slam
    same_count = sum(1 for c in recent_calls[-5:]
                     if c.get("tool") == last_tool and c.get("args") == last_args)
    if same_count >= 3:
        return f"tool_slam: {last_tool} called {same_count}x with identical args"
    
    # Stuck retry
    if last_error:
        error_count = sum(1 for c in recent_calls[-5:]
                         if c.get("error") == last_error)
        if error_count >= 3:
            return f"stuck_retry: same error repeated {error_count}x"
    
    # Token grind
    outputs = [c.get("output", "")[:300] for c in recent_calls[-5:]]
    if _ngram_overlap(outputs) > 0.85:
        return "token_grind: near-identical output across calls"
    
    return "unknown_loop"


def _generate_suggestions(errors: list) -> list:
    """Generate fix suggestions for validation errors."""
    suggestions = []
    for err in errors:
        if err.get("error") == "required":
            suggestions.append(f"Add field '{err.get('field')}' to the call arguments")
        elif err.get("error") == "type_mismatch":
            suggestions.append(
                f"Convert '{err.get('field')}' to {err.get('expected')} type")
        elif err.get("error") == "unknown_field":
            suggestions.append(f"Remove unknown field: {err.get('message')}")
    return suggestions


# ---------------------------------------------------------------------------
# State Management (in-memory with optional SQLite persistence)
# ---------------------------------------------------------------------------

_breaker_states: dict[str, dict] = {}
_budgets: dict[str, dict] = {}
_configs: dict[str, dict] = {}


def _get_breaker_state(session_id: str) -> dict:
    """Get or create circuit breaker state."""
    if session_id not in _breaker_states:
        _breaker_states[session_id] = {
            "state": "closed",
            "break_count": 0,
            "last_break_time": 0,
            "break_times": [],
            "cooling_period": 60,
            "max_breaks": 5,
        }
    return _breaker_states[session_id]


def _save_breaker_state(session_id: str, state: dict):
    """Save circuit breaker state."""
    _breaker_states[session_id] = state


def _get_budget(session_id: str) -> dict:
    """Get or create context budget."""
    if session_id not in _budgets:
        _budgets[session_id] = {
            "stated_window": 131072,
            "effective_capacity": int(131072 * 0.33),
            "used_tokens": 0,
            "step_count": 0,
            "compaction_count": 0,
            "last_compaction_at": None,
        }
    return _budgets[session_id]


def _save_budget(session_id: str, budget: dict):
    """Save context budget."""
    _budgets[session_id] = budget


def _get_config(session_id: str) -> dict:
    """Get or create config."""
    if session_id not in _configs:
        _configs[session_id] = {
            "effective_capacity_ratio": 0.33,
            "compaction_threshold": 0.9,
            "sliding_window_steps": 5,
            "loop_threshold": 0.8,
            "cooling_period": 60,
            "max_breaks": 5,
        }
    return _configs[session_id]


def _save_config(session_id: str, config: dict):
    """Save config."""
    _configs[session_id] = config
