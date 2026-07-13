"""Routing Commands — Task Classifier & Model Router for the small-model-harness.

Layer 1 of the five-layer architecture: routes each task to the most
cost-effective model tier that can handle it reliably.

Provides:
- classify_task(): Rule-based task complexity classifier (no LLM call)
- route_task(): Tier assignment + cascade logic
- harness_classify_task(): MCP tool wrapper
- harness_route(): MCP tool wrapper
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier Definitions
# ---------------------------------------------------------------------------

# Model tier definitions. These map to Josh's hardware: AMD RX 7900 GRE (16GB)
# with Qwen3-9B class models running locally, DeepSeek V4 Flash via OpenCode
# for cloud fallback.

TIER_DEFINITIONS: dict[str, dict[str, Any]] = {
    "t1": {
        "max_score": 2,
        "max_tools": 2,
        "max_steps": 10,
        "effective_context": 8192,
        "label": "T1 (<4B) — Simple, single-step tasks",
        "suitable_for": ["extraction", "formatting", "single tool call", "known patterns"],
        "models": ["phi-4-mini:3.8b", "smollm3:3b", "llama-3.2-3b"],
    },
    "t2": {
        "max_score": 5,
        "max_tools": 5,
        "max_steps": 25,
        "effective_context": 16384,
        "label": "T2 (4-8B) — Moderate multi-step tasks",
        "suitable_for": ["multi-step", "common reasoning", "document Q&A", "code generation"],
        "models": ["qwen3:8b", "llama-3.2-8b", "qwen3:4b"],
    },
    "t3": {
        "max_score": 8,
        "max_tools": 10,
        "max_steps": 50,
        "effective_context": 32768,
        "label": "T3 (9-12B) — Complex reasoning",
        "suitable_for": [
            "bug diagnosis",
            "planning",
            "architecture",
            "security review",
            "complex code generation",
        ],
        "models": ["qwen3-30b-a3b", "ornith-1.0-9b", "qwen3.5-9b-deepseek-v4-flash"],
    },
    "t4": {
        "max_score": 999,
        "max_tools": 20,
        "max_steps": 100,
        "effective_context": 131072,
        "label": "T4 — Cloud frontier fallback",
        "suitable_for": ["security critical", "high stakes", "persistent failure recovery"],
        "models": ["opencode/deepseek-v4-flash-free"],
        "provider": "opencode-zen",
    },
}

TIER_KEYS = ["t1", "t2", "t3", "t4"]

# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class TaskProfile:
    """Classification result for a task."""

    task: str
    tier: str  # "t1" | "t2" | "t3" | "t4"
    complexity: str  # "simple" | "moderate" | "complex" | "critical"
    score: int  # 0-10
    reasoning_depth: str  # "none" | "shallow" | "deep" | "multi-step"
    tool_count_estimate: int
    context_estimate: str  # "small" | "medium" | "large"
    confidence: float  # 0.0-1.0 heuristic confidence
    features: list[str] = field(default_factory=list)
    semantic_dispersion: float = field(default=1.0)  # 0.0-1.0 from Phase 6
    signal_flags: list[str] = field(default_factory=list)  # Phase 6 signal flags


@dataclass
class RouteDecision:
    """Result of routing a classified task to a tier."""

    task_profile: TaskProfile
    recommended_tier: str
    current_tier: str
    action: str  # "stay" | "cascade_up" | "cascade_down"
    reason: str
    can_handle: bool
    cascade_path: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classification Heuristics
# ---------------------------------------------------------------------------

# Keywords that raise complexity
_COMPLEXITY_KEYWORDS: dict[str, int] = {
    # Debug & diagnosis
    "debug": 2,
    "diagnose": 2,
    "trace": 2,
    "root cause": 3,
    "regression": 2,
    "why": 1,
    "explain": 1,
    "investigate": 2,
    # Planning & architecture
    "plan": 3,
    "design": 3,
    "architect": 3,
    "strategy": 3,
    "roadmap": 3,
    "migrate": 3,
    "refactor": 3,
    "restructure": 3,
    # Security
    "security": 2,
    "vulnerability": 3,
    "exploit": 3,
    "audit": 2,
    "cve": 3,
    "penetration test": 4,
    "threat": 3,
    # Multi-step
    "multi-step": 2,
    "deploy": 2,
    "orchestrate": 2,
    "coordinate": 2,
    "pipeline": 2,
    "workflow": 2,
    # Analysis
    "analyze": 2,
    "evaluate": 2,
    "compare": 1,
    "synthesize": 2,
    "summarize": 1,
    "optimize": 2,
    # Domain-specific
    "multi-agent": 3,
    "rag": 2,
    "fine-tune": 3,
    "training": 3,
    "reinforcement learning": 4,
    # Risk indicators
    "production": 2,
    "critical": 3,
    "blocking": 2,
    "revenue": 2,
}

# Domains that always require at least T2
_TIER_TWO_MINIMUM_KEYWORDS: list[str] = [
    "security review",
    "auth",
    "authentication",
    "authorization",
    "database",
    "migration",
    "schema change",
    "data model",
    "api design",
    "contract",
    "protocol",
]

# Signal words for very simple tasks
_SIMPLE_INDICATORS: list[str] = [
    "format",
    "convert",
    "translate",
    "spell check",
    "paraphrase",
    "rename",
    "copy",
    "move",
    "list",
    "show",
    "simple",
    "one-step",
    "single",
    "trivial",
]


def _estimate_tool_count(task: str) -> int:
    """Estimate the number of tool calls a task might need.

    Uses heuristics rather than an LLM call. Generally conservative
    — overestimating is safer than underestimating.
    """
    task_lower = task.lower()
    count = 1

    # Search/read/scan operations
    searches = len(re.findall(r"\b(find|search|look up|check|verify|validate)\b", task_lower))
    count += searches

    # Write operations
    writes = len(re.findall(r"\b(create|write|edit|modify|update|patch|fix|add)\b", task_lower))
    count += writes

    # File operations
    files = len(
        re.findall(r"\b(read|open|examine|inspect|review)\s+(file|code|log|config)", task_lower)
    )
    count += files

    # Multi-step connectors
    connectors = len(re.findall(r"\b(then|after|next|first.*then|and\s+then)\b", task_lower))
    count += connectors

    return min(count, 15)


def _estimate_context_size(task: str) -> tuple[str, int]:
    """Estimate context size needed.

    Returns (label, estimated_tokens).
    """
    # Rough: 1 token ≈ 4 chars in English
    char_length = len(task)
    estimated_tokens = char_length // 4

    # Tools with large outputs add context pressure
    if any(kw in task.lower() for kw in ["search", "scan", "audit", "analyze", "review"]):
        estimated_tokens += 4000

    if any(kw in task.lower() for kw in ["codebase", "repository", "entire", "full"]):
        estimated_tokens += 8000

    if estimated_tokens < 4096:
        return ("small", estimated_tokens)
    if estimated_tokens < 16384:
        return ("medium", estimated_tokens)
    return ("large", estimated_tokens)


def _extract_features(task: str) -> list[str]:
    """Extract relevant features from the task description."""
    features: list[str] = []
    task_lower = task.lower()

    for keyword in _COMPLEXITY_KEYWORDS:
        if keyword in task_lower:
            features.append(keyword)

    for indicator in _SIMPLE_INDICATORS:
        if indicator in task_lower:
            features.append(f"simple:{indicator}")

    for keyword in _TIER_TWO_MINIMUM_KEYWORDS:
        if keyword in task_lower:
            features.append(f"t2_min:{keyword}")

    if len(task.split()) < 10:
        features.append("short_query")

    if len(task.split()) > 100:
        features.append("long_query")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in features:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


# ---------------------------------------------------------------------------
# Classification Engine
# ---------------------------------------------------------------------------


def _compute_score(features: list[str], tool_count: int, context_tokens: int) -> int:
    """Compute a complexity score (0-10) from extracted features."""
    score = 0
    task_lower = " ".join(features).lower()

    # Base score from keyword weights
    for keyword, weight in _COMPLEXITY_KEYWORDS.items():
        if keyword in task_lower:
            score += weight

    # Tool count contribution (more tools = more complex)
    score += tool_count // 2

    # Context size contribution
    if context_tokens > 8000:
        score += 1
    if context_tokens > 16000:
        score += 1

    # Cap at 10
    return min(score, 10)


def _score_to_tier(score: int, features: list[str]) -> str:
    """Map a complexity score to a tier.

    Args:
        score: Complexity score (0-10).
        features: Extracted features for special-case overrides.

    Returns:
        Tier string: "t1", "t2", "t3", or "t4".

    """
    # Override for security-critical tasks
    feature_text = " ".join(features)
    if any(
        kw in feature_text
        for kw in [
            "vulnerability",
            "exploit",
            "cve",
            "penetration test",
            "threat",
            "reinforcement learning",
        ]
    ):
        return "t3"  # Minimum T3 for security-sensitive tasks

    # Override for tasks that need at least T2
    if any(f.startswith("t2_min:") for f in features):
        return max("t2", _score_to_tier(score, []), key=lambda t: TIER_KEYS.index(t))

    # Override for very simple tasks
    if any(f.startswith("simple:") for f in features) and score <= 1:
        return "t1"

    # Standard mapping
    for tier_key in TIER_KEYS:
        definition = TIER_DEFINITIONS[tier_key]
        if score <= definition["max_score"]:
            return tier_key

    return "t4"


def _estimate_confidence(task: str, features: list[str], score: int) -> float:
    """Estimate classification confidence.

    Short, clear tasks get higher confidence. Ambiguous or conflicting
    features reduce confidence.
    """
    word_count = len(task.split())
    feature_count = len(features)

    # Base: very short or very clear tasks
    if word_count < 5 and feature_count > 0:
        return 0.95
    if word_count < 10:
        return 0.85

    # Feature-rich tasks
    if feature_count >= 3:
        return 0.80

    # Short query with few features
    if word_count < 20 and feature_count == 0:
        return 0.50  # Not enough signal

    # Long query with few features — unclear
    if word_count > 50 and feature_count <= 2:
        return 0.40

    # Default
    return 0.70


def _classify_tier(task: str) -> str:
    """Determine the tier for a task using feature extraction + scoring.

    This is a fast, rule-based classifier. No LLM call needed.
    """
    features = _extract_features(task)
    tool_count = _estimate_tool_count(task)
    context_label, context_tokens = _estimate_context_size(task)
    score = _compute_score(features, tool_count, context_tokens)
    tier = _score_to_tier(score, features)
    return tier


def _classify_reasoning_depth(features: list[str]) -> str:
    """Classify the reasoning depth needed."""
    feature_text = " ".join(features)

    if any(
        kw in feature_text
        for kw in ["plan", "design", "architect", "strategy", "roadmap", "multi-agent"]
    ):
        return "multi-step"
    if any(
        kw in feature_text
        for kw in [
            "debug",
            "diagnose",
            "root cause",
            "investigate",
            "analyze",
            "evaluate",
            "synthesize",
        ]
    ):
        return "deep"
    if any(kw in feature_text for kw in ["explain", "why", "compare", "summarize", "optimize"]):
        return "shallow"
    return "none"


def _classify_complexity(score: int) -> str:
    """Map a numeric score to a complexity label."""
    if score <= 2:
        return "simple"
    if score <= 5:
        return "moderate"
    if score <= 8:
        return "complex"
    return "critical"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_task(task: str) -> TaskProfile:
    """Classify a task and return a TaskProfile.

    This is the primary classification function. It performs a fast,
    rule-based analysis without calling an LLM.

    Args:
        task: The task description or user request.

    Returns:
        A TaskProfile with classification results.

    """
    if not task or not task.strip():
        return TaskProfile(
            task=task or "",
            tier="t1",
            complexity="simple",
            score=0,
            reasoning_depth="none",
            tool_count_estimate=1,
            context_estimate="small",
            confidence=1.0,
            features=[],
        )

    features = _extract_features(task)
    tool_count = _estimate_tool_count(task)
    context_label, context_tokens = _estimate_context_size(task)
    score = _compute_score(features, tool_count, context_tokens)
    tier = _score_to_tier(score, features)
    complexity = _classify_complexity(score)
    reasoning_depth = _classify_reasoning_depth(features)
    confidence = _estimate_confidence(task, features, score)

    return TaskProfile(
        task=task[:500],  # Truncate very long tasks
        tier=tier,
        complexity=complexity,
        score=score,
        reasoning_depth=reasoning_depth,
        tool_count_estimate=tool_count,
        context_estimate=context_label,
        confidence=confidence,
        features=features[:20],
        semantic_dispersion=confidence,  # Heuristic: high confidence = low dispersion
        signal_flags=[],
    )


def route_task(
    task: str,
    current_tier: str = "t2",
    failure_count: int = 0,
    available_tiers: list[str] | None = None,
) -> RouteDecision:
    """Classify a task and make a routing decision.

    Args:
        task: The task description.
        current_tier: The currently active model tier.
        failure_count: Number of consecutive failures on this task.
        available_tiers: Which tiers are available (None = all).

    Returns:
        A RouteDecision with the routing recommendation.

    """
    profile = classify_task(task)

    if available_tiers is None:
        available_tiers = list(TIER_KEYS)

    # Determine recommended tier from classification
    recommended_tier = profile.tier

    # Ensure recommended tier is available
    available_set = set(available_tiers)
    if recommended_tier not in available_set:
        # Find the next available lower or higher tier
        for t in TIER_KEYS:
            if t in available_set and TIER_KEYS.index(t) >= TIER_KEYS.index(recommended_tier):
                recommended_tier = t
                break
        else:
            # Fallback to the highest available tier
            recommended_tier = max(available_tiers, key=lambda t: TIER_KEYS.index(t))

    # Build the cascade path (tiers between recommended and max)
    current_idx = TIER_KEYS.index(current_tier)
    recommended_idx = TIER_KEYS.index(recommended_tier)
    max_idx = min(TIER_KEYS.index(max(available_tiers)), len(TIER_KEYS) - 1)

    # Determine action
    tier_value = lambda t: TIER_KEYS.index(t)  # noqa: E731

    if failure_count > 0:
        # On failure, escalate one tier up if possible
        next_tier_idx = min(current_idx + failure_count, max_idx)
        recommended_tier = TIER_KEYS[next_tier_idx]
        action = "cascade_up"
        reason = (
            f"Escalating after {failure_count} failure(s). "
            f"Moving from {TIER_DEFINITIONS[current_tier]['label']} "
            f"to {TIER_DEFINITIONS[recommended_tier]['label']}."
        )
    elif tier_value(recommended_tier) < tier_value(current_tier):
        action = "cascade_down"
        reason = (
            f"Task classified as simpler than current tier. "
            f"Downgrading from {TIER_DEFINITIONS[current_tier]['label']} "
            f"to {TIER_DEFINITIONS[recommended_tier]['label']}."
        )
    elif tier_value(recommended_tier) > tier_value(current_tier):
        action = "cascade_up"
        reason = (
            f"Task complexity exceeds current tier capacity. "
            f"Upgrading from {TIER_DEFINITIONS[current_tier]['label']} "
            f"to {TIER_DEFINITIONS[recommended_tier]['label']}."
        )
    else:
        action = "stay"
        reason = (
            f"Current tier ({TIER_DEFINITIONS[current_tier]['label']}) "
            f"is appropriate for this task."
        )

    # Build alternatives list
    alternatives = [t for t in available_tiers if t != recommended_tier]

    # Build cascade path
    start_idx = TIER_KEYS.index(current_tier if action == "stay" else current_tier)
    end_idx = TIER_KEYS.index(recommended_tier)
    if start_idx <= end_idx:
        cascade_path = [TIER_KEYS[i] for i in range(start_idx, end_idx + 1)]
    else:
        cascade_path = [TIER_KEYS[i] for i in range(end_idx, start_idx + 1)]

    # Can this tier handle it?
    tier_def = TIER_DEFINITIONS[recommended_tier]
    can_handle = (
        profile.score <= tier_def["max_score"]
        and profile.tool_count_estimate <= tier_def["max_tools"]
    )

    return RouteDecision(
        task_profile=profile,
        recommended_tier=recommended_tier,
        current_tier=current_tier,
        action=action,
        reason=reason,
        can_handle=can_handle,
        cascade_path=cascade_path,
        alternatives=alternatives,
    )


# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------


def harness_classify_task(task: str) -> str:
    """Classify a task and return the profile as JSON.

    MCP tool wrapper around classify_task().

    Args:
        task: The task description or user request.

    Returns:
        JSON string with the TaskProfile.

    """
    profile = classify_task(task)
    return json.dumps(asdict(profile), indent=2)


def harness_route(
    task: str,
    current_tier: str = "t2",
    failure_count: int = 0,
    available_tiers: str | None = None,
) -> str:
    """Classify a task and return a routing decision as JSON.

    MCP tool wrapper around route_task().

    Args:
        task: The task description or user request.
        current_tier: The currently active model tier (default: "t2").
        failure_count: Number of consecutive failures on this task.
        available_tiers: Comma-separated list of available tiers
            (e.g. "t1,t2,t3"). Default: all tiers.

    Returns:
        JSON string with the RouteDecision.

    """
    if available_tiers:
        tiers = [t.strip() for t in available_tiers.split(",")]
    else:
        tiers = None

    decision = route_task(
        task=task,
        current_tier=current_tier,
        failure_count=failure_count,
        available_tiers=tiers,
    )

    return json.dumps(
        {
            "recommended_tier": decision.recommended_tier,
            "current_tier": decision.current_tier,
            "action": decision.action,
            "reason": decision.reason,
            "can_handle": decision.can_handle,
            "cascade_path": decision.cascade_path,
            "alternatives": decision.alternatives,
            "task_profile": asdict(decision.task_profile),
        },
        indent=2,
    )
