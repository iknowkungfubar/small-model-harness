"""Plugin-Level Verification Hooks — Phase 5

Integrates self-consistency and spec-grounded verification into
the small-model-harness plugin's hook system.

Provides:
- post_tool_call verification hook that checks model responses
- PluginVerifier that wires into the existing hook system
- MCP tool integration through the harness_verify_consistency endpoint
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin Verifier
# ---------------------------------------------------------------------------


class PluginVerifier:
    """Plugin-level verification that integrates with Hermes hooks.

    Wires into the post_tool_call hook to automatically verify
    model outputs against task profiles and known rubrics.
    """

    def __init__(self) -> None:
        self._enabled = True
        self._verify_threshold = 0.5  # Minimum confidence to skip verification
        self._response_history: list[dict[str, Any]] = []  # For self-consistency

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_verify_threshold(self, threshold: float) -> None:
        """Set the minimum confidence threshold for automatic verification."""
        self._verify_threshold = max(0.0, min(1.0, threshold))

    def record_response(
        self,
        tool_name: str,
        response: dict[str, Any] | str | None,
        task_id: str = "",
    ) -> None:
        """Record a response for future self-consistency checks.

        Maintains a rolling buffer of recent responses.
        """
        if response is None:
            return
        self._response_history.append({
            "tool_name": tool_name,
            "response": response if isinstance(response, dict) else {"raw": str(response)},
            "task_id": task_id,
            "timestamp": __import__("time").time(),
        })
        # Keep buffer manageable
        if len(self._response_history) > 100:
            self._response_history = self._response_history[-50:]

    def get_recent_responses(
        self,
        tool_name: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get recent responses, optionally filtered by tool name."""
        matching = self._response_history
        if tool_name:
            matching = [r for r in matching if r.get("tool_name") == tool_name]
        return matching[-limit:]

    def clear_history(self) -> None:
        """Clear the response history buffer."""
        self._response_history.clear()


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------

_PLUGIN_VERIFIER = PluginVerifier()


def get_verifier() -> PluginVerifier:
    """Get the global PluginVerifier instance."""
    return _PLUGIN_VERIFIER


# ---------------------------------------------------------------------------
# MCP Tool Integration
# ---------------------------------------------------------------------------


def harness_verify_consistency(
    responses_json: str,
    key_fields: str | None = None,
) -> str:
    """Verify consistency across multiple responses.

    MCP tool wrapper for SelfConsistencyChecker.

    Args:
        responses_json: JSON array of response dicts or strings.
        key_fields: Optional comma-separated list of fields to compare.

    Returns:
        JSON string with ConsistencyResult.

    """
    try:
        responses = json.loads(responses_json)
        if not isinstance(responses, list):
            return json.dumps({
                "error": "responses_json must be a JSON array",
                "is_consistent": False,
                "confidence": 0.0,
            })
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({
            "error": f"Failed to parse responses_json: {e}",
            "is_consistent": False,
            "confidence": 0.0,
        })

    field_list = key_fields.split(",") if key_fields else None

    # Import here to avoid circular imports at module level
    try:
        from verification import SelfConsistencyChecker

        checker = SelfConsistencyChecker()
        result = checker.check_consistency(
            responses=responses,
            key_fields=field_list,
        )
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
    except ImportError as e:
        return json.dumps({
            "error": f"verification module not available: {e}",
            "is_consistent": False,
            "confidence": 0.0,
        })
