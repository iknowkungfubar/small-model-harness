"""
Context Budget Manager — token tracking and budget enforcement.

Enforces the 1/3 effective window rule based on Chroma Research
findings (Hong, Troynikov, Huber, Jul 2025): ALL 18 tested models
show universal length-driven performance degradation. NVIDIA and
Chroma recommend keeping prompts under 1/3 of the stated context
window.

Also tracks: per-session token totals, step counts, compaction
events, and provides forced compaction when thresholds are exceeded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default budget ratios
EFFECTIVE_CAPACITY_RATIO = 3  # stated_window // 3 = effective capacity
COMPACTION_THRESHOLD = 0.80   # Compaction recommended at 80% utilization
WARN_THRESHOLD = 0.65         # Warning at 65% utilization
CRITICAL_THRESHOLD = 0.90     # Block new work at 90% utilization

# Rough token estimation (chars / 4)
CHARS_PER_TOKEN = 4

# Per-step overhead estimate
STEP_OVERHEAD_TOKENS = 250  # System prompt + instructions per step


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class BudgetState:
    """Current budget state for a session."""
    total_used_tokens: int = 0
    step_count: int = 0
    compaction_count: int = 0
    largest_step_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "total_used_tokens": self.total_used_tokens,
            "step_count": self.step_count,
            "compaction_count": self.compaction_count,
            "largest_step_tokens": self.largest_step_tokens,
        }


@dataclass
class BudgetCheckResult:
    """Result of checking a tool call against the budget."""
    allowed: bool
    message: str = ""
    utilization: float = 0.0
    effective_capacity: int = 0
    used_tokens: int = 0
    needs_compaction: bool = False


# ---------------------------------------------------------------------------
# Budget Manager
# ---------------------------------------------------------------------------


class ContextBudgetManager:
    """Manages context window budgets with the 1/3 effective window rule.

    Research basis:
    - Chroma Context Rot (Jul 2025): 18 models tested, ALL show
      universal length-driven performance degradation
    - NVIDIA recommendation: keep prompts under 1/3 of stated window
    - Effective capacity: stated_window // 3
    - For large models (128K stated): effective ~42K tokens

    Args:
        model_window: The model's stated context window in tokens.
                      Default 32000 (Qwen3-9B effective limit).
    """

    def __init__(self, model_window: int = 32000):
        self.stated_window = model_window
        self.effective_capacity = model_window // EFFECTIVE_CAPACITY_RATIO
        self._state = BudgetState()

    @property
    def utilization(self) -> float:
        """Current utilization as a ratio of effective capacity."""
        if self.effective_capacity <= 0:
            return 0.0
        return self._state.total_used_tokens / self.effective_capacity

    @property
    def needs_compaction(self) -> bool:
        """True if utilization exceeds compaction threshold."""
        return self.utilization >= COMPACTION_THRESHOLD

    def record_step(self, tool_name: str, args: dict | None, result: Any) -> int:
        """Record a tool call step and update token estimates.

        Args:
            tool_name: Name of the tool called.
            args: Arguments to the tool.
            result: Result from the tool.

        Returns:
            Estimated tokens consumed by this step.
        """
        step_tokens = STEP_OVERHEAD_TOKENS  # Base overhead

        # Estimate args tokens
        if args:
            args_text = str(args)
            step_tokens += len(args_text) // CHARS_PER_TOKEN

        # Estimate result tokens
        if result is not None:
            result_text = str(result)
            result_tokens = len(result_text) // CHARS_PER_TOKEN
            step_tokens += result_tokens

        self._state.total_used_tokens += step_tokens
        self._state.step_count += 1

        if step_tokens > self._state.largest_step_tokens:
            self._state.largest_step_tokens = step_tokens

        logger.debug(
            "Budget: step %d +%d tokens (total: %d/%d, %.0f%%)",
            self._state.step_count, step_tokens,
            self._state.total_used_tokens, self.effective_capacity,
            self.utilization * 100,
        )

        return step_tokens

    def check_before_call(self, tool_name: str) -> BudgetCheckResult:
        """Check budget before allowing a tool call.

        The check uses effective capacity (1/3 of stated window).
        Returns allowed=False if call would exceed capacity.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            BudgetCheckResult with allowed flag and status info.
        """
        current_util = self.utilization

        if current_util >= CRITICAL_THRESHOLD:
            return BudgetCheckResult(
                allowed=False,
                message=(
                    f"Context budget critical ({current_util:.0%}). "
                    f"Effective capacity ({self.effective_capacity}) "
                    f"is exhausted. Compaction required before new work."
                ),
                utilization=current_util,
                effective_capacity=self.effective_capacity,
                used_tokens=self._state.total_used_tokens,
                needs_compaction=True,
            )

        if current_util >= COMPACTION_THRESHOLD:
            return BudgetCheckResult(
                allowed=True,
                message=(
                    f"Context budget high ({current_util:.0%}). "
                    f"Compaction recommended."
                ),
                utilization=current_util,
                effective_capacity=self.effective_capacity,
                used_tokens=self._state.total_used_tokens,
                needs_compaction=True,
            )

        if current_util >= WARN_THRESHOLD:
            return BudgetCheckResult(
                allowed=True,
                message=(
                    f"Context budget at {current_util:.0%} "
                    f"({self._state.total_used_tokens}/{self.effective_capacity})"
                ),
                utilization=current_util,
                effective_capacity=self.effective_capacity,
                used_tokens=self._state.total_used_tokens,
                needs_compaction=False,
            )

        return BudgetCheckResult(
            allowed=True,
            utilization=current_util,
            effective_capacity=self.effective_capacity,
            used_tokens=self._state.total_used_tokens,
        )

    def mark_compaction(self) -> None:
        """Register a compaction event and reduce tracked usage.

        Compaction reduces the total used tokens by keeping only
        the last 30% of steps (sliding window). This mimics the
        actual effect of context compaction.
        """
        self._state.compaction_count += 1
        self._state.total_used_tokens = max(
            int(self._state.total_used_tokens * 0.3),
            0,
        )
        logger.info(
            "Budget: compaction #%d applied — tokens reduced to %d",
            self._state.compaction_count,
            self._state.total_used_tokens,
        )

    def can_handle_step(
        self,
        estimated_tokens: int = 5000,
    ) -> bool:
        """Check if the budget can accommodate an estimated next step.

        Args:
            estimated_tokens: Rough estimate of next step's token cost.

        Returns:
            True if the step fits within effective capacity.
        """
        projected = self._state.total_used_tokens + estimated_tokens
        return projected <= self.effective_capacity

    def get_available_tokens(self) -> int:
        """Get remaining token budget before effective capacity."""
        return max(self.effective_capacity - self._state.total_used_tokens, 0)

    def stats(self) -> dict:
        """Return current budget statistics."""
        return {
            "stated_window": self.stated_window,
            "effective_capacity": self.effective_capacity,
            "used_tokens": self._state.total_used_tokens,
            "available_tokens": self.get_available_tokens(),
            "utilization": self.utilization,
            "steps": self._state.step_count,
            "compactions": self._state.compaction_count,
            "needs_compaction": self.needs_compaction,
            "largest_step_tokens": self._state.largest_step_tokens,
        }

    def reset(self) -> None:
        """Full reset of budget state."""
        self._state = BudgetState()
        logger.info("ContextBudgetManager: reset")
