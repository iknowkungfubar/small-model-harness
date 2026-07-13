"""
Circuit Breaker — 3-state breaker for small model failure escalation.

Monitors loop detection scores and breaks the execution chain before
the model enters a catastrophic doom loop. Three states:
  - CLOSED: Normal operation
  - OPEN: Breaker tripped, escalation required
  - HALF_OPEN: Test state after cooling period

On break: escalates to next model tier (T1→T2→T3→T4).
On consecutive breaks: permanently escalates to human review.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

BreakerState = Literal["closed", "open", "half_open"]
EscalationAction = Literal["allow", "block", "escalate_tier", "escalate_human", "retry_different"]


@dataclass
class CircuitDecision:
    """Decision from the circuit breaker."""
    action: EscalationAction
    state: BreakerState
    message: str = ""
    retry_after: float = 0.0   # seconds until half-open retry allowed
    escalation_count: int = 0
    target_tier: int | None = None  # suggested model tier on escalate

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "state": self.state,
            "message": self.message,
            "retry_after": self.retry_after,
            "escalation_count": self.escalation_count,
            "target_tier": self.target_tier,
        }


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """3-state circuit breaker with cooling timer and tier escalation.

    The breaker tracks per-session state and escalates through model
    tiers (T1→T2→T3→T4, on_user) on consecutive breaks. Based on the
    observation that small models (<12B) enter doom loops at elevated
    rates (22.9% for Qwen3.5-4B) and need automated escalation.

    State machine:
        CLOSED → (loop_score > threshold) → OPEN
        OPEN → (cooling_timer expired) → HALF_OPEN
        HALF_OPEN → (if success) → CLOSED
        HALF_OPEN → (if failure) → OPEN (escalated)

    Tier cascade:
        T1 (<4B) → T2 (4-8B) → T3 (9-12B) → T4 (cloud frontier) → human
    """

    MAX_BREAKS_BEFORE_HUMAN = 5  # After 5 breaks, always escalate to human

    def __init__(
        self,
        cooling_seconds: float = 30.0,
        loop_threshold: float = 0.75,
        max_breaks_before_human: int = MAX_BREAKS_BEFORE_HUMAN,
    ):
        self.cooling_seconds = cooling_seconds
        self.loop_threshold = loop_threshold
        self.max_breaks_before_human = max_breaks_before_human
        self._state: BreakerState = "closed"
        self._last_break_time: float = 0.0
        self._break_count: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0
        self._tier: int = 1  # Start at T1

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def tier(self) -> int:
        return self._tier

    def check(self, loop_score: float) -> CircuitDecision:
        """Evaluate whether to allow, block, or escalate.

        Args:
            loop_score: Score from LoopDetector.score().overall (0.0-1.0)

        Returns:
            CircuitDecision with the recommended action.
        """
        if self._state == "open":
            return self._handle_open()

        if self._state == "half_open":
            return self._handle_half_open(loop_score)

        # CLOSED state
        if loop_score >= self.loop_threshold:
            return self._trip_breaker(loop_score)
        elif loop_score >= self.loop_threshold * 0.7:
            # Warning zone — don't break but return advisory
            return CircuitDecision(
                action="allow",
                state="closed",
                message=f"Loop risk elevated ({loop_score:.2f}) — monitoring",
                escalation_count=self._break_count,
            )

        return CircuitDecision(
            action="allow",
            state="closed",
            escalation_count=self._break_count,
        )

    def register_success(self) -> None:
        """Register a successful tool call (no loop detected)."""
        self._success_count += 1
        if self._state == "half_open":
            logger.info("CircuitBreaker: half-open trial succeeded — returning to CLOSED")
            self._state = "closed"
            self._break_count = 0

    def register_break(self, loop_score: float | None = None) -> CircuitDecision:
        """Register a circuit break event, return escalation decision.

        This is called when a loop has been positively confirmed —
        not just warned, but detected. It increments the break counter
        and escalates through model tiers.
        """
        self._failure_count += 1
        self._break_count += 1
        return self._trip_breaker(loop_score or 1.0)

    def _trip_breaker(self, loop_score: float) -> CircuitDecision:
        """Trip the breaker and determine escalation."""
        self._state = "open"
        self._last_break_time = time.time()
        self._tier = min(self._tier + 1, 5)  # 5 = human review

        if self._break_count >= self.max_breaks_before_human:
            return self._human_escalation()

        # Determine target tier
        if self._tier >= 5:
            return self._human_escalation()

        tier_names = {1: "T1 (<4B)", 2: "T2 (4-8B)", 3: "T3 (9-12B)", 4: "T4 (cloud)"}

        return CircuitDecision(
            action="escalate_tier",
            state="open",
            message=(
                f"Loop detected (score={loop_score:.2f}). "
                f"Escalating from T{self._tier - 1} to T{self._tier} "
                f"({tier_names.get(self._tier, 'human')}). "
                f"Cool-off: {self.cooling_seconds}s."
            ),
            retry_after=self.cooling_seconds,
            escalation_count=self._break_count,
            target_tier=self._tier,
        )

    def _human_escalation(self) -> CircuitDecision:
        """Escalate to human review."""
        return CircuitDecision(
            action="escalate_human",
            state="open",
            message=(
                f"Circuit breaker tripped {self._break_count} times. "
                f"Model Tier {self._tier - 1} consistently fails. "
                f"Manual human review required."
            ),
            retry_after=0.0,
            escalation_count=self._break_count,
        )

    def _handle_open(self) -> CircuitDecision:
        """Handle check while breaker is OPEN."""
        elapsed = time.time() - self._last_break_time

        if elapsed >= self.cooling_seconds:
            # Cooling period expired — try half-open
            logger.info(
                "CircuitBreaker: cooling period elapsed (%.1fs >= %.1fs) — "
                "transitioning to HALF_OPEN",
                elapsed, self.cooling_seconds,
            )
            self._state = "half_open"
            return CircuitDecision(
                action="allow",
                state="half_open",
                message="Trial run — breaker is testing recovery. One failure re-trips.",
                escalation_count=self._break_count,
            )

        remaining = self.cooling_seconds - elapsed
        return CircuitDecision(
            action="block",
            state="open",
            message=f"Circuit breaker OPEN. Retry in {remaining:.0f}s (cooling).",
            retry_after=remaining,
            escalation_count=self._break_count,
        )

    def _handle_half_open(self, loop_score: float) -> CircuitDecision:
        """Handle check while breaker is HALF_OPEN."""
        if loop_score >= self.loop_threshold:
            # Re-tripped
            self._state = "open"
            self._last_break_time = time.time()
            self._break_count += 1

            logger.warning(
                "CircuitBreaker: HALF_OPEN trial failed (score=%.2f) — "
                "returning to OPEN",
                loop_score,
            )

            if self._break_count >= self.max_breaks_before_human:
                return self._human_escalation()

            return CircuitDecision(
                action="block",
                state="open",
                message="Half-open trial failed — breaker re-opened.",
                retry_after=self.cooling_seconds,
                escalation_count=self._break_count,
            )

        # Success on half-open — returning to closed
        return CircuitDecision(
            action="allow",
            state="half_open",
            message="Half-open trial clean — if this succeeds, breaker resets.",
            escalation_count=self._break_count,
        )

    def reset(self) -> None:
        """Full reset to CLOSED state."""
        self._state = "closed"
        self._last_break_time = 0.0
        self._break_count = 0
        self._tier = 1
        logger.info("CircuitBreaker: reset to CLOSED")

    def stats(self) -> dict:
        """Return current breaker statistics."""
        return {
            "state": self._state,
            "tier": self._tier,
            "break_count": self._break_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "cooling_seconds": self.cooling_seconds,
            "loop_threshold": self.loop_threshold,
        }
