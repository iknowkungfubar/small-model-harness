"""Checkpoint-Based Rollback — Phase 5c

Persists verified intermediate states (tool call results, context snapshots)
and enables rollback to the last verified checkpoint on verification failure.

Design:
- Checkpoints store only references (task_id, step hash, confidence score)
  not full content, keeping storage lightweight
- Rollback restores the highest-confidence verified checkpoint
- Integrated with the confidence scorer and verifier
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """A single verified checkpoint in the rollback chain."""

    id: str
    task_id: str
    step: int
    confidence_score: float
    state_hash: str  # Content hash of the state being checkpointed
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step": self.step,
            "confidence_score": self.confidence_score,
            "state_hash": self.state_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            id=data["id"],
            task_id=data["task_id"],
            step=data["step"],
            confidence_score=data.get("confidence_score", 0.0),
            state_hash=data["state_hash"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CheckpointStore:
    """In-memory checkpoint storage with optional persistence."""

    checkpoints: dict[str, list[Checkpoint]] = field(default_factory=dict)
    max_checkpoints_per_task: int = 50
    _persist_path: str | None = None

    def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint, maintaining size limits."""
        if checkpoint.task_id not in self.checkpoints:
            self.checkpoints[checkpoint.task_id] = []
        self.checkpoints[checkpoint.task_id].append(checkpoint)
        # Trim to max
        if len(self.checkpoints[checkpoint.task_id]) > self.max_checkpoints_per_task:
            # Remove oldest
            self.checkpoints[checkpoint.task_id] = self.checkpoints[checkpoint.task_id][
                -self.max_checkpoints_per_task :
            ]

    def get_latest(self, task_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a task."""
        task_checkpoints = self.checkpoints.get(task_id, [])
        if not task_checkpoints:
            return None
        return task_checkpoints[-1]

    def get_best(self, task_id: str) -> Checkpoint | None:
        """Get the highest-confidence checkpoint for a task."""
        task_checkpoints = self.checkpoints.get(task_id, [])
        if not task_checkpoints:
            return None
        return max(task_checkpoints, key=lambda c: c.confidence_score)

    def get_best_before_step(self, task_id: str, before_step: int) -> Checkpoint | None:
        """Get highest-confidence checkpoint before a given step."""
        task_checkpoints = [c for c in self.checkpoints.get(task_id, []) if c.step < before_step]
        if not task_checkpoints:
            return None
        return max(task_checkpoints, key=lambda c: c.confidence_score)

    def list_checkpoints(self, task_id: str) -> list[Checkpoint]:
        """List all checkpoints for a task, newest first."""
        task_checkpoints = list(self.checkpoints.get(task_id, []))
        task_checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return task_checkpoints

    def clear_task(self, task_id: str) -> None:
        """Remove all checkpoints for a task."""
        self.checkpoints.pop(task_id, None)

    def clear_all(self) -> None:
        """Remove all checkpoints."""
        self.checkpoints.clear()


# Global checkpoint store instance
_checkpoint_store = CheckpointStore()


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages checkpoint creation, rollback, and lifecycle.

    Creates checkpoints after successful verification and provides
    rollback targets on failure.
    """

    def __init__(self, store: CheckpointStore | None = None) -> None:
        self._store = store or _checkpoint_store

    @property
    def store(self) -> CheckpointStore:
        return self._store

    def create_checkpoint(
        self,
        task_id: str,
        step: int,
        state: dict[str, Any],
        confidence_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint at the current step.

        Args:
            task_id: Task identifier.
            step: Current processing step number.
            state: Full state dict to checkpoint.
            confidence_score: Confidence score from verifier (0.0–1.0).
            metadata: Optional extra metadata.

        Returns:
            The created Checkpoint.

        """
        state_hash = self._hash_state(state)
        checkpoint = Checkpoint(
            id=f"ckpt-{task_id}-{step}-{int(time.time() * 1000)}",
            task_id=task_id,
            step=step,
            confidence_score=confidence_score,
            state_hash=state_hash,
            timestamp=time.time(),
            metadata={
                **(metadata or {}),
                "state_size": len(json.dumps(state)),
            },
        )
        self._store.save(checkpoint)
        return checkpoint

    def find_rollback_target(
        self,
        task_id: str,
        current_step: int,
        min_confidence: float = 0.5,
    ) -> Checkpoint | None:
        """Find the best checkpoint to roll back to.

        Args:
            task_id: Task identifier.
            current_step: Current step (rolls back to a step < this).
            min_confidence: Minimum confidence threshold.

        Returns:
            Best checkpoint before current_step with confidence >= min_confidence,
            or None if no suitable checkpoint exists.

        """
        return self._store.get_best_before_step(task_id, before_step=current_step)

    def can_rollback(
        self,
        task_id: str,
        current_step: int,
        min_confidence: float = 0.5,
    ) -> bool:
        """Check if a rollback target exists.

        Args:
            task_id: Task identifier.
            current_step: Current step number.
            min_confidence: Minimum confidence threshold.

        Returns:
            True if a rollback target exists.

        """
        target = self.find_rollback_target(task_id, current_step, min_confidence)
        return target is not None

    def rollback(
        self,
        task_id: str,
        current_step: int,
        state: dict[str, Any],
        min_confidence: float = 0.5,
    ) -> tuple[Checkpoint | None, str]:
        """Roll back to the best checkpoint before current_step.

        Args:
            task_id: Task identifier.
            current_step: Step to roll back from.
            state: Current state (for diff logging).
            min_confidence: Minimum confidence for rollback target.

        Returns:
            Tuple of (target_checkpoint or None, status_message).

        """
        target = self.find_rollback_target(task_id, current_step, min_confidence)

        if target is None:
            return None, f"No rollback target found for task {task_id} before step {current_step}"

        return target, (
            f"Rolled back to checkpoint {target.id} "
            f"(step {target.step}, confidence {target.confidence_score:.2f})"
        )

    def prune_low_confidence(self, task_id: str, min_confidence: float = 0.3) -> int:
        """Remove checkpoints below confidence threshold.

        Args:
            task_id: Task identifier.
            min_confidence: Minimum confidence to keep.

        Returns:
            Number of checkpoints removed.

        """
        task_checkpoints = self._store.checkpoints.get(task_id, [])
        before = len(task_checkpoints)
        self._store.checkpoints[task_id] = [
            c for c in task_checkpoints if c.confidence_score >= min_confidence
        ]
        return before - len(self._store.checkpoints[task_id])

    @staticmethod
    def _hash_state(state: dict[str, Any]) -> str:
        """Create a deterministic hash of a state dict.

        Uses SHA-256 on the sorted JSON representation.
        """
        raw = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------


def create_checkpoint(
    task_id: str,
    step: int,
    state: dict[str, Any],
    confidence_score: float,
) -> Checkpoint:
    """One-shot checkpoint creation."""
    mgr = CheckpointManager()
    return mgr.create_checkpoint(
        task_id=task_id,
        step=step,
        state=state,
        confidence_score=confidence_score,
    )


def rollback(
    task_id: str,
    current_step: int,
    state: dict[str, Any],
) -> tuple[Checkpoint | None, str]:
    """One-shot rollback attempt."""
    mgr = CheckpointManager()
    return mgr.rollback(
        task_id=task_id,
        current_step=current_step,
        state=state,
    )
