"""Input/Output Guardrails — Phase 7

Adds production security guardrails for the small-model-harness:

7a: Input Guardrails
  - Prompt injection detection (heuristic patterns)
  - Jailbreak detection (role-playing, hypothetical framing)
  - PII scanning on input (credit cards, SSN, API keys)

7b: Output Guardrails
  - PII scanning on output (data leakage prevention)
  - Topic boundary enforcement (off-topic detection)
  - Tool call argument boundary validation

Integration:
  Plugin pre_tool_call: run input guardrails on user message
  Plugin post_tool_call: run output guardrails on model response
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known prompt injection patterns
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (
        "ignore_previous",
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|commands|directions)",
    ),
    (
        "new_instructions",
        r"(?i)(you\s+are\s+(now|not\s+an?\s+ai)|forget\s+(everything|all)|start\s+over)",
    ),
    (
        "system_prompt_override",
        r"(?i)(system\s+(prompt|message|instruction)|you\s+must\s+now|your\s+new\s+(role|persona))",
    ),
    ("delimited_override", r"(?i)(\[system\]|<system>|\"\"\"system|\'\'\'system|--system)"),
    (
        "role_switch",
        r"(?i)act\s+as\s+(if|though)\s+you\s+(are|were)|roleplay\s+as|pretend\s+(to\s+be|you(\'re| are))",
    ),
    (
        "hypothetical_override",
        r"(?i)in\s+a\s+hypothetical\s+scenario|fictional\s+scenario|for\s+the\s+sake\s+of\s+this\s+exercise",
    ),
    ("token_leak", r"(?i)(api[_-]?key|secret[_-]?key|auth[_-]?token|bearer[\s=]+\w{20,})"),
    (
        "data_leak_request",
        r"(?i)(show\s+me\s+(your\s+)?(prompt|instructions|system|persona)|what\s+(are|is)\s+your\s+prompt)",
    ),
    ("chain_break", r"(?i)(|\||separator|delimiter).{0,10}(break|ignore|skip|bypass)"),
    (
        "output_format_override",
        r"(?i)(output\s+(format|mode)|respond\s+in\s+(json|xml|yaml))\s+(instead|rather|not)",
    ),
]

# Jailbreak patterns
JAILBREAK_PATTERNS: list[tuple[str, str]] = [
    ("dan_mode", r"(?i)dan\s+(mode|jailbreak|bypass|override)|do\s+anything\s+now"),
    (
        "character_roleplay",
        r"(?i)you\s+are\s+(now\s+)?(a\s+)?(character|person|human|assistant)\s+(named|called)\s+\w+",
    ),
    (
        "ethical_bypass",
        r"(?i)(for\s+(educational|research|testing|academic)\s+(purposes|reasons)|for\s+educational\s+and\s+research\s+purposes\s+only)",
    ),
    (
        "security_bypass",
        r"(?i)(bypass|circumvent|override|disable)\s+(security|safety|guardrail|restriction|filter|limit)",
    ),
    (
        "no_restrictions",
        r"(?i)(no\s+(restrictions|limits|boundaries|rules|constraints)|unrestricted|unfiltered|uncensored)",
    ),
]

# PII patterns (input — used to detect PII being sent to the model)
INPUT_PII_PATTERNS: list[tuple[str, str]] = [
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ("api_key_generic", r"(?i)(?:api[_-]?key|secret|token)[:\s=]*['\"]?[a-z0-9_\-]{16,}['\"]?"),
    ("aws_key", r"(?i)AKIA[0-9A-Z]{16}"),
    ("github_token", r"(?i)gh[pousr]_[A-Za-z0-9_]{36,}"),
    ("email", r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    ("phone", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    ("ip_address", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
]

# Output PII patterns (stricter — detect data leakage)
OUTPUT_PII_PATTERNS: list[tuple[str, str]] = [
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ("aws_key", r"(?i)AKIA[0-9A-Z]{16}"),
    ("github_token", r"(?i)gh[pousr]_[A-Za-z0-9_]{36,}"),
]

# Off-topic keywords (if ALL these appear in output, it's off-topic for tool calls)
OFF_TOPIC_KEYWORDS: list[re.Pattern] = [
    re.compile(
        r"(?i)\b(stock|price|invest|market|weather|news|sports|celebrity|politics|religion)\b"
    ),
]


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    score: float  # 0.0 (safe) to 1.0 (dangerous)
    flags: list[str]  # e.g., ["injection:ignore_previous", "pii:ssn"]
    details: list[dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""  # "allow" | "review" | "block"


@dataclass
class GuardrailConfig:
    """Configuration for guardrail behavior."""

    enabled: bool = True
    injection_detection: bool = True
    jailbreak_detection: bool = True
    pii_scan_input: bool = True
    pii_scan_output: bool = True
    topic_boundary: bool = True
    arg_boundary_validation: bool = True
    block_threshold: float = 0.7  # Score above this blocks
    warn_threshold: float = 0.3  # Score above this warns


# ---------------------------------------------------------------------------
# Input Guardrails
# ---------------------------------------------------------------------------


class InputGuardrails:
    """Input-side guardrail checks.

    Run before sending a user message to the model.
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()

    def check(self, text: str) -> GuardrailResult:
        """Run all input guardrail checks on user-provided text.

        Args:
            text: The user input or task description.

        Returns:
            GuardrailResult with pass/fail, flags, and recommendation.

        """
        if not self.config.enabled or not text:
            return GuardrailResult(passed=True, score=0.0, flags=[], recommendation="allow")

        all_flags: list[str] = []
        all_details: list[dict[str, Any]] = []
        max_score = 0.0

        # 1. Prompt injection detection
        if self.config.injection_detection:
            for name, pattern in INJECTION_PATTERNS:
                matches = list(re.finditer(pattern, text))
                if matches:
                    flag = f"injection:{name}"
                    all_flags.append(flag)
                    all_details.append({
                        "type": "injection",
                        "pattern": name,
                        "matches": [m.group()[:100] for m in matches[:3]],
                    })
                    # Score contribution: high for direct injection
                    max_score = max(max_score, 0.8)

        # 2. Jailbreak detection
        if self.config.jailbreak_detection:
            for name, pattern in JAILBREAK_PATTERNS:
                matches = list(re.finditer(pattern, text))
                if matches:
                    flag = f"jailbreak:{name}"
                    all_flags.append(flag)
                    all_details.append({
                        "type": "jailbreak",
                        "pattern": name,
                        "matches": [m.group()[:100] for m in matches[:3]],
                    })
                    max_score = max(max_score, 0.9)

        # 3. PII scanning on input
        if self.config.pii_scan_input:
            for name, pattern in INPUT_PII_PATTERNS:
                matches = list(re.finditer(pattern, text))
                if matches:
                    flag = f"pii_input:{name}"
                    all_flags.append(flag)
                    all_details.append({
                        "type": "pii_input",
                        "pattern": name,
                        "matches": [m.group()[:100] for m in matches[:3]],
                    })
                    max_score = max(max_score, 0.6)

        # Determine recommendation
        recommendation = self._classify(max_score, all_flags)

        return GuardrailResult(
            passed=max_score < self.config.block_threshold,
            score=max_score,
            flags=all_flags,
            details=all_details,
            recommendation=recommendation,
        )

    def _classify(self, score: float, flags: list[str]) -> str:
        """Classify the severity of a guardrail result."""
        if score >= self.config.block_threshold:
            return "block"
        if score >= self.config.warn_threshold:
            return "review"
        return "allow"


# ---------------------------------------------------------------------------
# Output Guardrails
# ---------------------------------------------------------------------------


class OutputGuardrails:
    """Output-side guardrail checks.

    Run on model responses before delivering to the user or executing
    as a tool call.
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()

    def check(self, output: str | dict[str, Any]) -> GuardrailResult:
        """Run all output guardrail checks on model output.

        Args:
            output: The model's response (string or parsed dict).

        Returns:
            GuardrailResult with pass/fail.

        """
        if not self.config.enabled:
            return GuardrailResult(passed=True, score=0.0, flags=[], recommendation="allow")

        # Convert to string for pattern matching
        if isinstance(output, dict):
            text = json.dumps(output)
        else:
            text = str(output)

        all_flags: list[str] = []
        all_details: list[dict[str, Any]] = []
        max_score = 0.0

        # 1. PII scanning on output (leakage prevention)
        if self.config.pii_scan_output:
            for name, pattern in OUTPUT_PII_PATTERNS:
                matches = list(re.finditer(pattern, text))
                if matches:
                    flag = f"pii_output:{name}"
                    all_flags.append(flag)
                    all_details.append({
                        "type": "pii_output",
                        "pattern": name,
                        "matches": [m.group()[:100] for m in matches[:3]],
                    })
                    max_score = max(max_score, 0.8)

        # 2. Topic boundary enforcement
        if self.config.topic_boundary and isinstance(output, dict):
            topic_violations = self._check_topic_boundary(output)
            if topic_violations:
                for violation in topic_violations:
                    all_flags.append(f"off_topic:{violation}")
                    all_details.append({
                        "type": "off_topic",
                        "pattern": violation,
                    })
                max_score = max(max_score, 0.5)

        # 3. Argument boundary validation (for tool call dicts)
        if self.config.arg_boundary_validation and isinstance(output, dict):
            arg_issues = self._validate_arguments(output)
            if arg_issues:
                for issue in arg_issues:
                    all_flags.append(f"arg_boundary:{issue}")
                    all_details.append({
                        "type": "arg_boundary",
                        "issue": issue,
                    })
                max_score = max(max_score, 0.4)

        recommendation = self._classify(max_score, all_flags)

        return GuardrailResult(
            passed=max_score < self.config.block_threshold,
            score=max_score,
            flags=all_flags,
            details=all_details,
            recommendation=recommendation,
        )

    def _check_topic_boundary(self, output: dict[str, Any]) -> list[str]:
        """Check if output tool call is off-topic.

        Looks at the tool name and arguments for known off-topic keywords.
        """
        violations: list[str] = []
        text_to_check = json.dumps(output)

        for pattern in OFF_TOPIC_KEYWORDS:
            if pattern.search(text_to_check):
                violations.append(pattern.pattern[:40])

        return violations[:5]

    def _validate_arguments(self, output: dict[str, Any]) -> list[str]:
        """Validate tool call argument boundaries.

        Checks for:
        - Extremely long string values
        - Path traversal in file paths
        - Integer values outside reasonable ranges
        """
        issues: list[str] = []
        args = output.get("arguments", output.get("args", output.get("params", {})))

        if not isinstance(args, dict):
            return []

        for key, value in args.items():
            # Check string length
            if isinstance(value, str) and len(value) > 10000:
                issues.append(f"string_too_long:{key}({len(value)}chars)")

            # Check for path traversal
            if isinstance(value, str) and ".." in value and "/" in value:
                issues.append(f"path_traversal:{key}")

            # Check integer ranges
            if isinstance(value, int):
                if abs(value) > 1_000_000_000:
                    issues.append(f"extreme_value:{key}={value}")

        return issues[:10]

    def _classify(self, score: float, flags: list[str]) -> str:
        """Classify the severity of a guardrail result."""
        if score >= self.config.block_threshold:
            return "block"
        if score >= self.config.warn_threshold:
            return "review"
        return "allow"


# ---------------------------------------------------------------------------
# Combined Guardrail System
# ---------------------------------------------------------------------------


class GuardrailSystem:
    """Combined input/output guardrail system.

    Provides a single entry point for running all guardrail checks
    at both input and output stages.
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()
        self.input_guardrails = InputGuardrails(self.config)
        self.output_guardrails = OutputGuardrails(self.config)

    def check_input(self, text: str) -> GuardrailResult:
        """Run input guardrails."""
        return self.input_guardrails.check(text)

    def check_output(self, output: str | dict[str, Any]) -> GuardrailResult:
        """Run output guardrails."""
        return self.output_guardrails.check(output)

    def check_both(
        self, text: str, output: str | dict[str, Any]
    ) -> tuple[GuardrailResult, GuardrailResult]:
        """Run both input and output guardrails."""
        return (
            self.input_guardrails.check(text),
            self.output_guardrails.check(output),
        )


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------


def check_input(text: str, config: GuardrailConfig | None = None) -> GuardrailResult:
    """One-shot input guardrail check."""
    return InputGuardrails(config).check(text)


def check_output(
    output: str | dict[str, Any], config: GuardrailConfig | None = None
) -> GuardrailResult:
    """One-shot output guardrail check."""
    return OutputGuardrails(config).check(output)
