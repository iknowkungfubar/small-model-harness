"""Self-Consistency & Specification-Grounded Verification — Phase 5

Detects semantic errors that structural validation misses by comparing
multiple responses (self-consistency) and checking against a formal
verification rubric (spec-grounded verification).

Three components:
  5a: Self-Consistency — compare key fields across multiple responses
  5b: Spec-Grounded Verification — check against expected schema/constraints
  5c: Verification Results — unified verdict with anomaly report
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class FieldComparison:
    """Result of comparing one field across responses."""

    field_name: str
    values: list[Any]
    all_match: bool
    most_common_value: Any | None
    agreement_ratio: float  # 0.0–1.0


@dataclass
class ConsistencyResult:
    """Result of self-consistency check across multiple responses."""

    is_consistent: bool
    confidence: float  # 0.0–1.0
    comparisons: list[FieldComparison]
    total_fields: int
    matching_fields: int
    agreement_ratio: float  # Overall agreement across all fields
    anomalies: list[str]


@dataclass
class VerificationRubric:
    """A rubric defining expected output characteristics.

    Used by spec-grounded verification to check model outputs
    against declarative constraints.
    """

    required_fields: list[str] = field(default_factory=list)
    forbidden_values: dict[str, list[Any]] = field(default_factory=dict)
    expected_types: dict[str, str] = field(default_factory=dict)
    numeric_ranges: dict[str, dict[str, float]] = field(default_factory=dict)
    custom_checks: list[dict[str, Any]] = field(default_factory=list)
    tool_name: str = ""
    max_tokens: int = 0
    invariances: list[str] = field(default_factory=list)  # Semantic invariants to check


@dataclass
class SpecVerificationResult:
    """Result of spec-grounded verification."""

    passes: bool
    score: float  # 0.0–1.0
    failures: list[str]
    warnings: list[str]
    field_results: dict[str, bool]
    tool_name: str
    total_checks: int
    passed_checks: int


@dataclass
class VerificationReport:
    """Complete verification result combining consistency + spec checks."""

    task_id: str
    tool_name: str
    overall_pass: bool
    overall_confidence: float  # 0.0–1.0
    consistency: ConsistencyResult | None
    spec_verification: SpecVerificationResult | None
    recommendation: str  # "proceed" | "retry" | "escalate" | "block"
    anomalies: list[str]


# ---------------------------------------------------------------------------
# Component 5a: Self-Consistency
# ---------------------------------------------------------------------------


class SelfConsistencyChecker:
    """Check self-consistency across multiple responses to the same task.

    Compares key fields (tool name, argument values) across N responses
    to detect disagreement, hallucination, or instability.
    """

    def check_consistency(
        self,
        responses: list[dict[str, Any] | str],
        key_fields: list[str] | None = None,
        tolerance: float = 0.7,
    ) -> ConsistencyResult:
        """Check consistency across multiple responses.

        Args:
            responses: N responses (parsed dicts or JSON strings).
            key_fields: Specific fields to compare (None = auto-detect).
            tolerance: Minimum agreement ratio to pass (default 0.7).

        Returns:
            ConsistencyResult with comparison details.

        """
        if len(responses) < 2:
            return ConsistencyResult(
                is_consistent=True,
                confidence=0.5,
                comparisons=[],
                total_fields=0,
                matching_fields=0,
                agreement_ratio=1.0,
                anomalies=["single_response_insufficient"],
            )

        # Parse responses if needed
        parsed: list[dict[str, Any]] = []
        for r in responses:
            if isinstance(r, str):
                try:
                    p = json.loads(r)
                    if isinstance(p, dict):
                        parsed.append(p)
                    else:
                        parsed.append({"value": p})
                except (json.JSONDecodeError, TypeError):
                    parsed.append({"raw": r})
            elif isinstance(r, dict):
                parsed.append(r)
            else:
                parsed.append({"raw": str(r)})

        if len(parsed) < 2:
            return ConsistencyResult(
                is_consistent=True,
                confidence=0.5,
                comparisons=[],
                total_fields=0,
                matching_fields=0,
                agreement_ratio=1.0,
                anomalies=["unparseable_responses"],
            )

        # Auto-detect fields if not specified
        if key_fields is None:
            key_fields = self._auto_detect_fields(parsed)

        # Compare each field
        comparisons: list[FieldComparison] = []
        matching = 0
        anomalies: list[str] = []

        for field in key_fields:
            values = [p.get(field) for p in parsed]
            # Normalize None -> None for comparison
            non_none = [v for v in values if v is not None]

            if not non_none:
                matches = True
                most_common = None
                agreement = 1.0
            else:
                # Count occurrences of each value
                str_values = [
                    json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
                    for v in non_none
                ]
                from collections import Counter

                value_counts = Counter(str_values)
                most_common_str, count = value_counts.most_common(1)[0]
                most_common = non_none[str_values.index(most_common_str)]
                agreement = count / len(non_none)
                matches = all(v == most_common for v in non_none)

            if matches:
                matching += 1

            comparisons.append(
                FieldComparison(
                    field_name=field,
                    values=values,
                    all_match=matches,
                    most_common_value=most_common,
                    agreement_ratio=agreement,
                )
            )

            if not matches:
                # Check if disagreement is near-tie or clear outlier
                str_values = [
                    json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
                    for v in values
                    if v is not None
                ]
                if str_values:
                    from collections import Counter

                    vc = Counter(str_values)
                    if len(vc) >= 2:
                        top_two = vc.most_common(2)
                        # Check all-different first: every value appears exactly once
                        if all(count == 1 for _, count in vc.items()):
                            anomalies.append(f"all_different_on_field:{field}")
                        elif top_two[0][1] == top_two[1][1]:
                            anomalies.append(f"tie_on_field:{field}")
                        else:
                            anomalies.append(f"disagreement_on_field:{field}")

        total = len(key_fields)
        agreement_ratio = matching / total if total > 0 else 1.0
        is_consistent = agreement_ratio >= tolerance

        return ConsistencyResult(
            is_consistent=is_consistent,
            confidence=agreement_ratio,
            comparisons=comparisons,
            total_fields=total,
            matching_fields=matching,
            agreement_ratio=agreement_ratio,
            anomalies=anomalies,
        )

    @staticmethod
    def _auto_detect_fields(responses: list[dict[str, Any]]) -> list[str]:
        """Auto-detect fields to compare from response structure.

        Picks fields present in at least half of responses, preferring
        common tool-call fields.
        """
        from collections import Counter

        all_keys: list[str] = []
        for resp in responses:
            all_keys.extend(resp.keys())

        field_counts = Counter(all_keys)
        threshold = max(1, len(responses) // 2)

        # Priority order for field importance
        priority_fields = [
            "tool_name",
            "tool",
            "name",
            "action",
            "arguments",
            "args",
            "params",
            "parameters",
            "input",
            "query",
            "target",
            "value",
            "result",
            "output",
            "reasoning",
            "thought",
            "explanation",
        ]

        # Build result: priority fields first, then others
        detected: list[str] = []
        seen: set[str] = set()
        for field in priority_fields:
            if field in field_counts and field_counts[field] >= threshold:
                detected.append(field)
                seen.add(field)

        for field, count in field_counts.most_common():
            if field not in seen and count >= threshold:
                detected.append(field)
                seen.add(field)

        return detected[:20]  # Cap at 20 fields


# ---------------------------------------------------------------------------
# Component 5b: Spec-Grounded Verification
# ---------------------------------------------------------------------------


class SpecGroundedVerifier:
    """Check model outputs against a declarative verification rubric.

    Unlike self-consistency (which compares responses to each other),
    spec-grounded verification checks each response against known
    constraints — expected fields, types, ranges, forbidden values.
    """

    def verify(
        self,
        response: dict[str, Any] | str,
        rubric: VerificationRubric,
    ) -> SpecVerificationResult:
        """Verify a single response against a rubric.

        Args:
            response: Parsed response or JSON string.
            rubric: VerificationRubric with expected characteristics.

        Returns:
            SpecVerificationResult with pass/fail and details.

        """
        # Parse response
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                return SpecVerificationResult(
                    passes=False,
                    score=0.0,
                    failures=["unparseable_json"],
                    warnings=[],
                    field_results={},
                    tool_name=rubric.tool_name,
                    total_checks=1,
                    passed_checks=0,
                )
        else:
            parsed = response

        if not isinstance(parsed, dict):
            return SpecVerificationResult(
                passes=False,
                score=0.0,
                failures=["response_not_an_object"],
                warnings=[],
                field_results={},
                tool_name=rubric.tool_name,
                total_checks=1,
                passed_checks=0,
            )

        failures: list[str] = []
        warnings: list[str] = []
        field_results: dict[str, bool] = {}
        total_checks = 0
        passed_checks = 0

        # 1. Check required fields
        for field in rubric.required_fields:
            total_checks += 1
            if field in parsed and parsed[field] is not None:
                field_results[f"required:{field}"] = True
                passed_checks += 1
            else:
                field_results[f"required:{field}"] = False
                failures.append(f"missing_required_field:{field}")

        # 2. Check forbidden values
        for field, forbidden_values in rubric.forbidden_values.items():
            if field in parsed:
                for f_val in forbidden_values:
                    total_checks += 1
                    if parsed[field] == f_val:
                        field_results[f"forbidden:{field}={f_val}"] = False
                        failures.append(f"forbidden_value:{field}={f_val}")
                    else:
                        field_results[f"forbidden:{field}={f_val}"] = True
                        passed_checks += 1

        # 3. Check expected types
        for field, expected_type in rubric.expected_types.items():
            if field in parsed and parsed[field] is not None:
                total_checks += 1
                type_ok = self._check_type(parsed[field], expected_type)
                field_results[f"type:{field}"] = type_ok
                if type_ok:
                    passed_checks += 1
                else:
                    failures.append(
                        f"type_mismatch:{field} expected={expected_type} "
                        f"got={type(parsed[field]).__name__}"
                    )

        # 4. Check numeric ranges
        for field, range_spec in rubric.numeric_ranges.items():
            if field in parsed and isinstance(parsed[field], (int, float)):
                total_checks += 1
                value = parsed[field]
                range_ok = True
                if "min" in range_spec and value < range_spec["min"]:
                    range_ok = False
                if "max" in range_spec and value > range_spec["max"]:
                    range_ok = False
                field_results[f"range:{field}"] = range_ok
                if range_ok:
                    passed_checks += 1
                else:
                    failures.append(f"range_violation:{field}={value} range={range_spec}")

        # 5. Run custom checks
        for i, check in enumerate(rubric.custom_checks):
            total_checks += 1
            check_type = check.get("type", "")
            check_field = check.get("field", "")
            check_value = check.get("value")

            if check_type == "not_empty" and check_field:
                val = parsed.get(check_field)
                if (val and (isinstance(val, str) and val.strip())) or (
                    isinstance(val, list) and val
                ):
                    passed_checks += 1
                    field_results[f"custom_{i}:{check_type}:{check_field}"] = True
                else:
                    field_results[f"custom_{i}:{check_type}:{check_field}"] = False
                    failures.append(f"custom_check_failed:{check_type}:{check_field}")
            elif check_type == "contains" and check_field:
                val = parsed.get(check_field)
                target = check.get("target", "")
                if isinstance(val, str) and target in val:
                    passed_checks += 1
                    field_results[f"custom_{i}:{check_type}:{check_field}"] = True
                else:
                    field_results[f"custom_{i}:{check_type}:{check_field}"] = False
                    failures.append(f"custom_check_failed:{check_type}:{check_field}={target}")
            elif check_type == "not_equals" and check_field:
                val = parsed.get(check_field)
                target = check.get("value")
                if val != target:
                    passed_checks += 1
                    field_results[f"custom_{i}:{check_type}:{check_field}"] = True
                else:
                    field_results[f"custom_{i}:{check_type}:{check_field}"] = False
                    failures.append(f"custom_check_failed:{check_type}:{check_field}!={target}")
            else:
                # Unknown check type — skip
                total_checks -= 1

        # Calculate score
        score = passed_checks / total_checks if total_checks > 0 else 1.0
        passes = score >= 0.8  # 80% threshold

        return SpecVerificationResult(
            passes=passes,
            score=score,
            failures=failures,
            warnings=warnings,
            field_results=field_results,
            tool_name=rubric.tool_name,
            total_checks=total_checks,
            passed_checks=passed_checks,
        )

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if a value matches an expected type.

        Args:
            value: The value to check.
            expected_type: One of "string", "integer", "number",
                "boolean", "array", "object".

        """
        type_map: dict[str, type | tuple[type, type]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        py_type = type_map.get(expected_type)
        if py_type is None:
            return True  # Unknown type — pass
        return isinstance(value, py_type)

    def build_rubric_from_schema(
        self, schema: dict[str, Any], tool_name: str = ""
    ) -> VerificationRubric:
        """Build a VerificationRubric from a JSON Schema dict.

        Extracts required fields, types, enums, and numeric ranges.
        """
        rubric = VerificationRubric(tool_name=tool_name)

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Required fields
        rubric.required_fields = required if isinstance(required, list) else []

        # Expected types
        for field_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_type = prop_schema.get("type")
                if prop_type:
                    rubric.expected_types[field_name] = prop_type

                # Enum values → forbidden values for out-of-range checks
                enum_vals = prop_schema.get("enum")
                if enum_vals and field_name in (required or []):
                    pass  # Don't add enum constraints as type checks

                # Numeric ranges
                if prop_type in ("integer", "number"):
                    range_spec: dict[str, float] = {}
                    if "minimum" in prop_schema:
                        range_spec["min"] = prop_schema["minimum"]
                    if "maximum" in prop_schema:
                        range_spec["max"] = prop_schema["maximum"]
                    if range_spec:
                        rubric.numeric_ranges[field_name] = range_spec

        return rubric

    def create_consistency_rubric(
        self,
        tool_name: str,
        key_fields: list[str],
        expected_types: dict[str, str] | None = None,
    ) -> VerificationRubric:
        """Create a verification rubric for consistency checking.

        Used when you want spec-grounded verification alongside or
        instead of self-consistency.
        """
        return VerificationRubric(
            tool_name=tool_name,
            required_fields=key_fields,
            expected_types=expected_types or {},
            custom_checks=[{"type": "not_empty", "field": field} for field in key_fields],
        )


# ---------------------------------------------------------------------------
# Combined Verification
# ---------------------------------------------------------------------------


class Verifier:
    """Combined verifier that runs both self-consistency and spec-grounded checks.

    Returns a unified VerificationReport with routing recommendation.
    """

    def __init__(self) -> None:
        self.consistency_checker = SelfConsistencyChecker()
        self.spec_verifier = SpecGroundedVerifier()

    def verify(
        self,
        task_id: str,
        tool_name: str,
        responses: list[dict[str, Any] | str],
        rubric: VerificationRubric | None = None,
        key_fields: list[str] | None = None,
    ) -> VerificationReport:
        """Run full verification pipeline.

        Args:
            task_id: Unique task identifier.
            tool_name: Name of the tool being verified.
            responses: One or more responses to verify.
            rubric: Optional spec-grounded verification rubric.
            key_fields: Optional fields for consistency check.

        Returns:
            VerificationReport with combined results.

        """
        anomalies: list[str] = []

        # Consistency check (requires 2+ responses)
        consistency: ConsistencyResult | None = None
        if len(responses) >= 2:
            consistency = self.consistency_checker.check_consistency(
                responses=responses, key_fields=key_fields
            )
            if consistency and not consistency.is_consistent:
                anomalies.extend(consistency.anomalies)

        # Spec-grounded verification
        spec_result: SpecVerificationResult | None = None
        if rubric:
            # Verify the first response against the rubric
            spec_result = self.spec_verifier.verify(
                response=responses[0] if responses else "{}",
                rubric=rubric,
            )
            if spec_result and not spec_result.passes:
                anomalies.extend(f"spec_fail:{f}" for f in spec_result.failures)

        # Overall confidence
        confidence_scores: list[float] = []
        if consistency:
            confidence_scores.append(consistency.confidence)
        if spec_result:
            confidence_scores.append(spec_result.score)

        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        )

        # Recommendation
        recommendation = self._recommend(
            overall_pass=len(anomalies) == 0,
            confidence=overall_confidence,
            anomalies=anomalies,
        )

        return VerificationReport(
            task_id=task_id,
            tool_name=tool_name,
            overall_pass=len(anomalies) == 0,
            overall_confidence=overall_confidence,
            consistency=consistency,
            spec_verification=spec_result,
            recommendation=recommendation,
            anomalies=anomalies,
        )

    @staticmethod
    def _recommend(
        overall_pass: bool,
        confidence: float,
        anomalies: list[str],
    ) -> str:
        """Generate routing recommendation from verification state."""
        if not overall_pass:
            if confidence < 0.3 or len(anomalies) >= 3:
                return "block"
            return "retry"
        if confidence < 0.5:
            return "escalate"
        if confidence < 0.7:
            return "verify_again"
        return "proceed"


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------


def check_consistency(
    responses: list[dict[str, Any] | str],
    key_fields: list[str] | None = None,
) -> ConsistencyResult:
    """One-shot self-consistency check."""
    checker = SelfConsistencyChecker()
    return checker.check_consistency(responses=responses, key_fields=key_fields)


def verify_against_rubric(
    response: dict[str, Any] | str,
    rubric: VerificationRubric,
) -> SpecVerificationResult:
    """One-shot spec-grounded verification."""
    verifier = SpecGroundedVerifier()
    return verifier.verify(response=response, rubric=rubric)
