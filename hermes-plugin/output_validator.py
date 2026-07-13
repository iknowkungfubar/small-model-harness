"""Output Validator — Phase 4 Tier 1: Post-hoc validation + retry.

Validates LLM tool call output against defined schemas and provides
a retry loop with escalating feedback. This is the default path for
LM Studio and other API-only backends without logit masking access.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ToolCallResult:
    """Result of a tool call validation."""

    valid: bool
    parsed: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "parsed": self.parsed,
            "errors": self.errors,
            "warnings": self.warnings,
            "attempts": self.attempts,
        }


class ValidationError(Exception):
    """Raised when validation fails definitively."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Code fence pattern for extracting JSON from markdown
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.+?)\n?```",
    re.DOTALL,
)

# Built-in tool schemas (same structure as validator.py's TOOL_SCHEMAS)
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "terminal": {
        "properties": {
            "command": {"type": "string", "min_length": 1},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
            "workdir": {"type": "string"},
            "background": {"type": "boolean"},
            "pty": {"type": "boolean"},
        },
        "required": ["command"],
    },
    "write_file": {
        "properties": {
            "path": {"type": "string", "min_length": 1},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    "patch": {
        "properties": {
            "mode": {"type": "string", "enum": ["replace", "patch"]},
            "path": {"type": "string", "min_length": 1},
            "old_string": {"type": "string", "min_length": 1},
            "new_string": {"type": "string"},
        },
        "required": ["mode", "path", "old_string", "new_string"],
    },
    "read_file": {
        "properties": {
            "path": {"type": "string", "min_length": 1},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
        },
        "required": ["path"],
    },
    "search_files": {
        "properties": {
            "pattern": {"type": "string", "min_length": 1},
            "target": {"type": "string", "enum": ["content", "files"]},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
    },
    "web_search": {
        "properties": {
            "query": {"type": "string", "min_length": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["query"],
    },
    "memory": {
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
            "target": {"type": "string", "enum": ["memory", "user"]},
            "content": {"type": "string"},
        },
        "required": ["action", "target"],
    },
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class OutputValidator:
    """Post-hoc validator for LLM tool call output.

    Validates LLM responses against tool schemas and provides a retry
    loop with escalating feedback. Designed for API-only backends where
    token-level constrained decoding is unavailable.

    Features:
    - JSON extraction from code fences
    - Structural validation (types, required fields, enums, ranges)
    - Multi-level retry feedback (error → schema → example)
    - Configurable max retries
    """

    def __init__(self, schemas: dict[str, dict[str, Any]] | None = None) -> None:
        self._schemas = dict(schemas) if schemas else dict(TOOL_SCHEMAS)

    def register_schema(self, tool_name: str, schema: dict[str, Any]) -> None:
        """Register or override a tool schema."""
        self._schemas[tool_name] = schema

    def get_schema(self) -> dict[str, dict[str, Any]]:
        """Get the current schema registry."""
        return dict(self._schemas)

    # ------------------------------------------------------------------
    # JSON Validation
    # ------------------------------------------------------------------

    def validate_json(self, response: str) -> ToolCallResult:
        """Parse and validate that response is valid JSON object.

        Steps:
        1. Check for empty/whitespace-only response
        2. Extract JSON from code fence if present
        3. Parse as JSON
        4. Verify it's a dict/object (not array or primitive)

        Args:
            response: Raw text response from the LLM.

        Returns:
            ToolCallResult with parsed data or errors.

        """
        raw = response or ""

        # Check empty
        if not raw.strip():
            return ToolCallResult(
                valid=False,
                errors=["Empty response from model — no content to validate"],
                raw_response=raw,
            )

        # Extract from code fence
        fence_match = _CODE_FENCE_RE.search(raw)
        if fence_match:
            raw = fence_match.group(1).strip()

        # Try to parse JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            return ToolCallResult(
                valid=False,
                errors=[f"Invalid JSON: {e.msg} (line {e.lineno}, col {e.colno})"],
                raw_response=raw,
            )

        # Must be a dict (object), not array or primitive
        if not isinstance(parsed, dict):
            return ToolCallResult(
                valid=False,
                errors=[
                    f"Expected JSON object, got {type(parsed).__name__}: {json.dumps(parsed)[:200]}"
                ],
                parsed=parsed if isinstance(parsed, dict) else None,
                raw_response=raw,
            )

        return ToolCallResult(valid=True, parsed=parsed, raw_response=raw)

    # ------------------------------------------------------------------
    # Schema Validation
    # ------------------------------------------------------------------

    def validate_against_schema(
        self,
        parsed: dict[str, Any],
        schema: dict[str, Any],
    ) -> ToolCallResult:
        """Validate a parsed JSON object against a tool schema.

        Checks:
        - Required field presence
        - Type constraints (string, integer, boolean, object)
        - Enum value constraints
        - String length bounds
        - Numeric range bounds
        - Nested object validation

        Args:
            parsed: The parsed JSON object.
            schema: The tool schema dict with 'properties' and 'required'.

        Returns:
            ToolCallResult with validation result.

        """
        errors: list[str] = []

        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # Check required fields
        for f_name in required_fields:
            if f_name not in parsed or parsed[f_name] is None:
                errors.append(f"Missing required field: '{f_name}'")

        # Check field types and constraints
        for field, value in parsed.items():
            if field not in properties:
                continue  # Extra fields allowed (permissive)

            prop = properties[field]
            expected_type: str | None = prop.get("type")

            # Type check
            if expected_type and value is not None:
                type_error = self._check_type(field, value, expected_type)
                if type_error:
                    errors.append(type_error)
                    continue  # No further checks if type is wrong

            # String checks
            if isinstance(value, str):
                min_len = prop.get("min_length")
                if min_len is not None and len(value) < min_len:
                    errors.append(f"'{field}' too short: {len(value)} chars, minimum {min_len}")
                max_len = prop.get("max_length")
                if max_len is not None and len(value) > max_len:
                    errors.append(f"'{field}' too long: {len(value)} chars, maximum {max_len}")

            # Numeric checks
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                minimum = prop.get("minimum")
                if minimum is not None and value < minimum:
                    errors.append(f"'{field}' below minimum: {value} < {minimum}")
                maximum = prop.get("maximum")
                if maximum is not None and value > maximum:
                    errors.append(f"'{field}' above maximum: {value} > {maximum}")

            # Enum check
            enum_values: list[Any] | None = prop.get("enum")
            if enum_values is not None and value not in enum_values:
                errors.append(
                    f"'{field}' invalid value '{value}'. "
                    f"Must be one of: {', '.join(str(e) for e in enum_values)}"
                )

            # Object check (nested)
            if expected_type == "object" and isinstance(value, dict):
                prop.get("properties", {})
                nested_required = prop.get("required", [])
                for nf in nested_required:
                    if nf not in value:
                        errors.append(f"Missing nested required field: '{field}.{nf}'")

        if errors:
            return ToolCallResult(valid=False, errors=errors, parsed=parsed)

        return ToolCallResult(valid=True, parsed=parsed)

    # ------------------------------------------------------------------
    # Combined Tool Call Validation
    # ------------------------------------------------------------------

    def validate_tool_call(
        self,
        response: str,
        tool_name: str,
        schema_override: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        """Validate a complete tool call response.

        Steps:
        1. Parse JSON from response
        2. Validate against tool schema

        Args:
            response: Raw LLM response text.
            tool_name: Expected tool name for schema lookup.
            schema_override: Optional schema to use instead of registered one.

        Returns:
            ToolCallResult with full validation result.

        """
        # Step 1: Parse JSON
        json_result = self.validate_json(response)
        if not json_result.valid:
            return json_result

        assert json_result.parsed is not None

        # Step 2: Schema validation
        schema = schema_override or self._schemas.get(tool_name)
        if schema is None:
            # No schema registered — pass with warning
            return ToolCallResult(
                valid=True,
                parsed=json_result.parsed,
                warnings=[f"No schema registered for tool '{tool_name}'"],
                raw_response=response,
            )

        schema_result = self.validate_against_schema(json_result.parsed, schema)
        schema_result.raw_response = response
        schema_result.attempts = 1
        return schema_result

    # ------------------------------------------------------------------
    # Retry Loop
    # ------------------------------------------------------------------

    def validate_and_retry(
        self,
        tool_name: str,
        llm_call: Callable[[str | None], str],
        max_retries: int = 3,
        schema_override: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        """Call the LLM and retry with escalating feedback on validation failure.

        Escalation levels:
        1. Error message (first retry): "Invalid output: {error}"
        2. Schema reminder (second retry): "Expected schema: {schema}"
        3. Example (third retry): "{error} — Use this format: {example}"

        Args:
            tool_name: Name of the tool being called.
            llm_call: Function that calls the LLM and returns raw response.
                Takes optional feedback string for the model.
            max_retries: Maximum number of retries (default: 3).
            schema_override: Optional schema override.

        Returns:
            ToolCallResult from the final attempt.

        """
        schema = schema_override or self._schemas.get(tool_name)
        last_result = ToolCallResult(valid=False, errors=["No attempts made"])

        total_allowed = max(1, max_retries)

        for attempt in range(1, total_allowed + 1):
            # Prepare feedback (None for first attempt)
            feedback: str | None = None
            if attempt > 1:
                feedback = self._build_feedback(tool_name, last_result, schema, attempt)

            # Call LLM
            try:
                raw_response = llm_call(feedback)
            except Exception as e:
                logger.warning("LLM call failed on attempt %d: %s", attempt, e)
                last_result = ToolCallResult(
                    valid=False,
                    errors=[f"LLM call error: {e}"],
                    attempts=attempt,
                    raw_response="",
                )
                continue

            # Validate
            result = self.validate_tool_call(raw_response, tool_name, schema)
            result.attempts = attempt
            result.raw_response = raw_response

            if result.valid:
                return result

            last_result = result

        return last_result

    def _build_feedback(
        self,
        tool_name: str,
        last_result: ToolCallResult,
        schema: dict[str, Any] | None,
        attempt: int,
    ) -> str:
        """Build escalating feedback message for retry.

        Level 1 (retry 1): Error summary only.
        Level 2 (retry 2): Error + schema reminder.
        Level 3 (retry 3+): Error + schema + example.
        """
        error_text = "; ".join(last_result.errors[:3])

        if attempt == 2:
            # Level 1: error message
            return (
                f"Your previous response was invalid: {error_text}\n"
                "Please correct your response and try again."
            )

        if attempt == 3:
            # Level 2: error + schema reminder
            schema_text = self._schema_to_text(schema) if schema else ""
            return (
                f"Your previous response was invalid: {error_text}\n\n"
                f"The expected output must match this schema:\n{schema_text}\n\n"
                "Please format your response correctly."
            )

        # Level 3: error + example
        schema_text = self._schema_to_text(schema) if schema else ""
        example = self._generate_example(tool_name, schema)
        return (
            f"Your previous response was invalid: {error_text}\n\n"
            f"Expected schema:\n{schema_text}\n\n"
            f"Example correct response:\n{example}\n\n"
            "Generate your output in exactly this format."
        )

    def _schema_to_text(self, schema: dict[str, Any]) -> str:
        """Convert schema to human-readable text."""
        props = schema.get("properties", {})
        required = schema.get("required", [])
        lines = ["{"]
        for f_name, prop in props.items():
            req_mark = " (required)" if f_name in required else ""
            ptype = prop.get("type", "any")
            enum_vals = prop.get("enum")
            if enum_vals:
                ptype = f"enum: {', '.join(str(e) for e in enum_vals)}"
            lines.append(f'  "{field}": {ptype}{req_mark}')
        lines.append("}")
        return "\n".join(lines)

    def _generate_example(self, tool_name: str, schema: dict[str, Any] | None) -> str:
        """Generate an example tool call for the schema."""
        if not schema:
            return f'{{"tool_name": "{tool_name}"}}'

        props = schema.get("properties", {})
        example: dict[str, Any] = {}
        for f_name, prop in props.items():
            ptype = prop.get("type", "string")
            enum_vals = prop.get("enum")
            if enum_vals:
                example[field] = enum_vals[0]
            elif ptype == "string":
                example[field] = f"<{field}>"
            elif ptype == "integer":
                example[field] = 0
            elif ptype == "number":
                example[field] = 0.0
            elif ptype == "boolean":
                example[field] = True
            elif ptype == "array":
                example[field] = []
            elif ptype == "object":
                example[field] = {}
            else:
                example[field] = f"<{field}>"

        return json.dumps(example, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_type(field: str, value: Any, expected_type: str) -> str | None:
        """Check if value matches expected type.

        Returns error string or None if valid.
        """
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        py_type = type_map.get(expected_type)
        if py_type is None:
            return None  # Unknown type

        if isinstance(value, py_type):
            return None

        # bool is subclass of int in Python
        if expected_type == "integer" and isinstance(value, bool):
            return f"'{field}' is boolean, expected integer"
        if expected_type == "boolean" and isinstance(value, int):
            return None  # Accept 0/1 as booleans

        actual = type(value).__name__
        return f"'{field}' wrong type: expected {expected_type}, got {actual}"
