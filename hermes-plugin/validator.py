"""Schema Validator — pre-call validation for tool arguments.

Validates tool call arguments against expected schemas before the
call is dispatched to the model. Catches format drift early —
before it cascades into multi-turn failures.

Based on the finding that Qwen3-32B achieves 87% tool-selection
accuracy vs GPT-4o's 92% (Ganglani, Jun 2026). The 5% gap is
primarily malformed structured output — which this validator catches.

Supports:
  - Required field presence
  - Type constraints (int, float, str, bool, list, dict)
  - String length bounds
  - Numeric range bounds
  - Enum value constraints
  - Pattern/regex validation for string fields
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a tool call schema validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Built-in tool schemas (Hermes-aware)
# ---------------------------------------------------------------------------

# Minimal schemas for common Hermes tools that small models commonly
# malform. Drawn from the actual Hermes tool API — these catch the
# most common format drift failure modes.
TOOL_SCHEMAS: dict[str, dict] = {
    "terminal": {
        "required": ["command"],
        "properties": {
            "command": {
                "type": "string",
                "min_length": 1,
                "max_length": 32000,
            },
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "maximum": 600,
            },
            "workdir": {
                "type": "string",
                "max_length": 4096,
            },
            "background": {
                "type": "boolean",
            },
            "pty": {
                "type": "boolean",
            },
        },
    },
    "write_file": {
        "required": ["path", "content"],
        "properties": {
            "path": {
                "type": "string",
                "min_length": 1,
                "max_length": 4096,
                "pattern": r"^[~/]|\w:",
            },
            "content": {
                "type": "string",
                "max_length": 10000000,
            },
        },
    },
    "patch": {
        "required": ["mode", "old_string", "new_string"],
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["replace", "patch"],
            },
            "path": {
                "type": "string",
                "min_length": 1,
            },
            "old_string": {
                "type": "string",
                "min_length": 1,
            },
            "new_string": {
                "type": "string",
            },
        },
    },
    "read_file": {
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "min_length": 1,
                "max_length": 4096,
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
            },
        },
    },
    "search_files": {
        "required": ["pattern"],
        "properties": {
            "pattern": {
                "type": "string",
                "min_length": 1,
                "max_length": 4096,
            },
            "target": {
                "type": "string",
                "enum": ["content", "files"],
            },
            "path": {
                "type": "string",
                "max_length": 4096,
            },
        },
    },
    "web_search": {
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "min_length": 1,
                "max_length": 500,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
            },
        },
    },
    "web_extract": {
        "required": ["urls"],
        "properties": {
            "urls": {
                "type": "array",
                "min_items": 1,
                "max_items": 5,
            },
            "char_limit": {
                "type": "integer",
                "minimum": 2000,
            },
        },
    },
    "memory": {
        "required": ["action", "target"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
            },
            "content": {
                "type": "string",
                "max_length": 1500,
            },
        },
    },
    "skill_view": {
        "required": ["name"],
        "properties": {
            "name": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Pre-call schema validator for tool arguments.

    Validates tool call arguments against known schemas. Catches:
    - Missing required fields
    - Wrong types (string instead of integer, etc.)
    - Out-of-range values
    - Invalid enum choices
    - Regex pattern violations
    - Length/constraint violations
    """

    def __init__(self):
        self._schemas = dict(TOOL_SCHEMAS)

    def register_schema(self, tool_name: str, schema: dict) -> None:
        """Register or override a tool schema."""
        self._schemas[tool_name] = schema
        logger.debug("Registered schema for %s", tool_name)

    def validate(self, tool_name: str, args: dict | None) -> ValidationResult:
        """Validate tool call arguments against known schema.

        Args:
            tool_name: The name of the tool being called.
            args: The arguments dict from the model.

        Returns:
            ValidationResult with valid flag and errors/warnings.

        """
        result = ValidationResult(valid=True)

        if args is None:
            args = {}

        schema = self._schemas.get(tool_name)
        if schema is None:
            # No schema registered — pass through with warning
            result.warnings.append(f"No schema registered for '{tool_name}'")
            return result

        # 1. Required fields
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in args or args[field_name] is None:
                result.errors.append(f"Missing required field '{field_name}' for '{tool_name}'")

        # 2. Property validation
        properties = schema.get("properties", {})
        for field_name, field_value in args.items():
            if field_value is None:
                continue  # null is handled above for required fields

            prop_schema = properties.get(field_name)
            if prop_schema is None:
                continue  # No schema for this field

            error = self._validate_field(field_name, field_value, prop_schema)
            if error:
                result.errors.append(error)

        if result.errors:
            result.valid = False

        return result

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        schema: dict,
    ) -> str | None:
        """Validate a single field against its schema. Returns error or None."""
        expected_type = schema.get("type")
        if expected_type:
            type_error = self._check_type(field_name, value, expected_type)
            if type_error:
                return type_error

        # String-specific validations
        if isinstance(value, str):
            min_length = schema.get("min_length")
            if min_length is not None and len(value) < min_length:
                return f"'{field_name}' too short: {len(value)} < {min_length}"

            max_length = schema.get("max_length")
            if max_length is not None and len(value) > max_length:
                return f"'{field_name}' too long: {len(value)} > {max_length}"

            pattern = schema.get("pattern")
            if pattern and not re.match(pattern, value):
                return f"'{field_name}' does not match pattern: {pattern}"

        # Numeric-specific validations
        if isinstance(value, (int, float)):
            minimum = schema.get("minimum")
            if minimum is not None and value < minimum:
                return f"'{field_name}' below minimum: {value} < {minimum}"

            maximum = schema.get("maximum")
            if maximum is not None and value > maximum:
                return f"'{field_name}' above maximum: {value} > {maximum}"

        # Enum validation
        enum_values = schema.get("enum")
        if enum_values is not None and value not in enum_values:
            return (
                f"'{field_name}' invalid value '{value}'. "
                f"Must be one of: {', '.join(str(e) for e in enum_values)}"
            )

        # Array-specific validations
        if isinstance(value, list):
            min_items = schema.get("min_items")
            if min_items is not None and len(value) < min_items:
                return f"'{field_name}' has too few items: {len(value)} < {min_items}"
            max_items = schema.get("max_items")
            if max_items is not None and len(value) > max_items:
                return f"'{field_name}' has too many items: {len(value)} > {max_items}"

        return None

    @staticmethod
    def _check_type(
        field_name: str,
        value: Any,
        expected_type: str,
    ) -> str | None:
        """Check if value has the expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        py_type = type_map.get(expected_type)
        if py_type is None:
            return None  # Unknown type — skip check

        if isinstance(value, py_type):
            return None

        # Special case: bool is a subclass of int in Python
        if expected_type == "integer" and isinstance(value, bool):
            return f"'{field_name}' has wrong type: expected integer, got boolean"

        # Special case: can't convert int to bool meaningfully
        if expected_type == "boolean" and isinstance(value, int):
            return None  # Accept 0/1 for boolean

        actual = type(value).__name__
        return f"'{field_name}' wrong type: expected {expected_type}, got {actual}"
