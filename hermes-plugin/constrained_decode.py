"""Phase 4 — Tier 2 Constrained Decoding Engine.

Provides XGrammar GBNF grammar compilation from JSON Schema, backend type
detection, constrained decoding integration, and hybrid Tier 2 → Tier 1
fallback validation.

This is a bounded implementation: GBNF grammar generation is done in pure
Python. Actual XGrammar C++ library integration is TBD.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class GrammarCompilationError(Exception):
    """Raised when a schema cannot be compiled to a GBNF grammar."""


# ---------------------------------------------------------------------------
# XGrammarCompiler — JSON Schema → GBNF grammar strings
# ---------------------------------------------------------------------------


class XGrammarCompiler:
    """Compiles JSON Schema into XGrammar GBNF grammar strings.

    Uses pure-Python grammar generation that mirrors the grammar structure
    XGrammar produces from ``xgrammar.Grammar.from_json_schema()``.

    Compiled grammars are cached by schema content fingerprint.
    """

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def compile(self, schema: dict[str, Any]) -> str:
        """Compile a JSON Schema dict to a GBNF grammar string.

        Args:
            schema: A JSON Schema dict (must have ``type: "object"``).

        Returns:
            A GBNF grammar string usable with XGrammar's grammar-guided
            generation or compatible backends (vLLM, llama.cpp).

        Raises:
            GrammarCompilationError: If the schema is empty or cannot be
                compiled.

        """
        if not schema:
            msg = "Cannot compile empty schema"
            raise GrammarCompilationError(msg)

        fingerprint = self._fingerprint(schema)
        cached = self._cache.get(fingerprint)
        if cached is not None:
            return cached

        grammar = self._schema_to_gbnf(schema)
        self._cache[fingerprint] = grammar
        return grammar

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _fingerprint(schema: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()

    def _schema_to_gbnf(self, schema: dict[str, Any]) -> str:
        schema_type = schema.get("type", "object")
        if schema_type != "object":
            msg = f"Cannot compile non-object schema type: {schema_type}"
            raise GrammarCompilationError(msg)

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Build property-level rules
        rules: list[str] = []
        for prop_name, prop_schema in properties.items():
            rule_name = self._sanitise_rule_name(prop_name)
            type_rule = self._type_to_gbnf_rule(prop_name, prop_schema, rules)
            rules.append(f'{rule_name} ::= "\\"{prop_name}\\"" ws ":" ws {type_rule}')

        # Object root (required properties first, then optional alternation)
        prop_order = list(properties.keys())
        req_rules = [self._sanitise_rule_name(p) for p in prop_order if p in required]
        opt_rules = [self._sanitise_rule_name(p) for p in prop_order if p not in required]

        if req_rules and not opt_rules:
            pairs = ' "," ws '.join(req_rules)
            root = f'root ::= "{{" ws {pairs} ws "}}" ws'
        elif req_rules and opt_rules:
            req_seq = ' "," ws '.join(req_rules)
            opt_alt = " | ".join(opt_rules)
            root = f'root ::= "{{" ws {req_seq} ("," ws ({opt_alt}))* ws "}}" ws'
        elif not req_rules and opt_rules:
            opt_alt = " | ".join(opt_rules)
            root = f'root ::= "{{" ws ({opt_alt}) ("," ws ({opt_alt}))* ws "}}" ws'
        else:
            root = 'root ::= "{" ws "}" ws'

        # Base GBNF type rules  (standard XGrammar-compatible definitions)
        base = self._base_type_rules()

        return "\n".join([*rules, root, base])

    @staticmethod
    def _sanitise_rule_name(name: str) -> str:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        if not safe or safe[0].isdigit():
            safe = "p_" + safe
        return safe

    def _type_to_gbnf_rule(
        self,
        prop_name: str,
        prop_schema: dict[str, Any],
        rules: list[str],
    ) -> str:
        """Map a JSON Schema property to a GBNF rule reference or inline."""
        if "enum" in prop_schema:
            values = [json.dumps(v) for v in prop_schema["enum"]]
            return "ws ".join(values) + " ws"

        prop_type = prop_schema.get("type", "string")

        if prop_type == "object":
            # Inline a nested object rule
            nested_name = self._sanitise_rule_name(prop_name) + "_nested"
            nested_props = prop_schema.get("properties", {})
            nested_required = set(prop_schema.get("required", []))

            sub_rules: list[str] = []
            for sub_name, sub_schema in nested_props.items():
                sub_rule = self._sanitise_rule_name(sub_name)
                sub_type = self._type_to_gbnf_rule(sub_name, sub_schema, rules)
                sub_rules.append(f'{sub_rule} ::= "\\"{sub_name}\\"" ws ":" ws {sub_type}')

            sub_order = list(nested_props.keys())
            sub_req = [self._sanitise_rule_name(p) for p in sub_order if p in nested_required]
            sub_opt = [self._sanitise_rule_name(p) for p in sub_order if p not in nested_required]

            if sub_req:
                seq = ' "," ws '.join(sub_req)
                if sub_opt:
                    alt = " | ".join(sub_opt)
                    nested_rule = f'{nested_name} ::= "{{" ws {seq} ("," ws ({alt}))* ws "}}" ws'
                else:
                    nested_rule = f'{nested_name} ::= "{{" ws {seq} ws "}}" ws'
            elif sub_opt:
                alt = " | ".join(sub_opt)
                nested_rule = f'{nested_name} ::= "{{" ({alt}) ("," ws ({alt}))* ws "}}" ws'
            else:
                nested_rule = f'{nested_name} ::= "{{" ws "}}" ws'

            rules.extend(sub_rules)
            rules.append(nested_rule)
            return nested_name

        if prop_type == "array":
            items = prop_schema.get("items", {})
            if items:
                items.get("type", "value")
                # Register a wrapper rule for typed arrays
                array_rule_name = self._sanitise_rule_name(prop_name) + "_items"
                typed = self._type_to_gbnf_rule(prop_name + "_elem", items, rules)
                array_rule = f'{array_rule_name} ::= "[" ws ({typed}) ("," ws ({typed}))* ws "]" ws'
                rules.append(array_rule)
                return array_rule_name
            return "array"

        return prop_type  # string, integer, number, boolean — all have base rules

    @staticmethod
    def _base_type_rules() -> str:
        """Standard GBNF type definitions (JSON-compatible)."""
        return """\
string ::= "\\"" characters "\\"" ws
characters ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
integer ::= "0" | [1-9] [0-9]* ws
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)? ws
boolean ::= "true" | "false" ws
null ::= "null" ws
array ::= "[" ws (value ("," ws value)*)? ws "]" ws
value ::= object | array | string | integer | number | boolean | null
object ::= "{" ws (member ("," ws member)*)? ws "}" ws
member ::= string ws ":" ws value
ws ::= [ \\t\\n]*"""


# ---------------------------------------------------------------------------
# BackendDetector — identify inference backend from provider config
# ---------------------------------------------------------------------------


class BackendDetector:
    """Detects the LLM inference backend type from provider configuration.

    Only vLLM and llama.cpp backends support XGrammar grammar constraints.
    LM Studio and OpenAI-compatible APIs use Tier 1 post-hoc validation.
    """

    GRAMMAR_BACKENDS = frozenset({"vllm", "llamacpp"})

    @staticmethod
    def detect(config: dict[str, Any]) -> str | None:
        """Detect the backend type from a provider config dict.

        Args:
            config: Dict with keys like ``provider``, ``base_url``, ``model``.

        Returns:
            One of ``"vllm"``, ``"llamacpp"``, ``"lm_studio"``, or ``None``
            if the backend cannot be identified.

        """
        provider = (config.get("provider") or "").lower()

        # Explicit provider names
        if "vllm" in provider:
            return "vllm"
        if "llamacpp" in provider or "llama.cpp" in provider:
            return "llamacpp"
        if "lm_studio" in provider or "lmstudio" in provider:
            return "lm_studio"

        # Detect by URL pattern for OpenAI-compatible endpoints
        base_url = (config.get("base_url") or "").lower()
        if "localhost:1234" in base_url or "127.0.0.1:1234" in base_url:
            return "lm_studio"

        # Generic OpenAI-compatible (no specific backend)
        if "opencode" in provider or "openai" in provider:
            return None

        return None

    @staticmethod
    def supports_grammar(backend_type: str | None) -> bool:
        """Whether the backend type supports XGrammar grammar constraints.

        Only vLLM and llama.cpp support grammar-guided generation.
        """
        if backend_type is None:
            return False
        return backend_type in BackendDetector.GRAMMAR_BACKENDS


# ---------------------------------------------------------------------------
# ConstrainedDecoder — grammar-constrained LLM generation
# ---------------------------------------------------------------------------


class ConstrainedDecoder:
    """Wrapper that applies XGrammar grammar constraints during generation.

    When the backend supports grammar constraints (vLLM, llama.cpp), the
    decoder passes the GBNF grammar to the backend API. For backends without
    grammar support, it falls through to direct passthrough.
    """

    def __init__(
        self,
        backend_type: str,
        grammar: str | None = None,
    ) -> None:
        self.backend_type = backend_type
        self._grammar = grammar
        self.available = BackendDetector.supports_grammar(backend_type)

    @property
    def available(self) -> bool:
        return self._available

    @available.setter
    def available(self, value: bool) -> None:
        self._available = value

    def generate(self, prompt: str, llm_call: Callable[[str], str]) -> Any:
        """Generate a constrained output for the given prompt.

        If the decoder is *available* (grammar-supporting backend) and a
        grammar was provided, it invokes the LLM with the grammar constraint
        and returns the **parsed** result (a dict/list/primitive).

        If unavailable (no grammar support), it passes through the raw string
        response unchanged — Tier 1 post-hoc validation handles it.

        Args:
            prompt: The prompt to send to the LLM.
            llm_call: A callable that takes the prompt (plus optional grammar
                injection) and returns the raw response string.

        Returns:
            Parsed Python object (when grammar is active) or raw string
            (passthrough fallback).

        """
        if self.available and self._grammar:
            # In production, the grammar would be injected into the backend
            # API call (e.g., ``grammar=`` param in vLLM / llama.cpp).
            # For this bounded implementation, we call the LLM and then
            # attempt to parse the result.
            raw = llm_call(prompt)
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw

        # Passthrough: no grammar support
        return llm_call(prompt)


# ---------------------------------------------------------------------------
# HybridValidator — Tier 2 constrained decode with Tier 1 fallback
# ---------------------------------------------------------------------------


@dataclass
class ToolCallResult:
    """Result of a tool call validation attempt."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    attempts: int = 1
    raw_response: str = ""
    parsed: dict[str, Any] | None = None


class HybridValidator:
    """Combines Tier 2 (constrained decode) and Tier 1 (post-hoc) validation.

    When the backend supports grammar constraints, Tier 2 is attempted first.
    If the constrained output fails post-hoc validation, it falls back to
    Tier 1's retry-with-feedback loop.
    """

    def __init__(
        self,
        backend_type: str,
        schema: dict[str, Any] | None = None,
        output_validator: Any | None = None,
    ) -> None:
        self.backend_type = backend_type
        self.schema = schema
        self.output_validator = output_validator
        self.tier2_available = BackendDetector.supports_grammar(backend_type)

        self._compiler: XGrammarCompiler | None = None
        self._grammar: str | None = None

        if self.tier2_available and schema:
            self._compiler = XGrammarCompiler()
            try:
                self._grammar = self._compiler.compile(schema)
            except GrammarCompilationError:
                self.tier2_available = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        schema: dict[str, Any] | None = None,
        output_validator: Any | None = None,
    ) -> HybridValidator:
        """Create a HybridValidator from a provider config dict.

        Automatically detects the backend type using
        :class:`BackendDetector`.
        """
        detector = BackendDetector()
        backend = detector.detect(config)
        return cls(
            backend_type=backend or "unknown",
            schema=schema,
            output_validator=output_validator,
        )

    def validate(
        self,
        prompt: str,
        llm_call: Callable[[str | None], str],
    ) -> ToolCallResult:
        """Validate an LLM response through Tier 2 → Tier 1 fallback.

        Args:
            prompt: The prompt to send (for Tier 2 decode).
            llm_call: Callable; takes optional feedback string, returns raw
                response text.

        Returns:
            A ToolCallResult with validation status.

        """
        # Tier 2: constrained decode
        if self.tier2_available and self._grammar:
            decoder = ConstrainedDecoder(
                backend_type=self.backend_type,
                grammar=self._grammar,
            )
            result = decoder.generate(prompt, llm_call)

            if isinstance(result, dict) and self.schema:
                # Validate against schema
                if self.output_validator is not None:
                    vr = self.output_validator.validate_against_schema(result, self.schema)
                    if vr.valid:
                        return ToolCallResult(
                            valid=True,
                            attempts=1,
                            parsed=result,
                        )

                # Simple schema validation fallback
                required = set(self.schema.get("required", []))
                self.schema.get("properties", {})
                missing = [p for p in required if p not in result]
                if not missing:
                    return ToolCallResult(
                        valid=True,
                        attempts=1,
                        parsed=result,
                    )

            # Tier 2 produced invalid output — fall back to Tier 1
            return self._tier1_fallback(prompt, llm_call)

        # No Tier 2 available — go straight to Tier 1
        return self._tier1_fallback(prompt, llm_call)

    def _tier1_fallback(
        self,
        prompt: str,
        llm_call: Callable[[str | None], str],
    ) -> ToolCallResult:
        """Fallback to post-hoc validation with retry loop."""
        if self.output_validator is not None:
            return self.output_validator.validate_and_retry(
                tool_name="hybrid",
                llm_call=llm_call,
                max_retries=3,
                schema_override=self.schema,
            )

        # No output validator — just try once and report
        raw = llm_call(prompt)
        try:
            parsed = json.loads(raw)
            if self.schema:
                required = set(self.schema.get("required", []))
                if all(p in parsed for p in required):
                    return ToolCallResult(valid=True, attempts=1, parsed=parsed)
            else:
                return ToolCallResult(valid=True, attempts=1, parsed=parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        return ToolCallResult(valid=False, errors=["No valid output after fallback"], attempts=1)
