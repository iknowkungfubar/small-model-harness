# Strategic Implementation Plan

> Plugin + MCP server for Hermes Agent to bridge the gap between small local models (1B-12B)
> and frontier-level performance. Five-phase build over 5 weeks.

---

## Phase 1: Plugin Foundation — Pre-Call Validation + Circuit Breaker (Week 1)

### Deliverable
Working Hermes plugin with `pre_tool_call` and `post_tool_call` hooks implementing Layers 2 and 4.

### Files
```
~/.hermes/profiles/dev/plugins/small-model-harness/
├── plugin.yaml           # Plugin manifest, hook registration
├── plugin.py             # Main plugin entry point
├── validator.py          # Schema validation + loop detection
├── circuit_breaker.py    # 3-state circuit breaker
├── loop_detector.py      # N-gram overlap + tool diversity scanner
└── __init__.py
```

### Implementation Steps

#### Step 1.1: Plugin skeleton (Day 1)
- Create `plugin.yaml` with `pre_tool_call` and `post_tool_call` hooks
- Create `plugin.py` with async hook handlers
- Register with `hermes plugins enable small-model-harness`
- Verify it loads: `hermes session start` → check for load message

```python
# plugin.py
from typing import Any

async def pre_tool_call(tool_name: str, args: dict, context: dict) -> dict:
    """Validate tool call before execution."""
    # Will implement: schema validation, loop detection, budget check, circuit breaker
    return {"action": "allow"}  # pass-through initially

async def post_tool_call(tool_name: str, args: dict, result: Any, context: dict) -> None:
    """Track tool call for loop detection and budget."""
    pass
```

#### Step 1.2: Schema validator (Day 1-2)
- Parse Hermes tool schemas from the runtime context
- Validate tool call arguments against JSON Schema
- Return validation errors with specific field-level info
- Integration: read schemas from `context["tool_schemas"]` if available, or MCP tool list

**Key methods:**
```python
def validate_schema(tool_name: str, args: dict, schemas: dict) -> ValidationResult:
    """Validate args against the tool's JSON Schema."""
    
def validate_required_fields(args: dict, schema: dict) -> list[str]:
    """Check all required fields are present."""
    
def validate_field_types(args: dict, schema: dict) -> list[str]:
    """Check type constraints on provided fields."""
```

#### Step 1.3: Loop detector (Day 2-3)
- Track last N tool calls per session
- Implement 4-signal ensemble:
  1. N-gram overlap (4-gram, sliding window of 5)
  2. Tool call diversity ratio
  3. Latency stability (coefficient of variation)
  4. Content stagnation (output similarity)
- Weighted scoring → single confidence score 0.0-1.0

```python
class LoopDetector:
    def __init__(self, window: int = 8):
        self.window = window
    
    def score(self, recent_calls: list[CallRecord]) -> float:
        """Weighted ensemble scoring. Returns 0.0-1.0."""
    
    def detect_pattern(self, recent_calls: list[CallRecord]) -> Optional[LoopPattern]:
        """Classify loop type if detected."""
```

**Loop pattern classification:**
| Pattern | Signature | Action |
|---------|-----------|--------|
| Tool slam | Same tool + same args × N | Block + escalate tier |
| Token grind | Output n-gram overlap >0.9 | Block + change temperature |
| Stuck retry | Same error returned × N | Block + retry with diff approach |
| Hallucination loop | Model generates plausible but wrong output repeatedly | Route to T3+ for verification |

#### Step 1.4: Circuit breaker (Day 3-4)
- 3-state machine (closed/open/half-open)
- Per-session state tracking
- Cooling timer
- Escalation on break count threshold

```python
class CircuitBreaker:
    states = ("closed", "open", "half_open")
    
    def check(self, session_id: str, loop_score: float) -> CircuitDecision:
        """Check if call should proceed."""
    
    def register_success(self, session_id: str):
        """Register successful completion."""
    
    def register_break(self, session_id: str) -> CircuitDecision:
        """Register circuit break and determine escalation."""
```

#### Step 1.5: Hook wiring (Day 5)
- Wire all components into `pre_tool_call` handler
- Test with real small models (Qwen3-4B/8B)
- Measure overhead: target <10ms per validation

### Success Criteria Phase 1
- [ ] Plugin loads without errors
- [ ] Schema validation catches incorrect tool args
- [ ] Loop detector identifies repetitive patterns with >80% accuracy
- [ ] Circuit breaker blocks >3 consecutive identical calls
- [ ] <10ms overhead per tool call

---

## Phase 2: Context Management (Week 2)

### Deliverable
Layer 5 (Context Budget Manager) added to plugin + MCP server with budget tracking and compaction.

### Files
```
~/.hermes/profiles/dev/plugins/small-model-harness/
├── context_budget.py     # Token tracking + budget calculation

~/.hermes/profiles/dev/mcp/small-model-harness/  # NEW
├── server.py             # MCP server entry point
├── context_commands.py   # harnexx_context_status, harnexx_compact
└── config.py             # Shared config
```

### Implementation Steps

#### Step 2.1: Token tracker (Day 1-2)
- Track per-session: total_tokens, active_tokens, step_count
- Calculate effective capacity (stated_window / 3)
- Monitor compaction thresholds (90%)
- Store in SQLite for persistence across HHTP sessions

```python
class ContextBudget:
    def __init__(self, stated_window: int):
        self.stated_window = stated_window
        self.effective_capacity = stated_window // 3
        self.used_tokens = 0
        self.step_count = 0
        self.compaction_count = 0
    
    @property
    def utilization(self) -> float: ...
    
    @property
    def needs_compaction(self) -> bool: ...
```

#### Step 2.2: MCP server skeleton (Day 2-3)
- Create MCP server at `~/.hermes/profiles/dev/mcp/small-model-harness/server.py`
- Register with Hermes: `hermes mcp add small-model-harness`
- Implement `harness_context_status` tool
- Implement `harness_compact` tool
- Test: `hermes tool call harness_context_status`

#### Step 2.3: Compaction engine (Day 3-4)
- Sliding window: keep last N complete steps
- Summarize older steps into bullet-point format
- Structured summary: participant, action, result, key_data
- Verify summary quality against original

```python
def compact(session: ContextBudget, steps: list[Step]) -> CompactionResult:
    """Compact old steps into summary."""
    
def summarize_steps(steps: list[Step]) -> str:
    """Generate structured bullet-point summary."""
```

#### Step 2.4: Hook integration (Day 5)
- `pre_tool_call`: check budget → block/allow
- `post_tool_call`: update token counts
- `pre_verify`: check if compaction needed
- Wire all components

### Success Criteria Phase 2
- [ ] MCP server tools respond correctly
- [ ] Context budget correctly tracks tokens
- [ ] Compaction triggers at 90% threshold
- [ ] Sliding window keeps correct steps
- [ ] Summary preserves critical information

---

## Phase 3: Model Router (Week 3)

### Deliverable
MCP server tools for Layer 1 (task classification + model routing) with cascade.

### Files
```
~/.hermes/profiles/dev/mcp/small-model-harness/
├── router.py             # Task classifier + model router + cascade logic
├── tier_config.py        # Model tier definitions
└── health.py             # Model health tracking
```

### Implementation Steps

#### Step 3.1: Task classifier (Day 1-2)
- Rule-based lightwX weight classifier (no LLM call needed for routing)
- Heuristics: keyword matching, tool count, step estimate
- Output: TaskProfile(tier, complexity, reasoning_depth, tool_count)

```python
def classify_task(task: str, tools: list[str]) -> TaskProfile:
    """Classify task without LLM call."""
    score = 0
    # Complexity heuristics
    if any(kw in task.lower() for kw in ["debug", "why", "diagnose"]):
        score += 2
    if any(kw in task.lower() for kw in ["plan", "desigt", "architect"]):
        score += 3
    # Tool count heuristics
    if len(tools) > 5:
        score += 2
    elif len(tools) > 2:
        score += 1
    # Map to tier
    ...
```

#### Step 3.2: Model router (Day 2-3)
- Configurable tier assignemnts
- Override rules (security → T3+, multi-step reasoning → T2+)
- Model availability check
- Historical success rate tracking

```python
def route(profile: TaskProfile, available_models: dict) -> RouteResult:
    """Route task to best model tier."""
    
def check_model_health(model_name: str) -> HealthStatus:
    """Check if model is responding correctly."""
```

#### Step 3.3: Cascade with confidence gate (Day 3-4)
- Cascade flow: T1 → T2 → T3 → T4
- Confidence threshold per tier (configurable, default 0.7)
- Context preservation across cascade
- Rate limiting: min 5s between cascades

```python
def should_cascade(current_tier: str, confidence: float) -> bool:
    """Deteimine if cascade is needed."""
    thresholds = {"t1": 0.7, "t2": 0.7, "t3": 0.6, "t4": 0.0}
    return confidence < thresholds[current_tier]

async def cascade(route_result: RouteResult, task) -> CascadeResult:
    """Execute cascade to next tier."""
```

#### Step 3.4: MCP tools (Day 4-5)
- `harness_classify_task` — classify input text
- `harness_route` — route to model tier
- `harness_cascade` — escalate to next tier
- Register in MCP server

### Success Criteria Phase 3
- [ ] Task classification matches human judgment 80%+ of tests
- [ ] Cascade correctly escalates low-confidence tasks
- [ ] Route respects tier limits (max_tools, max_steps)
- [ ] Down model is detected and bypassed

---

## Phase 4: Constrained Decoding Integration (Week 4)

### Deliverable
XGrammar/Outlines integration for guaranteed structured output.

### Implementation

#### Step 4.1: XGrammar integration for local models (Day 1-3)
- Check if llama.cpp (LM Studio backend) has XGrammar support
- Generate GBNF grammar from tool schemas
- Cache compiled grammars by schema hash
- Measure mask generation overhead (<1ms expected)

```python
def build_tool_grammar(tool_schemas: list[dict]) -> str:
    """Convert tool schemas to GBNF grammar."""
    
def cached_grammar(schema_hash: str, schemas: list[dict]) -> str:
    """Get or create compiled ngrammar."""
```

#### Step 4.2: Post-hoc validation fallback for cloud models (Day 3-4)
- When constrained decoding isn't available (cloud API)
- Validate output format after generation
- Retry with different prompt on format failure
- Max 3 retries before cascade

```python
def validate_output(output: str, expected_schema: dict) -> ValidationResult:
    """Validate output against expected schma."""
    
def reformat_with_retry(output: str, model: str, max_retries: int) -> str:
    """Retry with format correction prompt."""
```

#### Step 4.3: Integration (Day 5)
- Plugin detects backend type (local/cloud)
- Routes to appropriate constraint engine
- Falls back gracefully when constrained decoding unavailable

### Success Criteria Phase 4
- [ ] Local models produce 100% valid structured output
- [ ] Grammar generation completes in <1ms
- [ ] Cloud fallback succeeds within 3 retries 95%+ of time
- [ ] Backward compatible — existing tool calls unchanged

---

## Phase 5: Hardening + Observability (Week 5)

### Deliverable
Metrics, test suite, documentation, edge case handling.

### Implementation

#### Step 5.1: Prometheus metrics
- Export: loop_detections_total, circuit_breaks_total, compactions_total, cascades_total, tokens_saved, latency_distribution
- Labels: model_tier, session_id, tool_name

#### Step 5.2: Test suite
- Unit tests per layer (120+ tests target)
- Integration tests with real small models
- Loop detection accuracy benchmark
- Circuit breaker state machine verification
- Context compaction quality tests

#### Step 5.3: Documentation
- AGENTS.md integration
- README with architecture diagram
- Configuration reference
- Troubleshooting guide

#### Step 5.4: Edge cases
- Cold start (no history → no loop detection)
- Multi-session state (circuit breaker persistence)
- Model crash during operation
- Configuration reload without restart

### Success Criteria Phase 5
- [ ] All phases tested and passing
- [ ] Documentation complete
- [ ] Metrics dashboard functional
- [ ] Graceful degradation on all error paths

---

## Deployment Checklist

Pre-launch verification:

- [ ] Phase 1: Plugin validates, detects loops, breaks circuits
- [ ] Phase 2: Context budget tracks, compacts, enforces limits
- [ ] Phase 3: Router classifies, routes, cascades correctly
- [ ] Phase 4: Output constraint eliminates format errors
- [ ] Phase 5: Metrics, tests, documentation complete
- [ ] Integration: Compatible with existing sdlc-quality-hooks plugin
- [ ] Integration: Compatible with Hermes Forge MCP server
- [ ] Performance: <20ms overhead on tool call total
- [ ] Reliability: No false positives on >100 test cases per layer

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| False positive loop detection | Blocks valid agent work | Conservative thresholds (0.8), half-open recovery, configurable |
| Constrained decoding breaks model output | Schema-conforming but wrong content | Still validates semantic content, cascades on low confidence |
| Context compaction loses information | Degraded performance | Keep last 5 steps intact, verify summaries |
| XGrammar incompatibility with LM Studio version | No constrained decoding | Fallback to post-hoc validation |
| Plugin conflicts with sdlc-quality-hooks | Both hooks execute | Designed to complement, not conflict. Proof: both modify pre_tool_call. |
<jupyter_text><｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="write_file">
<｜｜DSML｜｜parameter name="content" string="true"># Small-Model Harness Plugin Implementation

> Reference implementation for the Hermes plugin component.
> Implements Layers 2, 4, 5 (pre-call validation, circuit breaker, context budget).

## plugin.yaml

```yaml
name: small-model-harness
version: 0.1.0
description: "5-layer defensive harness for small local LLMs (1B-12B)"
author: "TurinTech Solutions"
hooks:
  pre_tool_call: pre_tool_call_handler
  post_tool_call: post_tool_call_handler
  pre_verify: pre_verify_handler
config_schema:
  type: object
  properties:
    harness:
      type: object
      properties:
        enabled:
          type: boolean
          default: true
        routing:
          type: object
          properties:
            default_tier:
              type: string
              enum: [t1, t2, t3, t4]
              default: "t2"
            cascade_enabled:
              type: boolean
              default: true
            confidence_threshold:
              type: number
              default: 0.7
        circuit_breaker:
          type: object
          properties:
            enabled:
              type: boolean
              default: true
            loop_threshold:
              type: number
              default: 0.8
            cooling_period_seconds:
              type: integer
              default: 60
            max_breaks_per_session:
              type: integer
              default: 5
        context:
          type: object
          properties:
            effective_capacity_ratio:
              type: number
              default: 0.33
            compaction_threshold:
              type: number
              default: 0.9
            sliding_window_steps:
              type: integer
              default: 5
        validation:
          type: object
          properties:
            schema_check:
              type: boolean
              default: true
            max_consecutive_same_tool:
              type: integer
              default: 3
```

## plugin.py

```python
"""Small-Model Harness Plugin for Hermes Agent.

Five-layer defensive harness for running 1B-12B parameter models
in production agentic workflows.

Hooks:
  pre_tool_call: Schema validation, loop detection, circuit breaker, budget check
  post_tool_call: History tracking, token accounting
  pre_verify: Context budget coercion
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    tool: str
    args: dict
    output: str
    latency: float
    timestamp: float
    error: Optional[str] = None

@dataclass
class LoopPattern:
    pattern_type: str  # tool_slam, token_grind, stuck_retry, hallucination_loop
    confidence: float
    detail: str

@dataclass
class CircuitDecision:
    allow: bool
    state: str  # closed, open, half_open
    note: str = ""
    escalate: bool = False
    tier_override: Optional[str] = None

@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    escalate: bool = False
    needs_compaction: bool = False

@dataclass
class ContextBudget:
    session_id: str
    stated_window: int
    effective_capacity: int
    used_tokens: int = 0
    step_count: int = 0
    compaction_count: int = 0
    last_compaction_at: Optional[float] = None
    
    @property
    def utilization(self) -> float:
        return self.used_tokens / max(self.effective_capacity, 1)
    
    @property
    def needs_compaction(self) -> bool:
        return self.utilization > 0.9 or self.step_count > 25
    
    @property
    def output_headroom(self) -> int:
        return int(self.effective_capacity * 0.1)

# ---------------------------------------------------------------------------
# Session state (per-session)
# ---------------------------------------------------------------------------

class SessionState:
    """Holds runtime state for one agent session."""
    
    def __init__(self, session_id: str, stated_window: int = 131072):
        self.session_id = session_id
        self.recent_calls: list[CallRecord] = []
        self.budget = ContextBudget(
            session_id=session_id,
            stated_window=stated_window,
            effective_capacity=int(stated_window * 0.33)
        )
        self.break_count = 0
        self.break_times: list[float] = []
        self.breaker_state = "closed"  # closed, open, half_open
        self.last_break_time = 0.0
        self.cooling_period = 60
        self.max_breaks = 5
        self.max_breaks_window = 600  # 10 minutes

# ---------------------------------------------------------------------------
# Schema Validator (Layer 2)
# ---------------------------------------------------------------------------

class SchemaValidator:
    """Validates tool call arguments against their schemas."""
    
    @staticmethod
    def validate(tool_name: str, args: dict, schemas: dict) -> ValidationResult:
        """Validate args against the tool's schema."""
        if tool_name not in schemas:
            return ValidationResult(valid=True)  # unknown tools pass through
        
        schema = schemas[tool_name]
        required = schema.get("required", [])
        
        # Check required fields
        for field in required:
            if field not in args:
                return ValidationResult(
                    valid=False,
                    reason=f"Missing required field '{field}' in tool '{tool_name}'"
                )
        
        # Check field types
        for field, value in args.items():
            expected_type = _resolve_type(schema, field)
            if expected_type and not _type_matches(value, expected_type):
                return ValidationResult(
                    valid=False,
                    reason=f"Field '{field}' expects {expected_type}, got {type(value).__name__}"
                )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_required_fields(args: dict, schema: dict) -> list[str]:
        missing = []
        for field in schema.get("required", []):
            if field not in args:
                missing.append(field)
        return missing
    
    @staticmethod
    def validate_field_types(args: dict, schema: dict) -> list[str]:
        errors = []
        for field, value in args.items():
            expected = _resolve_type(schema, field)
            if expected and not _type_matches(value, expected):
                errors.append(f"{field}: expected {expected}, got {type(value).__name__}")
        return errors


def _resolve_type(schema: dict, field: str) -> Optional[str]:
    """Resolve a field's expected JSON Schema type."""
    properties = schema.get("properties", {})
    if field not in properties:
        return None
    return properties[field].get("type")


def _type_matches(value: Any, expected_type: str) -> bool:
    """Check if a Python value matches a JSON Schema type."""
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": (list, tuple),
        "object": dict,
        "null": type(None),
    }
    py_types = mapping.get(expected_type, None)
    if py_types is None:
        return True  # unknown type passes
    return isinstance(value, py_types)

# ---------------------------------------------------------------------------
# Loop Detector (Layer 4)
# ---------------------------------------------------------------------------

class LoopDetector:
    """Detects repetitive, looping, or stuck patterns in tool calls."""
    
    def __init__(self, window: int = 8):
        self.window = window
    
    def score(self, recent_calls: list[CallRecord]) -> float:
        """Weighted ensemble score. Returns 0.0 (no loop) to 1.0 (definite loop)."""
        if len(recent_calls) < 4:
            return 0.0
        
        scores = []
        
        # Signal 1: Token repetition (40% weight)
        outputs = [c.output[:500] for c in recent_calls[-5:]]
        ngram_score = self._ngram_overlap(outputs)
        scores.append((ngram_score, 0.4))
        
        # Signal 2: Tool call diversity (30% weight)
        recent_tools = [c.tool for c in recent_calls[-self.window:]]
        unique_ratio = len(set(recent_tools)) / max(len(recent_tools), 1)
        scores.append((1.0 - unique_ratio, 0.3))
        
        # Signal 3: Latency stability (15% weight)
        latencies = [c.latency for c in recent_calls[-5:] if c.latency > 0]
        if len(latencies) >= 3:
            import statistics
            cv = statistics.stdev(latencies) / max(statistics.mean(latencies), 0.001)
            stability_score = 1.0 - min(cv, 1.0)
            scores.append((stability_score, 0.15))
        
        # Signal 4: Content stagnation (15% weight)
        if len(recent_calls) >= 4:
            outputs_trimmed = [c.output[:200] for c in recent_calls[-4:]]
            sim_score = self._semantic_similarity(outputs_trimmed)
            scores.append((sim_score, 0.15))
        
        return sum(score * weight for score, weight in scores)
    
    def detect_pattern(self, recent_calls: list[CallRecord]) -> Optional[LoopPattern]:
        """Classify the loop pattern type."""
        if len(recent_calls) < 4:
            return None
        
        last = recent_calls[-1]
        
        # Pattern 1: Tool slam — same tool + same args repeatedly
        same_tool_count = sum(1 for c in recent_calls[-5:] 
                              if c.tool == last.tool and c.args == last.args)
        if same_tool_count >= 3:
            return LoopPattern(
                pattern_type="tool_slam",
                confidence=min(same_tool_count / 5, 1.0),
                detail=f"Same tool '{last.tool}' called {same_tool_count}x with same args"
            )
        
        # Pattern 2: Token grind — output n-gram overlap
        outputs = [c.output[:500] for c in recent_calls[-5:]]
        if self._ngram_overlap(outputs) > 0.9:
            return LoopPattern(
                pattern_type="token_grind",
                confidence=0.9,
                detail="Near-identical output across consecutive calls"
            )
        
        # Pattern 3: Stuck retry — same error returned repeatedly
        if last.error:
            error_count = sum(1 for c in recent_calls[-5:] if c.error == last.error)
            if error_count >= 3:
                return LoopPattern(
                    pattern_type="stuck_retry",
                    confidence=min(error_count / 5, 1.0),
                    detail=f"Same error '{last.error[:100]}' repeated {error_count}x"
                )
        
        return None
    
    def _ngram_overlap(self, texts: list[str], n: int = 4) -> float:
        """Compute n-gram overlap ratio between consecutive texts."""
        if len(texts) < 2:
            return 0.0
        
        def get_ngrams(text: str, n: int) -> set:
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        overlaps = []
        for i in range(1, len(texts)):
            prev = get_ngrams(texts[i-1], n)
            curr = get_ngrams(texts[i], n)
            if prev and curr:
                jaccard = len(prev & curr) / len(prev | curr)
                overlaps.append(jaccard)
        
        return sum(overlaps) / max(len(overlaps), 1)
    
    def _semantic_similarity(self, texts: list[str]) -> float:
        """Approximate semantic similarity via token overlap on key content."""
        if len(texts) < 2:
            return 0.0
        
        def normalize(text: str) -> set:
            return set(text.lower().split())
        
        similarities = []
        for i in range(1, len(texts)):
            t1 = normalize(texts[i-1])
            t2 = normalize(texts[i])
            if t1 and t2:
                jaccard = len(t1 & t2) / len(t1 | t2)
                similarities.append(jaccard)
        
        return sum(similarities) / max(len(similarities), 1)

# ---------------------------------------------------------------------------
# Circuit Breaker (Layer 4)
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Per-session 3-state circuit breaker with escalation."""
    
    def __init__(self, state: SessionState):
        self.state = state
    
    def check(self, loop_score: float) -> CircuitDecision:
        """Check if the tool call should proceed."""
        now = time.time()
        
        # Prune old break records
        self.state.break_times = [
            t for t in self.state.break_times
            if now - t < self.state.max_breaks_window
        ]
        
        # State: OPEN
        if self.state.breaker_state == "open":
            if now - self.state.last_break_time > self.state.cooling_period:
                self.state.breaker_state = "half_open"
                return CircuitDecision(allow=True, state="half_open",
                                       note="Half-open test request")
            remaining = int(self.state.cooling_period - 
                           (now - self.state.last_break_time))
            return CircuitDecision(allow=False, state="open",
                                   note=f"Cooling: {remaining}s remaining")
        
        # Check loop threshold
        if loop_score > 0.8:
            self.state.break_count += 1
            self.state.break_times.append(now)
            self.state.last_break_time = now
            
            # Max breaks → lock to T4
            if self.state.break_count >= self.state.max_breaks:
                self.state.breaker_state = "open"
                return CircuitDecision(
                    allow=False, state="open",
                    note=f"Max breaks ({self.state.max_breaks}) exceeded. Locked to T4.",
                    escalate=True, tier_override="t4"
                )
            
            self.state.breaker_state = "open"
            return CircuitDecision(
                allow=False, state="open",
                note=f"Loop detected (score={loop_score:.2f}). Circuit broken.",
                escalate=True
            )
        
        # State: HALF_OPEN → CLOSED on success
        if self.state.breaker_state == "half_open":
            self.state.breaker_state = "closed"
        
        return CircuitDecision(allow=True, state=self.state.breaker_state)
    
    def register_success(self):
        """Register a successful tool call."""
        self.state.breaker_state = "closed"

# ---------------------------------------------------------------------------
# Context Budget Manager (Layer 5)
# ---------------------------------------------------------------------------

class ContextBudgetManager:
    """Tracks and enforces context budget per session."""
    
    def __init__(self, budget: ContextBudget):
        self.budget = budget
    
    def check(self) -> ValidationResult:
        """Check if the context budget allows another call."""
        if self.budget.needs_compaction:
            return ValidationResult(
                valid=False,
                reason=f"Context at {self.budget.utilization:.0%} (threshold: 90%). "
                       f"Compact before next call.",
                needs_compaction=True
            )
        return ValidationResult(valid=True)
    
    def track_tool_call(self, tool_output: str, token_count: int):
        """Update budget after a tool call."""
        self.budget.used_tokens += token_count
        self.budget.step_count += 1
    
    def compact(self, steps: list) -> dict:
        """Compact context by summarizing old steps."""
        keep = 5  # sliding window
        if len(steps) <= keep:
            return {"steps_before": len(steps), "steps_after": len(steps),
                    "tokens_freed": 0}
        
        summarize = steps[:-keep]
        intact = steps[-keep:]
        
        tokens_before = sum(s.get("tokens", 0) for s in summarize)
        # Summary is ~300 tokens
        tokens_after = 300
        
        self.budget.used_tokens -= (tokens_before - tokens_after)
        self.budget.compaction_count += 1
        self.budget.last_compaction_at = time.time()
        
        return {
            "steps_before": len(summarize),
            "steps_after": 1,
            "tokens_freed": tokens_before - tokens_after,
            "new_active_tokens": tokens_after + sum(s.get("tokens", 0) for s in intact)
        }

# ---------------------------------------------------------------------------
# Main Plugin Class
# ---------------------------------------------------------------------------

_sessions: dict[str, SessionState] = {}
_validator = SchemaValidator()
_detector = LoopDetector()


def _get_session(session_id: str) -> SessionState:
    """Get or create session state."""
    if session_id not in _sessions:
        _sessions[session_id] = SessionState(session_id)
    return _sessions[session_id]


async def pre_tool_call_handler(
    tool_name: str,
    args: dict,
    context: dict
) -> dict:
    """
    Hermes pre_tool_call hook.
    
    Returns:
        {"action": "allow"} — proceed with the tool call
        {"action": "block", "reason": "..."} — prevent execution
        {"action": "escalate", "reason": "..."} — block + escalate tier
    """
    if not context.get("harness_enabled", True):
        return {"action": "allow"}
    
    session_id = context.get("session_id", "default")
    session = _get_session(session_id)
    
    # Layer 2: Schema validation
    schemas = context.get("tool_schemas", {})
    validation = _validator.validate(tool_name, args, schemas)
    if not validation.valid:
        return {"action": "block", "reason": validation.reason}
    
    # Layer 4: Loop detection + Circuit breaker
    loop_score = _detector.score(session.recent_calls)
    if loop_score > 0.8:
        pattern = _detector.detect_pattern(session.recent_calls)
    
    breaker = CircuitBreaker(session)
    decision = breaker.check(loop_score)
    
    if not decision.allow:
        if decision.escalate:
            return {
                "action": "escalate",
                "reason": decision.note,
                "tier_override": decision.tier_override
            }
        return {"action": "block", "reason": decision.note}
    
    # Layer 5: Context budget check
    budget_mgr = ContextBudgetManager(session.budget)
    budget_check = budget_mgr.check()
    if budget_check.needs_compaction:
        return {"action": "delay", "reason": budget_check.reason}
    
    return {"action": "allow"}


async def post_tool_call_handler(
    tool_name: str,
    args: dict,
    result: Any,
    context: dict
) -> None:
    """Hermes post_tool_call hook - track state."""
    session_id = context.get("session_id", "default")
    session = _get_session(session_id)
    
    # Record the call
    record = CallRecord(
        tool=tool_name,
        args=args,
        output=str(result)[:1000] if result else "",
        latency=context.get("latency", 0.0),
        timestamp=time.time(),
        error=context.get("error", None)
    )
    session.recent_calls.append(record)
    
    # Keep only last N records
    if len(session.recent_calls) > 100:
        session.recent_calls = session.recent_calls[-100:]
    
    # Update context budget
    token_count = context.get("token_count", 0)
    budget_mgr = ContextBudgetManager(session.budget)
    budget_mgr.track_tool_call(str(result)[:200], token_count)


async def pre_verify_handler(context: dict) -> dict:
    """Hermes pre_verify hook - coercion to compact if needed."""
    session_id = context.get("session_id", "default")
    session = _get_session(session_id)
    
    budget_mgr = ContextBudgetManager(session.budget)
    
    if budget_mgr.budget.needs_compaction:
        return {
            "action": "enforce",
            "reason": f"Context at {budget_mgr.budget.utilization:.0%}. "
                     f"Compacting before next turn.",
            "commands": ["compact_context"]
        }
    
    return {"action": "pass"}
```

## __init__.py

```python
"""Small-Model Harness - Hermes Plugin."""
```

## Circuit Breaker State Machine Reference

```
        ┌─────────────────────────────┐
        │          CLOSED              │  Normal operation
        │  All requests pass through  │
        └─────────────┬───────────────┘
                      │
                      │ Loop confidence > threshold (0.8)
                      │
                      ▼
        ┌─────────────────────────────┐
        │           OPEN               │  Blocking
        │  Requests blocked           │
        │  Timer: 60s cooling         │
        └─────────────┬───────────────┘
                      │
                      │ Timer expires
                      │
                      ▼
        ┌─────────────────────────────┐
        │        HALF-OPEN             │  Testing
        │  Single request allowed     │
        └─────────────┬───────────────┘
                      │
              ┌───────┴───────┐
              │               │
          Success          Failure
              │               │
              ▼               ▼
        ┌──────────┐  ┌──────────────┐
        │  CLOSED  │  │    OPEN      │
        └──────────┘  └──────────────┘
```

## Loop Detection Threshold Guidelines

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.7 | Sensitive — catches mild looping, some false positives | High-cost tool calls (writes, network) |
| 0.8 | Balanced — catches real loops, few false positives | Default for most operations |
| 0.9 | Conservative — only catches severe loops | Read-only operations where loop cost is low |
| Custom | Per-tool threshold override | e.g., web_search 0.9, write_file 0.7 |
