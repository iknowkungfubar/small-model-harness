# Small-Model Harness Architecture

> Five-layer defensive architecture for running 1B–12B parameter models in production agentic workflows.
> Designed for integration with Hermes Agent as a plugin + MCP server pair.

---

## 1. Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │            ENTRY POINT                   │
                    │   User Request / Agent Task Dispatch     │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
     LAYER 1        │         TASK CLASSIFIER                 │
     ROUTING        │   Route to appropriate model tier       │
                    │   Tier 1: <4B (simple, single-step)     │
                    │   Tier 2: 4-8B (moderate complexity)    │
                    │   Tier 3: 9-12B (complex reasoning)     │
                    │   Tier 4: Cloud frontier (hardest 10%)  │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
     LAYER 2        │      PRE-CALL VALIDATION                │
     GUARDRAIL      │  • Schema validation of tool call args  │
                    │  • Doom loop pattern detection           │
                    │  • Rate limit / budget check             │
                    │  • Replay attack protection              │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
     LAYER 3        │    CONSTRAINED DECODING ENGINE          │
     OUTPUT         │  • XGrammar / Outlines token masking    │
     ENFORCEMENT    │  • JSON Schema guaranteed output        │
                    │  • Tool call format enforcement          │
                    │  • No free-text generation unmonitored   │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
     LAYER 4        │      CIRCUIT BREAKER                    │
     LOOP           │  • Detect repetitive token patterns     │
     DETECTION      │  • Track per-session loop probability   │
                    │  • 3-state: closed → open → half-open   │
                    │  • Automatic model tier escalation       │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
     LAYER 5        │    CONTEXT BUDGET MANAGER               │
     CONTEXT        │  • Token tracking across all tools      │
     MANAGEMENT     │  • Sliding window with summarization    │
                    │  • Effective capacity: 1/3 stated window│
                    │  • Forced compaction at 90% threshold   │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │         OUTPUT VERIFIER                 │
                    │  • Schema compliance check              │
                    │  • Semantic consistency scan            │
                    │  • Action verification (did tool run?)  │
                    │  • Escalation to human if uncertain     │
                    └─────────────────────────────────────────┘
```

## 2. Layer Details

### Layer 1: Task Classifier & Model Router

**Component:** MCP Server Tool `harness_classify_task` + `harness_route`

**Purpose:** Route each task to the most cost-effective model tier that can handle it reliably.

**Architecture:**

```
User Request
    │
    ▼
┌──────────────────┐
│ Classification   │  Dimensions:
│ Engine           │  • Complexity: simple / moderate / complex / critical
│                  │  • Tool count: 0, 1-2, 3-5, 6+
│                  │  • Reasoning depth: none, shallow, deep, multi-step
│                  │  • Failure cost: low, medium, high, critical
│                  │  • Context needed: small (<4K), medium (4-16K), large (16K+)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tier Assignment  │  Rules:
│ Engine           │  • Score-weighted voting across dimensions
│                  │  • Override for critical failure cost → min T3
│                  │  • Override for multi-step reasoning → min T2
│                  │  • Historical success rates adjust tier up/down
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Cascade Gate     │  If current tier confidence < threshold:
│                  │  • Route to next higher tier
│                  │  • Cascade flow: T1 → T2 → T3 → T4
│                  │  • Each escalation carries step context
│                  │  • Model health check before routing
└──────────────────┘
```

**Classification algorithm (lightweight, rule-based):**

```python
def classify_task(task: str, tools: list[str]) -> TaskProfile:
    """Classify task complexity without an LLM call."""
    score = 0
    reasoning_depth = "none"
    
    # Heuristics
    if any(word in task.lower() for word in ["debug", "explain", "why", "diagnose"]):
        score += 2
        reasoning_depth = "shallow"
    if any(word in task.lower() for word in ["plan", "design", "architect", "strategy"]):
        score += 3
        reasoning_depth = "deep"
    if any(word in task.lower() for word in ["security", "vulnerability", "exploit"]):
        score += 2  # don't automatically route security to large model
    
    tool_count = len(tools)
    if tool_count > 5:
        score += 2
    elif tool_count > 2:
        score += 1
    
    # Complexity bucket
    if score <= 2:
        complexity = "simple"
    elif score <= 4:
        complexity = "moderate"
    elif score <= 6:
        complexity = "complex"
    else:
        complexity = "critical"
    
    # Tier assignment
    tier_map = {
        "simple": "t1",
        "moderate": "t2",
        "complex": "t3",
        "critical": "t3"  # escalate through cascade if needed
    }
    
    return TaskProfile(
        tier=tier_map[complexity],
        complexity=complexity,
        reasoning_depth=reasoning_depth,
        tool_count=tool_count,
        score=score
    )
```

**Model tier definitions (configurable):**

| Tier | Models | Max Tools | Max Steps | Context Limit | Suitable For |
|------|--------|-----------|-----------|---------------|-------------|
| T1 | Phi-4-mini 3.8B, SmolLM3-3B, Llama 3.2-3B | 2 | 10 | 8K effective | Formatting, extraction, single tool call, known patterns |
| T2 | Qwen3-8B, Llama 3.2-8B, Qwen3-4B | 5 | 25 | 16K effective | Multi-step tasks, moderate reasoning, code gen |
| T3 | Qwen3-30B-A3B, Ornith-1.0-9B, Qwen3-14B | 10 | 50 | 32K effective | Complex reasoning, bug diagnosis, planning |
| T4 | Cloud frontier (Claude/GPT-4o) | 20 | 100 | 128K | Security-sensitive, high-stakes, hardest 10% |

### Layer 2: Pre-Call Validation

**Component:** Plugin hook `pre_tool_call` + MCP tool `harness_validate_call`

**Purpose:** Prevent bad tool calls before they execute.

**Validation pipeline:**

```python
def validate_tool_call(tool_name: str, args: dict, session: Session) -> ValidationResult:
    """Run all validation checks and return result."""
    
    # 1. Schema validation
    schema = get_tool_schema(tool_name)
    if not schema:
        return ValidationResult(valid=False, reason=f"Unknown tool: {tool_name}")
    
    schema_errors = validate_against_schema(args, schema)
    if schema_errors:
        return ValidationResult(valid=False, reason=str(schema_errors))
    
    # 2. Loop pattern detection
    loop_score = detection_ensemble(session.recent_calls, tool_name, args)
    if loop_score > 0.9:
        return ValidationResult(
            valid=False, reason="Loop detected (confidence: {loop_score:.2f})",
            escalate=True
        )
    
    # 3. Duplicate call detection
    if session.recent_calls and len(session.recent_calls) >= 2:
        last_two = session.recent_calls[-2:]
        if all(c.tool == tool_name and c.args == args for c in last_two):
            return ValidationResult(
                valid=False, reason="Same tool call repeated consecutively"
            )
    
    # 4. Budget check
    budget = session.context_budget
    effective_capacity = budget.stated_window * 0.33
    if budget.used_tokens > effective_capacity * 0.9:
        return ValidationResult(
            valid=False, reason="Context budget exceeded 90% of effective capacity. Compact first.",
            needs_compaction=True
        )
    
    return ValidationResult(valid=True)
```

**Doom loop detection ensemble:**

```python
def detection_ensemble(recent_calls: list[CallRecord], 
                       current_tool: str, current_args: dict) -> float:
    """Weighted ensemble of 4 detection signals. Returns 0.0-1.0."""
    
    if len(recent_calls) < 4:
        return 0.0  # not enough history
    
    scores = []
    
    # Signal 1: Token repetition (40% weight)
    outputs = [c.output for c in recent_calls[-5:]]
    ngram_overlap = compute_ngram_overlap(outputs, n=4)  # 4-gram
    scores.append((ngram_overlap, 0.4))
    
    # Signal 2: Tool call diversity (30% weight)
    recent_tools = [c.tool for c in recent_calls[-8:]]
    unique_ratio = len(set(recent_tools)) / max(len(recent_tools), 1)
    tool_diversity_score = 1.0 - unique_ratio  # low diversity → high loop risk
    scores.append((tool_diversity_score, 0.3))
    
    # Signal 3: Latency stability (15% weight)
    recent_latencies = [c.latency for c in recent_calls[-5:]]
    if recent_latencies:
        latency_std = statistics.stdev(recent_latencies)
        latency_mean = statistics.mean(recent_latencies)
        cv = latency_std / max(latency_mean, 1)  # coefficient of variation
        stability_score = 1.0 - min(cv, 1.0)  # very stable → suspicious (deterministic loop)
        scores.append((stability_score, 0.15))
    
    # Signal 4: Content stagnation (15% weight)
    if len(recent_calls) >= 4:
        outputs_trimmed = [c.output[:200] for c in recent_calls[-4:]]
        semantic_sim = compute_semantic_similarity(outputs_trimmed)
        scores.append((semantic_sim, 0.15))
    
    return sum(score * weight for score, weight in scores)
```

### Layer 3: Constrained Decoding Engine

**Component:** MCP integration with XGrammar/Outlines

**Purpose:** Force guaranteed valid structured output. Eliminate format drift.

**Architecture:**

```
LLM Inference Engine (LM Studio / llama.cpp / vLLM)
    │
    ▼
┌───────────────────────────────────────────┐
│ Constrained Decoding Adapter              │
│                                           │
│  ┌───────────────────────────────────┐    │
│  │ Schema Compiler                   │    │
│  │ • JSON Schema → Grammar file      │    │
│  │ • Caches compiled schemas         │    │
│  │ • Multi-schema composition        │    │
│  └──────────────┬────────────────────┘    │
│                 │                          │
│  ┌──────────────▼────────────────────┐    │
│  │ Token Mask Generator              │    │
│  │ • XGrammar: 40μs mask gen         │    │
│  │ • Returns valid next tokens       │    │
│  │ • Only tokens matching schema     │    │
│  └──────────────┬────────────────────┘    │
│                 │                          │
│  ┌──────────────▼────────────────────┐    │
│  │ Integration Backend               │    │
│  │ • Local: XGrammar C library       │    │
│  │ • Cloud: Post-hoc check + retry   │    │
│  │ • Fallback: Outlines Python lib   │    │
│  └───────────────────────────────────┘    │
└───────────────────────────────────────────┘
```

**Integration points with Hermes:**

1. **LM Studio backend:** XGrammar integrates via llama.cpp's built-in grammar support. The plugin generates a GBNF grammar from the tool schemas and passes it to the model at inference time.

2. **Cloud backend:** Constrained decoding not available natively. Use post-hoc validation + retry with max 3 attempts. The MCP server's `harness_validate_call` serves as the validator.

3. **Hermes Forge integration:** Forge's existing rescue_tool_call handles malformed output but is best-effort. The constrained decoding engine PREVENTS malformed output entirely — eliminates the need for rescue in most cases.

**Grammar generation pattern:**

```python
def build_grammar(tool_schemas: list[dict]) -> str:
    """Build a GBNF grammar that constrains output to valid tool calls."""
    # For each tool, generate:
    #   tool-name "(" json-schema-args ")"
    # 
    # Combined grammar:
    #   root ::= tool-call
    #   tool-call ::= tool-name "(" json-object ")"
    #   tool-name ::= "search_files" | "read_file" | "web_search" | ...
    #   json-object ::= "{" pair ("," pair)* "}"
    #   pair ::= key ":" value
    #   key ::= "\"" string "\""
    #   value ::= string | number | boolean | "null" | "[" values "]" | "{" object "}"
    #   ... (schema-specific constraints)
    
    # Key optimization: compile once, cache by schema hash
    grammar = compile_to_gbnf(tool_schemas)
    return grammar
```

### Layer 4: Circuit Breaker

**Component:** Plugin state machine + MCP tool `harness_circuit_break`

**Purpose:** Detect and break cycles before they exhaust context.

**State machine:**

```
        ┌─────────────────────────────┐
        │          CLOSED              │  ← Normal operation
        │  All requests flow through  │
        └─────────────┬───────────────┘
                      │
                      │ Loop detection confidence > threshold
                      ▼
        ┌─────────────────────────────┐
        │           OPEN               │  ← Breaking
        │  Requests blocked            │
        │  Timer: cooling_period (60s) │
        └─────────────┬───────────────┘
                      │
                      │ Timer expires
                      ▼
        ┌─────────────────────────────┐
        │        HALF-OPEN             │  ← Testing
        │  Single test request passes │
        │  Success → CLOSED           │
        │  Failure → OPEN             │
        └─────────────────────────────┘
```

**Session-level state machine:**

```python
class SessionCircuitBreaker:
    """Per-session circuit breaker state."""
    
    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = self.STATE_CLOSED
        self.break_count = 0
        self.last_break_time = 0
        self.cooling_period = 60  # seconds
        self.max_breaks = 5
        self.max_breaks_window = 600  # 10 minutes
        self.break_times: list[float] = []
        
    def check(self, loop_score: float) -> CircuitDecision:
        """Check if this request can proceed."""
        # Window maintenance
        now = time.time()
        self.break_times = [t for t in self.break_times if now - t < self.max_breaks_window]
        
        if self.state == self.STATE_OPEN:
            if now - self.last_break_time > self.cooling_period:
                self.state = self.STATE_HALF_OPEN
                return CircuitDecision(allow=True, state=self.state, note="Half-open test")
            return CircuitDecision(allow=False, state=self.state, 
                                   note=f"Cooling: {int(self.cooling_period - (now - self.last_break_time))}s remaining")
        
        if loop_score > 0.8:
            self.break_count += 1
            self.break_times.append(now)
            self.last_break_time = now
            
            if self.break_count >= self.max_breaks:
                return CircuitDecision(allow=False, state=self.STATE_OPEN,
                                       note="Max breaks exceeded. Locking to T4 for session.",
                                       escalate=True, tier_override="t4")
            
            self.state = self.STATE_OPEN
            return CircuitDecision(allow=False, state=self.STATE_OPEN,
                                   note=f"Loop detected (score: {loop_score:.2f}). Broken.",
                                   escalate=True)
        
        return CircuitDecision(allow=True, state=self.state)
```

**Escalation matrix on circuit break:**

| Break # | Action | Model Escalation |
|---------|--------|-----------------|
| 1 | Reprompt with different system hint (temperature += 0.2) | Same tier |
| 2 | Escalate to next model tier | T1→T2, T2→T3, T3→T4 |
| 3 | Kill chain, return partial + error | — |
| 4+ | Lock session to T4 for all requests | T4 forced |
| Window max | Halt agent, require human intervention | — |

### Layer 5: Context Budget Manager

**Component:** Plugin state tracking + MCP tool `harness_context_status` + `harness_compact`

**Purpose:** Prevent context rot by keeping effective context under reliable limit.

**Core rules:**

1. **Effective capacity = stated_window / 3**
   - Model claims 128K → effective max is ~42K
   - Model claims 32K → effective max is ~10K
   - Reservations: 10% output headroom → useable max is ~90% of effective

2. **Compaction triggers:**
   - Active tokens > 90% of effective capacity
   - Step count > sliding_window_steps * 2 (too many small steps)
   - Tool output > 60% of active tokens (too much noise)
   - Compaction count in last 10 steps > 3 (too much churn)

3. **Sliding window strategy:**
   - Keep last N complete steps (N=5 default)
   - Summarize steps [0, N-3] into structured summary
   - Keep steps [N-2, N] intact for context continuity
   - Store summaries as bullet points, not free text

**Compaction algorithm:**

```python
def compact_context(session: Session) -> CompactionResult:
    """Compact session context to free budget. Returns new token count."""
    
    steps = session.steps
    keep_last = session.sliding_window_steps  # default 5
    
    if len(steps) <= keep_last:
        return CompactionResult(
            steps_before=len(steps), steps_after=len(steps),
            tokens_freed=0, note="Too few steps to compact"
        )
    
    # Identify steps to summarize
    summarize_steps = steps[:-keep_last]
    intact_steps = steps[-keep_last:]
    
    # Generate summary
    summary = summarize_steps_to_bullets(summarize_steps)
    # summary is a compact structured string: ~200-500 tokens
    
    # Replace summarized steps with single summary step
    session.steps = [SummaryStep(content=summary)] + intact_steps
    
    tokens_before = sum(s.tokens for s in summarize_steps)
    tokens_after = estimate_tokens(summary)
    
    return CompactionResult(
        steps_before=len(summarize_steps),
        steps_after=1,
        tokens_freed=tokens_before - tokens_after,
        new_active_tokens=tokens_after + sum(s.tokens for s in intact_steps)
    )
```

**Budget tracking per session:**

```python
@dataclass
class ContextBudget:
    session_id: str
    stated_window: int       # e.g., 131072 for 128K model
    effective_capacity: int  # stated_window * 0.33
    used_tokens: int         # current active context tokens
    step_count: int
    compaction_count: int
    last_compaction_at: Optional[float]
    
    @property
    def utilization(self) -> float:
        return self.used_tokens / self.effective_capacity
    
    @property 
    def needs_compaction(self) -> bool:
        return self.utilization > 0.9 or self.step_count > 25
    
    @property
    def output_headroom(self) -> int:
        return int(self.effective_capacity * 0.1)  # 10% reserved for output
```

## 3. Cross-Layer Interactions

### Normal flow:
```
Request → L1 classify/route → L2 validate (pass) → Model → L3 constrain → L4 check (pass) → L5 track → Done
```

### Error recovery flow (loop detected):
```
Request → L1 → L2 validate (pass) → Model → L3 constrain → L4 check (FAIL: loop)
    │
    ├─→ L1: escalate to T2
    │    └─→ Reprompt at higher tier
    │         └─→ L4 check (pass) → L5 track → Done
    │
    └─→ L5: compact context after escalation
```

### Error recovery flow (context budget exceeded):
```
Request → L1 → L2 (FAIL: budget exceeded)
    │
    └─→ Call L5 compact
         └─→ Re-validate
              └─→ Continue
```

### Cascading failure flow (3 consecutive breaks):
```
Request → L1 → L2 → ... → L4 break → L1 escalate → L4 break → L1 escalate → L4 break
    │
    └─→ L4: Kill chain
         └─→ Return partial results + error report
```

## 4. Integration with Hermes

### Plugin hooks:

```yaml
# plugin.yaml
name: small-model-harness
version: 0.1.0
description: "5-layer defensive harness for small local LLMs (1B-12B)"
hooks:
  pre_tool_call: pre_tool_call_handler
  post_tool_call: post_tool_call_handler
  pre_verify: pre_verify_handler
```

**pre_tool_call:** Layers 2, 4 — validate call, check circuit breaker, check context budget
**post_tool_call:** Layers 4, 5 — update loop detection history, update token tracking
**pre_verify:** Layer 5 — check if compaction needed before next model call

### MCP Server tools:

MCP server provides the analysis/decision tools that the plugin enforces:

```
harness_classify_task   → L1 task classification
harness_route            → L1 model routing
harness_validate_call    → L2 validation
harness_detect_loop      → L4 loop detection
harness_circuit_break    → L4 state management
harness_context_status   → L5 budget status
harness_compact          → L5 context compaction
```

### Interaction with existing Hermes systems:

| Hermes System | How Harness Complements |
|--------------|------------------------|
| Hermes Forge | Forge = general tool validation. Harness = small-model-specific failure prevention (loops, context rot, format drift). Forge rescues; Harness prevents. |
| sdlc-quality-hooks | sdlc = build-level safety (rm -rf, git force, .env). Harness = model-level safety (loops, budget, validation). |
| Context budget | Harness provides MORE aggressive budget (1/3 rule) for small models. Coexists with general budget — harness limit fires first. |
| Multi-model fallback | Harness provides TASK-CLASSIFIED routing (by complexity) vs simple availability-based fallback. |
| Long context handling | Harness provides SMALL-MODEL-SPECIFIC rules (effective window 1/3, aggressive compaction, sliding window). |

## 5. Configuration

Full YAML config reference (applies to both plugin and MCP):

```yaml
# small-model-harness.yaml
harness:
  enabled: true
  
  routing:
    default_tier: "t2"
    cascade_enabled: true
    confidence_threshold: 0.7
    cascade_interval_seconds: 5  # min time between cascades
    tiers:
      t1:
        models: ["phi-4-mini:3.8b", "smollm3:3b"]
        max_tools: 2
        max_steps: 10
        effective_context: 8192
      t2:
        models: ["qwen3:8b", "llama3.2:8b", "qwen3:4b"]
        max_tools: 5
        max_steps: 25
        effective_context: 16384
      t3:
        models: ["qwen3-30b-a3b", "ornith-1.0-9b", "qwen3:14b"]
        max_tools: 10
        max_steps: 50
        effective_context: 32768
      t4:
        models: []  # cloud
        provider: "opencode-zen"
        max_tools: 20
        max_steps: 100
        effective_context: 131072
  
  circuit_breaker:
    enabled: true
    loop_threshold: 0.8
    cooling_period_seconds: 60
    max_breaks_per_session: 5
    break_window_minutes: 10
    tier_escalation_on_break: true
    rate_limit_calls_per_minute: 30
  
  context:
    effective_capacity_ratio: 0.33
    compaction_threshold: 0.9
    sliding_window_steps: 5
    reserved_output_ratio: 0.10
    max_steps_before_compaction: 25
    summarization_model: "qwen3:4b"  # can be cheaper than main model
  
  validation:
    schema_check: true
    duplicate_detection: true
    max_consecutive_same_tool: 3
    loop_detection_window: 8  # recent calls to check
  
  output_constraint:
    engine: "xgrammar"  # xgrammar, outlines, post-hoc
    cache_compiled_schemas: true
    max_retries_on_format_fail: 3
    fallback_to_post_hoc: true
```
