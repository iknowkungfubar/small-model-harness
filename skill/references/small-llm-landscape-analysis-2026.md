# Small Local LLM Landscape Analysis (July 2026)

> Comprehensive analysis of 1B-12B parameter models, their capabilities, critical pitfalls,
> and the architectural framework required to bridge the gap to frontier-level performance.
> Based on primary research: Chroma Context Rot study, Liquid AI Antidoom paper,
> Qwen3 family benchmarks, Hermes Forge architecture, and industry surveys.

---

## Part 1: The Small Model Landscape (1B–12B, July 2026)

### 1.1 Current SOTA Models

| Model | Size Range | Context | License | Key Strength | Key Weakness |
|-------|-----------|---------|---------|--------------|-------------|
| Qwen3 dense (0.6B–32B) | 0.6B, 1.7B, 4B, 8B, 14B, 32B | 128K | Apache 2.0 | Hybrid thinking mode, tool calling | Format drift on long chains |
| Qwen3 MoE (30B-A3B) | 30B total, 3B active | 128K | Apache 2.0 | Near 32B quality at 3B cost | MoE routing overhead |
| Gemma 4 (E4B, 9B, 12B) | 4B, 9B, 12B | 32K-128K | Gemma | Multimodal, native audio | Highest doom loop rates |
| Phi-4-mini | 3.8B | 128K | MIT | Top sub-4B benchmarks | Smaller knowledge base |
| SmolLM3 | 3B | 32K | Apache 2.0 | Optimized for mobile | Limited agent capability |
| Llama 3.2 (1B, 3B, 8B) | 1B, 3B, 8B | 128K | Llama 3 | Broad ecosystem support | Older architecture |
| Ornith-1.0-9B | 9B | 262K | MIT | Massive context for size | Tool-looping tendency |
| DeepSeek R1 distill (1.5B–8B) | 1.5B–8B | 128K | MIT | Reasoning capability | Distillation quality loss |

### 1.2 Parameter Efficiency Trends

The defining trend of 2025-2026: **smaller models are closing the gap faster than expected.**

- **Qwen3-4B** rivals Qwen2.5-72B — an **18x parameter reduction** with comparable output quality
- **Qwen3-8B** outperforms Qwen2.5-14B across 50%+ of benchmarks
- Small model distillation from reasoning teachers (DeepSeek R1, Qwen3) produces capable reasoners at 3-8B
- MoE architectures (Qwen3-30B-A3B, 235B-A22B) deliver frontier capability at fraction of active params

**Implication for agent work:** A 4-8B model today is more capable for tool use than a 70B model from 18 months ago. The performance gap to frontier models is narrowing, but the **reliability gap** persists.

---

## Part 2: Critical Pitfalls — The Full Diagnosis

### 2.1 Doom Loops (Repetitive Death Spirals)

**The research:** Liquid AI's Antidoom paper (July 7, 2026), arXiv:2606.13705.

**Definition:** The model emits a span, then repeats the same span again and again until the context window is exhausted. NOT self-correction, NOT verbose reasoning — true pathological repetition.

**Measured rates by model:**
| Model | Loop Rate (Greedy) | After Antidoom |
|-------|-------------------|----------------|
| LFM2.5-2.6B | 10.2% | 1.4% |
| Qwen3.5-4B | 22.9% | 1.0% |
| Gemma 4 E4B | ~15% (estimated) | N/A yet |

**Root cause:** Overtrained tokens at specific positions. Common starters: `the`, `So`, `Alternatively`, `Wait`, `But`. These tokens have pathologically high probability at certain decoder positions due to training distribution artifacts in instruction-tuning.

**Why small models are more vulnerable:**
- Smaller parameter count = less capacity for diverse reasoning paths
- Instruction-tuned on narrower data distributions
- Stronger mode collapse under greedy sampling
- Long thinking traces amplify any positional bias

**Existing fixes:**
1. **Antidoom/FTPO** (training-level): ~2hr dataset gen + 1-2hr LoRA training. Cuts loops 90-95%. Surgical — targets one token position.
2. **Repetition penalty** (inference-level): Band-aid. Reduces all token probabilities equally, damaging output quality. Can cause topic drift.
3. **Max-steps hard cap**: Blunt instrument. Limits all chains, not just looping ones.

### 2.2 Context Rot (Length-Driven Degradation)

**The research:** Chroma study (July 2025, updated 2026), 18 models tested. Also supported by NVIDIA engineering guidelines (April 2026).

**Core finding:** EVERY model degrades with input length. This is NOT "lost in the middle" (position-based) — it is **length-driven**. Same content placed in a longer context produces worse results.

**Key numbers:**
- **NVIDIA's production rule:** Keep prompt under 1/3 of stated window, rest reserved for output + headroom.
- **Repeated Words task (simplest test):** Even frontier models show 30-60% accuracy degradation from 4K to 128K tokens.
- **Needle-Question Similarity:** Semantic matches degrade faster than lexical matches as context grows.
- **Multi-step reasoning (realistic):** Expected degradation is MORE severe than controlled tests — because synthesis compounds position and length effects.

**Small model impact:**
- A 32K context model like Llama 3.2-3B has ~10K effective reliable context
- A 128K context model like Qwen3-4B has ~40K effective (per NVIDIA 1/3 rule)
- For agent work, every step adds tokens: if each tool call consumes 2K tokens, a 20-step chain is already at 40K tokens — exceeding effective capacity

### 2.3 Instruction Following Fragility

**The research:** Kunal Ganglani Qwen3 agent testing (June 2026), multiple production surveys.

**Tool selection accuracy:**
| Model | First-attempt accuracy |
|-------|----------------------|
| GPT-4o | ~92% |
| Qwen3-32B | ~87% |
| Qwen3-30B-A3B | ~84% |
| Qwen3-8B | ~78% |
| Phi-4-mini | ~75% |
| Llama 3.2-8B | ~72% |
| Gemma 4-9B | ~70% |

**Format drift cascade:** A malformed response at step N in a 15-step chain has a >50% chance of derailing ALL subsequent steps. Each malformed response compounds — the model's context now contains broken XML/JSON, making the next prediction harder.

**Critical finding:** The gap between open and closed models is NOT intelligence at this point — it's **instruction-following reliability at scale**.

### 2.4 Hallucination + Reasoning Paradox

**Small model hallucination rates:**
| Model | HHEM Rate | Notes |
|-------|-----------|-------|
| Phi-4 (14B) | ~3-5% | Lowest among small models |
| Qwen3-8B | ~5-7% | Slightly higher than Phi-4 |
| Gemma 4 E4B | ~45-49% | Significantly higher |
| Llama 3.2-8B | ~8-10% | Middle of pack |

**The reasoning paradox:** Thinking/reasoning mode amplifies hallucination 2-3x. The model "reasons" confidently about fabricated facts. This affects all models but hits small ones hardest because:
- Less world knowledge to ground reasoning
- Stronger distribution mode collapse under extended generation
- Cannot effectively self-correct without a larger model's judgment

### 2.5 Training Data Cutoffs

| Model Family | Estimated Cutoff |
|-------------|-----------------|
| Qwen3 | April 2025 |
| Gemma 4 | Late 2024 |
| Phi-4 | Late 2024 |
| Llama 3.2 | Mid 2024 |
| DeepSeek R1 distill | Early 2025 |

**Impact for software development:**
- Knowledge of libraries released after cutoff = zero
- Must rely on tool calling to search/read docs
- Cannot answer questions about very recent API changes
- Agent must be designed to fetch real-time information, not rely on parametric knowledge

---

## Part 3: Architectural Framework — The Small-Model Harness

### 3.1 Design Principles

1. **Assume failure, verify everything** — Every output is suspect until validated
2. **Compensate for size with infrastructure** — The harness provides what the model lacks
3. **Route intelligently** — Not every task needs a frontier model; not every task can be handled by a 4B
4. **Detect and break loops** — Doom loops are the #1 silent killer of small-model agents
5. **Bound context ruthlessly** — Context rot is the #2 killer; keep effective context under 1/3 of stated window
6. **Constrained output by default** — Never let the model choose output format; force it with grammar/token masking

### 3.2 Five-Layer Defensive Architecture

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

### 3.3 Layer Details

#### Layer 1: Task Classifier & Model Router

**Purpose:** Route each task to the appropriate model tier based on complexity, required reasoning depth, and failure sensitivity.

**Implementation:**
- **Lightweight classifier** (can be rules-based or a tiny embedding model like nomic-embed-text)
- Classification dimensions: complexity (simple/moderate/complex), tool count (0/1-2/3+), reasoning depth needed (none/some/deep), failure cost (low/medium/high)
- **Four tiers:**
  - T1 (<4B, fast): Formatting, simple extraction, single tool call, known patterns
  - T2 (4-8B, balanced): Multi-step tasks, moderate reasoning, code generation
  - T3 (9-12B, capable): Complex reasoning, bug diagnosis, planning
  - T4 (cloud frontier, reserved): Hardest 10% — security-sensitive, high-stakes decisions

**Cascade with confidence gate:**
```
T1 → if confidence < threshold → T2
T2 → if confidence < threshold → T3
T3 → if confidence < threshold → T4 (cloud)
```

**Model selection matrix (July 2026):**

| Task Type | T1 Model | T2 Model | T3 Model | T4 Fallback |
|-----------|----------|----------|----------|-------------|
| Code gen (simple) | Phi-4-mini 3.8B | Qwen3-8B | Qwen3-30B-A3B | GPT-4o / Claude |
| Code gen (complex) | — | Qwen3-8B | Qwen3-30B-A3B | Claude Opus 4 |
| Bug diagnosis | — | — | Qwen3-30B-A3B | GPT-4o |
| Tool calling (1-2 tools) | Phi-4-mini | Qwen3-8B | Qwen3-30B-A3B | Claude Opus 4 |
| Tool calling (3+ tools) | — | — | Qwen3-30B-A3B | GPT-4o |
| Summarization | Phi-4-mini | Qwen3-8B | — | — |
| Data extraction | SmolLM3-3B | Qwen3-4B | — | — |
| Planning/Strategy | — | — | Qwen3-30B-A3B | Claude Opus 4 |
| Security-sensitive | — | — | — | GPT-4o / Claude |

#### Layer 2: Pre-Call Validation

**Purpose:** Prevent bad tool calls before they execute. This is the FIRST line of defense against doom loops and cascading failures.

**Sub-components:**
1. **Schema validator:** Check tool call arguments against function schema before sending to model for execution. Catches parameter name hallucinations, wrong types, missing required fields.
2. **Doom loop pattern detector:** Monitor the last N tool calls for repetitive patterns. Common patterns:
   - Same tool called >3 times with same/similar arguments
   - Same error returned >2 times (stuck in retry loop)
   - Token sequence shows repetition markers (repeated phrases, identical reasoning steps)
   - Execution time per step not decreasing (not converging)
3. **Budget check:** Track token spend per task. If approaching budget, force simplification or escalation.
4. **Risk scanner:** Flag dangerous operations (file deletion, network writes, config changes) for human approval.

**Implementation in Hermes:** `pre_tool_call` hook — exactly like sdlc-quality-hooks but focused on LLM-specific guardrails.

#### Layer 3: Constrained Decoding Engine

**Purpose:** Force the model to produce valid structured output. This ELIMINATES the format drift problem that plagues small models.

**Implementation options:**
1. **XGrammar** (CMU): 40μs token mask generation, ~100x faster than prior methods. Integrates with llama.cpp, vLLM, SGLang. JSON Schema guaranteed.
2. **Outlines:** Python library, integrates with multiple backends. Regex, JSON, and type-constrained generation.
3. **llguidance:** Microsoft's guided generation. Supports multiple grammar formats.

**For small models, XGrammar is preferred:**
- Minimal inference overhead (critical for slow small models)
- Schema-level guarantees eliminate ALL parsing errors
- 17% "creativity cost" documented but acceptable for agent tool calling (where creativity is NOT desired)

**Integration with Hermes:**
- Hermes already uses tool schemas — the constrained decoding engine wraps the LLM backend
- For locally hosted models (LM Studio, llama.cpp), XGrammar is a C library that plugs into the inference engine
- For cloud models, the output is validated post-hoc (schema check + retry)

#### Layer 4: Circuit Breaker

**Purpose:** Detect and break loops before they exhaust context or waste compute.

**Three-state model:**
```
[CLOSED] → Normal operation. All requests pass.
    ↓ Trigger: loop detection confidence > 0.7
[OPEN] → Requests blocked. Circuit breaker fires.
    ↓ Timeout: cooling period (30-120s)
[HALF-OPEN] → Test request allowed.
    ↓ Success: back to CLOSED
    ↓ Failure: back to OPEN
```

**Detection signals (weighted ensemble):**
1. **Token repetition score** (40% weight): N-gram overlap ratio between recent outputs. If >0.8, strong loop signal.
2. **Tool call diversity** (30% weight): Same tool called with same args consecutively. Entropy of tool names in sliding window.
3. **Latency stability** (15% weight): If successive calls have identical compute time, suggests deterministic path (loop).
4. **Content stagnation** (15% weight): Output tokens are near-identical across turns. Semantic similarity of consecutive outputs.

**Escalation on circuit break:**
1. First break in session: Reprompt with different temperature + system hint
2. Second break: Escalate to next model tier (T1 → T2, T2 → T3, T3 → T4 cloud)
3. Third break: Kill the chain, return partial results + error report
4. Session-level: Track break frequency. If >5 breaks in 10 minutes, lock to T4 for all requests.

#### Layer 5: Context Budget Manager

**Purpose:** Prevent context rot by keeping effective context under the reliable limit.

**Rules:**
1. **Effective capacity = stated window / 3** (NVIDIA rule, backed by Chroma study)
2. **Active context = last N complete steps** (sliding window, N=3-5)
3. **Compaction triggers:**
   - When active context > 90% of effective capacity → force compaction
   - When step count > 20 → summarize early steps
   - When context contains >50% tool output → truncate/compress tool outputs
4. **Summarization strategy:** Replace the oldest steps with a compressed summary until active context is under 70% of effective capacity
5. **Reserved headroom:** Always reserve 10% of context for the model's output generation

**Implementation:**
- Track per-session: total_tokens, active_tokens, step_count, compaction_count
- On each turn boundary, check active_tokens against effective_capacity
- When compaction needed: summarize steps [0, N-3] into a single summary step, keep steps [N-2, N] intact
- Store summaries in a lossy but structured format (not free text — bullet points or schema)

### 3.4 Cross-Layer Interactions

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Classify → Route → Cascade                          │
│ Output: model_tier, task_profile, initial_budget            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent Loop (per step):                                       │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Layer 2  │──▶│  Model   │──▶│ Layer 3  │──▶│ Layer 4  │ │
│  │ Validate │   │ Inference│   │ Constrain│   │ Check    │ │
│  │ Pre-call │   │          │   │ Output   │   │ Circuit  │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│                                                      │       │
│  ┌──────────┐                                       │       │
│  │ Layer 5  │◄──────────────────────────────────────┘       │
│  │ Context  │   (loop detected → escalate tier)             │
│  │ Budget   │   (normal → continue)                         │
│  └──────────┘                                               │
│                                                              │
│  Layer 2-4 feedback to Layer 5: budget consumption tracked  │
│  Layer 4 breach → Layer 1 re-route to higher tier          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Output Verification                                          │
│ • Schema check                                               │
│ • Action verification (did the tool actually run?)          │
│ • Consistency scan                                           │
│ • Result OR error report                                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 Failure Mode Coverage Matrix

| Failure Mode | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
|-------------|---------|---------|---------|---------|---------|
| Doom loops | T3/T4 preferred | Pattern detect | — | Break circuit | — |
| Context rot | — | — | — | — | Budget limit |
| Tool call format drift | — | Schema validate | XGrammar | — | — |
| Hallucination | Route to larger model | — | Constrain output | — | — |
| Cascading failure | Cascade to higher tier | Pre-call validation | — | Circuit break | — |
| Budget exhaustion | Allocate per task | Per-call budget | — | — | Compaction |
| Stuck in retry loop | — | Duplicate detect | — | Break circuit | — |
| Training data cutoff | — | — | — | — | — (needs RAG) |
| Wrong tool selection | Route to better model | Schema mismatch | — | Pattern detect | — |

---

## Part 4: Strategic Implementation Plan for Hermes

### 4.1 Architecture Decision

**Two-component system:**

1. **`small-model-harness` Hermes Plugin** — Hooks into the agent loop (pre_tool_call, post_tool_call, pre_verify) for deterministic enforcement. Cannot be bypassed by the agent. Handles Layers 2, 4, 5.

2. **`small-model-harness` MCP Server** — Provides tools for task classification, constrained decoding, circuit breaker status, and context management. Handles Layers 1, 3. Supplementary to the plugin — provides the analysis/decision tools the plugin enforces.

**Why both:**
- Plugins provide DETERMINISTIC enforcement (like sdlc-quality-hooks)
- MCP servers provide RICH ANALYSIS tools the agent can call
- Together they mirror the sdlc-quality-hooks + sdlc-quality-gates pattern

### 4.2 Implementation Phases

#### Phase 1: Foundation (Week 1)
**Deliverable:** Hermes plugin with Layers 2 + 4 (pre-call validation + circuit breaker)

- `plugin.py` with pre_tool_call hook
- Schema validator for tool calls
- Doom loop pattern detector (n-gram overlap, tool call diversity)
- Circuit breaker with 3-state model
- Basic retry budget

#### Phase 2: Context Management (Week 2)
**Deliverable:** Layer 5 (context budget manager) in plugin + MCP server

- Token tracking per session
- Sliding window summarization
- Forced compaction triggers
- Effective capacity calculator (1/3 rule)
- MCP server: `harness_context_status`, `harness_compact`

#### Phase 3: Model Router (Week 3)
**Deliverable:** MCP server with Layer 1 (task classifier + model router)

- Task complexity classifier
- Model tier definitions (configurable)
- Cascade with confidence gate
- Model health tracking
- MCP tools: `harness_classify_task`, `harness_route`, `harness_cascade`

#### Phase 4: Constrained Decoding (Week 4)
**Deliverable:** Integration with XGrammar/Outlines for Layer 3

- XGrammar integration for local models
- Fallback to post-hoc validation for cloud models
- Token mask generation for tool schemas
- Schema compilation cache

#### Phase 5: Hardening + Observability (Week 5)
**Deliverable:** Metrics, dashboards, edge cases

- Prometheus metrics for all layers
- Loop detection accuracy tuning
- Fallback chain testing
- Documentation + AGENTS.md integration

### 4.3 Hermes Plugin API

```python
# plugin.yaml
name: small-model-harness
version: 0.1.0
hooks:
  pre_tool_call: pre_tool_call
  post_tool_call: post_tool_call
  pre_verify: pre_verify
```

**Key hook implementations:**

```python
async def pre_tool_call(tool_name: str, args: dict, context: dict) -> dict:
    """Validate tool call before execution.
    
    Returns:
        {"action": "allow"}
        {"action": "block", "reason": "..."} 
        {"action": "escalate", "reason": "..."}
    """
    # 1. Schema validation
    schema_result = validate_schema(tool_name, args)
    if not schema_result.valid:
        return {"action": "block", "reason": f"Schema mismatch: {schema_result.error}"}
    
    # 2. Loop pattern detection
    loop_score = detect_loop_pattern(tool_name, args, context["recent_calls"])
    if loop_score > 0.8:
        # Circuit breaker open
        circuit_state = get_circuit_state(context["session_id"])
        if circuit_state == "open":
            return {"action": "block", "reason": "Circuit breaker open. Cooling period active."}
        elif loop_score > 0.9:
            open_circuit(context["session_id"])
            return {"action": "escalate", "reason": "Loop detected. Escalating to higher model tier."}
    
    # 3. Budget check
    budget = get_context_budget(context["session_id"])
    if budget.used > budget.effective_capacity:
        return {"action": "delay", "reason": "Context budget exceeded. Compacting before next call."}
    
    return {"action": "allow"}
```

### 4.4 MCP Server Tools

| Tool | Purpose | Layer | Input | Output |
|------|---------|-------|-------|--------|
| `harness_classify_task` | Classify task complexity | L1 | task_description, tools_available | tier, confidence, reasoning |
| `harness_route` | Route to model tier | L1 | task_classification, model_availability | model_name, tier, fallback_chain |
| `harness_validate_call` | Validate tool call | L2 | tool_schema, args, session_history | valid, errors, suggestions |
| `harness_detect_loop` | Check for looping | L4 | recent_calls (list) | loop_score, pattern_type, recommendation |
| `harness_circuit_break` | Manage circuit breaker | L4 | session_id, action (check/open/close/reset) | state, history, cooling_remaining |
| `harness_context_status` | Show context budget | L5 | session_id | used_tokens, effective_capacity, step_count, compaction_count |
| `harness_compact` | Force context compaction | L5 | session_id, strategy (summarize/truncate/drop) | new_active_tokens, summary_length |
| `harness_config` | Get/set harness config | ALL | session_id, config_updates | current_config |

### 4.5 Integration with Existing Hermes Infrastructure

The small-model-harness integrates with and extends existing components:

| Existing Component | How Harness Integrates |
|-------------------|----------------------|
| Hermes Forge | Harness replaces forge for small-model-specific concerns. Forge continues for general LLM guardrails (rescue, step ordering, general validation). Harness focuses on small-model failure modes specifically. |
| sdlc-quality-hooks | Harness adds model-level guardrails to sdlc-quality-hooks' build-level guardrails. Pre_tool_call is extended with ML-specific checks alongside the existing rm -rf / and git push --force blocks. |
| Context budget | Harness provides a more aggressive, small-model-aware budget (1/3 effective capacity) vs the general context budget. They coexist — harness's budget is the MORE restrictive one that takes effect. |
| Multi-model routing | Harness adds the cascade-with-confidence-gate pattern to Hermes' existing fallback chain. Implements task classification that Hermes' fallback chain doesn't have. |

### 4.6 Configuration Schema

```yaml
# small-model-harness config
harness:
  enabled: true
  
  routing:
    default_tier: "t2"  # t1, t2, t3, t4
    cascade_enabled: true
    confidence_threshold: 0.7
    tiers:
      t1:
        models: ["phi-4-mini:3.8b", "smollm3:3b"]
        max_tools: 2
        max_steps: 10
      t2:
        models: ["qwen3:8b", "llama3.2:8b"]
        max_tools: 5
        max_steps: 25
      t3:
        models: ["qwen3-30b-a3b", "qwen3:14b", "ornith-1.0-9b"]
        max_tools: 10
        max_steps: 50
      t4:
        models: []  # cloud — configured via provider settings
        provider: "opencode-zen"
        max_tools: 20
        max_steps: 100
  
  circuit_breaker:
    enabled: true
    loop_detection_threshold: 0.8
    cooling_period_seconds: 60
    max_breaks_per_session: 5
    break_window_minutes: 10
  
  context:
    effective_capacity_ratio: 0.33  # 1/3 of stated window
    compaction_threshold: 0.9       # compact at 90% of effective capacity
    sliding_window_steps: 5
    reserved_output_ratio: 0.10
    summarization_model: "qwen3:4b"  # can be smaller than main model
  
  validation:
    schema_check: true
    duplicate_detection: true
    max_consecutive_same_tool: 3
  
  output_constraint:
    engine: "xgrammar"  # xgrammar, outlines, post-hoc
    cache_compiled_schemas: true
```

### 4.6 Skill Integration

This is delivered as a skill package under `meta/small-model-harness/` with:
- `SKILL.md` — Main skill with workflow guidance
- `references/small-llm-landscape-analysis-2026.md` — This research document
- `references/harness-architecture.md` — Architecture details
- `references/plugin-api-reference.md` — API documentation
- `plugin/` — Plugin directory with plugin.yaml + plugin.py
- `mcp/` — MCP server code
