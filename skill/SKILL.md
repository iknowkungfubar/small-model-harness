# Small-Model Harness

> Defensive harness for running 1B–12B parameter models in production agentic workflows.
> Five-layer architecture: routing → validation → constraint → circuit break → context management.
> Delivered as a Hermes plugin (deterministic enforcement) + MCP server (analysis tools).

## When To Load This Skill

- Running any local model under 12B parameters for agentic work (coding, tool calling, multi-step tasks)
- Debugging "model loops forever" / "model makes same mistake repeatedly" / "model ignores instructions"
- Setting up a new local LLM for a production or development agent flow
- Tuning context window / budget limits for a smaller model
- Designing an architecture that mixes small local models with cloud fallback

## The Small Model Problem (July 2026)

Small models (1B–12B) have narrowed the quality gap dramatically vs. frontier models — Qwen3-4B rivals Qwen2.5-72B, and Qwen3-32B achieves 87% tool-selection accuracy vs GPT-4o's 92%. **But the reliability gap persists.** Research from Chroma, Liquid AI, NVIDIA, and Ganglani identifies three critical failure modes:

### Critical Failure Mode 1: Doom Loops

Small thinking models enter repetitive death spirals. Qwen3.5-4B has a **22.9% loop rate** under greedy sampling. Antidoom (FTPO) training cuts this to ~1%, but requires dedicated fine-tuning. **Without mitigation, every 5th hard problem will loop to context exhaustion.**

### Critical Failure Mode 2: Context Rot

All models degrade with input length — not position-driven ("lost in the middle") but **length-driven**. Chroma tested 18 models (including all modern frontier models). NVIDIA recommends keeping prompt under **1/3 of stated window**. For a 32K context model, effective reliable capacity is ~10K tokens — about 5 tool calls before degradation.

### Critical Failure Mode 3: Format Drift / Tool Calling Fragility

Qwen3-32B: 87% tool-selection accuracy vs GPT-4o's 92%. The gap widens on long chains — a single malformed response at step N cascades through all subsequent steps. Constrained decoding (XGrammar at 40μs mask generation) eliminates parsing errors but has a ~17% "creativity cost."

## The Five-Layer Architecture

```
              ┌─────────────────────────────────────────┐
              │        ENTRY POINT / TASK DISPATCH      │
              └────────────────┬────────────────────────┘
                               │
              ┌────────────────▼────────────────────────┐
 LAYER 1      │         TASK CLASSIFIER                 │
 ROUTING      │  Classify complexity, route to model    │
              │  T1 (<4B) → T2 (4-8B) → T3 (9-12B) →   │
              │  T4 (cloud) cascade with confidence     │
              └────────────────┬────────────────────────┘
                               │
              ┌────────────────▼────────────────────────┐
 LAYER 2      │      PRE-CALL VALIDATION                │
 GUARDRAIL    │  Schema check, loop detection, budget   │
              │  Blocks bad calls BEFORE execution      │
              └────────────────┬────────────────────────┘
                               │
              ┌────────────────▼────────────────────────┐
 LAYER 3      │     CONSTRAINED DECODING ENGINE         │
 OUTPUT       │  XGrammar token masking (40μs)          │
 ENFORCEMENT  │  Guarantee valid JSON/tool call output  │
              └────────────────┬────────────────────────┘
                               │
              ┌────────────────▼────────────────────────┐
 LAYER 4      │       CIRCUIT BREAKER                   │
 LOOP         │  3-state: closed → open → half-open     │
 DETECTION    │  Detect loops, break circuits, escalate  │
              └────────────────┬────────────────────────┘
                               │
              ┌────────────────▼────────────────────────┐
 LAYER 5      │     CONTEXT BUDGET MANAGER              │
 CONTEXT      │  Track tokens, enforce 1/3 rule,        │
 MANAGEMENT   │  compact when >90% of effective capacity │
              └─────────────────────────────────────────┘
```

## Quick Start

### 1. Load this skill

```
skill_view(name='small-model-harness')
```

### 2. Enable the plugin

```bash
hermes plugins enable small-model-harness
```

### 3. Add the MCP server

```bash
hermes mcp add small-model-harness \
  --cmd "python3 ~/.hermes/profiles/dev/mcp/small-model-harness/server.py"
```

### 4. Configure tiers for your available models

Edit `harness.routing.tiers` in the plugin config to match your model stack. Example for a setup with Qwen3-4B (local) + OpenCode Zen (cloud):

```yaml
harness:
  routing:
    tiers:
      t1: ~  # no sub-4B model
      t2:
        models: ["qwen3:8b"]  # or your smallest capable local model
        max_tools: 3
        max_steps: 15
      t3:
        models: ["qwen3-30b-a3b", "ornith-1.0-9b"]
        max_tools: 8
        max_steps: 40
      t4:
        provider: "opencode-zen"
        models: ["opencode/deepseek-v4-flash-free"]
```

### 5. When the circuit breaker fires

Check context and consider escalation:

```python
# The plugin blocks the call and suggests escalation
# Run this MCP tool to check:
tool_result = harness_circuit_break(session_id="...", action="check")
# Returns: {state: "open", cooling_remaining: 42, breaks_this_session: 2}

# To reset:
harness_circuit_break(session_id="...", action="reset")
```

## Workflow: Setting Up a New Small Model

1. **Identify your models** — What's available locally? Qwen3-4B/8B? Phi-4-mini? Gemma 3/4?

2. **Configure tiers** — Classify by effective capability, not parameter count:
   - T1: <4B models for simple extraction/single tool
   - T2: 4-8B for multi-step/standard reasoning
   - T3: 9-12B or MoE for complex work
   - T4: Cloud fallback for the hardest 10%

3. **Set effective context** — For each model, calculate: `effective = stated_window / 3`
   - 32K model → ~10K effective → tightly pace your prompts
   - 128K model → ~42K effective → more room but still bounded
   - 262K model (Ornith) → ~87K effective → approaching useful

4. **Enable circuit breaker** — Always. The cost of one doom loop is loss of the entire session.

5. **Run the awareness check:**
   ```
   harness_context_status(session_id="test")
   # Should show: stated_window, effective_capacity, utilization%, step_count
   ```

## MCP Server Tools Reference

| Tool | Layer | Purpose |
|------|-------|---------|
| `harness_classify_task` | L1 | Classify task complexity → tier suggestion |
| `harness_route` | L1 | Route task to specific model tier |
| `harness_validate_call` | L2 | Validate tool call args against schema |
| `harness_detect_loop` | L4 | Score recent calls for loop patterns |
| `harness_circuit_break` | L4 | Manage circuit breaker state |
| `harness_context_status` | L5 | Show current context budget |
| `harness_compact` | L5 | Trigger context compaction |
| `harness_config` | ALL | Get/set harness configuration |

## Anti-Rationalization Table

| Rationalization | Truth | Consequence |
|----------------|-------|-------------|
| "This model says 128K context, so I can use 100K prompts" | Effective capacity is 1/3 of stated window | Context rot degrades output quality silently |
| "Loops only happen on frontier models" | 22.9% doom loop rate on Qwen3.5-4B | Every ~5th hard problem exhausts context |
| "Format drift won't happen with good prompts" | Even system prompts with examples degrade on long chains | A malformed response at step 12 cascades every subsequent step |
| "Repetition penalty fixes looping" | Repetition penalty damages output quality | Reduced creativity AND doesn't fix the root cause |
| "My small model is good enough, I don't need routing" | 87% vs 92% tool accuracy gap compounds over 20+ steps | 15%+ failure rate on complex chains vs 4% for frontier |
| "Constrained decoding makes output worse" | 17% creativity cost is irrelevant for structured output | Valid JSON is infinitely more useful than creative malformed XML |
| "The harness adds too much latency" | Loop detection: <10ms. Circuit breaker: <1ms. XGrammar: 40μs. | The harness adds ~50ms to a call that takes 2-30s. Negligible. |

## Pitfalls

### 1. False positive loop detection
**Problem:** The harness blocks a legitimate complex reasoning trace that looks like a loop.
**Fix:** The circuit breaker's half-open state allows test requests. If the model recovers, the breaker closes. If thresholds are too sensitive, adjust `loop_threshold` downward (higher = less sensitive).

### 2. Context compaction loses information
**Problem:** Summarizing old steps drops details needed to understand the current state.
**Fix:** The sliding window keeps the last 5 steps intact. Only older steps are summarized. If you need full history, increase `sliding_window_steps` or use the Hermes memory system for persistent state.

### 3. Very long running agent sessions
**Problem:** After 50+ tool calls, even with compaction, context is exhausted.
**Fix:** Use Hermes' native context budget (compression at 85% threshold, target ratio 0.75) as a secondary defense. The harness fires first with its 1/3 rule; the general budget fires second.

### 4. Multi-model inconsistency
**Problem:** Cascading between tiers changes model behavior — T2 might solve something T1 couldn't, or vice versa.
**Fix:** Always pass the full step context on cascade (not just the latest prompt). For T4 cloud cascade, use the same system prompt minus small-model-specific instructions.

### 5. XGrammar backend incompatibility
**Problem:** XGrammar C library may not be compiled into the local llama.cpp backend.
**Fix:** The harness falls back to post-hoc validation with retry. Check `constraint_engine` in `harness_config` to verify XGrammar is active.

### 6. Plugin conflicts with sdlc-quality-hooks
**Problem:** Both plugins modify pre_tool_call with block decisions.
**Fix:** Both return `{"action": "allow"/"block"}`. Hermes evaluates both and blocks if EITHER plugin blocks. The harness adds model-level enforcement alongside sdlc's build-level enforcement. No conflict — they're additive.

## Related Skills

- `local-llm-optimization` — LM Studio configuration, model selection, GPU tuning
- `sdlc-quality-hooks` — Build-level deterministic enforcement (complementary to harness's model-level enforcement)
- `hermes-forge` — General LLM validation / rescue / step ordering
- `context-engineering` — System prompt architecture and context optimization
- `debugging-and-error-recovery` — Debugging methodology for agent failures

## Version

v0.1.0 — July 2026
