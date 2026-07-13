# Small-Model Harness

**Defensive harness for running 1B–12B parameter models in production agentic workflows.**

Five-layer architecture: routing → validation → constraint → circuit break → context management. Delivered as a Hermes plugin (deterministic enforcement) + MCP server (analysis tools).

## The Problem

Small models (1B–12B) have dramatically closed the quality gap with frontier models, but the **reliability gap persists**:

| Failure Mode | Impact | Source |
|---|---|---|
| **Doom Loops** — repetitive death spirals under greedy sampling | Qwen3.5-4B: **22.9% loop rate** | Antidoom (FTPO) |
| **Context Rot** — degradation with input length, not position | Effective window = ~1/3 of stated window | Chroma/NVIDIA |
| **Format Drift** — tool call fragility on long chains | 87% accuracy vs GPT-4o's 92% | Qwen3-32B eval |

## Architecture

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
              │  (NOT YET IMPLEMENTED)                  │
              └────────────────┬────────────────────────┘
                               │
              ┌────────────────▼────────────────────────┐
 LAYER 4      │       CIRCUIT BREAKER                   │
 LOOP         │  3-state: closed → open → half-open     │
 DETECTION    │  Detect loops, break circuits, escalate  │
              └────────────────┬────────────────────────┘
                               │
              ┌────────────────▼────────────────────────┐
 LAYER 5      │      CONTEXT BUDGET                     │
 CONTEXT      │  Sliding window compaction              │
              │  1/3 effective window rule              │
              └─────────────────────────────────────────┘
```

## Components

| Component | What It Does | Status |
|---|---|---|
| **Hermes Plugin** | `pre_tool_call` hooks — schema validation, loop detection, circuit breaker, context budget, routing awareness | ✅ Phase 1 |
| **MCP Server** | 5 tools: context status, compaction, task classification, routing, reset | ✅ Phase 2+3 |
| **Task Classifier** | Rule-based complexity scoring and model tier assignment | ✅ Phase 3 |
| **Context Router** | Tier cascade (T1→T2→T3→T4) with failure-based escalation | ✅ Phase 3 |
| **Output Enforcement** | Constrained decoding (XGrammar/Outlines) for guaranteed valid output | ⏳ Planned |
| **Output Verifier** | Post-generation validation of tool call correctness | 📋 Planned |

## Installation

### As a Hermes Plugin

```bash
# Copy plugin to Hermes plugins directory
cp -r hermes-plugin ~/.hermes/profiles/dev/plugins/small-model-harness

# Enable it
hermes plugins enable small-model-harness
```

### As an MCP Server

Add to `~/.hermes/config.yaml` or `~/.hermes/profiles/dev/config.yaml`:

```yaml
mcp_servers:
  small-model-harness:
    command: python3
    args: ["/path/to/mcp-server/server.py"]
    enabled: true
```

## MCP Server Tools

| Tool | Description |
|---|---|
| `harness_context_status` | Query context budget utilization for a session |
| `harness_compact` | Compact session context (sliding window summarization) |
| `harness_classify_task` | Classify task complexity and suggest model tier |
| `harness_route` | Route task to model tier with cascade logic |
| `harness_reset` | Reset all harness state for a session |

## Tier Reference

| Tier | Model Size | Example Models | Use Case |
|---|---|---|---|
| T1 | <4B | SmolLM3, phi-4-mini | Simple extraction, classification |
| T2 | 4-8B | Qwen3-8B, Llama 3.2-8B | Single tool calls, basic routing |
| T3 | 9-12B | Ornith-1.0-9B, Qwen3-30B-A3B | Multi-step, reasoning, planning |
| T4 | Cloud | DeepSeek V4, GPT-4o | Complex chains, security-critical |

## Development

```bash
# Setup
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Test MCP server
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python3 mcp-server/server.py
```

## License

MIT
