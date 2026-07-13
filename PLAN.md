# Small-Model Harness — Hallucination Prevention Plan

**Date:** 2026-07-12
**Context:** Phases 1-3 (plugin, MCP server, context router) complete and pushed to GitHub. This plan covers Phases 4-7 to close the remaining reliability gaps.

---

## Research Summary

Based on July 2026 primary sources (arXiv, ACL, AAAI, production engineering blogs), the hallucination prevention landscape breaks into six complementary layers:

| Technique | Latency | Effectiveness | Local Model Compat? |
|---|---|---|---|
| **Constrained Decoding** (XGrammar/Outlines) | 40μs/token | 100% format compliance | Yes, with logit access |
| **Post-hoc Validation + Retry** (Instructor) | <5ms + retry | 95%+ structural validity | Yes, API-only |
| **Self-Consistency** (multiple samples) | 3-5x tokens | +8-14% accuracy | Yes, but costly |
| **Specification-Grounded Verification** | 1x latency | +18% accuracy | Yes |
| **Semantic Entropy** | 5-10x cost | Detects confabulations | Yes |
| **Confidence-Based Routing** | ~0ms | Tier escalation | Yes |

Key insight from production systems surveyed: **no single technique covers every failure mode.** Production agents use defense-in-depth. The most effective architectures combine structural guarantees (constrained decoding) with semantic verification (spec-grounded checking) at branch points.

---

## Phase 4: Constrained Decoding Engine (Layer 3)

**Goal:** Guarantee valid JSON tool call output from local models. Two-tier strategy since LM Studio doesn't expose logits.

### Tier 1: Post-hoc Validation + Retry (LM Studio, API-only backends)

- Implement **Instructor/Pydantic pattern**: define tool call schemas as Pydantic models, validate every output, retry with error feedback on failure
- Max 3 retries with escalating feedback (error message → schema reminder → example)
- **Edges:** Empty response, malformed JSON, valid JSON but wrong schema, correct schema but invalid values (enum, range)
- **Integration:** Add `validate_and_retry()` as a harness utility called after every tool-call response
- **Expected result:** ~95%+ structural validity on first attempt, ~99%+ with retries

### Tier 2: Token-Level Constrained Decoding (llama.cpp/vLLM backends)

- Implement **XGrammar** integration for models served via llama.cpp or vLLM
- Compile tool call JSON Schema → context-free grammar → token mask at startup
- ~40μs/token overhead, zero structural failures
- **Integration:** Add as optional backend; auto-detect when model is served via compatible endpoint
- **Expected result:** 100% structural validity, 0% parsing errors

### Tier 3: Hybrid

- Tier 2 when available (vLLM/llama.cpp), fallback to Tier 1 (LM Studio API)
- Plugin detects backend type from provider configuration at init time

### Files to create/modify

| File | Action | Description |
|---|---|---|
| `mcp-server/constrained_decode.py` | CREATE | XGrammar integration + schema compilation |
| `hermes-plugin/output_validator.py` | CREATE | Post-hoc validation with Pydantic + retry |
| `hermes-plugin/__init__.py` | MODIFY | Wire output_validator as post_tool_call hook |
| `tests/test_constrained_decode.py` | CREATE | 30+ tests |
| `tests/test_output_validator.py` | CREATE | 30+ tests |

---

## Phase 5: Self-Consistency & Verification (Layer 6 — New)

**Goal:** Catch semantic errors (wrong tool, wrong arguments, hallucinated facts) that structural validation misses.

### Component 5a: Lightweight Self-Consistency

- For T1-T2 tasks: sample 2 responses, compare key fields (tool name, argument values)
- For T3-T4 tasks: sample 3 responses, majority vote on tool selection
- **Cost control:** Only apply at propagation-risk nodes (branch points, destructive operations)
- **Integration:** MCP tool `harness_verify_consistency(task, responses[])` returning consistency score + anomalies

### Component 5b: Specification-Grounded Verification

- Define **verification rubric** for each task: expected output schema, semantic constraints, invariant checks
- Pass to a **dedicated verifier** with fresh context (no contamination from executor context)
- Verifier can be same model at T3+ or a smaller model (phi-4-mini) for T1-T2
- **Key insight from Sherlock research:** verification in parallel with speculative execution → 48.7% time reduction

### Component 5c: Checkpoint-Based Rollback

- Persist verified intermediate states (tool call results, context snapshots)
- On verification failure, roll back to last verified checkpoint
- **Integration:** Context budget manager tracks checkpoint boundaries

### Files to create/modify

| File | Action | Description |
|---|---|---|
| `mcp-server/verification.py` | CREATE | Self-consistency + spec-grounded verification |
| `mcp-server/harness_verify_consistency` tool | ADD | MCP tool for consistency checking |
| `hermes-plugin/verifier.py` | CREATE | Plugin-level verification hooks |
| `mcp-server/checkpoint_manager.py` | CREATE | Checkpoint-based rollback |
| `tests/test_verification.py` | CREATE | 40+ tests |

---

## Phase 6: Confidence Scoring & Semantic Uncertainty

**Goal:** Detect when the model is uncertain or confabulating, before verification.

### Component 6a: Token Probability Aggregation

- Collect per-token log probabilities from LM Studio API (when available)
- Compute: mean probability, min probability, entropy per token
- Flag: bottom-10% probability → "uncertain" status in TaskProfile

### Component 6b: Semantic Entropy (lightweight)

- Sample N=3 responses at temperature 0.7
- Cluster by semantic similarity (embedding-based, using sentence-transformers)
- High dispersion → potential confabulation → escalate to verification

### Component 6c: Confidence Score Integration

- Merge token probability signal + semantic entropy + self-consistency into unified `confidence_score` (0.0-1.0)
- Route: confidence < 0.5 → escalate to T4 or human
- Confidence 0.5-0.7 → run verification
- Confidence > 0.7 → proceed normally

### Integration

- Return confidence_score in `harness_classify_task` TaskProfile
- Plugin pre_verify hook checks confidence before continuing
- MCP tool `harness_estimate_confidence(task, responses[])` for ad-hoc queries

### Files to create/modify

| File | Action | Description |
|---|---|---|
| `mcp-server/confidence.py` | CREATE | Token probability + semantic entropy |
| `routing_commands.py` | MODIFY | Add confidence to TaskProfile |
| `tests/test_confidence.py` | CREATE | 25+ tests |

---

## Phase 7: Guardrails & Defense-in-Depth

**Goal:** Add input/output guardrails for production security, complementing the existing structural layers.

### Component 7a: Input Guardrails

- Prompt injection detection (heuristic: known patterns, separator tokens, system prompt override attempts)
- Jailbreak detection (role-playing, hypothetical framing)
- PII scanning on input (credit cards, SSN, API keys)

### Component 7b: Output Guardrails

- PII scanning on output (data leakage prevention)
- Rate limit enforcement
- Topic boundary enforcement (prevent off-topic tool calls)
- Tool call argument boundary validation (URLs, file paths, numeric ranges)

### Integration

- Plugin pre_tool_call: run input guardrails on user message
- Plugin post_tool_call: run output guardrails on model response
- Implement as pluggable guard providers (regex, NeMo Guardrails integration, custom)

### Files to create/modify

| File | Action | Description |
|---|---|---|
| `hermes-plugin/guardrails.py` | CREATE | Input/output guardrail system |
| `hermes-plugin/guardrails_pii.py` | CREATE | PII patterns and scanning |
| `tests/test_guardrails.py` | CREATE | 35+ tests |

---

## Architecture Evolution

### Before (Phases 1-3)

```
Entry → Router → Validation → [No Constraint] → Circuit Breaker → Budget
```

### After (Phases 4-7)

```
Entry → Router → Validation → Constrained Decode → Confidence Score →
        ┌──────────────────────────────┐
        │  Circuit Breaker + Guardrails │
        └──────────────────────────────┘
        → Verify Consistency → Spec-Grounded Verifier →
        → Confidence > 0.7? → Execute
        → Else → Retry/Escalate/Rollback
        → Context Budget → Checkpoint Save
```

## Implementation Order & Dependencies

```
Phase 4 (Constrained Decode)
  ├─ Tier 1: Post-hoc validation + retry (no deps)
  └─ Tier 2: XGrammar integration (needs lm-format-enforcer or xgrammar pkg)
       └─ Dependency: confirm LM Studio or vLLM backend supports logit masking
            ↓
Phase 6 (Confidence Scoring) — can start after Phase 4 Tier 1
  ├─ Token probability aggregation (needs LM Studio logprobs API)
  └─ Semantic entropy (no deps beyond sentence-transformers)
       ↓
Phase 5 (Self-Consistency & Verification)
  ├─ Spec-grounded verification (needs Phase 6 confidence signal)
  └─ Checkpoint rollback (needs Phase 2 budget manager)
       ↓
Phase 7 (Guardrails) — independent, can run parallel
```

## Key Design Decisions

1. **No single point of failure.** Every layer can degrade gracefully. If XGrammar isn't available, post-hoc validation handles it. If confidence scoring isn't available, verification handles it.

2. **LM Studio is the primary constraint.** It exposes an OpenAI-compatible API but not logits. This means:
   - Phase 4 Tier 1 (post-hoc validation + retry) is the default path
   - Phase 4 Tier 2 (XGrammar) only activates for vLLM/llama.cpp backends
   - Token probability signals (Phase 6) depend on `logprobs=true` in API calls (LM Studio supports this)

3. **Cost-conscious verification.** Self-consistency at 3-5x cost is applied selectively (high-risk nodes only), not every step. Most tasks use spec-grounded verification at 1x cost.

4. **Checkpoint cost.** Checkpoints store only references (session_id, step hash) not full content. DB storage is negligible.

## Test Strategy

| Phase | New Tests | Key Behaviors |
|---|---|---|
| P4 | 60+ | Schema validation, retry loop, XGrammar compile, hybrid mode |
| P5 | 40+ | Consistency scoring, verifier rubric, checkpoint save/rollback |
| P6 | 25+ | Probability aggregation, entropy calculation, confidence threshold |
| P7 | 35+ | Input/output blocking, PII detection, rate limiting |

Total: ~160 new tests across 4 phases.

## Success Metrics

| Metric | Current | Target | Phase |
|---|---|---|---|
| Structural validity (tool call JSON) | ~87% (Qwen3-32B baseline) | >99% | P4 |
| Semantic correctness | ~87% tool selection | >93% | P5 |
| Loop detection | ~1% (after antidoom) | <0.5% | P5+P6 |
| False positive rate (verification) | N/A | <5% | P5 |
| Guard coverage (known attacks) | None | >90% | P7 |
| Latency overhead (average) | 0ms (no verification) | <150ms | P4-P7 |
