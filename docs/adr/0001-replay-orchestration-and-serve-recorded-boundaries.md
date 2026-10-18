# ADR 0001: Replay Orchestration While Serving Recorded Model Choices and Tool Attempts

- **Status:** Accepted
- **Date:** 2026-10-18

## Context

When a tool-calling agent fails in production, engineers want to step through the agent graph to inspect branching logic, state transitions, retry loops, and deadline guards. Two competing approaches exist:

1. **Re-invoke the model at `temperature=0` against live or mocked tools.**
2. **Re-execute the real orchestration code while serving recorded model choices, tool attempt outcomes, and ambient framework draws from a trace.**

## Decision

We re-execute the actual graph orchestration loop (`execute_agent_with_driver` in `src/agent_trace_replay/agent.py`) while injecting a `ReplayDriver` that serves recorded stochastic choice points, recorded tool attempt sequences (errors and results), and (at L2+) recorded service timings and tie-break draws.

## Alternatives Considered

### Re-invoking the LLM at `temperature=0`

Even at `temperature=0`, re-invoking a model or re-sampling choice points after prompt, state, or tool-output drift debugs a *different* trajectory than the one that paged the on-call engineer. In our order-desk agent, the two stochastic choice points (`tight_stock_policy` and `notice_template`) and the document tie-break draw stand in for sampled model decisions. At capture level L1 (where tool results are served from the trace but ambient tie-break draws and service timings are re-drawn), 312 out of 466 failed runs diverge on RNG draws and 154 diverge on clock checks.

## Consequences

- **Positive:** We test the real orchestration code path (including conditional edges, retry counters, and state transformations) with 100.0% determinism at L2 and L3.
- **Negative:** Replay verifies how the orchestrator handles the recorded world and model responses; it cannot tell you whether an updated prompt would have caused an external LLM to emit a different tool call without starting a fresh live run.
