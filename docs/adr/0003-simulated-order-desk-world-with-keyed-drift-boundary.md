# ADR 0003: Simulated Order-Desk World With Keyed SHA-256 Drift vs. External APIs

- **Status:** Accepted
- **Date:** 2026-10-26

## Context

To measure how world drift degrades live rerun and checkpoint-resume reproducibility across thousands of incidents, we needed an environment with mutable tables, transient flakiness, and time-evolving state.

## Decision

We built a self-contained, deterministic `OrderDeskWorld` (`src/agent_trace_replay/world.py`) backed by `draw(seed, *key)` (`src/agent_trace_replay/rng.py`), which hashes canonicalized key tuples with SHA-256 into `[0, 1)`. Item prices at epoch `E` are a pure function of `(seed, item, E)`, transient tool errors are keyed on `(seed, "flaky", attempt_tag, call_index, tool_name)`, and every write appends an `EffectEntry` billed to `attempt_tag`.

## Alternatives Considered

### Recording against live SaaS APIs or local Dockerized microservices

Live APIs require API keys, incur network variance that depends on the reviewer's machine and location, and make a 2,000-run drift sweep slow and non-reproducible. Keyed SHA-256 streams give independent namespaces for price drift, service latency jitter, flakiness, and tie-break choices without any global PRNG state.

## What Transfers and What Does Not

- **Transfers directly to production:** The capture ladder boundary (why L0 and L1 fail when orchestration branches on service time or tie-breaks), the checkpoint-vs-trace side-effect hazard, and volatile field exclusion in state hashing.
- **Cannot be claimed from this simulation:** Real LLM token-sampling drift across model weight updates, third-party API schema breaking changes, and multi-agent write races on shared database rows.
