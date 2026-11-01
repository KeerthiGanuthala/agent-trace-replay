# ADR 0004: SHA-256 Fingerprints Over Canonical JSON Excluding Volatile Observability Fields

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

At capture level L3, we want to detect and localize state corruption or orchestration bugs to the exact step where the agent's state first diverged from the recorded run, without storing a full deep copy of the state dictionary at every step.

## Decision

After each step, we serialize the agent state to canonical sorted-key JSON (`json.dumps(..., sort_keys=True, separators=(",", ":"))`) after stripping `VOLATILE_STATE_KEYS` (`last_call_latency_ms`, `step_wall_timestamp_ms`, `attempt_tag`) and record its 64-character SHA-256 hex digest as `state_fingerprint`.

## Alternatives Considered

### 1. Storing full state snapshots at every step

Storing full state copies turns a trace into a sequence of checkpoints, bloating trace size linearly with state payload size. Hashing canonical JSON adds only 1,208.6 bytes per run on average from L2 (3,943.0 B) to L3 (5,151.6 B)—a 2.30x ratio over L0—while pinning 100.0% of state-corrupting mutations (`dropped_state_field`, `skipped_backoff`, `fudged_budget_threshold`, `extra_retry`) at the exact step they occur.

### 2. Hashing the raw state dictionary without excluding volatile fields

We measured the exact cost of leaving `step_wall_timestamp_ms`, `last_call_latency_ms`, and `attempt_tag` inside the state hash in Experiment 3: including volatile fields causes 2000 out of 2,000 L3 replays to fail at Step 0 (`plan`) with `DivergenceCause.VOLATILE_FINGERPRINT`, even when the execution is otherwise bit-identical.

## Consequences

- **Positive:** Compact 64-byte hex digests per step that are invariant to wall-clock replay time and session IDs, yet immediately catch silent state drops (`dropped_state_field` is caught 5.65 steps before completion despite being 100.0% invisible to final status checks).
- **Negative:** Maintaining `VOLATILE_STATE_KEYS` requires discipline whenever new telemetry counters are added to the state dictionary.
