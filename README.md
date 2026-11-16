# agent-trace-replay

**A checkpoint lets you resume; only a trace lets you replay.**

When a tool-calling agent fails in production, you cannot debug it by re-running it live or resuming from its last state checkpoint: the first run already mutated the world (`reserve_stock`, `create_ticket`, `send_notice`, `kv_put`), and prices, inventory, and service latencies have drifted since the incident.

Across a cohort of 2000 seeded order-desk runs that produce 466 failed or non-standard runs (a cohort failure rate of 23.30%):

- **Re-running failed incidents live is a second incident:** re-running those 466 failures after 3 epochs of world drift fires 1153 duplicate write side effects (487 duplicate tickets, 128 duplicate stock reservations, 105 duplicate customer notices, and 433 duplicate KV audit writes), flips 103 failures into phantom successes, and reproduces the exact trajectory in only 7 of 466 cases (a trajectory reproduction rate of 1.50%).
- **Resuming from a pre-failure checkpoint is not replay:** resuming those same 466 incidents from their pre-failure checkpoint against the drifted world re-fires 735 live write side effects and quietly turns 33 incidents into phantom successes (`0.0%` exact state reproduction). Replaying their captured L2 traces achieves 100.0% bit-identical reproduction with 0 re-fired writes.

---

## Quickstart (Reproduce Every Number in One Command, Zero API Keys)

```bash
git clone <your-repo-url>
cd agent-trace-replay
make verify
```

To reproduce the headline Checkpoint Resume vs. Trace Replay table directly:

```bash
PYTHONPATH=src python3 experiments/exp2_checkpoint_vs_trace.py
```

---

## Headline Results

### 1. Checkpoint Resume vs. Trace Replay (Experiment 2)

Resuming from the pre-failure state snapshot (`resume_from_checkpoint`) vs. replaying the captured L2 trace (`replay_trace`) across all 466 failed runs after 3 epochs of drift (`docs/experiments/exp2_checkpoint_vs_trace.json`):

| Failure Kind | Incidents | Checkpoint Resume Exact | Checkpoint Phantom Successes | Checkpoint Re-Fired Writes | L2 Trace Replay Exact | L2 Trace Re-Fired Writes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deadline_exceeded` | 236 | 0 (0.0%) | 0 | 236 | 236 (100.0%) | 0 |
| `budget_escalated` | 131 | 0 (0.0%) | 14 | 289 | 131 (100.0%) | 0 |
| `stock_backordered` | 75 | 0 (0.0%) | 0 | 149 | 75 (100.0%) | 0 |
| `retry_exhausted` | 24 | 0 (0.0%) | 19 | 61 | 24 (100.0%) | 0 |
| **TOTAL** | **466** | **0 (0.0%)** | **33 (7.08%)** | **735** | **466 (100.0%)** | **0** |

Why checkpoint resume lies during debugging:
- When a `retry_exhausted` failure is resumed from its pre-failure snapshot, the transient flakiness draw is keyed on the new attempt, so 19 of 24 incidents (`79.17%`) succeed on resume and fire downstream fulfillment writes.
- When a `budget_escalated` incident is resumed after 3 epochs of price drift, downward price moves bring 14 of 131 quotes (`10.69%`) back under budget, turning a real escalation incident into a silent fulfillment that double-bills inventory and notifications.

### 2. The Capture Ladder: What Each Level Costs and What It Buys (Experiment 3)

We record each run once at full capture and project down to `L0`, `L1`, `L2`, and `L3` (`docs/experiments/exp3_capture_ladder.json`):

| Capture Level | Mean Trace Size | Overhead vs L0 | Exact Replay Rate | Aborted (Safe Mode) | Diverged | Attributed Divergence Causes |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **L0** (Transcript) | 2241.5 B | 1.00x | 0 / 466 (0.0%) | 255 (54.72%) | 211 (45.28%) | `uncaptured_result`: 255, `drifted_read`: 211 |
| **L1** (+ Tool Results) | 3112.8 B | 1.39x | 0 / 466 (0.0%) | 0 (0.00%) | 466 (100.00%) | `rng`: 312, `clock`: 154 |
| **L2** (+ Ambient Draws & Clocks) | 3943.0 B | 1.76x | 466 / 466 (100.0%) | 0 (0.00%) | 0 (0.00%) | None (100% bit-identical) |
| **L3** (+ Per-Step Fingerprints) | 5151.6 B | 2.30x | 466 / 466 (100.0%) | 0 (0.00%) | 0 (0.00%) | None + instant step-level bug localization |

- **At L0**, safe mode aborts 255 of 466 runs the moment the first uncaptured `WRITE` tool is reached, and 211 diverge on drifted reads (`get_price` or `get_stock` returning post-incident values).
- **At L1**, tool results are served from the trace so zero writes fire and zero reads drift, but 154 diverge on service-time clock checks and 312 diverge on ambient RNG tie-breaks (`lookup` related-document selection).
- **At L2**, adding the ambient values consumed by the framework (`service_ms`, `backoff_ms`, `tie_break_index`) costs only `1.76x` the byte size of an L0 transcript (`3,943.0 B` vs. `2,241.5 B`) and reaches `100.0%` bit-identical replay.

#### Why Volatile Observability Fields Poison Fingerprints

A state fingerprint is computed as SHA-256 over the canonical sorted-key JSON of the agent state after stripping `VOLATILE_STATE_KEYS` (`last_call_latency_ms`, `step_wall_timestamp_ms`, `attempt_tag`). If you do not exclude volatile fields before hashing, all 2000 of 2,000 replays diverge at Step 0 (`plan`) with `DivergenceCause.VOLATILE_FINGERPRINT` before a single tool is even called.

### 3. Bug Localization via L3 Per-Step Fingerprints (Experiment 4)

Why pay the extra `1,208.6 B` per trace from L2 (`3,943.0 B`) to L3 (`5,151.6 B`)? We injected four subtle engine bugs behind `MutationFlags` and replayed all 2,000 traces (`docs/experiments/exp4_localization.json`):

| Engine Mutation | Affected Runs | L3 Pinned Exact First Step | Mean Steps Earlier Than End | Invisible to Final-Status Check |
| :--- | ---: | ---: | ---: | ---: |
| `dropped_state_field` (drops `priority_tag` in `lookup`) | 1,998 | 1,998 (100.0%) | 5.65 steps | 1,998 / 1,998 (100.0%) |
| `skipped_backoff` (omits retry sleep on transient error) | 1,203 | 1,203 (100.0%) | 3.27 steps | 1,024 / 1,203 (85.12%) |
| `fudged_budget_threshold` (+5% budget slack in `price`) | 42 | 42 (100.0%) | 2.00 steps | 0 / 42 (0.0%) |
| `extra_retry` (+1 retry budget on transient error) | 24 | 24 (100.0%) | 1.00 steps | 0 / 24 (0.0%) |

- `dropped_state_field` corrupts `selected_doc` at Step 1 (`lookup`), which is invisible to final status in 1998 of 1,998 affected runs (`100.0%`), whereas L3 pins it at Step 1—catching the bug 5.65 steps earlier on average.
- `skipped_backoff` alters `elapsed_ms` whenever a transient error occurs, yet 85.12% of affected runs still finish with the same final status (`1,024 / 1,203`), masking the regression unless per-step L3 hashes are checked.

### 4. Drift Sweep & Analytic Trajectory Survival Model (Experiment 5)

Sweeping the world drift gap from `E+0` to `E+10` epochs between incident and debug (`docs/experiments/exp5_drift_sweep.json`, plotted in `docs/experiments/drift_sweep.svg`):

- Measured live rerun trajectory reproduction starts at 5.79% at E+0 (due to fresh flakiness, tie-break, and service-time draws) and decays to 1.93% at E+3 as item prices drift away from the recorded epoch.
- Our structured analytic survival model predicts 1.99% at E+3 (`src/agent_trace_replay/analytic.py`—which deduplicates price moves per item, accounts for stock bookkeeping, and runs a 5ms-grid dynamic program for the deadline guard), matching the measured `1.93%` closely, whereas a naive per-step hazard model predicts 0.55% because it falsely multiplies independent drift hazards across steps that read the same item.
- Across all 11 drift epochs (`E+0` through `E+10`), L2 trace replay remains completely flat at `100.0%`.

### 5. LangGraph `StateGraph` + `InMemorySaver` Integration (Experiment 6)

In `src/agent_trace_replay/langgraph_bridge.py` and `experiments/exp6_langgraph.py`, we wire the same order-desk tools and policy into a `StateGraph` with an `InMemorySaver` checkpointer and our `TraceGateway`. Across 200 seeded runs, InMemorySaver checkpoint resumption fires 200 duplicate writes when resuming from the pre-final node, while `TraceGateway` replay achieves `200 / 200 (100.0%)` bit-identical state hashes with `0` writes fired.

---

## Architecture & Repository Layout

```text
agent-trace-replay/
├── src/agent_trace_replay/
│   ├── rng.py               # SHA-256 keyed draw(seed, *key) -> [0, 1) with type-tagged namespaces
│   ├── world.py             # OrderDeskWorld: 12 SKUs, epoch price drift, flakiness, effect_log billing
│   ├── tools.py             # EffectClass (PURE, READ, WRITE) & strict ToolRegistry
│   ├── agent.py             # Hand-rolled graph loop, deadline guard, retries, volatile-stripped SHA-256
│   ├── trace.py             # Append-only JSONL serializer, L0-L3 projector, corruption parser
│   ├── replay.py            # ReplayDriver (EXACT/DIVERGED/ABORTED), rerun_live, resume_from_checkpoint
│   ├── analytic.py          # Keyed trajectory survival model + deadline DP vs naive hazard model
│   └── langgraph_bridge.py  # StateGraph + InMemorySaver + TraceGateway integration
├── experiments/             # 6 reproducible experiment scripts writing docs/experiments/*.json
├── tools/
│   ├── run_tests.py         # Unittest + sys.settrace line coverage runner -> docs/test_report.json
│   ├── collect_metrics.py   # Compiles machine-measured figures -> docs/metrics.json
│   ├── check_numbers.py     # Anchor-phrase figure guard + self-test sabotaging a figure
│   └── lint.py              # Offline AST, docstring, and formatting verifier
├── docs/
│   ├── adr/                 # 5 Architecture Decision Records (0001 through 0005)
│   ├── journal.md           # Session-by-session engineering build journal
│   ├── metrics.json         # Machine-generated single source of truth for all published figures
│   └── experiments/         # JSON outputs + drift_sweep.svg plot
└── tests/                   # Property test suite named as full-sentence behavioral specifications
```

The entire codebase is verified by 329 automated test checks (test methods + parameterized subtests) with 88.78% line coverage (`docs/test_report.json`), and every number quoted in this README is enforced by `tools/check_numbers.py`.

---

## What This Project Cannot Measure (And Out-of-Scope Boundaries)

1. **Real LLM token-sampling nondeterminism:** We model stochastic policy branching via keyed choice points (`tight_stock_policy`, `notice_template`, and document tie-breaks). While this captures orchestration branching under nondeterministic choices, it does not measure GPU floating-point batching variance or hosted model weight updates.
2. **External API schema drift:** `OrderDeskWorld` models value drift (prices and stock levels changing over epochs), not a third-party vendor renaming a JSON field mid-week.
3. **Multi-agent and concurrent interleavings (Out of Scope on purpose):** This engine targets single-agent sequential tool calls. Concurrent traces where two agents interleave writes on a shared table are out of scope for v1; the trigger that would justify v2 is the first production trace exhibiting cross-agent write causality (where an OpenTelemetry span importer with Lamport clocks becomes necessary).
