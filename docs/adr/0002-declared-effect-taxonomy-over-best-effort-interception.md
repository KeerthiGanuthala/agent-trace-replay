# ADR 0002: Mandatory Declared Effect Taxonomy (PURE, READ, WRITE) Over Best-Effort Interception

- **Status:** Accepted
- **Date:** 2026-10-24

## Context

During a replay session, re-firing a mutating tool (`reserve_stock`, `create_ticket`, `send_notice`, `kv_put`) against a live environment creates duplicate reservations, duplicate support tickets, and duplicate customer messages. We needed a mechanism to guarantee that a replay never mutates the world, even when fed an incomplete trace (such as L0 transcript logs).

## Decision

Every tool registered in `ToolRegistry` (`src/agent_trace_replay/tools.py`) must declare an explicit `EffectClass`:

- `EffectClass.PURE` (`calc`): Deterministic computation over arguments; safe to execute at any level.
- `EffectClass.READ` (`lookup_order`, `get_price`, `get_stock`): Reads world state without mutating tables; allowed in L0 live fallback (where drift can be observed and attributed to `DivergenceCause.DRIFTED_READ`).
- `EffectClass.WRITE` (`reserve_stock`, `create_ticket`, `send_notice`, `kv_put`): Mutates world tables and appends to `effect_log`.

Looking up or invoking an undeclared tool raises `UndeclaredToolError`. During replay in safe mode (`safe_mode=True`), reaching an uncaptured `WRITE` at L0 immediately aborts the replay (`ReplayStatus.ABORTED`) before the tool function is entered.

## Alternatives Considered

### Best-effort network/HTTP interception or read-only database proxies

Intercepting outbound sockets or wrapping database connections after the fact fails silently when a tool mutates in-memory state, writes to a local queue, or calls an unmocked SDK client. By making `EffectClass` a required registration contract on `ToolRegistry.register()`, any tool without an effect declaration cannot be called by the agent in the first place.

## Consequences

- **Positive:** Zero re-fired writes across all 466 failed runs in L0 safe mode (255 runs cleanly abort at the first uncaptured write, and 211 diverge earlier on drifted reads) and across all 466 runs at L1, L2, and L3.
- **Negative:** Every new tool added to the system requires an explicit effect classification by its author.
