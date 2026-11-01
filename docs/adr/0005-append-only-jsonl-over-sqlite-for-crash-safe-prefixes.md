# ADR 0005: Append-Only JSONL Trace Files Over SQLite

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Agent failures frequently involve unhandled exceptions, OOM kills, or hard process timeouts mid-run. The trace storage format must preserve every completed step up to the crash point and remain inspectable with standard Unix tools (`head`, `jq`, `grep`, `git diff`).

## Decision

We store traces as newline-delimited JSON (JSONL): one `header` line, one `step` line flushed per completed step, and one `end` line upon termination (`src/agent_trace_replay/trace.py`). `parse_trace(..., allow_partial=True)` validates and reconstructs a partial `TraceBundle` from any truncated prefix whose `end` record is missing.

## Alternatives Considered

### SQLite database per trace or shared SQLite WAL file

SQLite is binary, cannot be piped directly through line-oriented CLI filters, and if a worker process dies mid-transaction or corrupts a journal file on a network filesystem, recovering the prefix requires database repair tools. Append-only JSONL guarantees that every newline-terminated line flushed before a crash is a complete, self-contained JSON object.

## Consequences

- **Positive:** Human-readable diffs, zero external C-extension dependencies, strict line-by-line corruption diagnostics (`TraceCorruptionError` reports the exact 1-based line number and schema violation), and clean partial-prefix recovery.
- **Negative:** Cross-run analytics require scanning JSONL files rather than running SQL queries directly against a relational store.
