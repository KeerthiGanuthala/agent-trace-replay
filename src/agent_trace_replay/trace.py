"""Append-only JSONL trace format supporting four progressive capture levels (L0-L3)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from agent_trace_replay.agent import RunOutcome, StepRecord, ToolAttemptRecord


class CaptureLevel(IntEnum):
    """Progressive trace fidelity levels."""

    L0 = 0  # Transcript: actions, chosen options, attempt error pattern
    L1 = 1  # + Every tool result (reads and writes included)
    L2 = 2  # + Ambient values consumed by framework (service timings, tie-break draws)
    L3 = 3  # + Canonical state fingerprint after every step


class TraceCorruptionError(ValueError):
    """Raised when a JSONL trace stream is malformed, truncated, or fails schema checks."""


@dataclass
class TraceHeader:
    """Header record starting a JSONL trace file."""

    schema_version: str
    seed: int
    attempt_tag: str
    capture_level: CaptureLevel
    order_id: str


@dataclass
class TraceEnd:
    """Trailer record concluding a completed JSONL trace file."""

    status: str
    failure_kind: str | None
    step_count: int
    final_fingerprint: str | None


@dataclass
class TraceBundle:
    """Parsed in-memory representation of a trace at a given CaptureLevel."""

    header: TraceHeader
    steps: list[dict[str, Any]]
    end: TraceEnd | None
    partial: bool = False


def outcome_to_full_trace(outcome: RunOutcome) -> TraceBundle:
    """Convert a completed `RunOutcome` into a full L3 `TraceBundle`."""
    header = TraceHeader(
        schema_version="1.0",
        seed=outcome.seed,
        attempt_tag=outcome.attempt_tag,
        capture_level=CaptureLevel.L3,
        order_id=f"ORD-{outcome.seed:05d}",
    )
    step_dicts: list[dict[str, Any]] = []
    for s in outcome.steps:
        attempts_list = []
        for a in s.tool_attempts:
            attempts_list.append(
                {
                    "tool_name": a.tool_name,
                    "effect": a.effect,
                    "args": copy.deepcopy(a.args),
                    "call_index": a.call_index,
                    "attempt_index": a.attempt_index,
                    "error": a.error,
                    "result": copy.deepcopy(a.result),
                    "service_ms": a.service_ms,
                    "backoff_ms": a.backoff_ms,
                }
            )
        step_dicts.append(
            {
                "record_type": "step",
                "step_index": s.step_index,
                "node": s.node,
                "next_node": s.next_node,
                "choice_made": s.choice_made,
                "tie_break_index": s.tie_break_index,
                "elapsed_before_ms": s.elapsed_before_ms,
                "elapsed_after_ms": s.elapsed_after_ms,
                "tool_attempts": attempts_list,
                "state_fingerprint": s.state_fingerprint,
                "raw_fingerprint_with_volatile": s.raw_fingerprint_with_volatile,
            }
        )
    end = TraceEnd(
        status=outcome.status,
        failure_kind=outcome.failure_kind,
        step_count=len(outcome.steps),
        final_fingerprint=outcome.final_fingerprint,
    )
    return TraceBundle(header=header, steps=step_dicts, end=end, partial=False)


def project_trace(bundle: TraceBundle, level: CaptureLevel) -> TraceBundle:
    """Emit a lower-level view of the exact same recorded run without re-executing."""
    if level > bundle.header.capture_level:
        raise ValueError(
            f"Cannot project trace from {bundle.header.capture_level.name} up to {level.name}"
        )
    header = TraceHeader(
        schema_version=bundle.header.schema_version,
        seed=bundle.header.seed,
        attempt_tag=bundle.header.attempt_tag,
        capture_level=level,
        order_id=bundle.header.order_id,
    )
    projected_steps: list[dict[str, Any]] = []
    for s in bundle.steps:
        proj_attempts: list[dict[str, Any]] = []
        for a in s["tool_attempts"]:
            att: dict[str, Any] = {
                "tool_name": a["tool_name"],
                "effect": a["effect"],
                "args": copy.deepcopy(a["args"]),
                "call_index": a["call_index"],
                "attempt_index": a["attempt_index"],
                "error": a["error"],
            }
            if level >= CaptureLevel.L1:
                att["result"] = copy.deepcopy(a.get("result"))
            if level >= CaptureLevel.L2:
                att["service_ms"] = a.get("service_ms")
                att["backoff_ms"] = a.get("backoff_ms")
            proj_attempts.append(att)

        step_entry: dict[str, Any] = {
            "record_type": "step",
            "step_index": s["step_index"],
            "node": s["node"],
            "next_node": s["next_node"],
            "choice_made": s["choice_made"],
            "tool_attempts": proj_attempts,
        }
        if level >= CaptureLevel.L2:
            step_entry["tie_break_index"] = s.get("tie_break_index")
            step_entry["elapsed_before_ms"] = s.get("elapsed_before_ms")
            step_entry["elapsed_after_ms"] = s.get("elapsed_after_ms")
        if level >= CaptureLevel.L3:
            step_entry["state_fingerprint"] = s.get("state_fingerprint")
            step_entry["raw_fingerprint_with_volatile"] = s.get("raw_fingerprint_with_volatile")
        projected_steps.append(step_entry)

    end: TraceEnd | None = None
    if bundle.end is not None:
        end = TraceEnd(
            status=bundle.end.status,
            failure_kind=bundle.end.failure_kind,
            step_count=bundle.end.step_count,
            final_fingerprint=bundle.end.final_fingerprint if level >= CaptureLevel.L2 else None,
        )
    return TraceBundle(header=header, steps=projected_steps, end=end, partial=bundle.partial)


def serialize_trace(bundle: TraceBundle) -> str:
    """Serialize a `TraceBundle` into canonical append-only JSONL text."""
    lines: list[str] = []
    header_Line = {
        "record_type": "header",
        "schema_version": bundle.header.schema_version,
        "seed": bundle.header.seed,
        "attempt_tag": bundle.header.attempt_tag,
        "capture_level": int(bundle.header.capture_level),
        "order_id": bundle.header.order_id,
    }
    lines.append(json.dumps(header_Line, sort_keys=True, separators=(",", ":")))
    for s in bundle.steps:
        lines.append(json.dumps(s, sort_keys=True, separators=(",", ":")))
    if bundle.end is not None:
        end_line = {
            "record_type": "end",
            "status": bundle.end.status,
            "failure_kind": bundle.end.failure_kind,
            "step_count": bundle.end.step_count,
            "final_fingerprint": bundle.end.final_fingerprint,
        }
        lines.append(json.dumps(end_line, sort_keys=True, separators=(",", ":")))
    return chr(10).join(lines) + chr(10)


def parse_trace(jsonl_text: str, allow_partial: bool = False) -> TraceBundle:
    """Parse and validate a JSONL trace string, refusing corrupted files with a specific reason."""
    raw_lines = [ln for ln in jsonl_text.splitlines() if ln.strip()]
    if not raw_lines:
        raise TraceCorruptionError("Empty trace: missing header record")

    records: list[dict[str, Any]] = []
    for idx, line in enumerate(raw_lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceCorruptionError(f"Invalid JSON on line {idx + 1}: {exc.msg}") from exc
        if not isinstance(obj, dict) or "record_type" not in obj:
            raise TraceCorruptionError(f"Missing 'record_type' on line {idx + 1}")
        records.append(obj)

    first = records[0]
    if first.get("record_type") != "header":
        raise TraceCorruptionError(
            f"First record must be 'header', found '{first.get('record_type')}'"
        )
    if first.get("schema_version") != "1.0":
        raise TraceCorruptionError(
            f"Unsupported schema_version: {first.get('schema_version')!r}"
        )
    try:
        lvl = CaptureLevel(int(first["capture_level"]))
        seed = int(first["seed"])
        attempt_tag = str(first["attempt_tag"])
        order_id = str(first["order_id"])
    except (KeyError, ValueError, TypeError) as exc:
        raise TraceCorruptionError(f"Malformed header fields: {exc}") from exc

    header = TraceHeader(
        schema_version="1.0",
        seed=seed,
        attempt_tag=attempt_tag,
        capture_level=lvl,
        order_id=order_id,
    )

    steps: list[dict[str, Any]] = []
    end_rec: TraceEnd | None = None

    for idx, rec in enumerate(records[1:], start=2):
        rtype = rec.get("record_type")
        if rtype == "step":
            if end_rec is not None:
                raise TraceCorruptionError(f"Step record found after end record on line {idx}")
            expected_step_idx = len(steps)
            if rec.get("step_index") != expected_step_idx:
                raise TraceCorruptionError(
                    f"Non-contiguous step_index on line {idx}: expected {expected_step_idx}, got {rec.get('step_index')}"
                )
            if "node" not in rec or "next_node" not in rec or "tool_attempts" not in rec:
                raise TraceCorruptionError(f"Step record on line {idx} missing required keys")
            if lvl >= CaptureLevel.L3 and not rec.get("state_fingerprint"):
                raise TraceCorruptionError(
                    f"L3 trace missing state_fingerprint on step {expected_step_idx}"
                )
            steps.append(rec)
        elif rtype == "end":
            if end_rec is not None:
                raise TraceCorruptionError(f"Duplicate end record on line {idx}")
            if rec.get("step_count") != len(steps):
                raise TraceCorruptionError(
                    f"End record step_count ({rec.get('step_count')}) does not match steps ({len(steps)})"
                )
            end_rec = TraceEnd(
                status=str(rec.get("status")),
                failure_kind=rec.get("failure_kind"),
                step_count=int(rec["step_count"]),
                final_fingerprint=rec.get("final_fingerprint"),
            )
        else:
            raise TraceCorruptionError(f"Unknown record_type '{rtype}' on line {idx}")

    if end_rec is None and not allow_partial:
        raise TraceCorruptionError("Truncated trace: missing 'end' record")

    return TraceBundle(header=header, steps=steps, end=end_rec, partial=end_rec is None)
