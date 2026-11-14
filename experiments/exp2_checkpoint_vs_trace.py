#!/usr/bin/env python3
"""Experiment 2: Checkpoint vs Trace (Headline Table).

Resume each failure from its last pre-failure snapshot against the live drifted
world, versus replaying its L2 trace. Report re-fired writes, phantom successes,
and reproduction rate for each, broken down per failure kind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_trace_replay.agent import AgentConfig
from agent_trace_replay.replay import ReplayStatus, replay_trace, resume_from_checkpoint
from agent_trace_replay.trace import CaptureLevel, project_trace
from experiments._cohort import COHORT_SIZE, DRIFT_EPOCHS_INCIDENT_TO_DEBUG, build_cohort


def _select_pre_failure_snapshot(item):  # type: ignore[no-untyped-def]
    """Select the pre-failure checkpoint snapshot at the step that triggered the failure or escalation."""
    snaps = item.outcome.snapshots
    fk = item.outcome.failure_kind
    if fk in ("budget_escalated", "stock_backordered") and len(snaps) >= 3:
        # Step -1 is `finish`, Step -2 is `escalate`/`backorder`, Step -3 is `price`/`stock` where the branch failed
        return snaps[-3]
    if len(snaps) >= 2:
        return snaps[-2]
    return snaps[0]


def main() -> int:
    cohort = build_cohort(COHORT_SIZE)
    cfg = AgentConfig()
    failed_runs = [r for r in cohort if r.outcome.failure_kind is not None]

    kinds = ("deadline_exceeded", "budget_escalated", "stock_backordered", "retry_exhausted")
    per_kind: dict[str, dict[str, int | float]] = {
        k: {
            "incidents": 0,
            "resume_exact_match": 0,
            "resume_kind_match": 0,
            "resume_phantom_success": 0,
            "resume_refired_writes": 0,
            "trace_exact_match": 0,
            "trace_refired_writes": 0,
        }
        for k in kinds
    }

    for item in failed_runs:
        fk = str(item.outcome.failure_kind)
        bucket = per_kind[fk]
        bucket["incidents"] = int(bucket["incidents"]) + 1

        # 1. Live resume from pre-failure checkpoint snapshot
        w_res = item.world_after_run.clone()
        w_res.advance_epochs(DRIFT_EPOCHS_INCIDENT_TO_DEBUG)
        snap = _select_pre_failure_snapshot(item)
        res_out = resume_from_checkpoint(snap, w_res, cfg, attempt_tag="checkpoint_resume_1")

        dups = w_res.duplicate_effects_between("incident_0", "checkpoint_resume_1")
        bucket["resume_refired_writes"] = int(bucket["resume_refired_writes"]) + len(dups)
        if res_out.final_fingerprint == item.outcome.final_fingerprint:
            bucket["resume_exact_match"] = int(bucket["resume_exact_match"]) + 1
        if res_out.failure_kind == item.outcome.failure_kind:
            bucket["resume_kind_match"] = int(bucket["resume_kind_match"]) + 1
        if res_out.failure_kind is None:
            bucket["resume_phantom_success"] = int(bucket["resume_phantom_success"]) + 1

        # 2. Trace replay at L2
        w_rep = item.world_after_run.clone()
        w_rep.advance_epochs(DRIFT_EPOCHS_INCIDENT_TO_DEBUG)
        l2_bundle = project_trace(item.l3_bundle, CaptureLevel.L2)
        rep_res = replay_trace(
            l2_bundle,
            w_rep,
            cfg,
            attempt_tag="trace_replay_1",
            reference_l3_bundle=item.l3_bundle,
        )
        if rep_res.status == ReplayStatus.EXACT:
            bucket["trace_exact_match"] = int(bucket["trace_exact_match"]) + 1
        bucket["trace_refired_writes"] = int(bucket["trace_refired_writes"]) + rep_res.writes_re_fired

    tot_incidents = len(failed_runs)
    tot_res_exact = sum(int(v["resume_exact_match"]) for v in per_kind.values())
    tot_res_kind = sum(int(v["resume_kind_match"]) for v in per_kind.values())
    tot_res_phantom = sum(int(v["resume_phantom_success"]) for v in per_kind.values())
    tot_res_writes = sum(int(v["resume_refired_writes"]) for v in per_kind.values())
    tot_tr_exact = sum(int(v["trace_exact_match"]) for v in per_kind.values())
    tot_tr_writes = sum(int(v["trace_refired_writes"]) for v in per_kind.values())

    for k, bucket in per_kind.items():
        n = int(bucket["incidents"])
        bucket["resume_exact_rate_pct"] = round(100.0 * int(bucket["resume_exact_match"]) / n, 2) if n else 0.0
        bucket["resume_kind_rate_pct"] = round(100.0 * int(bucket["resume_kind_match"]) / n, 2) if n else 0.0
        bucket["resume_phantom_rate_pct"] = round(100.0 * int(bucket["resume_phantom_success"]) / n, 2) if n else 0.0
        bucket["trace_exact_rate_pct"] = round(100.0 * int(bucket["trace_exact_match"]) / n, 2) if n else 0.0

    data = {
        "experiment": "exp2_checkpoint_vs_trace",
        "cohort_size": COHORT_SIZE,
        "failed_runs": tot_incidents,
        "totals": {
            "incidents": tot_incidents,
            "resume_exact_match": tot_res_exact,
            "resume_exact_rate_pct": round(100.0 * tot_res_exact / tot_incidents, 2),
            "resume_kind_match": tot_res_kind,
            "resume_kind_rate_pct": round(100.0 * tot_res_kind / tot_incidents, 2),
            "resume_phantom_success": tot_res_phantom,
            "resume_phantom_rate_pct": round(100.0 * tot_res_phantom / tot_incidents, 2),
            "resume_refired_writes": tot_res_writes,
            "trace_exact_match": tot_tr_exact,
            "trace_exact_rate_pct": round(100.0 * tot_tr_exact / tot_incidents, 2),
            "trace_refired_writes": tot_tr_writes,
        },
        "per_failure_kind": per_kind,
    }

    out_path = ROOT / "docs" / "experiments" / "exp2_checkpoint_vs_trace.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    print("=== Experiment 2: Checkpoint Resume vs Trace Replay ===")
    print(f"{'Failure Kind':<20} | {'Incidents':>9} | {'Resume Exact':>12} | {'Phantom OK':>10} | {'Resume Writes':>13} | {'Trace Exact':>11} | {'Trace Writes':>12}")
    print("-" * 102)
    for k in kinds:
        b = per_kind[k]
        print(
            f"{k:<20} | {b['incidents']:>9} | "
            f"{b['resume_exact_match']:>5} ({b['resume_exact_rate_pct']:>4.1f}%) | "
            f"{b['resume_phantom_success']:>10} | "
            f"{b['resume_refired_writes']:>13} | "
            f"{b['trace_exact_match']:>4} ({b['trace_exact_rate_pct']:>5.1f}%) | "
            f"{b['trace_refired_writes']:>12}"
        )
    print("-" * 102)
    t = data["totals"]
    print(
        f"{'TOTAL':<20} | {t['incidents']:>9} | "
        f"{t['resume_exact_match']:>5} ({t['resume_exact_rate_pct']:>4.1f}%) | "
        f"{t['resume_phantom_success']:>10} | "
        f"{t['resume_refired_writes']:>13} | "
        f"{t['trace_exact_match']:>4} ({t['trace_exact_rate_pct']:>5.1f}%) | "
        f"{t['trace_refired_writes']:>12}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
