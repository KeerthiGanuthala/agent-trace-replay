#!/usr/bin/env python3
"""Experiment 1: Re-running is a second incident.

Take every failed run in the 2,000-seed cohort. Advance the world 3 epochs.
Re-run each live. Report: how many reproduce the exact trajectory, how many
reproduce even the failure kind, and how many duplicate side effects (tickets,
notices, reservations, kv writes) the debugging session itself fired.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_trace_replay.agent import AgentConfig
from agent_trace_replay.replay import rerun_live
from experiments._cohort import COHORT_SIZE, DRIFT_EPOCHS_INCIDENT_TO_DEBUG, build_cohort


def main() -> int:
    cohort = build_cohort(COHORT_SIZE)
    cfg = AgentConfig()

    kind_counts = Counter(r.outcome.failure_kind or "completed" for r in cohort)
    failed_runs = [r for r in cohort if r.outcome.failure_kind is not None]

    exact_trajectory_matches = 0
    same_failure_kind_matches = 0
    flipped_to_phantom_success = 0

    dup_by_tool: Counter[str] = Counter()
    total_duplicate_writes = 0
    runs_with_duplicate_writes = 0

    for item in failed_runs:
        w_debug = item.world_after_run.clone()
        w_debug.advance_epochs(DRIFT_EPOCHS_INCIDENT_TO_DEBUG)
        rerun_out = rerun_live(w_debug, cfg, attempt_tag="debug_rerun_1")

        orig_nodes = [s.node for s in item.outcome.steps]
        rerun_nodes = [s.node for s in rerun_out.steps]
        if (
            orig_nodes == rerun_nodes
            and rerun_out.status == item.outcome.status
            and rerun_out.failure_kind == item.outcome.failure_kind
            and rerun_out.final_state.get("selected_doc") == item.outcome.final_state.get("selected_doc")
            and rerun_out.final_state.get("unit_price") == item.outcome.final_state.get("unit_price")
            and rerun_out.final_state.get("tight_stock_policy") == item.outcome.final_state.get("tight_stock_policy")
            and rerun_out.final_state.get("notice_template") == item.outcome.final_state.get("notice_template")
            and [len(s.tool_attempts) for s in rerun_out.steps] == [len(s.tool_attempts) for s in item.outcome.steps]
        ):
            exact_trajectory_matches += 1

        if rerun_out.failure_kind == item.outcome.failure_kind:
            same_failure_kind_matches += 1
        if rerun_out.failure_kind is None:
            flipped_to_phantom_success += 1

        dups = w_debug.duplicate_effects_between("incident_0", "debug_rerun_1")
        if dups:
            runs_with_duplicate_writes += 1
        total_duplicate_writes += len(dups)
        for d in dups:
            dup_by_tool[d.tool_name] += 1

    n_failed = len(failed_runs)
    data = {
        "experiment": "exp1_rerun_incident",
        "cohort_size": COHORT_SIZE,
        "drift_epochs": DRIFT_EPOCHS_INCIDENT_TO_DEBUG,
        "failed_runs": n_failed,
        "failure_rate_pct": round(100.0 * n_failed / COHORT_SIZE, 2),
        "cohort_breakdown": dict(sorted(kind_counts.items())),
        "exact_trajectory_reproduced": exact_trajectory_matches,
        "exact_trajectory_rate_pct": round(100.0 * exact_trajectory_matches / n_failed, 2),
        "same_failure_kind_reproduced": same_failure_kind_matches,
        "same_failure_kind_rate_pct": round(100.0 * same_failure_kind_matches / n_failed, 2),
        "phantom_successes_on_rerun": flipped_to_phantom_success,
        "phantom_success_rate_pct": round(100.0 * flipped_to_phantom_success / n_failed, 2),
        "total_duplicate_side_effects": total_duplicate_writes,
        "runs_firing_duplicate_writes": runs_with_duplicate_writes,
        "duplicate_effects_by_tool": {
            "reserve_stock": dup_by_tool.get("reserve_stock", 0),
            "create_ticket": dup_by_tool.get("create_ticket", 0),
            "send_notice": dup_by_tool.get("send_notice", 0),
            "kv_put": dup_by_tool.get("kv_put", 0),
        },
    }

    out_path = ROOT / "docs" / "experiments" / "exp1_rerun_incident.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    print("=== Experiment 1: Re-running Is a Second Incident ===")
    print(f"Cohort runs: {COHORT_SIZE} | Failed incidents: {n_failed} ({data['failure_rate_pct']:.2f}%)")
    print(f"Exact trajectory reproduced: {exact_trajectory_matches}/{n_failed} ({data['exact_trajectory_rate_pct']:.2f}%)")
    print(f"Same failure kind reproduced: {same_failure_kind_matches}/{n_failed} ({data['same_failure_kind_rate_pct']:.2f}%)")
    print(f"Flipped to phantom success:   {flipped_to_phantom_success}/{n_failed} ({data['phantom_success_rate_pct']:.2f}%)")
    print(f"Total duplicate writes fired: {total_duplicate_writes} across {runs_with_duplicate_writes} debug sessions")
    print(f"  - create_ticket: {dup_by_tool.get('create_ticket', 0)}")
    print(f"  - reserve_stock: {dup_by_tool.get('reserve_stock', 0)}")
    print(f"  - send_notice:   {dup_by_tool.get('send_notice', 0)}")
    print(f"  - kv_put:        {dup_by_tool.get('kv_put', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
