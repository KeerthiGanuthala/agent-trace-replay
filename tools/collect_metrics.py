#!/usr/bin/env python3
"""Collect every measured figure from test and experiment JSON reports into `docs/metrics.json`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXP_DIR = DOCS / "experiments"


def _read_json(path: Path) -> dict:  # type: ignore[type-arg]
    if not path.exists():
        raise FileNotFoundError(f"Missing required report file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    test_rep = _read_json(DOCS / "test_report.json")
    exp1 = _read_json(EXP_DIR / "exp1_rerun_incident.json")
    exp2 = _read_json(EXP_DIR / "exp2_checkpoint_vs_trace.json")
    exp3 = _read_json(EXP_DIR / "exp3_capture_ladder.json")
    exp4 = _read_json(EXP_DIR / "exp4_localization.json")
    exp5 = _read_json(EXP_DIR / "exp5_drift_sweep.json")
    exp6 = _read_json(EXP_DIR / "exp6_langgraph.json")

    sweep_by_e = {int(r["epoch_delta"]): r for r in exp5["sweep"]}

    metrics = {
        "cohort": {
            "total_runs": exp1["cohort_size"],
            "failed_runs": exp1["failed_runs"],
            "failure_rate_pct": exp1["failure_rate_pct"],
            "completed_runs": exp1["cohort_breakdown"]["completed"],
            "deadline_exceeded": exp1["cohort_breakdown"]["deadline_exceeded"],
            "budget_escalated": exp1["cohort_breakdown"]["budget_escalated"],
            "stock_backordered": exp1["cohort_breakdown"]["stock_backordered"],
            "retry_exhausted": exp1["cohort_breakdown"]["retry_exhausted"],
        },
        "tests": {
            "test_methods_run": test_rep["test_methods_run"],
            "subtests_run": test_rep["subtests_run"],
            "total_test_cases": test_rep["total_test_cases"],
            "line_coverage_pct": test_rep["line_coverage_pct"],
        },
        "exp1": {
            "exact_trajectory_reproduced": exp1["exact_trajectory_reproduced"],
            "exact_trajectory_rate_pct": exp1["exact_trajectory_rate_pct"],
            "same_failure_kind_reproduced": exp1["same_failure_kind_reproduced"],
            "same_failure_kind_rate_pct": exp1["same_failure_kind_rate_pct"],
            "phantom_successes_on_rerun": exp1["phantom_successes_on_rerun"],
            "phantom_success_rate_pct": exp1["phantom_success_rate_pct"],
            "total_duplicate_side_effects": exp1["total_duplicate_side_effects"],
            "runs_firing_duplicate_writes": exp1["runs_firing_duplicate_writes"],
            "dup_create_ticket": exp1["duplicate_effects_by_tool"]["create_ticket"],
            "dup_reserve_stock": exp1["duplicate_effects_by_tool"]["reserve_stock"],
            "dup_send_notice": exp1["duplicate_effects_by_tool"]["send_notice"],
            "dup_kv_put": exp1["duplicate_effects_by_tool"]["kv_put"],
        },
        "exp2": {
            "resume_exact_match": exp2["totals"]["resume_exact_match"],
            "resume_exact_rate_pct": exp2["totals"]["resume_exact_rate_pct"],
            "resume_kind_match": exp2["totals"]["resume_kind_match"],
            "resume_kind_rate_pct": exp2["totals"]["resume_kind_rate_pct"],
            "resume_phantom_success": exp2["totals"]["resume_phantom_success"],
            "resume_phantom_rate_pct": exp2["totals"]["resume_phantom_rate_pct"],
            "resume_refired_writes": exp2["totals"]["resume_refired_writes"],
            "trace_exact_match": exp2["totals"]["trace_exact_match"],
            "trace_exact_rate_pct": exp2["totals"]["trace_exact_rate_pct"],
            "trace_refired_writes": exp2["totals"]["trace_refired_writes"],
            "per_failure_kind": exp2["per_failure_kind"],
        },
        "exp3": {
            "L0": exp3["levels"]["L0"],
            "L1": exp3["levels"]["L1"],
            "L2": exp3["levels"]["L2"],
            "L3": exp3["levels"]["L3"],
            "volatile_step0_divergences": exp3["volatile_fingerprint_ablation"][
                "step_0_divergences_when_volatile_included"
            ],
            "volatile_step0_divergence_rate_pct": exp3["volatile_fingerprint_ablation"][
                "step_0_divergence_rate_pct"
            ],
        },
        "exp4": exp4["mutations"],
        "exp5": {
            "e0_rerun_exact_pct": sweep_by_e[0]["rerun_exact_pct"],
            "e3_rerun_exact_pct": sweep_by_e[3]["rerun_exact_pct"],
            "e5_rerun_exact_pct": sweep_by_e[5]["rerun_exact_pct"],
            "e10_rerun_exact_pct": sweep_by_e[10]["rerun_exact_pct"],
            "e10_rerun_kind_pct": sweep_by_e[10]["rerun_kind_match_pct"],
            "e10_l2_exact_pct": sweep_by_e[10]["l2_replay_exact_pct"],
            "e3_analytic_pct": sweep_by_e[3]["analytic_predicted_exact_pct"],
            "e3_naive_pct": sweep_by_e[3]["naive_hazard_predicted_pct"],
        },
        "exp6": {
            "runs_evaluated": exp6["runs_evaluated"],
            "inmemory_saver_duplicate_writes_fired": exp6["inmemory_saver_duplicate_writes_fired"],
            "gateway_trace_replay_writes_fired": exp6["gateway_trace_replay_writes_fired"],
            "gateway_trace_replay_exact_matches": exp6["gateway_trace_replay_exact_matches"],
            "gateway_trace_replay_exact_rate_pct": exp6["gateway_trace_replay_exact_rate_pct"],
        },
    }

    out_path = DOCS / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    print(f"[collect_metrics] Wrote {out_path.relative_to(ROOT)} with all machine-measured figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
