#!/usr/bin/env python3
"""Experiment 3: The Capture Ladder (L0, L1, L2, L3) + Volatile Fingerprint Ablation.

Replay every failure at L0, L1, L2, L3 against the drifted world. Report exact-replay
rate, abort rate, divergence causes, and mean trace size in bytes per level. Also
measure the step-1 divergence rate when volatile fields are not excluded from state
fingerprints.
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
from agent_trace_replay.replay import ReplayStatus, replay_trace
from agent_trace_replay.trace import CaptureLevel, project_trace, serialize_trace
from experiments._cohort import COHORT_SIZE, DRIFT_EPOCHS_INCIDENT_TO_DEBUG, build_cohort


def main() -> int:
    cohort = build_cohort(COHORT_SIZE)
    cfg = AgentConfig()
    failed_runs = [r for r in cohort if r.outcome.failure_kind is not None]
    n_failed = len(failed_runs)

    levels_summary: dict[str, dict[str, int | float | dict[str, int]]] = {}

    for lvl in (CaptureLevel.L0, CaptureLevel.L1, CaptureLevel.L2, CaptureLevel.L3):
        exact_cnt = 0
        diverged_cnt = 0
        aborted_cnt = 0
        total_bytes = 0
        causes: Counter[str] = Counter()

        for item in failed_runs:
            proj = project_trace(item.l3_bundle, lvl)
            raw_text = serialize_trace(proj)
            total_bytes += len(raw_text.encode("utf-8"))

            w_rep = item.world_after_run.clone()
            w_rep.advance_epochs(DRIFT_EPOCHS_INCIDENT_TO_DEBUG)
            res = replay_trace(
                proj,
                w_rep,
                cfg,
                attempt_tag=f"replay_{lvl.name.lower()}",
                safe_mode=True,
                reference_l3_bundle=item.l3_bundle,
            )
            if res.status == ReplayStatus.EXACT:
                exact_cnt += 1
            elif res.status == ReplayStatus.ABORTED:
                aborted_cnt += 1
                if res.cause:
                    causes[res.cause.value] += 1
            else:
                diverged_cnt += 1
                if res.cause:
                    causes[res.cause.value] += 1

        levels_summary[lvl.name] = {
            "capture_level": int(lvl),
            "runs_evaluated": n_failed,
            "mean_bytes": round(total_bytes / n_failed, 1),
            "total_bytes": total_bytes,
            "exact_replays": exact_cnt,
            "exact_rate_pct": round(100.0 * exact_cnt / n_failed, 2),
            "aborted_replays": aborted_cnt,
            "abort_rate_pct": round(100.0 * aborted_cnt / n_failed, 2),
            "diverged_replays": diverged_cnt,
            "diverged_rate_pct": round(100.0 * diverged_cnt / n_failed, 2),
            "divergence_causes": dict(sorted(causes.items())),
        }

    # Volatile fingerprint ablation across all 2,000 runs at L3
    volatile_diverged_step0 = 0
    for item in cohort:
        res_vol = replay_trace(
            item.l3_bundle,
            item.world_after_run,
            cfg,
            attempt_tag="replay_volatile_check",
            include_volatile_in_fingerprint=True,
        )
        if res_vol.status == ReplayStatus.DIVERGED and res_vol.diverged_step == 0:
            volatile_diverged_step0 += 1

    l0_bytes = float(levels_summary["L0"]["mean_bytes"])
    for lvl_name, entry in levels_summary.items():
        entry["byte_overhead_vs_l0_ratio"] = round(float(entry["mean_bytes"]) / l0_bytes, 2)

    data = {
        "experiment": "exp3_capture_ladder",
        "cohort_size": COHORT_SIZE,
        "failed_runs": n_failed,
        "levels": levels_summary,
        "volatile_fingerprint_ablation": {
            "runs_tested": COHORT_SIZE,
            "step_0_divergences_when_volatile_included": volatile_diverged_step0,
            "step_0_divergence_rate_pct": round(100.0 * volatile_diverged_step0 / COHORT_SIZE, 2),
        },
    }

    out_path = ROOT / "docs" / "experiments" / "exp3_capture_ladder.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    print("=== Experiment 3: The Capture Ladder ===")
    print(f"{'Level':<6} | {'Mean Bytes':>10} | {'Overhead':>8} | {'Exact Replay':>16} | {'Aborted':>14} | {'Diverged':>14} | {'Primary Causes'}")
    print("-" * 105)
    for lvl_name in ("L0", "L1", "L2", "L3"):
        row = levels_summary[lvl_name]
        cause_str = ", ".join(f"{k}:{v}" for k, v in row["divergence_causes"].items()) or "none"  # type: ignore[union-attr]
        print(
            f"{lvl_name:<6} | {row['mean_bytes']:>10.1f} | "
            f"{row['byte_overhead_vs_l0_ratio']:>7.2f}x | "
            f"{row['exact_replays']:>4}/{n_failed} ({row['exact_rate_pct']:>5.1f}%) | "
            f"{row['aborted_replays']:>4} ({row['abort_rate_pct']:>5.1f}%) | "
            f"{row['diverged_replays']:>4} ({row['diverged_rate_pct']:>5.1f}%) | {cause_str}"
        )
    print(
        chr(10) + f"Volatile field ablation: {volatile_diverged_step0}/{COHORT_SIZE} "
        f"({data['volatile_fingerprint_ablation']['step_0_divergence_rate_pct']:.2f}%) diverge at Step 0 when volatile fields are hashed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
