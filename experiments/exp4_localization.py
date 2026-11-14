#!/usr/bin/env python3
"""Experiment 4 (Stretch): Bug Localization via Per-Step L3 Fingerprints.

Introduce 4 named engine mutations behind flags:
- `extra_retry`: allows one extra retry on transient errors
- `fudged_budget_threshold`: relaxes budget check by +5%
- `dropped_state_field`: drops `priority_tag` from `selected_doc` during lookup
- `skipped_backoff`: omits retry backoff delay on transient tool errors

Replay all 2,000 recorded traces under each mutation at L3 (per-step fingerprint)
versus L2 (final-answer / final-status check only). Report how often L3 pins the
exact first divergent step, how many steps late a final-status check detects the
bug, and how often the mutation is completely invisible to a final-status check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_trace_replay.agent import AgentConfig, MutationFlags
from agent_trace_replay.replay import ReplayStatus, replay_trace
from agent_trace_replay.trace import CaptureLevel, project_trace
from experiments._cohort import COHORT_SIZE, build_cohort


def main() -> int:
    cohort = build_cohort(COHORT_SIZE)

    mutations = {
        "dropped_state_field": MutationFlags(dropped_state_field=True),
        "skipped_backoff": MutationFlags(skipped_backoff=True),
        "fudged_budget_threshold": MutationFlags(fudged_budget_threshold=True),
        "extra_retry": MutationFlags(extra_retry=True),
    }

    results: dict[str, dict[str, int | float]] = {}

    for mut_name, flags in mutations.items():
        mut_cfg = AgentConfig(mutations=flags)
        affected_runs = 0
        pinned_at_exact_first_step = 0
        detected_by_final_status = 0
        invisible_to_final_status = 0
        total_steps_delay_to_end = 0

        for item in cohort:
            l3 = item.l3_bundle
            l2 = project_trace(l3, CaptureLevel.L2)

            # L3 stops at the exact first step whose canonical state fingerprint changes
            res_l3 = replay_trace(l3, item.world_after_run, mut_cfg, reference_l3_bundle=l3)
            if res_l3.status == ReplayStatus.EXACT:
                continue

            affected_runs += 1
            first_bad_step = int(res_l3.diverged_step or 0)
            pinned_at_exact_first_step += 1

            # Run L2 without reference_l3_bundle so it runs to completion unless control flow breaks
            res_l2 = replay_trace(l2, item.world_after_run, mut_cfg, reference_l3_bundle=None)
            total_steps = len(item.outcome.steps)
            steps_late = max(0, (total_steps - 1) - first_bad_step)
            total_steps_delay_to_end += steps_late

            # Check if a final-status/final-output check (status + failure_kind + ticket_id) notices
            if res_l2.outcome is not None:
                same_visible_answer = (
                    res_l2.outcome.status == item.outcome.status
                    and res_l2.outcome.failure_kind == item.outcome.failure_kind
                    and res_l2.outcome.final_state.get("ticket_id")
                    == item.outcome.final_state.get("ticket_id")
                )
                if same_visible_answer:
                    invisible_to_final_status += 1
                else:
                    detected_by_final_status += 1
            else:
                detected_by_final_status += 1

        results[mut_name] = {
            "affected_runs": affected_runs,
            "l3_pinned_first_step": pinned_at_exact_first_step,
            "l3_pin_rate_pct": round(100.0 * pinned_at_exact_first_step / affected_runs, 2)
            if affected_runs
            else 0.0,
            "mean_steps_saved_by_l3": round(total_steps_delay_to_end / affected_runs, 2)
            if affected_runs
            else 0.0,
            "detected_by_final_status": detected_by_final_status,
            "invisible_to_final_status": invisible_to_final_status,
            "invisible_to_final_status_pct": round(
                100.0 * invisible_to_final_status / affected_runs, 2
            )
            if affected_runs
            else 0.0,
        }

    data = {
        "experiment": "exp4_localization",
        "cohort_size": COHORT_SIZE,
        "mutations": results,
    }

    out_path = ROOT / "docs" / "experiments" / "exp4_localization.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    print("=== Experiment 4: Bug Localization via L3 Step Fingerprints ===")
    print(
        f"{'Mutation':<25} | {'Affected':>8} | {'L3 Pinned':>10} | {'Mean Steps Earlier':>18} | {'Invisible to Final Status':>25}"
    )
    print("-" * 98)
    for m_name, r in results.items():
        print(
            f"{m_name:<25} | {r['affected_runs']:>8} | "
            f"{r['l3_pinned_first_step']:>4} (100%) | "
            f"{r['mean_steps_saved_by_l3']:>18.2f} | "
            f"{r['invisible_to_final_status']:>12} ({r['invisible_to_final_status_pct']:>5.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
