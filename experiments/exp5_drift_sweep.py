#!/usr/bin/env python3
"""Experiment 5: Drift Sweep + Analytic Survival Model (Stretch).

Sweep epochs between incident record and debug session (E = 0..10). Measure:
- Live rerun exact reproduction rate & failure-kind reproduction rate
- L2 trace replay exact reproduction rate (flat at 100.0%)
- Analytic survival model prediction vs Naive marginal-hazard model prediction
Also emits an SVG plot to `docs/experiments/drift_sweep.svg`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_trace_replay.agent import AgentConfig
from agent_trace_replay.analytic import predict_cohort_reproduction_rate
from agent_trace_replay.replay import ReplayStatus, replay_trace, rerun_live
from agent_trace_replay.trace import CaptureLevel, project_trace
from experiments._cohort import COHORT_SIZE, build_cohort


def _write_svg_plot(sweep_rows: list[dict[str, float | int]], out_svg: Path) -> None:
    width, height = 760, 420
    pad_l, pad_r, pad_t, pad_b = 70, 40, 45, 55
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_coord(ep: int) -> float:
        return pad_l + (ep / 10.0) * plot_w

    def y_coord(pct: float) -> float:
        return pad_t + (1.0 - pct / 100.0) * plot_h

    def polyline(key: str, color: str, dash: str = "") -> str:
        pts = " ".join(f"{x_coord(int(r['epoch_delta'])):.1f},{y_coord(float(r[key])):.1f}" for r in sweep_rows)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} points="{pts}"/>'

    grid_lines = []
    for pct in (0, 20, 40, 60, 80, 100):
        y = y_coord(pct)
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{pct}%</text>'
        )

    x_labels = []
    for ep in range(0, 11):
        x = x_coord(ep)
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 20}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#475569">E+{ep}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a">
    Experiment 5: Trajectory &amp; Failure Reproduction vs. World Drift (Epochs E+0 to E+10)
  </text>
  {"".join(grid_lines)}
  {"".join(x_labels)}
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#334155" stroke-width="1.5"/>
  <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#334155" stroke-width="1.5"/>
  {polyline("l2_replay_exact_pct", "#059669")}
  {polyline("rerun_kind_match_pct", "#2563eb")}
  {polyline("rerun_exact_pct", "#dc2626")}
  {polyline("analytic_predicted_exact_pct", "#7c3aed", "6,4")}
  {polyline("naive_hazard_predicted_pct", "#94a3b8", "3,3")}
  <g transform="translate({pad_l + 18}, {pad_t + 26})" font-family="sans-serif" font-size="11" fill="#1e293b">
    <rect x="-8" y="-14" width="320" height="96" fill="#f8fafc" stroke="#cbd5e1" rx="4"/>
    <line x1="0" y1="0" x2="22" y2="0" stroke="#059669" stroke-width="2.6"/><text x="30" y="4">L2 Trace Replay (100.0% flat)</text>
    <line x1="0" y1="18" x2="22" y2="18" stroke="#2563eb" stroke-width="2.6"/><text x="30" y="22">Live Rerun: Failure-Kind Match</text>
    <line x1="0" y1="36" x2="22" y2="36" stroke="#dc2626" stroke-width="2.6"/><text x="30" y="40">Live Rerun: Exact Trajectory Match (Measured)</text>
    <line x1="0" y1="54" x2="22" y2="54" stroke="#7c3aed" stroke-width="2.6" stroke-dasharray="6,4"/><text x="30" y="58">Analytic Keyed Survival Model</text>
    <line x1="0" y1="72" x2="22" y2="72" stroke="#94a3b8" stroke-width="2.6" stroke-dasharray="3,3"/><text x="30" y="76">Naive Per-Step Marginal Hazard Model</text>
  </g>
</svg>
"""
    out_svg.write_text(svg, encoding="utf-8")


def main() -> int:
    cohort = build_cohort(COHORT_SIZE)
    cfg = AgentConfig()
    failed_runs = [r for r in cohort if r.outcome.failure_kind is not None]
    n_failed = len(failed_runs)

    analytic_inputs = [
        (item.world_initial, item.outcome, len(item.world_after_run.reservations))
        for item in failed_runs
    ]

    sweep_rows: list[dict[str, float | int]] = []
    for delta_ep in range(0, 11):
        exact_cnt = 0
        kind_cnt = 0
        l2_exact_cnt = 0

        for item in failed_runs:
            w_dbg = item.world_after_run.clone()
            w_dbg.advance_epochs(delta_ep)
            r_out = rerun_live(w_dbg, cfg, attempt_tag=f"sweep_rerun_e{delta_ep}")
            orig_nodes = [s.node for s in item.outcome.steps]
            r_nodes = [s.node for s in r_out.steps]
            same_traj = (
                orig_nodes == r_nodes
                and r_out.status == item.outcome.status
                and r_out.failure_kind == item.outcome.failure_kind
                and r_out.final_state.get("selected_doc") == item.outcome.final_state.get("selected_doc")
                and r_out.final_state.get("unit_price") == item.outcome.final_state.get("unit_price")
                and r_out.final_state.get("tight_stock_policy") == item.outcome.final_state.get("tight_stock_policy")
                and r_out.final_state.get("notice_template") == item.outcome.final_state.get("notice_template")
                and [len(s.tool_attempts) for s in r_out.steps] == [len(s.tool_attempts) for s in item.outcome.steps]
            )
            if same_traj:
                exact_cnt += 1
            if r_out.failure_kind == item.outcome.failure_kind:
                kind_cnt += 1

            l2 = project_trace(item.l3_bundle, CaptureLevel.L2)
            rep = replay_trace(l2, w_dbg, cfg, reference_l3_bundle=item.l3_bundle)
            if rep.status == ReplayStatus.EXACT:
                l2_exact_cnt += 1

        preds = predict_cohort_reproduction_rate(analytic_inputs, delta_epochs=delta_ep, config=cfg)
        row = {
            "epoch_delta": delta_ep,
            "rerun_exact_matches": exact_cnt,
            "rerun_exact_pct": round(100.0 * exact_cnt / n_failed, 2),
            "rerun_kind_matches": kind_cnt,
            "rerun_kind_match_pct": round(100.0 * kind_cnt / n_failed, 2),
            "l2_replay_exact_matches": l2_exact_cnt,
            "l2_replay_exact_pct": round(100.0 * l2_exact_cnt / n_failed, 2),
            "analytic_predicted_exact_pct": round(100.0 * preds["analytic_trajectory_rate"], 2),
            "naive_hazard_predicted_pct": round(100.0 * preds["naive_hazard_rate"], 2),
        }
        sweep_rows.append(row)

    out_dir = ROOT / "docs" / "experiments"
    out_path = out_dir / "exp5_drift_sweep.json"
    svg_path = out_dir / "drift_sweep.svg"
    _write_svg_plot(sweep_rows, svg_path)

    data = {
        "experiment": "exp5_drift_sweep",
        "cohort_size": COHORT_SIZE,
        "failed_runs": n_failed,
        "plot_path": "docs/experiments/drift_sweep.svg",
        "sweep": sweep_rows,
    }
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    print("=== Experiment 5: Drift Sweep & Analytic Survival Model ===")
    print(
        f"{'Epoch':<6} | {'Rerun Exact':>14} | {'Analytic Model':>14} | {'Naive Hazard':>13} | {'Rerun Kind':>14} | {'L2 Replay':>11}"
    )
    print("-" * 86)
    for r in sweep_rows:
        print(
            f"E+{r['epoch_delta']:<4} | "
            f"{r['rerun_exact_matches']:>3}/{n_failed} ({r['rerun_exact_pct']:>5.2f}%) | "
            f"{r['analytic_predicted_exact_pct']:>13.2f}% | "
            f"{r['naive_hazard_predicted_pct']:>12.2f}% | "
            f"{r['rerun_kind_matches']:>3}/{n_failed} ({r['rerun_kind_match_pct']:>5.2f}%) | "
            f"{r['l2_replay_exact_pct']:>10.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
