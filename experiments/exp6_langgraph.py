#!/usr/bin/env python3
"""Experiment 6 (Stretch): StateGraph + InMemorySaver Checkpoint Resume vs Trace Replay.

Run 200 seeded orders through `StateGraph` + `InMemorySaver`. For every run that
executes writes (`reserve_stock`, `create_ticket`, `send_notice`, `kv_put`),
compare resuming from the pre-final checkpoint via `InMemorySaver` against
replaying the recorded trace through `TraceGateway`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_trace_replay.agent import initial_agent_state
from agent_trace_replay.langgraph_bridge import (
    HAS_EXTERNAL_LANGGRAPH,
    InMemorySaver,
    TraceGateway,
    build_langgraph_order_agent,
)
from agent_trace_replay.world import OrderDeskWorld


def main() -> int:
    n_runs = 200
    checkpoint_resumes_evaluated = 0
    checkpoint_duplicate_writes = 0
    gateway_replay_writes = 0
    gateway_exact_matches = 0

    for seed in range(1, n_runs + 1):
        w = OrderDeskWorld(seed=seed)
        saver = InMemorySaver()
        gw_rec = TraceGateway(world=w, mode="record", attempt_tag="lg_rec_0")
        app = build_langgraph_order_agent(gw_rec, saver)

        init_s = initial_agent_state(w.primary_order_id, "lg_rec_0")
        final_s = app.invoke(init_s, thread_id=f"thread_{seed}")
        rec_bundle = gw_rec.to_trace_bundle(final_s)

        cps = saver.list_checkpoints(f"thread_{seed}")
        if len(cps) >= 2:
            checkpoint_resumes_evaluated += 1
            pre_last_cp = cps[-2]
            w_res = w.clone()
            w_res.advance_epochs(2)
            gw_res = TraceGateway(world=w_res, mode="live_resume", attempt_tag="lg_res_1")
            app_res = build_langgraph_order_agent(gw_res, saver)
            app_res.invoke(None, thread_id=f"thread_{seed}_res", resume_from_checkpoint=pre_last_cp)
            checkpoint_duplicate_writes += gw_res.writes_fired

        w_rep = w.clone()
        w_rep.advance_epochs(2)
        gw_rep = TraceGateway(
            world=w_rep, mode="replay", attempt_tag="lg_rep_1", replay_bundle=rec_bundle
        )
        app_rep = build_langgraph_order_agent(gw_rep, InMemorySaver())
        rep_s = app_rep.invoke(
            initial_agent_state(w.primary_order_id, "lg_rep_1"), thread_id=f"thread_{seed}_rep"
        )
        gateway_replay_writes += gw_rep.writes_fired
        rep_bundle = gw_rep.to_trace_bundle(rep_s)
        if (
            rec_bundle.end is not None
            and rep_bundle.end is not None
            and rec_bundle.end.final_fingerprint == rep_bundle.end.final_fingerprint
        ):
            gateway_exact_matches += 1

    data = {
        "experiment": "exp6_langgraph",
        "external_langgraph_installed": HAS_EXTERNAL_LANGGRAPH,
        "runs_evaluated": n_runs,
        "checkpoint_resumes_evaluated": checkpoint_resumes_evaluated,
        "inmemory_saver_duplicate_writes_fired": checkpoint_duplicate_writes,
        "gateway_trace_replay_writes_fired": gateway_replay_writes,
        "gateway_trace_replay_exact_matches": gateway_exact_matches,
        "gateway_trace_replay_exact_rate_pct": round(100.0 * gateway_exact_matches / n_runs, 2),
    }

    out_path = ROOT / "docs" / "experiments" / "exp6_langgraph.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    print("=== Experiment 6: StateGraph + InMemorySaver vs TraceGateway Replay ===")
    print(f"Runs evaluated: {n_runs}")
    print(f"InMemorySaver checkpoint resumes: {checkpoint_resumes_evaluated} -> {checkpoint_duplicate_writes} duplicate writes fired")
    print(f"TraceGateway L3 replays:          {gateway_exact_matches}/{n_runs} ({data['gateway_trace_replay_exact_rate_pct']:.1f}%) exact, {gateway_replay_writes} writes fired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
