"""Shared cohort runner for the 2,000-seed benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from agent_trace_replay.agent import AgentConfig, RunOutcome, run_agent_live
from agent_trace_replay.trace import TraceBundle, outcome_to_full_trace
from agent_trace_replay.world import OrderDeskWorld

COHORT_SIZE = 2000
DRIFT_EPOCHS_INCIDENT_TO_DEBUG = 3


@dataclass
class CohortRun:
    seed: int
    world_initial: OrderDeskWorld
    world_after_run: OrderDeskWorld
    outcome: RunOutcome
    l3_bundle: TraceBundle


def build_cohort(n_seeds: int = COHORT_SIZE) -> list[CohortRun]:
    cfg = AgentConfig()
    runs: list[CohortRun] = []
    for seed in range(1, n_seeds + 1):
        w0 = OrderDeskWorld(seed=seed)
        w_run = w0.clone()
        out = run_agent_live(w_run, cfg, attempt_tag="incident_0")
        l3 = outcome_to_full_trace(out)
        runs.append(
            CohortRun(
                seed=seed,
                world_initial=w0,
                world_after_run=w_run,
                outcome=out,
                l3_bundle=l3,
            )
        )
    return runs
