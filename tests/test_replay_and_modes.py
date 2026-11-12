"""Property tests for M4, M5, and Stretch goals: replay, rerun, resume, mutations, and LangGraph."""

from __future__ import annotations

import unittest

from agent_trace_replay.agent import AgentConfig, MutationFlags, initial_agent_state, run_agent_live
from agent_trace_replay.analytic import predict_cohort_reproduction_rate
from agent_trace_replay.langgraph_bridge import (
    InMemorySaver,
    TraceGateway,
    build_langgraph_order_agent,
)
from agent_trace_replay.replay import (
    DivergenceCause,
    ReplayStatus,
    replay_trace,
    rerun_live,
    resume_from_checkpoint,
)
from agent_trace_replay.trace import CaptureLevel, outcome_to_full_trace, project_trace
from agent_trace_replay.world import OrderDeskWorld


class TestReplayRerunResume(unittest.TestCase):
    def test_l2_and_l3_replay_are_bit_identical_across_cohort_and_fire_zero_writes(self) -> None:
        cfg = AgentConfig()
        for seed in range(1, 81):
            with self.subTest(seed=seed):
                w = OrderDeskWorld(seed=seed)
                out = run_agent_live(w, cfg, attempt_tag="incident_0")
                effects_before = len(w.effect_log)
                # Even after world advances 5 epochs, L2 and L3 replay must be 100% bit-identical
                w.advance_epochs(5)
                l3 = outcome_to_full_trace(out)
                l2 = project_trace(l3, CaptureLevel.L2)

                res_l2 = replay_trace(l2, w, cfg, attempt_tag="replay_l2", reference_l3_bundle=l3)
                self.assertEqual(res_l2.status, ReplayStatus.EXACT)
                self.assertEqual(res_l2.writes_re_fired, 0)
                self.assertEqual(len(w.effect_log), effects_before)
                assert res_l2.outcome is not None
                self.assertEqual(res_l2.outcome.final_fingerprint, out.final_fingerprint)

                res_l3 = replay_trace(l3, w, cfg, attempt_tag="replay_l3", reference_l3_bundle=l3)
                self.assertEqual(res_l3.status, ReplayStatus.EXACT)
                self.assertEqual(res_l3.writes_re_fired, 0)

    def test_l3_replay_with_volatile_fields_included_diverges_at_step_zero(self) -> None:
        cfg = AgentConfig()
        for seed in range(1, 21):
            with self.subTest(seed=seed):
                w = OrderDeskWorld(seed=seed)
                out = run_agent_live(w, cfg, attempt_tag="incident_0")
                l3 = outcome_to_full_trace(out)
                res = replay_trace(
                    l3,
                    w,
                    cfg,
                    attempt_tag="replay_volatile",
                    include_volatile_in_fingerprint=True,
                )
                self.assertEqual(res.status, ReplayStatus.DIVERGED)
                self.assertEqual(res.diverged_step, 0)
                self.assertEqual(res.cause, DivergenceCause.VOLATILE_FINGERPRINT)

    def test_l0_safe_mode_aborts_on_first_write_and_unsafe_mode_re_fires_write(self) -> None:
        cfg = AgentConfig()
        w = OrderDeskWorld(seed=1)
        out = run_agent_live(w, cfg, attempt_tag="incident_0")
        l3 = outcome_to_full_trace(out)
        l0 = project_trace(l3, CaptureLevel.L0)

        safe_res = replay_trace(l0, w, cfg, safe_mode=True, reference_l3_bundle=l3)
        self.assertEqual(safe_res.status, ReplayStatus.ABORTED)
        self.assertEqual(safe_res.cause, DivergenceCause.UNCAPTURED_RESULT)
        self.assertEqual(safe_res.writes_re_fired, 0)

        unsafe_res = replay_trace(l0, w, cfg, safe_mode=False, reference_l3_bundle=l3)
        self.assertGreater(unsafe_res.writes_re_fired, 0)

    def test_l1_replay_attributes_clock_or_rng_divergence_when_ambient_draws_shift(self) -> None:
        cfg = AgentConfig()
        diverged_causes = set()
        for seed in range(1, 60):
            w = OrderDeskWorld(seed=seed)
            out = run_agent_live(w, cfg, attempt_tag="incident_0")
            l3 = outcome_to_full_trace(out)
            l1 = project_trace(l3, CaptureLevel.L1)
            res = replay_trace(l1, w, cfg, attempt_tag="replay_l1_diff_tag", reference_l3_bundle=l3)
            if res.status == ReplayStatus.DIVERGED and res.cause is not None:
                diverged_causes.add(res.cause)
        self.assertIn(DivergenceCause.CLOCK, diverged_causes)
        self.assertIn(DivergenceCause.RNG, diverged_causes)

    def test_mutation_flags_pin_divergent_step_at_l3(self) -> None:
        w = OrderDeskWorld(seed=2)
        base_cfg = AgentConfig()
        out = run_agent_live(w, base_cfg, attempt_tag="incident_0")
        l3 = outcome_to_full_trace(out)

        mut_cfg = AgentConfig(mutations=MutationFlags(dropped_state_field=True))
        res = replay_trace(l3, w, mut_cfg, reference_l3_bundle=l3)
        self.assertEqual(res.status, ReplayStatus.DIVERGED)
        self.assertEqual(res.diverged_step, 1)  # `lookup` is step 1 where selected_doc is written
        self.assertEqual(res.cause, DivergenceCause.CODE_CHANGE)

    def test_rerun_and_checkpoint_resume_execute_against_live_world(self) -> None:
        w = OrderDeskWorld(seed=5)
        cfg = AgentConfig()
        out = run_agent_live(w, cfg, attempt_tag="incident_0")
        w.advance_epochs(3)
        rerun_out = rerun_live(w, cfg, attempt_tag="rerun_1")
        self.assertEqual(rerun_out.seed, 5)

        snap = out.snapshots[-2]
        resume_out = resume_from_checkpoint(snap, w, cfg, attempt_tag="resume_1")
        self.assertEqual(resume_out.seed, 5)

    def test_langgraph_checkpoint_resume_refires_writes_while_gateway_replay_fires_zero(self) -> None:
        w = OrderDeskWorld(seed=2, flaky_rate=0.0)
        saver = InMemorySaver()
        gw_rec = TraceGateway(world=w, mode="record", attempt_tag="lg_0")
        app = build_langgraph_order_agent(gw_rec, saver)

        init_s = initial_agent_state(w.primary_order_id, "lg_0")
        final_s = app.invoke(init_s, thread_id="t1")
        self.assertEqual(final_s["status"], "completed")
        recorded_writes = gw_rec.writes_fired
        self.assertGreaterEqual(recorded_writes, 3)

        # Resume from checkpoint right after `reserve` (so `ticket`, `notify`, `finish` re-run live)
        cps = saver.list_checkpoints("t1")
        reserve_cp = [cp for cp in cps if cp.node == "reserve"][0]

        gw_resume = TraceGateway(world=w, mode="live_resume", attempt_tag="lg_resume_1")
        app_resume = build_langgraph_order_agent(gw_resume, saver)
        app_resume.invoke(None, thread_id="t1_resume", resume_from_checkpoint=reserve_cp)
        self.assertGreater(gw_resume.writes_fired, 0)

        # Gateway trace replay from the start fires 0 live writes and matches final fingerprint
        bundle = gw_rec.to_trace_bundle(final_s)
        gw_replay = TraceGateway(
            world=w, mode="replay", attempt_tag="lg_replay_1", replay_bundle=bundle
        )
        app_replay = build_langgraph_order_agent(gw_replay, InMemorySaver())
        replay_s = app_replay.invoke(initial_agent_state(w.primary_order_id, "lg_replay_1"), thread_id="t1_rep")
        self.assertEqual(gw_replay.writes_fired, 0)
        self.assertEqual(
            bundle.end.final_fingerprint if bundle.end else None,
            gw_replay.to_trace_bundle(replay_s).end.final_fingerprint,
        )

    def test_analytic_survival_model_returns_bounded_rates(self) -> None:
        cfg = AgentConfig()
        sample = []
        for s in range(1, 25):
            w0 = OrderDeskWorld(seed=s)
            w_run = w0.clone()
            out = run_agent_live(w_run, cfg)
            sample.append((w0, out, len(w_run.reservations)))
        preds = predict_cohort_reproduction_rate(sample, delta_epochs=3, config=cfg)
        self.assertGreaterEqual(preds["analytic_trajectory_rate"], 0.0)
        self.assertLessEqual(preds["analytic_trajectory_rate"], 1.0)
        self.assertGreaterEqual(preds["naive_hazard_rate"], 0.0)
