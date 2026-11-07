"""Replay, rerun, and checkpoint-resume engines with divergence attribution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_trace_replay.agent import (
    AgentConfig,
    LiveDriver,
    RunOutcome,
    StepRecord,
    StepSnapshot,
    ToolAttemptRecord,
    compute_state_fingerprint,
    execute_agent_with_driver,
    run_agent_live,
)
from agent_trace_replay.rng import draw, draw_bool, draw_int
from agent_trace_replay.tools import EffectClass, ToolRegistry, build_default_registry
from agent_trace_replay.trace import CaptureLevel, TraceBundle
from agent_trace_replay.world import OrderDeskWorld


class ReplayStatus(str, Enum):
    """High-level outcome of replaying a recorded trace."""

    EXACT = "EXACT"
    DIVERGED = "DIVERGED"
    ABORTED = "ABORTED"


class DivergenceCause(str, Enum):
    """Attributed root cause when a replay diverges from the recorded trace."""

    DRIFTED_READ = "drifted_read"
    CLOCK = "clock"
    RNG = "rng"
    UNCAPTURED_RESULT = "uncaptured_result"
    CODE_CHANGE = "code_change"
    VOLATILE_FINGERPRINT = "volatile_fingerprint"


@dataclass
class ReplayResult:
    """Detailed outcome returned by `replay_trace`."""

    status: ReplayStatus
    diverged_step: int | None
    cause: DivergenceCause | None
    detail: str
    outcome: RunOutcome | None
    writes_re_fired: int


class _ReplayAbortSignal(RuntimeError):
    def __init__(self, step_index: int, detail: str) -> None:
        super().__init__(detail)
        self.step_index = step_index
        self.detail = detail


class _ReplayDivergenceSignal(RuntimeError):
    def __init__(self, step_index: int, cause: DivergenceCause, detail: str) -> None:
        super().__init__(detail)
        self.step_index = step_index
        self.cause = cause
        self.detail = detail


class ReplayDriver:
    """Execution driver that serves captured values according to the trace's CaptureLevel."""

    def __init__(
        self,
        bundle: TraceBundle,
        world: OrderDeskWorld,
        registry: ToolRegistry,
        config: AgentConfig,
        attempt_tag: str,
        safe_mode: bool = True,
        reference_l3_bundle: TraceBundle | None = None,
        include_volatile_in_fingerprint: bool = False,
    ) -> None:
        self.bundle = bundle
        self.world = world
        self.registry = registry
        self.config = config
        self.attempt_tag = attempt_tag
        self.safe_mode = safe_mode
        self.reference_l3 = reference_l3_bundle
        self.include_volatile = include_volatile_in_fingerprint
        self.writes_re_fired = 0
        self._attempt_cursors: dict[int, int] = {}
        self._pending_cause: DivergenceCause | None = None
        self._pending_detail: str = ""

    @property
    def level(self) -> CaptureLevel:
        return self.bundle.header.capture_level

    def _get_step_trace(self, step_index: int) -> dict[str, Any]:
        if step_index >= len(self.bundle.steps):
            raise _ReplayDivergenceSignal(
                step_index=step_index,
                cause=self._pending_cause or DivergenceCause.CODE_CHANGE,
                detail=self._pending_detail or f"Execution exceeded recorded step count ({len(self.bundle.steps)})",
            )
        return self.bundle.steps[step_index]

    def before_step(self, step_index: int, node: str, state: dict[str, Any], call_index: int) -> None:
        rec_step = self._get_step_trace(step_index)
        if rec_step["node"] != node:
            cause = self._pending_cause or DivergenceCause.CODE_CHANGE
            detail = self._pending_detail or f"Expected node '{rec_step['node']}', got '{node}'"
            raise _ReplayDivergenceSignal(step_index=step_index, cause=cause, detail=detail)

    def pick_tie_break(self, step_index: int, num_docs: int) -> int:
        rec_step = self._get_step_trace(step_index)
        if self.level >= CaptureLevel.L2 and rec_step.get("tie_break_index") is not None:
            return int(rec_step["tie_break_index"])
        # At L0/L1, tie-break draw uses the current replay attempt's ambient RNG
        drawn = draw_int(
            self.world.seed, 0, num_docs - 1, "tie_break", self.attempt_tag, step_index
        )
        if self.reference_l3 is not None and step_index < len(self.reference_l3.steps):
            ref_idx = self.reference_l3.steps[step_index].get("tie_break_index")
            if ref_idx is not None and drawn != int(ref_idx):
                self._pending_cause = DivergenceCause.RNG
                self._pending_detail = f"Ambient tie-break draw {drawn} != recorded {ref_idx} at step {step_index}"
        return drawn

    def pick_stochastic_choice(
        self, step_index: int, choice_name: str, options: tuple[str, ...]
    ) -> str:
        rec_step = self._get_step_trace(step_index)
        recorded_choice = rec_step.get("choice_made")
        if recorded_choice is not None:
            return str(recorded_choice)
        raise _ReplayDivergenceSignal(
            step_index=step_index,
            cause=DivergenceCause.CODE_CHANGE,
            detail=f"Unexpected stochastic choice '{choice_name}' at step {step_index}",
        )

    def _fresh_service_ms(self, tool_name: str, call_index: int) -> float:
        spec = self.registry.get(tool_name)
        jitter = draw(self.world.seed, "svc_jitter", self.attempt_tag, call_index, tool_name) * 9.0
        hiccup = 0.0
        if draw_bool(
            self.world.seed,
            self.config.hiccup_prob,
            "svc_hiccup",
            self.attempt_tag,
            call_index,
            tool_name,
        ):
            hiccup = self.config.hiccup_ms * (
                0.75 + 0.5 * draw(self.world.seed, "hiccup_mag", self.attempt_tag, call_index)
            )
        return round(spec.base_ms + jitter + hiccup, 3)

    def invoke_tool(
        self,
        step_index: int,
        node: str,
        tool_name: str,
        args: dict[str, Any],
        call_index: int,
        attempt_index: int,
        max_retries: int,
    ) -> ToolAttemptRecord:
        rec_step = self._get_step_trace(step_index)
        cursor = self._attempt_cursors.get(step_index, 0)
        recorded_attempts = rec_step["tool_attempts"]

        if cursor >= len(recorded_attempts):
            raise _ReplayDivergenceSignal(
                step_index=step_index,
                cause=self._pending_cause or DivergenceCause.CODE_CHANGE,
                detail=f"Extra tool attempt #{cursor} ({tool_name}) at step {step_index}",
            )

        rec_att = recorded_attempts[cursor]
        self._attempt_cursors[step_index] = cursor + 1

        if rec_att["tool_name"] != tool_name:
            raise _ReplayDivergenceSignal(
                step_index=step_index,
                cause=self._pending_cause or DivergenceCause.CODE_CHANGE,
                detail=f"Tool mismatch at step {step_index}: expected {rec_att['tool_name']}, got {tool_name}",
            )

        spec = self.registry.get(tool_name)

        # Determine service_ms and backoff_ms
        if self.level >= CaptureLevel.L2 and rec_att.get("service_ms") is not None:
            service_ms = float(rec_att["service_ms"])
            if self.config.mutations.skipped_backoff:
                backoff_ms = 0.0
            else:
                backoff_ms = float(rec_att.get("backoff_ms", 0.0))
        else:
            service_ms = self._fresh_service_ms(tool_name, call_index)
            will_retry = rec_att.get("error") is not None and attempt_index < max_retries
            backoff_ms = (
                0.0
                if (not will_retry or self.config.mutations.skipped_backoff)
                else round(self.config.backoff_ms * (attempt_index + 1), 3)
            )
            if self.reference_l3 is not None and step_index < len(self.reference_l3.steps):
                ref_atts = self.reference_l3.steps[step_index]["tool_attempts"]
                if cursor < len(ref_atts):
                    ref_ms = float(ref_atts[cursor]["service_ms"])
                    if abs(service_ms - ref_ms) > 1e-6 and self._pending_cause is None:
                        self._pending_cause = DivergenceCause.CLOCK
                        self._pending_detail = (
                            f"Ambient service_ms {service_ms} != recorded {ref_ms} at step {step_index}"
                        )

        # If the recorded attempt raised a transient error, serve that exact error
        if rec_att.get("error") is not None:
            return ToolAttemptRecord(
                tool_name=tool_name,
                effect=spec.effect.value,
                args=copy.deepcopy(args),
                call_index=call_index,
                attempt_index=attempt_index,
                service_ms=service_ms,
                backoff_ms=backoff_ms,
                error=str(rec_att["error"]),
                result=None,
            )

        # Recorded attempt succeeded: serve captured result at L1+, or handle L0
        if self.level >= CaptureLevel.L1:
            captured_res = copy.deepcopy(rec_att["result"])
            return ToolAttemptRecord(
                tool_name=tool_name,
                effect=spec.effect.value,
                args=copy.deepcopy(args),
                call_index=call_index,
                attempt_index=attempt_index,
                service_ms=service_ms,
                backoff_ms=backoff_ms,
                error=None,
                result=captured_res,
            )

        # Level L0: tool results were NOT captured
        if spec.effect == EffectClass.WRITE:
            if self.safe_mode:
                raise _ReplayAbortSignal(
                    step_index=step_index,
                    detail=f"Refused uncaptured WRITE tool '{tool_name}' at step {step_index} in safe mode",
                )
            self.writes_re_fired += 1
            live_res = spec.fn(self.world, self.attempt_tag, args)
            if self._pending_cause is None:
                self._pending_cause = DivergenceCause.UNCAPTURED_RESULT
                self._pending_detail = f"Uncaptured write '{tool_name}' re-executed live at step {step_index}"
            return ToolAttemptRecord(
                tool_name=tool_name,
                effect=spec.effect.value,
                args=copy.deepcopy(args),
                call_index=call_index,
                attempt_index=attempt_index,
                service_ms=service_ms,
                backoff_ms=backoff_ms,
                error=None,
                result=copy.deepcopy(live_res),
            )

        # L0 PURE or READ: run against live world
        live_res = spec.fn(self.world, self.attempt_tag, args)
        if self.reference_l3 is not None and step_index < len(self.reference_l3.steps):
            ref_atts = self.reference_l3.steps[step_index]["tool_attempts"]
            if cursor < len(ref_atts):
                ref_res = ref_atts[cursor].get("result")
                if ref_res != live_res:
                    self._pending_cause = DivergenceCause.DRIFTED_READ
                    self._pending_detail = (
                        f"Live {tool_name} returned {live_res} != recorded {ref_res} at step {step_index}"
                    )
        return ToolAttemptRecord(
            tool_name=tool_name,
            effect=spec.effect.value,
            args=copy.deepcopy(args),
            call_index=call_index,
            attempt_index=attempt_index,
            service_ms=service_ms,
            backoff_ms=backoff_ms,
            error=None,
            result=copy.deepcopy(live_res),
        )

    def on_step_complete(self, step_record: StepRecord, state: dict[str, Any]) -> None:
        rec_step = self._get_step_trace(step_record.step_index)
        used = self._attempt_cursors.get(step_record.step_index, 0)
        if used != len(rec_step["tool_attempts"]):
            raise _ReplayDivergenceSignal(
                step_index=step_record.step_index,
                cause=self._pending_cause or DivergenceCause.CODE_CHANGE,
                detail=f"Step {step_record.step_index} consumed {used} attempts, expected {len(rec_step['tool_attempts'])}",
            )

        # Check volatile fingerprint mismatch first if requested (for the README volatile-field proof)
        if self.include_volatile and self.level >= CaptureLevel.L3:
            rec_raw = rec_step.get("raw_fingerprint_with_volatile")
            if rec_raw and step_record.raw_fingerprint_with_volatile != rec_raw:
                raise _ReplayDivergenceSignal(
                    step_index=step_record.step_index,
                    cause=DivergenceCause.VOLATILE_FINGERPRINT,
                    detail=f"Volatile timestamp/attempt_tag poisoned state hash at step {step_record.step_index}",
                )

        # Check next_node transition
        if step_record.next_node != rec_step["next_node"]:
            cause = self._pending_cause or DivergenceCause.CODE_CHANGE
            detail = (
                self._pending_detail
                or f"Transition mismatch at step {step_record.step_index}: {step_record.node} -> {step_record.next_node} (expected {rec_step['next_node']})"
            )
            raise _ReplayDivergenceSignal(
                step_index=step_record.step_index, cause=cause, detail=detail
            )

        # At L3, verify step state fingerprint immediately
        if self.level >= CaptureLevel.L3:
            expected_fp = rec_step["state_fingerprint"]
            if step_record.state_fingerprint != expected_fp:
                cause = self._pending_cause or DivergenceCause.CODE_CHANGE
                detail = (
                    self._pending_detail
                    or f"Step {step_record.step_index} fingerprint mismatch at node '{step_record.node}'"
                )
                raise _ReplayDivergenceSignal(
                    step_index=step_record.step_index, cause=cause, detail=detail
                )


def replay_trace(
    bundle: TraceBundle,
    world: OrderDeskWorld,
    config: AgentConfig | None = None,
    registry: ToolRegistry | None = None,
    attempt_tag: str = "replay_0",
    safe_mode: bool = True,
    reference_l3_bundle: TraceBundle | None = None,
    include_volatile_in_fingerprint: bool = False,
    wall_clock_base_ms: float = 1_700_005_000_000.0,
) -> ReplayResult:
    """Replay `bundle` against `world` and classify outcome as EXACT, DIVERGED, or ABORTED."""
    cfg = config or AgentConfig()
    reg = registry or build_default_registry()
    driver = ReplayDriver(
        bundle=bundle,
        world=world,
        registry=reg,
        config=cfg,
        attempt_tag=attempt_tag,
        safe_mode=safe_mode,
        reference_l3_bundle=reference_l3_bundle,
        include_volatile_in_fingerprint=include_volatile_in_fingerprint,
    )
    try:
        outcome = execute_agent_with_driver(
            seed=bundle.header.seed,
            attempt_tag=attempt_tag,
            driver=driver,
            config=cfg,
            wall_clock_base_ms=wall_clock_base_ms,
        )
    except _ReplayAbortSignal as abort_exc:
        return ReplayResult(
            status=ReplayStatus.ABORTED,
            diverged_step=abort_exc.step_index,
            cause=DivergenceCause.UNCAPTURED_RESULT,
            detail=abort_exc.detail,
            outcome=None,
            writes_re_fired=driver.writes_re_fired,
        )
    except _ReplayDivergenceSignal as div_exc:
        return ReplayResult(
            status=ReplayStatus.DIVERGED,
            diverged_step=div_exc.step_index,
            cause=div_exc.cause,
            detail=div_exc.detail,
            outcome=None,
            writes_re_fired=driver.writes_re_fired,
        )

    # Check final fingerprint against reference L3 or bundle end record
    expected_final_fp = (
        reference_l3_bundle.end.final_fingerprint
        if (reference_l3_bundle and reference_l3_bundle.end)
        else (bundle.end.final_fingerprint if bundle.end else None)
    )
    if expected_final_fp is not None and outcome.final_fingerprint != expected_final_fp:
        cause = driver._pending_cause or DivergenceCause.CODE_CHANGE
        detail = (
            driver._pending_detail
            or "Final state fingerprint diverged from recorded execution"
        )
        # Attribute to first step where state fingerprint differed from reference L3 if available
        div_step = len(outcome.steps) - 1
        if reference_l3_bundle is not None:
            for idx, s in enumerate(outcome.steps):
                if idx < len(reference_l3_bundle.steps):
                    if s.state_fingerprint != reference_l3_bundle.steps[idx]["state_fingerprint"]:
                        div_step = idx
                        break
        return ReplayResult(
            status=ReplayStatus.DIVERGED,
            diverged_step=div_step,
            cause=cause,
            detail=detail,
            outcome=outcome,
            writes_re_fired=driver.writes_re_fired,
        )

    return ReplayResult(
        status=ReplayStatus.EXACT,
        diverged_step=None,
        cause=None,
        detail="Bit-identical replay verified by canonical state fingerprint",
        outcome=outcome,
        writes_re_fired=driver.writes_re_fired,
    )


def rerun_live(
    world: OrderDeskWorld,
    config: AgentConfig | None = None,
    registry: ToolRegistry | None = None,
    attempt_tag: str = "rerun_1",
) -> RunOutcome:
    """Re-run the agent from scratch against the live (already mutated and drifted) world."""
    return run_agent_live(
        world=world,
        config=config,
        registry=registry,
        attempt_tag=attempt_tag,
    )


def resume_from_checkpoint(
    snapshot: StepSnapshot,
    world: OrderDeskWorld,
    config: AgentConfig | None = None,
    registry: ToolRegistry | None = None,
    attempt_tag: str = "resume_1",
) -> RunOutcome:
    """Restore pre-step `snapshot` and continue execution live against `world`."""
    cfg = config or AgentConfig()
    reg = registry or build_default_registry()
    driver = LiveDriver(
        world=world,
        registry=reg,
        config=cfg,
        attempt_tag=attempt_tag,
    )
    return execute_agent_with_driver(
        seed=world.seed,
        attempt_tag=attempt_tag,
        driver=driver,
        config=cfg,
        start_state=snapshot.state,
        start_step_index=snapshot.step_index,
        start_call_index=snapshot.call_index,
    )
