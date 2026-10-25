"""Hand-written order-desk agent graph and orchestration framework with keyed nondeterminism."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_trace_replay.rng import draw, draw_bool, draw_choice, draw_int
from agent_trace_replay.tools import EffectClass, ToolRegistry, build_default_registry
from agent_trace_replay.world import OrderDeskWorld, TransientToolError

VOLATILE_STATE_KEYS: frozenset[str] = frozenset(
    {
        "last_call_latency_ms",
        "step_wall_timestamp_ms",
        "attempt_tag",
    }
)


@dataclass(frozen=True)
class MutationFlags:
    """Named bug flags for Experiment 4 (localization via per-step fingerprints)."""

    extra_retry: bool = False
    fudged_budget_threshold: bool = False
    dropped_state_field: bool = False
    skipped_backoff: bool = False


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the order-desk agent framework."""

    deadline_ms: float = 325.0
    max_retries: int = 2
    backoff_ms: float = 28.0
    hiccup_prob: float = 0.075
    hiccup_ms: float = 78.0
    mutations: MutationFlags = field(default_factory=MutationFlags)


@dataclass
class ToolAttemptRecord:
    """One physical or replayed tool invocation attempt within a step."""

    tool_name: str
    effect: str
    args: dict[str, Any]
    call_index: int
    attempt_index: int
    service_ms: float
    backoff_ms: float
    error: str | None = None
    result: dict[str, Any] | None = None


@dataclass
class StepRecord:
    """Complete record of one orchestration step."""

    step_index: int
    node: str
    next_node: str
    choice_made: str | None
    tie_break_index: int | None
    elapsed_before_ms: float
    elapsed_after_ms: float
    tool_attempts: list[ToolAttemptRecord]
    state_fingerprint: str
    raw_fingerprint_with_volatile: str


@dataclass
class StepSnapshot:
    """Pre-step checkpoint snapshot for `resume_from_checkpoint`."""

    step_index: int
    node: str
    state: dict[str, Any]
    call_index: int


@dataclass
class RunOutcome:
    """Result of executing or replaying the agent loop."""

    seed: int
    attempt_tag: str
    status: str  # "completed", "escalated", "backordered", "failed"
    failure_kind: str | None  # None, "deadline_exceeded", "retry_exhausted", "budget_escalated", "stock_backordered"
    final_state: dict[str, Any]
    final_fingerprint: str
    steps: list[StepRecord]
    snapshots: list[StepSnapshot]


def canonical_json(data: dict[str, Any], exclude_volatile: bool = True) -> str:
    """Serialize `data` into canonical sorted-key JSON, optionally stripping volatile keys."""
    if exclude_volatile:
        filtered = {k: v for k, v in data.items() if k not in VOLATILE_STATE_KEYS}
    else:
        filtered = dict(data)
    return json.dumps(filtered, sort_keys=True, separators=(",", ":"))


def compute_state_fingerprint(state: dict[str, Any], exclude_volatile: bool = True) -> str:
    """Return SHA-256 hex digest of canonical JSON state."""
    payload = canonical_json(state, exclude_volatile=exclude_volatile)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExecutionDriver(Protocol):
    """Strategy interface allowing the exact same graph orchestration loop to run live or replayed."""

    def before_step(self, step_index: int, node: str, state: dict[str, Any], call_index: int) -> None:
        ...

    def pick_tie_break(self, step_index: int, num_docs: int) -> int:
        ...

    def pick_stochastic_choice(
        self, step_index: int, choice_name: str, options: tuple[str, ...]
    ) -> str:
        ...

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
        ...

    def on_step_complete(self, step_record: StepRecord, state: dict[str, Any]) -> None:
        ...


class LiveDriver:
    """Live execution driver against an `OrderDeskWorld`."""

    def __init__(
        self,
        world: OrderDeskWorld,
        registry: ToolRegistry,
        config: AgentConfig,
        attempt_tag: str,
        clock_bias_ms: float = 0.0,
    ) -> None:
        self.world = world
        self.registry = registry
        self.config = config
        self.attempt_tag = attempt_tag
        self.clock_bias_ms = clock_bias_ms
        self.snapshots: list[StepSnapshot] = []

    def before_step(self, step_index: int, node: str, state: dict[str, Any], call_index: int) -> None:
        self.snapshots.append(
            StepSnapshot(
                step_index=step_index,
                node=node,
                state=copy.deepcopy(state),
                call_index=call_index,
            )
        )

    def pick_tie_break(self, step_index: int, num_docs: int) -> int:
        return draw_int(
            self.world.seed, 0, num_docs - 1, "tie_break", self.attempt_tag, step_index
        )

    def pick_stochastic_choice(
        self, step_index: int, choice_name: str, options: tuple[str, ...]
    ) -> str:
        return draw_choice(
            self.world.seed, options, "choice", choice_name, self.attempt_tag, step_index
        )

    def _draw_service_ms(self, tool_name: str, call_index: int) -> float:
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
        return round(spec.base_ms + jitter + hiccup + self.clock_bias_ms, 3)

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
        spec = self.registry.get(tool_name)
        service_ms = self._draw_service_ms(tool_name, call_index)
        try:
            if spec.effect != EffectClass.PURE:
                self.world.check_flaky(tool_name, self.attempt_tag, call_index)
            result = spec.fn(self.world, self.attempt_tag, args)
            return ToolAttemptRecord(
                tool_name=tool_name,
                effect=spec.effect.value,
                args=copy.deepcopy(args),
                call_index=call_index,
                attempt_index=attempt_index,
                service_ms=service_ms,
                backoff_ms=0.0,
                error=None,
                result=copy.deepcopy(result),
            )
        except TransientToolError as exc:
            will_retry = attempt_index < max_retries
            backoff = (
                0.0
                if (not will_retry or self.config.mutations.skipped_backoff)
                else round(self.config.backoff_ms * (attempt_index + 1), 3)
            )
            return ToolAttemptRecord(
                tool_name=tool_name,
                effect=spec.effect.value,
                args=copy.deepcopy(args),
                call_index=call_index,
                attempt_index=attempt_index,
                service_ms=service_ms,
                backoff_ms=backoff,
                error=str(exc),
                result=None,
            )

    def on_step_complete(self, step_record: StepRecord, state: dict[str, Any]) -> None:
        pass


def initial_agent_state(order_id: str, attempt_tag: str, start_wall_ms: float = 1_700_000_000_000.0) -> dict[str, Any]:
    """Create the initial canonical agent state dictionary."""
    return {
        "order_id": order_id,
        "node": "plan",
        "status": "running",
        "failure_kind": None,
        "elapsed_ms": 0.0,
        "item": None,
        "qty": 0,
        "budget": 0,
        "customer": None,
        "selected_doc": None,
        "unit_price": None,
        "quote_total": None,
        "available_stock": None,
        "tight_stock_policy": None,
        "reservation_id": None,
        "reserved_qty": 0,
        "ticket_id": None,
        "notice_id": None,
        "notice_template": None,
        "audit_kv_version": None,
        "retry_count": 0,
        # Volatile observability fields (excluded from canonical fingerprint)
        "last_call_latency_ms": 0.0,
        "step_wall_timestamp_ms": start_wall_ms,
        "attempt_tag": attempt_tag,
    }


def execute_agent_with_driver(
    seed: int,
    attempt_tag: str,
    driver: ExecutionDriver,
    config: AgentConfig,
    start_state: dict[str, Any] | None = None,
    start_step_index: int = 0,
    start_call_index: int = 0,
    wall_clock_base_ms: float = 1_700_000_000_000.0,
) -> RunOutcome:
    """Execute the order-desk graph using `driver` for all tools, RNG draws, and clocks."""
    order_id = f"ORD-{seed:05d}"
    state = (
        copy.deepcopy(start_state)
        if start_state is not None
        else initial_agent_state(order_id, attempt_tag, start_wall_ms=wall_clock_base_ms)
    )
    state["attempt_tag"] = attempt_tag

    step_index = start_step_index
    call_index = start_call_index
    steps: list[StepRecord] = []

    effective_max_retries = config.max_retries + (1 if config.mutations.extra_retry else 0)

    def _call_with_retries(
        current_node: str, tool_name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[ToolAttemptRecord], float]:
        nonlocal call_index
        attempts: list[ToolAttemptRecord] = []
        step_latency = 0.0
        for attempt_idx in range(effective_max_retries + 1):
            rec = driver.invoke_tool(
                step_index=step_index,
                node=current_node,
                tool_name=tool_name,
                args=args,
                call_index=call_index,
                attempt_index=attempt_idx,
                max_retries=effective_max_retries,
            )
            call_index += 1
            attempts.append(rec)
            step_latency = round(step_latency + rec.service_ms + rec.backoff_ms, 3)
            state["last_call_latency_ms"] = rec.service_ms
            if rec.error is None and rec.result is not None:
                return rec.result, attempts, step_latency
            state["retry_count"] = int(state.get("retry_count", 0)) + 1
        return None, attempts, step_latency

    while state["node"] not in ("TERMINAL",):
        current_node = str(state["node"])
        driver.before_step(step_index, current_node, state, call_index)

        elapsed_before = float(state["elapsed_ms"])
        choice_made: str | None = None
        tie_break_index: int | None = None
        tool_attempts: list[ToolAttemptRecord] = []
        step_delta_ms = 0.0

        # Deadline guard before executing non-terminal work steps (except give_up/finish)
        if current_node not in ("finish", "give_up") and elapsed_before > config.deadline_ms:
            state["failure_kind"] = "deadline_exceeded"
            next_node = "give_up"
        elif current_node == "plan":
            # Pure planning step
            step_delta_ms = 2.0
            state["last_call_latency_ms"] = 2.0
            next_node = "lookup"

        elif current_node == "lookup":
            res, attempts, step_delta_ms = _call_with_retries(
                "lookup", "lookup_order", {"order_id": state["order_id"]}
            )
            tool_attempts.extend(attempts)
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["item"] = res["item"]
                state["qty"] = res["qty"]
                state["budget"] = res["budget"]
                state["customer"] = res["customer"]
                docs = res["related_docs"]
                tie_break_index = driver.pick_tie_break(step_index, len(docs))
                selected = docs[tie_break_index]
                if config.mutations.dropped_state_field:
                    selected = {k: v for k, v in selected.items() if k != "priority_tag"}
                state["selected_doc"] = selected
                next_node = "price"

        elif current_node == "price":
            res, attempts, delta1 = _call_with_retries(
                "price", "get_price", {"item": state["item"]}
            )
            tool_attempts.extend(attempts)
            step_delta_ms = delta1
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["unit_price"] = res["unit_price"]
                handling = int(state["selected_doc"]["handling_fee"])
                calc_res, calc_attempts, delta2 = _call_with_retries(
                    "price",
                    "calc",
                    {
                        "op": "quote_total",
                        "unit_price": state["unit_price"],
                        "qty": state["qty"],
                        "handling_fee": handling,
                    },
                )
                tool_attempts.extend(calc_attempts)
                step_delta_ms = round(step_delta_ms + delta2, 3)
                assert calc_res is not None
                state["quote_total"] = calc_res["total"]

                budget_limit = float(state["budget"])
                if config.mutations.fudged_budget_threshold:
                    budget_limit = budget_limit * 1.05

                if float(state["quote_total"]) > budget_limit:
                    next_node = "escalate"
                else:
                    next_node = "stock"

        elif current_node == "escalate":
            res, attempts, step_delta_ms = _call_with_retries(
                "escalate",
                "create_ticket",
                {
                    "order_id": state["order_id"],
                    "kind": "budget_escalation",
                    "summary": f"Quote {state['quote_total']} exceeds budget {state['budget']}",
                },
            )
            tool_attempts.extend(attempts)
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["ticket_id"] = res["ticket_id"]
                state["status"] = "escalated"
                state["failure_kind"] = "budget_escalated"
                next_node = "finish"

        elif current_node == "stock":
            res, attempts, delta1 = _call_with_retries(
                "stock", "get_stock", {"item": state["item"]}
            )
            tool_attempts.extend(attempts)
            step_delta_ms = delta1
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                avail = int(res["available"])
                state["available_stock"] = avail
                req_qty = int(state["qty"])
                if avail < req_qty:
                    next_node = "backorder"
                else:
                    # Stochastic model choice point 1: when stock is tight (avail <= req_qty + 2),
                    # decide whether to reserve immediately or re-verify price first.
                    if avail <= req_qty + 2:
                        choice_made = driver.pick_stochastic_choice(
                            step_index,
                            "tight_stock_policy",
                            ("reserve_immediately", "recheck_price_first"),
                        )
                    else:
                        choice_made = "reserve_immediately"
                    state["tight_stock_policy"] = choice_made

                    if choice_made == "recheck_price_first":
                        re_res, re_attempts, delta2 = _call_with_retries(
                            "stock", "get_price", {"item": state["item"]}
                        )
                        tool_attempts.extend(re_attempts)
                        step_delta_ms = round(step_delta_ms + delta2, 3)
                        if re_res is None:
                            state["failure_kind"] = "retry_exhausted"
                            next_node = "give_up"
                        else:
                            state["unit_price"] = re_res["unit_price"]
                            next_node = "reserve"
                    else:
                        next_node = "reserve"

        elif current_node == "backorder":
            res, attempts, step_delta_ms = _call_with_retries(
                "backorder",
                "create_ticket",
                {
                    "order_id": state["order_id"],
                    "kind": "stock_backorder",
                    "summary": f"Need {state['qty']} of {state['item']}, only {state['available_stock']} in stock",
                },
            )
            tool_attempts.extend(attempts)
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["ticket_id"] = res["ticket_id"]
                state["status"] = "backordered"
                state["failure_kind"] = "stock_backordered"
                next_node = "finish"

        elif current_node == "reserve":
            res, attempts, step_delta_ms = _call_with_retries(
                "reserve",
                "reserve_stock",
                {
                    "order_id": state["order_id"],
                    "item": state["item"],
                    "qty": state["qty"],
                },
            )
            tool_attempts.extend(attempts)
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["reservation_id"] = res["reservation_id"]
                state["reserved_qty"] = res["reserved_qty"]
                next_node = "ticket"

        elif current_node == "ticket":
            res, attempts, step_delta_ms = _call_with_retries(
                "ticket",
                "create_ticket",
                {
                    "order_id": state["order_id"],
                    "kind": "fulfillment_dispatch",
                    "summary": f"Dispatch {state['reserved_qty']}x {state['item']} ({state['reservation_id']})",
                },
            )
            tool_attempts.extend(attempts)
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["ticket_id"] = res["ticket_id"]
                next_node = "notify"

        elif current_node == "notify":
            # Stochastic model choice point 2: notification template selection
            choice_made = driver.pick_stochastic_choice(
                step_index, "notice_template", ("concise_sms", "detailed_email")
            )
            state["notice_template"] = choice_made
            res, attempts, step_delta_ms = _call_with_retries(
                "notify",
                "send_notice",
                {
                    "order_id": state["order_id"],
                    "customer": state["customer"],
                    "template": choice_made,
                    "message": f"Order {state['order_id']} confirmed under ticket {state['ticket_id']}",
                },
            )
            tool_attempts.extend(attempts)
            if res is None:
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["notice_id"] = res["notice_id"]
                state["status"] = "completed"
                next_node = "finish"

        elif current_node == "finish":
            res, attempts, step_delta_ms = _call_with_retries(
                "finish",
                "kv_put",
                {
                    "key": f"audit:{state['order_id']}",
                    "value": {
                        "status": state["status"],
                        "ticket_id": state["ticket_id"],
                        "reservation_id": state["reservation_id"],
                        "notice_id": state["notice_id"],
                    },
                },
            )
            tool_attempts.extend(attempts)
            if res is None:
                state["status"] = "failed"
                state["failure_kind"] = "retry_exhausted"
                next_node = "give_up"
            else:
                state["audit_kv_version"] = res["version"]
                next_node = "TERMINAL"

        elif current_node == "give_up":
            state["status"] = "failed"
            res, attempts, step_delta_ms = _call_with_retries(
                "give_up",
                "create_ticket",
                {
                    "order_id": state["order_id"],
                    "kind": f"incident_{state['failure_kind'] or 'unknown'}",
                    "summary": f"Run aborted at {elapsed_before:.1f}ms ({state['failure_kind']})",
                },
            )
            tool_attempts.extend(attempts)
            if res is not None:
                state["ticket_id"] = res["ticket_id"]
            next_node = "TERMINAL"

        else:
            raise ValueError(f"Unknown graph node: {current_node}")

        elapsed_after = round(elapsed_before + step_delta_ms, 3)
        state["elapsed_ms"] = elapsed_after
        state["step_wall_timestamp_ms"] = round(
            float(state["step_wall_timestamp_ms"]) + step_delta_ms + 1.137, 3
        )
        state["node"] = next_node

        fp_clean = compute_state_fingerprint(state, exclude_volatile=True)
        fp_raw = compute_state_fingerprint(state, exclude_volatile=False)

        step_rec = StepRecord(
            step_index=step_index,
            node=current_node,
            next_node=next_node,
            choice_made=choice_made,
            tie_break_index=tie_break_index,
            elapsed_before_ms=elapsed_before,
            elapsed_after_ms=elapsed_after,
            tool_attempts=tool_attempts,
            state_fingerprint=fp_clean,
            raw_fingerprint_with_volatile=fp_raw,
        )
        steps.append(step_rec)
        driver.on_step_complete(step_rec, state)
        step_index += 1

    final_fp = compute_state_fingerprint(state, exclude_volatile=True)
    snapshots = getattr(driver, "snapshots", [])
    return RunOutcome(
        seed=seed,
        attempt_tag=attempt_tag,
        status=str(state["status"]),
        failure_kind=state["failure_kind"],
        final_state=copy.deepcopy(state),
        final_fingerprint=final_fp,
        steps=steps,
        snapshots=snapshots,
    )


def run_agent_live(
    world: OrderDeskWorld,
    config: AgentConfig | None = None,
    registry: ToolRegistry | None = None,
    attempt_tag: str = "run_0",
    clock_bias_ms: float = 0.0,
    wall_clock_base_ms: float = 1_700_000_000_000.0,
) -> RunOutcome:
    """Execute the order-desk agent live against `world`."""
    cfg = config or AgentConfig()
    reg = registry or build_default_registry()
    driver = LiveDriver(
        world=world,
        registry=reg,
        config=cfg,
        attempt_tag=attempt_tag,
        clock_bias_ms=clock_bias_ms,
    )
    return execute_agent_with_driver(
        seed=world.seed,
        attempt_tag=attempt_tag,
        driver=driver,
        config=cfg,
        wall_clock_base_ms=wall_clock_base_ms,
    )
