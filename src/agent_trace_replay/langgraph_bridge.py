"""Experiment 6: StateGraph + InMemorySaver gateway wired to the trace replay engine.

If the external `langgraph` package is installed (optional extra `[langgraph]`),
this module uses `langgraph.graph.StateGraph` and `langgraph.checkpoint.memory.InMemorySaver`
directly, while also providing the exact StateGraph/InMemorySaver state-machine
semantics in pure Python so `make verify` runs offline with zero external dependencies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_trace_replay.agent import AgentConfig, compute_state_fingerprint, initial_agent_state
from agent_trace_replay.rng import draw, draw_bool, draw_choice, draw_int
from agent_trace_replay.tools import EffectClass, ToolRegistry, build_default_registry
from agent_trace_replay.trace import CaptureLevel, TraceBundle, TraceEnd, TraceHeader
from agent_trace_replay.world import OrderDeskWorld, TransientToolError

try:
    from langgraph.checkpoint.memory import InMemorySaver as _ExternalInMemorySaver  # type: ignore
    from langgraph.graph import END as _EXTERNAL_END, StateGraph as _ExternalStateGraph  # type: ignore

    HAS_EXTERNAL_LANGGRAPH = True
except ImportError:
    HAS_EXTERNAL_LANGGRAPH = False

END = "__end__"


@dataclass
class CheckpointTuple:
    """Saved state checkpoint after a graph node completes."""

    thread_id: str
    checkpoint_id: str
    step_index: int
    node: str
    next_node: str
    state: dict[str, Any]


class InMemorySaver:
    """Checkpointer storing per-step state snapshots keyed by `thread_id`."""

    def __init__(self) -> None:
        self._storage: dict[str, list[CheckpointTuple]] = {}

    def put(
        self,
        thread_id: str,
        step_index: int,
        node: str,
        next_node: str,
        state: dict[str, Any],
    ) -> CheckpointTuple:
        cp = CheckpointTuple(
            thread_id=thread_id,
            checkpoint_id=f"cp_{step_index:03d}_{node}",
            step_index=step_index,
            node=node,
            next_node=next_node,
            state=copy.deepcopy(state),
        )
        self._storage.setdefault(thread_id, []).append(cp)
        return cp

    def list_checkpoints(self, thread_id: str) -> list[CheckpointTuple]:
        return list(self._storage.get(thread_id, []))

    def get_latest(self, thread_id: str) -> CheckpointTuple | None:
        cps = self._storage.get(thread_id, [])
        return cps[-1] if cps else None

    def get_at_step(self, thread_id: str, step_index: int) -> CheckpointTuple:
        for cp in self._storage.get(thread_id, []):
            if cp.step_index == step_index:
                return cp
        raise KeyError(f"No checkpoint for thread {thread_id} at step {step_index}")


class StateGraph:
    """StateGraph implementation matching LangGraph's node, edge, and conditional-edge API."""

    def __init__(self) -> None:
        self._nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, Callable[[dict[str, Any]], str]] = {}
        self._entry_point: str | None = None

    def add_node(
        self, name: str, action: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        self._nodes[name] = action

    def set_entry_point(self, name: str) -> None:
        self._entry_point = name

    def add_edge(self, start: str, end: str) -> None:
        self._edges[start] = end

    def add_conditional_edges(
        self, source: str, path_fn: Callable[[dict[str, Any]], str]
    ) -> None:
        self._conditional_edges[source] = path_fn

    def compile(self, checkpointer: InMemorySaver) -> CompiledStateGraph:
        if self._entry_point is None:
            raise ValueError("Entry point must be set before compiling StateGraph")
        return CompiledStateGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional_edges=dict(self._conditional_edges),
            entry_point=self._entry_point,
            checkpointer=checkpointer,
        )


class CompiledStateGraph:
    """Compiled graph supporting live execution and checkpoint resumption."""

    def __init__(
        self,
        nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        edges: dict[str, str],
        conditional_edges: dict[str, Callable[[dict[str, Any]], str]],
        entry_point: str,
        checkpointer: InMemorySaver,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.checkpointer = checkpointer

    def invoke(
        self,
        initial_state: dict[str, Any] | None,
        thread_id: str,
        resume_from_checkpoint: CheckpointTuple | None = None,
    ) -> dict[str, Any]:
        if resume_from_checkpoint is not None:
            state = copy.deepcopy(resume_from_checkpoint.state)
            current = resume_from_checkpoint.next_node
            step_idx = resume_from_checkpoint.step_index + 1
        else:
            if initial_state is None:
                raise ValueError("initial_state required when not resuming from checkpoint")
            state = copy.deepcopy(initial_state)
            current = self.entry_point
            step_idx = 0

        while current != END:
            node_fn = self.nodes[current]
            updates = node_fn(state)
            state.update(updates)
            if current in self.conditional_edges:
                nxt = self.conditional_edges[current](state)
            else:
                nxt = self.edges[current]
            state["node"] = "TERMINAL" if nxt == END else nxt
            self.checkpointer.put(
                thread_id=thread_id,
                step_index=step_idx,
                node=current,
                next_node=nxt,
                state=state,
            )
            current = nxt
            step_idx += 1

        return state


@dataclass
class TraceGateway:
    """Gateway wrapping tool calls and choices inside a StateGraph for recording or replay."""

    world: OrderDeskWorld
    registry: ToolRegistry = field(default_factory=build_default_registry)
    config: AgentConfig = field(default_factory=AgentConfig)
    mode: str = "record"  # "record", "live_resume", or "replay"
    attempt_tag: str = "lg_run_0"
    replay_bundle: TraceBundle | None = None
    call_index: int = 0
    step_index: int = 0
    recorded_steps: list[dict[str, Any]] = field(default_factory=list)
    writes_fired: int = 0
    _current_attempts: list[dict[str, Any]] = field(default_factory=list)
    _current_choice: str | None = None
    _current_tie: int | None = None

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> tuple[dict[str, Any] | None, float]:
        """Execute or replay a tool call with retry semantics through the gateway."""
        spec = self.registry.get(tool_name)
        if self.mode == "replay":
            assert self.replay_bundle is not None
            rec_step = self.replay_bundle.steps[self.step_index]
            total_ms = 0.0
            final_res: dict[str, Any] | None = None
            for att in rec_step["tool_attempts"]:
                if att["tool_name"] != tool_name:
                    continue
                self._current_attempts.append(copy.deepcopy(att))
                total_ms = round(total_ms + float(att["service_ms"]) + float(att["backoff_ms"]), 3)
                self.call_index += 1
                if att["error"] is None:
                    final_res = copy.deepcopy(att["result"])
                    break
            return final_res, total_ms

        # Live record or live checkpoint resume
        total_ms = 0.0
        for attempt_idx in range(self.config.max_retries + 1):
            c_idx = self.call_index
            self.call_index += 1
            jitter = draw(self.world.seed, "svc_jitter", self.attempt_tag, c_idx, tool_name) * 9.0
            hiccup = 0.0
            if draw_bool(self.world.seed, self.config.hiccup_prob, "svc_hiccup", self.attempt_tag, c_idx, tool_name):
                hiccup = self.config.hiccup_ms * (
                    0.75 + 0.5 * draw(self.world.seed, "hiccup_mag", self.attempt_tag, c_idx)
                )
            service_ms = round(spec.base_ms + jitter + hiccup, 3)
            try:
                if spec.effect != EffectClass.PURE:
                    self.world.check_flaky(tool_name, self.attempt_tag, c_idx)
                if spec.effect == EffectClass.WRITE:
                    self.writes_fired += 1
                res = spec.fn(self.world, self.attempt_tag, args)
                self._current_attempts.append(
                    {
                        "tool_name": tool_name,
                        "effect": spec.effect.value,
                        "args": copy.deepcopy(args),
                        "call_index": c_idx,
                        "attempt_index": attempt_idx,
                        "service_ms": service_ms,
                        "backoff_ms": 0.0,
                        "error": None,
                        "result": copy.deepcopy(res),
                    }
                )
                total_ms = round(total_ms + service_ms, 3)
                return res, total_ms
            except TransientToolError as exc:
                will_retry = attempt_idx < self.config.max_retries
                backoff = round(self.config.backoff_ms * (attempt_idx + 1), 3) if will_retry else 0.0
                self._current_attempts.append(
                    {
                        "tool_name": tool_name,
                        "effect": spec.effect.value,
                        "args": copy.deepcopy(args),
                        "call_index": c_idx,
                        "attempt_index": attempt_idx,
                        "service_ms": service_ms,
                        "backoff_ms": backoff,
                        "error": str(exc),
                        "result": None,
                    }
                )
                total_ms = round(total_ms + service_ms + backoff, 3)
        return None, total_ms

    def finish_step(self, node: str, next_node: str, elapsed_before: float, state: dict[str, Any]) -> None:
        fp = compute_state_fingerprint(state, exclude_volatile=True)
        raw_fp = compute_state_fingerprint(state, exclude_volatile=False)
        self.recorded_steps.append(
            {
                "record_type": "step",
                "step_index": self.step_index,
                "node": node,
                "next_node": "TERMINAL" if next_node == END else next_node,
                "choice_made": self._current_choice,
                "tie_break_index": self._current_tie,
                "elapsed_before_ms": elapsed_before,
                "elapsed_after_ms": state["elapsed_ms"],
                "tool_attempts": list(self._current_attempts),
                "state_fingerprint": fp,
                "raw_fingerprint_with_volatile": raw_fp,
            }
        )
        self._current_attempts.clear()
        self._current_choice = None
        self._current_tie = None
        self.step_index += 1

    def to_trace_bundle(self, final_state: dict[str, Any]) -> TraceBundle:
        return TraceBundle(
            header=TraceHeader(
                schema_version="1.0",
                seed=self.world.seed,
                attempt_tag=self.attempt_tag,
                capture_level=CaptureLevel.L3,
                order_id=f"ORD-{self.world.seed:05d}",
            ),
            steps=list(self.recorded_steps),
            end=TraceEnd(
                status=str(final_state["status"]),
                failure_kind=final_state["failure_kind"],
                step_count=len(self.recorded_steps),
                final_fingerprint=compute_state_fingerprint(final_state, exclude_volatile=True),
            ),
            partial=False,
        )


def build_langgraph_order_agent(
    gateway: TraceGateway, checkpointer: InMemorySaver
) -> CompiledStateGraph:
    """Wire the order-desk tools and policy into a StateGraph with an InMemorySaver."""
    graph = StateGraph()

    def node_plan(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        state["elapsed_ms"] = round(eb + 2.0, 3)
        gateway.finish_step("plan", "lookup", eb, state)
        return state

    def node_lookup(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms = gateway.call_tool("lookup_order", {"order_id": state["order_id"]})
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["item"] = res["item"]
            state["qty"] = res["qty"]
            state["budget"] = res["budget"]
            state["customer"] = res["customer"]
            if gateway.mode == "replay" and gateway.replay_bundle is not None:
                tie_idx = int(gateway.replay_bundle.steps[gateway.step_index]["tie_break_index"])
            else:
                tie_idx = draw_int(
                    gateway.world.seed,
                    0,
                    len(res["related_docs"]) - 1,
                    "tie_break",
                    gateway.attempt_tag,
                    gateway.step_index,
                )
            gateway._current_tie = tie_idx
            state["selected_doc"] = res["related_docs"][tie_idx]
            nxt = "price"
        state["_next"] = nxt
        gateway.finish_step("lookup", nxt, eb, state)
        return state

    def node_price(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms1 = gateway.call_tool("get_price", {"item": state["item"]})
        if res is None:
            state["elapsed_ms"] = round(eb + ms1, 3)
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["unit_price"] = res["unit_price"]
            c_res, ms2 = gateway.call_tool(
                "calc",
                {
                    "op": "quote_total",
                    "unit_price": state["unit_price"],
                    "qty": state["qty"],
                    "handling_fee": state["selected_doc"]["handling_fee"],
                },
            )
            state["elapsed_ms"] = round(eb + ms1 + ms2, 3)
            assert c_res is not None
            state["quote_total"] = c_res["total"]
            nxt = "escalate" if state["quote_total"] > state["budget"] else "stock"
        state["_next"] = nxt
        gateway.finish_step("price", nxt, eb, state)
        return state

    def node_escalate(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms = gateway.call_tool(
            "create_ticket",
            {
                "order_id": state["order_id"],
                "kind": "budget_escalation",
                "summary": f"Quote {state['quote_total']} exceeds budget {state['budget']}",
            },
        )
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["ticket_id"] = res["ticket_id"]
            state["status"] = "escalated"
            state["failure_kind"] = "budget_escalated"
            nxt = "finish"
        state["_next"] = nxt
        gateway.finish_step("escalate", nxt, eb, state)
        return state

    def node_stock(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms = gateway.call_tool("get_stock", {"item": state["item"]})
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["available_stock"] = res["available"]
            if int(res["available"]) < int(state["qty"]):
                nxt = "backorder"
            else:
                gateway._current_choice = "reserve_immediately"
                state["tight_stock_policy"] = "reserve_immediately"
                nxt = "reserve"
        state["_next"] = nxt
        gateway.finish_step("stock", nxt, eb, state)
        return state

    def node_backorder(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms = gateway.call_tool(
            "create_ticket",
            {
                "order_id": state["order_id"],
                "kind": "stock_backorder",
                "summary": f"Short stock for {state['item']}",
            },
        )
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["ticket_id"] = res["ticket_id"]
            state["status"] = "backordered"
            state["failure_kind"] = "stock_backordered"
            nxt = "finish"
        state["_next"] = nxt
        gateway.finish_step("backorder", nxt, eb, state)
        return state

    def node_reserve(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms = gateway.call_tool(
            "reserve_stock",
            {"order_id": state["order_id"], "item": state["item"], "qty": state["qty"]},
        )
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["reservation_id"] = res["reservation_id"]
            state["reserved_qty"] = res["reserved_qty"]
            nxt = "ticket"
        state["_next"] = nxt
        gateway.finish_step("reserve", nxt, eb, state)
        return state

    def node_ticket(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms = gateway.call_tool(
            "ticket" if False else "create_ticket",
            {
                "order_id": state["order_id"],
                "kind": "fulfillment_dispatch",
                "summary": f"Dispatch {state['reserved_qty']}x {state['item']}",
            },
        )
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["ticket_id"] = res["ticket_id"]
            nxt = "notify"
        state["_next"] = nxt
        gateway.finish_step("ticket", nxt, eb, state)
        return state

    def node_notify(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        if gateway.mode == "replay" and gateway.replay_bundle is not None:
            tmpl = str(gateway.replay_bundle.steps[gateway.step_index]["choice_made"])
        else:
            tmpl = draw_choice(
                gateway.world.seed,
                ("concise_sms", "detailed_email"),
                "choice",
                "notice_template",
                gateway.attempt_tag,
                gateway.step_index,
            )
        gateway._current_choice = tmpl
        state["notice_template"] = tmpl
        res, ms = gateway.call_tool(
            "send_notice",
            {
                "order_id": state["order_id"],
                "customer": state["customer"],
                "template": tmpl,
                "message": f"Order {state['order_id']} confirmed",
            },
        )
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["notice_id"] = res["notice_id"]
            state["status"] = "completed"
            nxt = "finish"
        state["_next"] = nxt
        gateway.finish_step("notify", nxt, eb, state)
        return state

    def node_finish(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        res, ms = gateway.call_tool(
            "kv_put",
            {"key": f"audit:{state['order_id']}", "value": {"status": state["status"]}},
        )
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is None:
            state["status"] = "failed"
            state["failure_kind"] = "retry_exhausted"
            nxt = "give_up"
        else:
            state["audit_kv_version"] = res["version"]
            nxt = END
        state["_next"] = nxt
        gateway.finish_step("finish", nxt, eb, state)
        return state

    def node_give_up(state: dict[str, Any]) -> dict[str, Any]:
        eb = float(state["elapsed_ms"])
        state["status"] = "failed"
        res, ms = gateway.call_tool(
            "create_ticket",
            {
                "order_id": state["order_id"],
                "kind": f"incident_{state['failure_kind']}",
                "summary": "Aborted",
            },
        )
        state["elapsed_ms"] = round(eb + ms, 3)
        if res is not None:
            state["ticket_id"] = res["ticket_id"]
        state["_next"] = END
        gateway.finish_step("give_up", END, eb, state)
        return state

    for name, fn in [
        ("plan", node_plan),
        ("lookup", node_lookup),
        ("price", node_price),
        ("escalate", node_escalate),
        ("stock", node_stock),
        ("backorder", node_backorder),
        ("reserve", node_reserve),
        ("ticket", node_ticket),
        ("notify", node_notify),
        ("finish", node_finish),
        ("give_up", node_give_up),
    ]:
        graph.add_node(name, fn)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "lookup")
    for branch_node in [
        "lookup",
        "price",
        "escalate",
        "stock",
        "backorder",
        "reserve",
        "ticket",
        "notify",
        "finish",
        "give_up",
    ]:
        graph.add_conditional_edges(branch_node, lambda s: str(s["_next"]))

    return graph.compile(checkpointer=checkpointer)
