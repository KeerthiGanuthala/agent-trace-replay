"""Declared tool effect taxonomy (PURE, READ, WRITE) and registry enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from agent_trace_replay.world import OrderDeskWorld


class EffectClass(str, Enum):
    """Every tool declares exactly one effect class."""

    PURE = "PURE"
    READ = "READ"
    WRITE = "WRITE"


class UndeclaredToolError(KeyError):
    """Raised when attempting to look up or invoke an undeclared tool."""


@dataclass(frozen=True)
class ToolSpec:
    """Registration metadata and implementation for a single tool."""

    name: str
    effect: EffectClass
    base_ms: float
    fn: Callable[[OrderDeskWorld, str, dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    """Strict registry of declared tools and their effect classes."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        effect: EffectClass,
        base_ms: float,
        fn: Callable[[OrderDeskWorld, str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        if not isinstance(effect, EffectClass):
            raise TypeError(f"effect must be EffectClass, got {effect!r}")
        if name in self._specs:
            raise ValueError(f"Tool already registered: {name}")
        self._specs[name] = ToolSpec(name=name, effect=effect, base_ms=base_ms, fn=fn)

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise UndeclaredToolError(f"Tool '{name}' is not declared in ToolRegistry")
        return self._specs[name]

    def effect_of(self, name: str) -> EffectClass:
        return self.get(name).effect

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs.keys())


def _tool_calc(world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]) -> dict[str, Any]:
    op = args.get("op", "quote_total")
    if op == "quote_total":
        unit_price = int(args["unit_price"])
        qty = int(args["qty"])
        handling_fee = int(args.get("handling_fee", 0))
        total = unit_price * qty + handling_fee
        return {"total": total, "unit_price": unit_price, "qty": qty, "handling_fee": handling_fee}
    raise ValueError(f"Unsupported calc op: {op}")


def _tool_lookup_order(
    world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]
) -> dict[str, Any]:
    order_id = str(args["order_id"])
    rec = world.orders[order_id]
    return {
        "order_id": rec.order_id,
        "item": rec.item,
        "qty": rec.qty,
        "budget": rec.budget,
        "customer": rec.customer,
        "related_docs": [dict(d) for d in rec.related_docs],
    }


def _tool_get_price(
    world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]
) -> dict[str, Any]:
    item = str(args["item"])
    unit_price = world.price_at_epoch(item, world.epoch)
    return {"item": item, "unit_price": unit_price, "epoch": world.epoch}


def _tool_get_stock(
    world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]
) -> dict[str, Any]:
    item = str(args["item"])
    available = int(world.stock.get(item, 0))
    return {"item": item, "available": available}


def _tool_reserve_stock(
    world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]
) -> dict[str, Any]:
    item = str(args["item"])
    qty = int(args["qty"])
    order_id = str(args["order_id"])
    avail = int(world.stock.get(item, 0))
    reserved_qty = min(avail, qty)
    world.stock[item] = max(0, avail - reserved_qty)
    reservation_id = f"RES-{len(world.reservations) + 1:04d}"
    record = {
        "reservation_id": reservation_id,
        "order_id": order_id,
        "item": item,
        "reserved_qty": reserved_qty,
        "remaining_stock": world.stock[item],
    }
    world.reservations.append(record)
    world._record_effect(attempt_tag, "reserve_stock", args, record)
    return dict(record)


def _tool_create_ticket(
    world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]
) -> dict[str, Any]:
    order_id = str(args["order_id"])
    kind = str(args["kind"])
    summary = str(args["summary"])
    ticket_id = f"TKT-{len(world.tickets) + 1:04d}"
    record = {
        "ticket_id": ticket_id,
        "order_id": order_id,
        "kind": kind,
        "summary": summary,
    }
    world.tickets.append(record)
    world._record_effect(attempt_tag, "create_ticket", args, record)
    return dict(record)


def _tool_send_notice(
    world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]
) -> dict[str, Any]:
    order_id = str(args["order_id"])
    customer = str(args["customer"])
    template = str(args["template"])
    message = str(args["message"])
    notice_id = f"NTC-{len(world.notices) + 1:04d}"
    record = {
        "notice_id": notice_id,
        "order_id": order_id,
        "customer": customer,
        "template": template,
        "message": message,
    }
    world.notices.append(record)
    world._record_effect(attempt_tag, "send_notice", args, record)
    return dict(record)


def _tool_kv_put(
    world: OrderDeskWorld, attempt_tag: str, args: dict[str, Any]
) -> dict[str, Any]:
    key = str(args["key"])
    value = args["value"]
    world.kv_store[key] = value
    record = {"key": key, "value": value, "version": len(world.effect_log) + 1}
    world._record_effect(attempt_tag, "kv_put", args, record)
    return dict(record)


def build_default_registry() -> ToolRegistry:
    """Construct the canonical 8-tool order-desk registry."""
    reg = ToolRegistry()
    reg.register("calc", EffectClass.PURE, base_ms=6.0, fn=_tool_calc)
    reg.register("lookup_order", EffectClass.READ, base_ms=22.0, fn=_tool_lookup_order)
    reg.register("get_price", EffectClass.READ, base_ms=18.0, fn=_tool_get_price)
    reg.register("get_stock", EffectClass.READ, base_ms=16.0, fn=_tool_get_stock)
    reg.register("reserve_stock", EffectClass.WRITE, base_ms=28.0, fn=_tool_reserve_stock)
    reg.register("create_ticket", EffectClass.WRITE, base_ms=25.0, fn=_tool_create_ticket)
    reg.register("send_notice", EffectClass.WRITE, base_ms=20.0, fn=_tool_send_notice)
    reg.register("kv_put", EffectClass.WRITE, base_ms=12.0, fn=_tool_kv_put)
    return reg
