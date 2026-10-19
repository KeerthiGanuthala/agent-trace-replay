"""Simulated deterministic order-desk world with keyed epoch price drift and effect billing."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from agent_trace_replay.rng import draw, draw_bool, draw_choice, draw_int

CATALOG_ITEMS: tuple[str, ...] = (
    "sku_servo_m4",
    "sku_valve_k9",
    "sku_relay_p2",
    "sku_sensor_t7",
    "sku_gasket_r1",
    "sku_pump_v8",
    "sku_filter_c3",
    "sku_bearing_b6",
    "sku_switch_s5",
    "sku_coupler_d0",
    "sku_nozzle_n8",
    "sku_clamp_x4",
)


class TransientToolError(RuntimeError):
    """Raised when a simulated transient tool failure occurs."""

    def __init__(self, tool_name: str, attempt_tag: str, call_index: int) -> None:
        super().__init__(
            f"transient error in {tool_name} (attempt_tag={attempt_tag}, call_index={call_index})"
        )
        self.tool_name = tool_name
        self.attempt_tag = attempt_tag
        self.call_index = call_index


@dataclass(frozen=True)
class EffectEntry:
    """One recorded write effect in the order-desk world."""

    seq: int
    attempt_tag: str
    epoch: int
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class OrderRecord:
    """Static order request parameters."""

    order_id: str
    item: str
    qty: int
    budget: int
    customer: str
    related_docs: list[dict[str, Any]]


@dataclass
class OrderDeskWorld:
    """Simulated order-desk environment."""

    seed: int
    drift_rate: float = 0.35
    drift_step: int = 9
    flaky_rate: float = 0.125
    epoch: int = 0
    stock: dict[str, int] = field(default_factory=dict)
    tickets: list[dict[str, Any]] = field(default_factory=list)
    notices: list[dict[str, Any]] = field(default_factory=list)
    kv_store: dict[str, Any] = field(default_factory=dict)
    reservations: list[dict[str, Any]] = field(default_factory=list)
    effect_log: list[EffectEntry] = field(default_factory=list)
    orders: dict[str, OrderRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stock:
            for item in CATALOG_ITEMS:
                self.stock[item] = draw_int(self.seed, 2, 14, "init_stock", item)
        if not self.orders:
            self._init_default_order()

    def _init_default_order(self) -> None:
        item = draw_choice(self.seed, CATALOG_ITEMS, "order_item")
        avail = self.stock[item]
        # Sometimes request slightly more than stock or tight stock to exercise branches
        if draw_bool(self.seed, 0.22, "order_tight_stock"):
            qty = avail
        elif draw_bool(self.seed, 0.045, "order_over_stock"):
            qty = avail + draw_int(self.seed, 1, 3, "order_over_delta")
        else:
            qty = draw_int(self.seed, 1, max(1, avail - 1), "order_normal_qty")

        base_unit = self.base_price(item)
        nominal_total = base_unit * qty
        # Tune budget margin so a controlled fraction exceeds budget or drifts over budget
        margin_Roll = draw(self.seed, "order_budget_margin")
        if margin_Roll < 0.042:
            # Over budget even at epoch 0
            budget = max(10, int(nominal_total * 0.90))
        elif margin_Roll < 0.24:
            # Tight budget sensitive to upward price drift or handling surcharge
            budget = nominal_total + draw_int(self.seed, 0, max(4, qty * self.drift_step), "tight_pad")
        else:
            budget = int(nominal_total * 1.35) + 25

        docs = [
            {
                "doc_id": f"doc_{self.seed}_std",
                "tier": "standard",
                "handling_fee": 0,
                "priority_tag": "routine",
            },
            {
                "doc_id": f"doc_{self.seed}_exp",
                "tier": "expedited",
                "handling_fee": 6,
                "priority_tag": "rush",
            },
            {
                "doc_id": f"doc_{self.seed}_ware",
                "tier": "warehouse_direct",
                "handling_fee": 3,
                "priority_tag": "bulk",
            },
        ]
        order_id = f"ORD-{self.seed:05d}"
        self.orders[order_id] = OrderRecord(
            order_id=order_id,
            item=item,
            qty=qty,
            budget=budget,
            customer=f"cust_{(self.seed % 97) + 1:03d}",
            related_docs=docs,
        )

    @property
    def primary_order_id(self) -> str:
        return f"ORD-{self.seed:05d}"

    def base_price(self, item: str) -> int:
        """Return deterministic epoch-0 price for `item` derived from seed alone."""
        if item not in CATALOG_ITEMS:
            raise KeyError(f"Unknown item: {item}")
        return draw_int(self.seed, 40, 160, "base_price", item)

    def price_at_epoch(self, item: str, epoch: int) -> int:
        """Pure function of (seed, item, epoch): base price plus cumulative epoch drift."""
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        price = self.base_price(item)
        for ep in range(1, epoch + 1):
            if draw_bool(self.seed, self.drift_rate, "drift_move", item, ep):
                direction = 1 if draw_bool(self.seed, 0.58, "drift_dir", item, ep) else -1
                price = max(10, price + direction * self.drift_step)
        return price

    def advance_epochs(self, n: int = 1) -> int:
        """Advance world clock by `n` epochs."""
        if n < 0:
            raise ValueError("n must be >= 0")
        self.epoch += n
        return self.epoch

    def check_flaky(self, tool_name: str, attempt_tag: str, call_index: int) -> None:
        """Raise TransientToolError with probability `flaky_rate` keyed on attempt_tag and call_index."""
        if draw_bool(self.seed, self.flaky_rate, "flaky", attempt_tag, call_index, tool_name):
            raise TransientToolError(tool_name=tool_name, attempt_tag=attempt_tag, call_index=call_index)

    def _record_effect(
        self, attempt_tag: str, tool_name: str, args: dict[str, Any], result: dict[str, Any]
    ) -> None:
        entry = EffectEntry(
            seq=len(self.effect_log) + 1,
            attempt_tag=attempt_tag,
            epoch=self.epoch,
            tool_name=tool_name,
            args=copy.deepcopy(args),
            result=copy.deepcopy(result),
        )
        self.effect_log.append(entry)

    def effects_for_attempt(self, attempt_tag: str) -> list[EffectEntry]:
        """Return all write effects billed to `attempt_tag`."""
        return [e for e in self.effect_log if e.attempt_tag == attempt_tag]

    def duplicate_effects_between(
        self, original_tag: str, debug_tag: str
    ) -> list[EffectEntry]:
        """Return write effects fired by `debug_tag` during a debugging session."""
        return [e for e in self.effect_log if e.attempt_tag == debug_tag]

    def clone(self) -> OrderDeskWorld:
        """Deep copy the world state for controlled side-by-side experiments."""
        return copy.deepcopy(self)
