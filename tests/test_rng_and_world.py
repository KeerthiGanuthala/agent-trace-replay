"""Property tests for M1: keyed randomness and the simulated order-desk world."""

from __future__ import annotations

import unittest

from agent_trace_replay.rng import draw, draw_bool, draw_choice, draw_int
from agent_trace_replay.tools import (
    EffectClass,
    ToolRegistry,
    UndeclaredToolError,
    build_default_registry,
)
from agent_trace_replay.world import CATALOG_ITEMS, OrderDeskWorld, TransientToolError


class TestKeyedRandomnessAndWorld(unittest.TestCase):
    def test_draw_returns_identical_float_for_identical_seed_and_key(self) -> None:
        for i in range(15):
            with self.subTest(i=i):
                a = draw(100 + i, "namespace", "step", i, True)
                b = draw(100 + i, "namespace", "step", i, True)
                self.assertEqual(a, b)
                self.assertGreaterEqual(a, 0.0)
                self.assertLess(a, 1.0)

    def test_draw_produces_independent_streams_across_namespaces(self) -> None:
        stream_a = [draw(42, "stream_a", idx) for idx in range(20)]
        stream_b = [draw(42, "stream_b", idx) for idx in range(20)]
        self.assertNotEqual(stream_a, stream_b)

    def test_draw_distinguishes_boolean_and_integer_key_types(self) -> None:
        # Ensure canonical key encoding prevents `True` and `1` from colliding
        self.assertNotEqual(draw(7, True), draw(7, 1))
        self.assertNotEqual(draw(7, "1"), draw(7, 1))
        self.assertNotEqual(draw(7, 1.5), draw(7, "1.5"))

    def test_draw_bool_respects_boundary_probabilities(self) -> None:
        self.assertFalse(draw_bool(99, 0.0, "never"))
        self.assertFalse(draw_bool(99, -0.1, "negative"))
        self.assertTrue(draw_bool(99, 1.0, "always"))
        self.assertTrue(draw_bool(99, 1.2, "above_one"))

    def test_draw_int_stays_within_inclusive_bounds_and_rejects_inverted_range(self) -> None:
        for seed in range(30):
            val = draw_int(seed, 5, 9, "bounds_check")
            self.assertGreaterEqual(val, 5)
            self.assertLessEqual(val, 9)
        self.assertEqual(draw_int(1, 7, 7, "singleton"), 7)
        with self.assertRaises(ValueError):
            draw_int(1, 10, 5, "invalid")

    def test_draw_choice_selects_from_sequence_and_rejects_empty_options(self) -> None:
        opts = ("alpha", "beta", "gamma")
        for seed in range(15):
            self.assertIn(draw_choice(seed, opts, "pick"), opts)
        with self.assertRaises(ValueError):
            draw_choice(1, (), "empty")

    def test_price_at_epoch_is_a_pure_function_of_seed_item_and_epoch(self) -> None:
        for seed in range(1, 21):
            w1 = OrderDeskWorld(seed=seed)
            w2 = OrderDeskWorld(seed=seed)
            w2.advance_epochs(5)
            for item in CATALOG_ITEMS[:4]:
                with self.subTest(seed=seed, item=item):
                    self.assertEqual(w1.price_at_epoch(item, 5), w2.price_at_epoch(item, 5))
                    self.assertGreaterEqual(w1.price_at_epoch(item, 5), 10)

    def test_unknown_catalog_item_or_negative_epoch_raises_error(self) -> None:
        w = OrderDeskWorld(seed=1)
        with self.assertRaises(KeyError):
            w.base_price("nonexistent_sku")
        with self.assertRaises(ValueError):
            w.price_at_epoch(CATALOG_ITEMS[0], -1)
        with self.assertRaises(ValueError):
            w.advance_epochs(-2)

    def test_mutable_tables_and_effect_log_bill_duplicates_per_attempt_tag(self) -> None:
        w = OrderDeskWorld(seed=12)
        reg = build_default_registry()
        order_id = w.primary_order_id
        item = w.orders[order_id].item
        initial_avail = w.stock[item]

        res1 = reg.get("reserve_stock").fn(
            w, "incident_0", {"order_id": order_id, "item": item, "qty": 1}
        )
        self.assertEqual(w.stock[item], initial_avail - 1)
        self.assertEqual(res1["reservation_id"], "RES-0001")

        # Second attempt (e.g. a careless debug session) fires duplicate write effects
        reg.get("reserve_stock").fn(
            w, "debug_rerun_1", {"order_id": order_id, "item": item, "qty": 1}
        )
        reg.get("create_ticket").fn(
            w,
            "debug_rerun_1",
            {"order_id": order_id, "kind": "fulfillment", "summary": "dup"},
        )

        self.assertEqual(len(w.effects_for_attempt("incident_0")), 1)
        self.assertEqual(len(w.effects_for_attempt("debug_rerun_1")), 2)
        dups = w.duplicate_effects_between("incident_0", "debug_rerun_1")
        self.assertEqual(len(dups), 2)
        self.assertEqual(dups[0].tool_name, "reserve_stock")
        self.assertEqual(dups[1].tool_name, "create_ticket")

    def test_flakiness_is_keyed_on_attempt_tag_and_call_index(self) -> None:
        w = OrderDeskWorld(seed=77, flaky_rate=1.0)
        with self.assertRaises(TransientToolError) as ctx:
            w.check_flaky("get_price", "run_0", 3)
        self.assertEqual(ctx.exception.tool_name, "get_price")
        self.assertEqual(ctx.exception.attempt_tag, "run_0")
        self.assertEqual(ctx.exception.call_index, 3)

    def test_tool_registry_refuses_undeclared_names_and_duplicate_registrations(self) -> None:
        reg = build_default_registry()
        with self.assertRaises(UndeclaredToolError):
            reg.get("drop_production_database")
        with self.assertRaises(ValueError):
            reg.register("calc", EffectClass.PURE, 5.0, lambda w, t, a: {})
        with self.assertRaises(TypeError):
            reg.register("bad_effect", "NOT_AN_ENUM", 5.0, lambda w, t, a: {})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            reg.get("calc").fn(OrderDeskWorld(seed=1), "run_0", {"op": "invalid_op"})
