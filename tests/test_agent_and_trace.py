"""Property tests for M2 and M3: agent orchestration, volatile fingerprint hygiene, and JSONL traces."""

from __future__ import annotations

import json
import unittest

from agent_trace_replay.agent import (
    AgentConfig,
    compute_state_fingerprint,
    initial_agent_state,
    run_agent_live,
)
from agent_trace_replay.trace import (
    CaptureLevel,
    TraceCorruptionError,
    outcome_to_full_trace,
    parse_trace,
    project_trace,
    serialize_trace,
)
from agent_trace_replay.world import OrderDeskWorld


class TestAgentAndTraceFormat(unittest.TestCase):
    def test_fingerprint_excludes_volatile_fields_and_diverges_when_volatile_included(self) -> None:
        """Story test pinning the bug we hit in M3:

        During initial M3 testing, we hashed `json.dumps(state, sort_keys=True)` directly.
        Every single replay diverged at Step 0 (`plan`) even when tool outputs and RNG draws
        were 100% identical, because `step_wall_timestamp_ms` and `attempt_tag` ("replay_0"
        vs "run_0") lived inside the state dictionary. Excluding `VOLATILE_STATE_KEYS`
        prior to canonical JSON hashing made step fingerprints invariant to wall-clock time
        and session tags while remaining sensitive to every semantic state mutation.
        """
        s1 = initial_agent_state("ORD-00001", "run_0", start_wall_ms=1700000000000.0)
        s2 = initial_agent_state("ORD-00001", "replay_99", start_wall_ms=1700999999999.0)
        s2["last_call_latency_ms"] = 42.125

        self.assertEqual(
            compute_state_fingerprint(s1, exclude_volatile=True),
            compute_state_fingerprint(s2, exclude_volatile=True),
        )
        self.assertNotEqual(
            compute_state_fingerprint(s1, exclude_volatile=False),
            compute_state_fingerprint(s2, exclude_volatile=False),
        )

    def test_trace_serializes_and_parses_across_all_four_capture_levels(self) -> None:
        for seed in range(1, 16):
            w = OrderDeskWorld(seed=seed)
            out = run_agent_live(w, AgentConfig())
            full_bundle = outcome_to_full_trace(out)
            for lvl in (CaptureLevel.L0, CaptureLevel.L1, CaptureLevel.L2, CaptureLevel.L3):
                with self.subTest(seed=seed, level=lvl.name):
                    projected = project_trace(full_bundle, lvl)
                    raw_jsonl = serialize_trace(projected)
                    parsed = parse_trace(raw_jsonl)
                    self.assertEqual(parsed.header.capture_level, lvl)
                    self.assertEqual(parsed.header.seed, seed)
                    self.assertEqual(len(parsed.steps), len(out.steps))
                    if lvl == CaptureLevel.L0:
                        for st in parsed.steps:
                            for att in st["tool_attempts"]:
                                self.assertNotIn("result", att)
                                self.assertNotIn("service_ms", att)
                    elif lvl == CaptureLevel.L1:
                        for st in parsed.steps:
                            for att in st["tool_attempts"]:
                                self.assertIn("result", att)
                                self.assertNotIn("service_ms", att)
                    elif lvl == CaptureLevel.L2:
                        for st in parsed.steps:
                            self.assertIn("elapsed_after_ms", st)
                            self.assertNotIn("state_fingerprint", st)
                    elif lvl == CaptureLevel.L3:
                        for st in parsed.steps:
                            self.assertIn("state_fingerprint", st)

    def test_project_trace_refuses_upward_projection(self) -> None:
        w = OrderDeskWorld(seed=3)
        out = run_agent_live(w, AgentConfig())
        l1 = project_trace(outcome_to_full_trace(out), CaptureLevel.L1)
        with self.assertRaises(ValueError):
            project_trace(l1, CaptureLevel.L3)

    def test_parse_trace_refuses_corrupted_jsonl_with_specific_reason(self) -> None:
        w = OrderDeskWorld(seed=4)
        out = run_agent_live(w, AgentConfig())
        valid_text = serialize_trace(outcome_to_full_trace(out))
        lines = valid_text.strip().splitlines()

        cases = [
            ("", "Empty trace"),
            ("{bad json" + chr(10) + "", "Invalid JSON on line 1"),
            (json.dumps({"hello": "world"}) + chr(10), "Missing 'record_type'"),
            (lines[1] + chr(10), "First record must be 'header'"),
            (
                json.dumps({**json.loads(lines[0]), "schema_version": "9.9"}) + chr(10),
                "Unsupported schema_version",
            ),
            (
                json.dumps({**json.loads(lines[0]), "capture_level": 99}) + chr(10),
                "Malformed header fields",
            ),
            (chr(10).join(lines[:-1]) + chr(10), "Truncated trace: missing 'end' record"),
            (
                chr(10).join([lines[0], lines[2], lines[-1]]) + chr(10),
                "Non-contiguous step_index",
            ),
            (
                chr(10).join([lines[0], json.dumps({"record_type": "step", "step_index": 0}), lines[-1]])
                + chr(10),
                "missing required keys",
            ),
            (
                chr(10).join([*lines, lines[-1]]) + chr(10),
                "Duplicate end record",
            ),
            (
                chr(10).join([*lines, lines[1]]) + chr(10),
                "Step record found after end record",
            ),
            (
                chr(10).join([lines[0], json.dumps({"record_type": "mystery"}), lines[-1]]) + chr(10),
                "Unknown record_type",
            ),
            (
                chr(10).join(
                    [
                        *lines[:-1],
                        json.dumps({**json.loads(lines[-1]), "step_count": 999}),
                    ]
                )
                + chr(10),
                "does not match steps",
            ),
        ]

        for corrupted, expected_msg in cases:
            with self.subTest(expected_msg=expected_msg):
                with self.assertRaises(TraceCorruptionError) as ctx:
                    parse_trace(corrupted)
                self.assertIn(expected_msg, str(ctx.exception))

    def test_append_only_jsonl_prefix_remains_readable_after_mid_run_crash(self) -> None:
        w = OrderDeskWorld(seed=8)
        out = run_agent_live(w, AgentConfig())
        valid_text = serialize_trace(outcome_to_full_trace(out))
        truncated = chr(10).join(valid_text.strip().splitlines()[:-1]) + chr(10)
        bundle = parse_trace(truncated, allow_partial=True)
        self.assertTrue(bundle.partial)
        self.assertIsNone(bundle.end)
        self.assertEqual(len(bundle.steps), len(out.steps))
