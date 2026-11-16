#!/usr/bin/env python3
"""Fail the build when any document quotes a figure that `docs/metrics.json` no longer produces.

Every figure is guarded by an exact contextual anchor phrase (`prefix {value} suffix`),
never bare digits. Run with `--self-test` to prove that mutating any single guarded figure
in memory causes the checker to fail cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "docs" / "metrics.json"


@dataclass(frozen=True)
class GuardedFigure:
    doc_rel_path: str
    metric_path: str
    template: str
    fmt: str = "{}"

    def expected_phrase(self, metrics: dict[str, Any]) -> str:
        cur: Any = metrics
        for part in self.metric_path.split("."):
            cur = cur[part]
        formatted_val = self.fmt.format(cur)
        return self.template.replace("{VAL}", formatted_val)


GUARDED_FIGURES: tuple[GuardedFigure, ...] = (
    # README.md headline and cohort figures
    GuardedFigure("README.md", "cohort.total_runs", "cohort of {VAL} seeded order-desk runs"),
    GuardedFigure("README.md", "cohort.failed_runs", "produce {VAL} failed or non-standard runs"),
    GuardedFigure("README.md", "cohort.failure_rate_pct", "failure rate of {VAL}%", "{:.2f}"),
    GuardedFigure("README.md", "exp1.total_duplicate_side_effects", "fires {VAL} duplicate write side effects"),
    GuardedFigure("README.md", "exp1.dup_create_ticket", "{VAL} duplicate tickets"),
    GuardedFigure("README.md", "exp1.dup_reserve_stock", "{VAL} duplicate stock reservations"),
    GuardedFigure("README.md", "exp1.dup_send_notice", "{VAL} duplicate customer notices"),
    GuardedFigure("README.md", "exp1.dup_kv_put", "{VAL} duplicate KV audit writes"),
    GuardedFigure("README.md", "exp1.exact_trajectory_reproduced", "reproduces the exact trajectory in only {VAL} of 466 cases"),
    GuardedFigure("README.md", "exp1.exact_trajectory_rate_pct", "trajectory reproduction rate of {VAL}%", "{:.2f}"),
    GuardedFigure("README.md", "exp1.phantom_successes_on_rerun", "flips {VAL} failures into phantom successes"),
    # README.md Experiment 2 (Checkpoint vs Trace)
    GuardedFigure("README.md", "exp2.resume_refired_writes", "re-fires {VAL} live write side effects"),
    GuardedFigure("README.md", "exp2.resume_phantom_success", "turns {VAL} incidents into phantom successes"),
    GuardedFigure("README.md", "exp2.trace_exact_rate_pct", "achieves {VAL}% bit-identical reproduction", "{:.1f}"),
    GuardedFigure("README.md", "exp2.trace_refired_writes", "with {VAL} re-fired writes"),
    # README.md Experiment 3 (Capture Ladder)
    GuardedFigure("README.md", "exp3.L0.mean_bytes", "| **L0** (Transcript) | {VAL} B |", "{:.1f}"),
    GuardedFigure("README.md", "exp3.L1.mean_bytes", "| **L1** (+ Tool Results) | {VAL} B |", "{:.1f}"),
    GuardedFigure("README.md", "exp3.L2.mean_bytes", "| **L2** (+ Ambient Draws & Clocks) | {VAL} B |", "{:.1f}"),
    GuardedFigure("README.md", "exp3.L3.mean_bytes", "| **L3** (+ Per-Step Fingerprints) | {VAL} B |", "{:.1f}"),
    GuardedFigure("README.md", "exp3.L0.aborted_replays", "safe mode aborts {VAL} of 466 runs"),
    GuardedFigure("README.md", "exp3.L0.diverged_replays", "and {VAL} diverge on drifted reads"),
    GuardedFigure("README.md", "exp3.L1.divergence_causes.clock", "{VAL} diverge on service-time clock checks"),
    GuardedFigure("README.md", "exp3.L1.divergence_causes.rng", "and {VAL} diverge on ambient RNG tie-breaks"),
    GuardedFigure("README.md", "exp3.volatile_step0_divergences", "all {VAL} of 2,000 replays diverge at Step 0"),
    # README.md Experiment 4 (Localization)
    GuardedFigure("README.md", "exp4.dropped_state_field.invisible_to_final_status", "invisible to final status in {VAL} of 1,998 affected runs"),
    GuardedFigure("README.md", "exp4.dropped_state_field.mean_steps_saved_by_l3", "catching the bug {VAL} steps earlier on average", "{:.2f}"),
    GuardedFigure("README.md", "exp4.skipped_backoff.invisible_to_final_status_pct", "{VAL}% of affected runs still finish with the same final status", "{:.2f}"),
    # README.md Experiment 5 & 6
    GuardedFigure("README.md", "exp5.e0_rerun_exact_pct", "starts at {VAL}% at E+0", "{:.2f}"),
    GuardedFigure("README.md", "exp5.e3_rerun_exact_pct", "decays to {VAL}% at E+3", "{:.2f}"),
    GuardedFigure("README.md", "exp5.e3_analytic_pct", "analytic survival model predicts {VAL}% at E+3", "{:.2f}"),
    GuardedFigure("README.md", "exp5.e3_naive_pct", "naive per-step hazard model predicts {VAL}%", "{:.2f}"),
    GuardedFigure("README.md", "exp6.inmemory_saver_duplicate_writes_fired", "InMemorySaver checkpoint resumption fires {VAL} duplicate writes"),
    # Suite metrics in README
    GuardedFigure("README.md", "tests.total_test_cases", "verified by {VAL} automated test checks"),
    GuardedFigure("README.md", "tests.line_coverage_pct", "with {VAL}% line coverage", "{:.2f}"),
    # ADR 0004 & Journal guarded figures
    GuardedFigure("docs/adr/0004-canonical-state-fingerprints-excluding-volatile-fields.md", "exp3.volatile_step0_divergences", "causes {VAL} out of 2,000 L3 replays to fail at Step 0"),
    GuardedFigure("docs/journal.md", "cohort.failed_runs", "landed at {VAL} failed runs out of 2,000"),
)


def verify_documents(
    metrics: dict[str, Any], doc_overrides: dict[str, str] | None = None
) -> list[str]:
    errors: list[str] = []
    cache: dict[str, str] = {}

    for item in GUARDED_FIGURES:
        if doc_overrides and item.doc_rel_path in doc_overrides:
            content = doc_overrides[item.doc_rel_path]
        else:
            if item.doc_rel_path not in cache:
                full_p = ROOT / item.doc_rel_path
                if not full_p.exists():
                    errors.append(f"Missing guarded document: {item.doc_rel_path}")
                    continue
                cache[item.doc_rel_path] = full_p.read_text(encoding="utf-8")
            content = cache[item.doc_rel_path]

        phrase = item.expected_phrase(metrics)
        if phrase not in content:
            errors.append(
                f"[{item.doc_rel_path}] Missing anchored figure for '{item.metric_path}': expected exact phrase {phrase!r}"
            )
    return errors


def run_self_test(metrics: dict[str, Any]) -> int:
    """Prove the checker fails when a figure in README.md is perturbed."""
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    first_guard = GUARDED_FIGURES[0]
    valid_phrase = first_guard.expected_phrase(metrics)
    corrupted_phrase = first_guard.template.replace("{VAL}", "999999")
    corrupted_readme = readme_text.replace(valid_phrase, corrupted_phrase, 1)

    errs = verify_documents(metrics, doc_overrides={"README.md": corrupted_readme})
    if not errs:
        print("[check_numbers] SELF-TEST FAILED: Perturbed figure was not caught!", file=sys.stderr)
        return 1
    print(f"[check_numbers] Self-test passed: Sabotaged figure caught cleanly -> {errs[0]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Prove checker catches a mutated figure")
    args = parser.parse_args()

    if not METRICS_PATH.exists():
        print(f"Missing {METRICS_PATH}; run tools/collect_metrics.py first.", file=sys.stderr)
        return 1

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    errors = verify_documents(metrics)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    print(
        f"[check_numbers] Verified {len(GUARDED_FIGURES)} anchor-guarded figures across README.md, ADRs, and journal.md."
    )
    if args.self_test:
        return run_self_test(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
