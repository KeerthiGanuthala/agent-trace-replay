"""Property tests for M6: anchor-phrase figure guard and self-test verification."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_numbers import GUARDED_FIGURES, run_self_test, verify_documents


class TestMeasurementDisciplineChecker(unittest.TestCase):
    def test_check_numbers_passes_on_actual_docs_and_fails_when_any_figure_is_broken(self) -> None:
        metrics_path = ROOT / "docs" / "metrics.json"
        if not metrics_path.exists():
            self.skipTest("metrics.json not generated yet")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(verify_documents(metrics), [])
        self.assertEqual(run_self_test(metrics), 0)

        # Verify every single guarded figure fails if its anchor value is altered
        for guard in GUARDED_FIGURES:
            with self.subTest(metric=guard.metric_path):
                orig = (ROOT / guard.doc_rel_path).read_text(encoding="utf-8")
                phrase = guard.expected_phrase(metrics)
                broken = orig.replace(phrase, guard.template.replace("{VAL}", "BROKEN_VAL"), 1)
                errs = verify_documents(metrics, doc_overrides={guard.doc_rel_path: broken})
                self.assertEqual(len(errs), 1)
