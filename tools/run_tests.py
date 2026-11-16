#!/usr/bin/env python3
"""Run test suite with stdlib `sys.settrace` line coverage and emit machine-readable report."""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src" / "agent_trace_replay"
TESTS_DIR = ROOT / "tests"
REPORT_PATH = ROOT / "docs" / "test_report.json"

sys.path.insert(0, str(ROOT / "src"))


class CountingTestResult(unittest.TextTestResult):
    """Counts both top-level test methods and parameterized subTests."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.subtest_count = 0

    def addSubTest(self, test, subtest, err) -> None:  # type: ignore[no-untyped-def]
        super().addSubTest(test, subtest, err)
        self.subtest_count += 1


def _executable_lines(py_file: Path) -> set[int]:
    """Extract runtime-executable statement line numbers inside functions/methods."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    lines: set[int] = set()

    for fn_node in ast.walk(tree):
        if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip Protocol stub methods whose body is just `...`
            if (
                len(fn_node.body) == 1
                and isinstance(fn_node.body[0], ast.Expr)
                and isinstance(fn_node.body[0].value, ast.Constant)
                and fn_node.body[0].value.value is Ellipsis
            ):
                continue
            for stmt in ast.walk(fn_node):
                if isinstance(
                    stmt,
                    (
                        ast.Return,
                        ast.Assign,
                        ast.AugAssign,
                        ast.AnnAssign,
                        ast.For,
                        ast.While,
                        ast.If,
                        ast.Raise,
                        ast.Assert,
                        ast.Try,
                    ),
                ):
                    lines.add(stmt.lineno)
                elif isinstance(stmt, ast.Expr) and not (
                    isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (str, type(Ellipsis)))
                ):
                    lines.add(stmt.lineno)
    return lines


def main() -> int:
    executed: dict[str, set[int]] = {}
    src_files = sorted(p.resolve() for p in SRC_DIR.glob("*.py"))
    src_map = {str(p): p for p in src_files}

    def tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        if event == "line":
            fname = frame.f_code.co_filename
            if fname in src_map:
                executed.setdefault(fname, set()).add(frame.f_lineno)
        return tracer

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2, resultclass=CountingTestResult)
    sys.settrace(tracer)
    try:
        result: CountingTestResult = runner.run(suite)  # type: ignore[assignment]
    finally:
        sys.settrace(None)

    per_file_cov: dict[str, dict[str, float | int]] = {}
    total_exec_lines = 0
    total_hit_lines = 0

    for p in src_files:
        exec_lines = _executable_lines(p)
        hit_lines = exec_lines.intersection(executed.get(str(p), set()))
        n_exec = len(exec_lines)
        n_hit = len(hit_lines)
        total_exec_lines += n_exec
        total_hit_lines += n_hit
        pct = round((100.0 * n_hit / n_exec), 2) if n_exec else 100.0
        per_file_cov[p.name] = {
            "executable_lines": n_exec,
            "covered_lines": n_hit,
            "coverage_pct": pct,
        }

    overall_cov_pct = round(100.0 * total_hit_lines / total_exec_lines, 2) if total_exec_lines else 100.0
    total_checks = result.testsRun + result.subtest_count
    report = {
        "test_methods_run": result.testsRun,
        "subtests_run": result.subtest_count,
        "total_test_cases": total_checks,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "executable_lines": total_exec_lines,
        "covered_lines": total_hit_lines,
        "line_coverage_pct": overall_cov_pct,
        "files": per_file_cov,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    print(
        f"\n[run_tests] methods={result.testsRun}, subtests={result.subtest_count}, "
        f"total_cases={total_checks}, coverage={overall_cov_pct:.2f}%"
    )
    if not result.wasSuccessful():
        return 1
    if overall_cov_pct < 85.0:
        print(f"Coverage {overall_cov_pct:.2f}% fell below 85.0% threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
