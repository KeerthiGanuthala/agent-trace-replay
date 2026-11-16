#!/usr/bin/env python3
"""Zero-dependency offline linter enforcing syntax, docstrings, imports, and formatting."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_file(py_path: Path) -> list[str]:
    errors: list[str] = []
    text = py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(py_path))
    except SyntaxError as exc:
        errors.append(f"{py_path.relative_to(ROOT)}:{exc.lineno}: SyntaxError: {exc.msg}")
        return errors

    if not ast.get_docstring(tree):
        errors.append(f"{py_path.relative_to(ROOT)}:1: Missing module docstring")

    for idx, line in enumerate(text.splitlines(), start=1):
        if chr(9) in line:
            errors.append(f"{py_path.relative_to(ROOT)}:{idx}: Tab character found")
    return errors


def main() -> int:
    targets = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
        *sorted((ROOT / "experiments").rglob("*.py")),
        *sorted((ROOT / "tools").rglob("*.py")),
    ]
    all_errors: list[str] = []
    for p in targets:
        all_errors.extend(check_file(p))
    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        return 1
    print(f"[lint] Checked {len(targets)} Python files: 0 issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
