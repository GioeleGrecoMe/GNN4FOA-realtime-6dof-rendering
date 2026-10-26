#!/usr/bin/env python3
"""Lightweight C/C++ delimiter-integrity gate for first-party GNN4FOA sources.

This is intentionally not a C++ parser. It strips comments and string/character
literals, then verifies {}, (), and [] nesting. Its purpose is to catch malformed
release edits (such as an extra namespace brace) before the expensive JUCE/Xcode
build starts. The real compiler remains authoritative.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Source"
EXCLUDED_DIRS = {"Eigen"}
SUFFIXES = {".cpp", ".h", ".hpp", ".c", ".mm"}
OPEN_TO_CLOSE = {"{": "}", "(": ")", "[": "]"}
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}


def first_party_files() -> list[Path]:
    files: list[Path] = []
    for path in SOURCE.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        rel = path.relative_to(SOURCE)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        files.append(path)
    return sorted(files)


def scrub(text: str) -> str:
    out = list(text)
    i = 0
    n = len(out)
    while i < n:
        if out[i] == "/" and i + 1 < n and out[i + 1] == "/":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and out[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if out[i] == "/" and i + 1 < n and out[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n:
                if out[i] == "*" and i + 1 < n and out[i + 1] == "/":
                    out[i] = out[i + 1] = " "
                    i += 2
                    break
                if out[i] != "\n":
                    out[i] = " "
                i += 1
            continue
        if out[i] in {'"', "'"}:
            quote = out[i]
            out[i] = " "
            i += 1
            while i < n:
                ch = out[i]
                if ch == "\\":
                    out[i] = " "
                    if i + 1 < n:
                        if out[i + 1] != "\n":
                            out[i + 1] = " "
                        i += 2
                    else:
                        i += 1
                    continue
                out[i] = "\n" if ch == "\n" else " "
                i += 1
                if ch == quote:
                    break
            continue
        i += 1
    return "".join(out)


def check_file(path: Path) -> list[str]:
    text = scrub(path.read_text(encoding="utf-8", errors="replace"))
    stack: list[tuple[str, int, int]] = []
    errors: list[str] = []
    line = 1
    col = 0
    for ch in text:
        if ch == "\n":
            line += 1
            col = 0
            continue
        col += 1
        if ch in OPEN_TO_CLOSE:
            stack.append((ch, line, col))
        elif ch in CLOSE_TO_OPEN:
            if not stack:
                errors.append(f"{path.relative_to(ROOT)}:{line}:{col}: unmatched closing {ch}")
                continue
            opener, open_line, open_col = stack.pop()
            expected = OPEN_TO_CLOSE[opener]
            if ch != expected:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line}:{col}: closing {ch} mismatches "
                    f"{opener} opened at {open_line}:{open_col} (expected {expected})"
                )
    for opener, open_line, open_col in reversed(stack):
        errors.append(
            f"{path.relative_to(ROOT)}:{open_line}:{open_col}: unmatched opening {opener}"
        )
    return errors


def main() -> int:
    files = first_party_files()
    if not files:
        print("ERROR: no first-party Source files found", file=sys.stderr)
        return 2
    errors: list[str] = []
    for path in files:
        errors.extend(check_file(path))
    if errors:
        print("FAIL: C/C++ delimiter-integrity check")
        for error in errors:
            print(error)
        return 1
    print(f"PASS: C/C++ delimiter integrity ({len(files)} first-party source files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
