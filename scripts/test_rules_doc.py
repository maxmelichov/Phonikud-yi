#!/usr/bin/env python
"""Execute docs/yiddish_phoneme_set.md — every rule's examples run live.

Parses each `### R..` section of the doc, extracts example-table rows
(`| input | expected | note |` with backticked cells), and asserts
hebrew_to_ipa(input, stress=True) == expected. `∅` means the token must be
quarantined (empty output). The inventory tables in section 1 are checked too.

A rule section with no examples, or any failing example, fails the run —
so the doc can never drift from the engine silently.

Run: .venv/bin/python scripts/test_rules_doc.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from yiddish_g2p import hebrew_to_ipa  # noqa: E402

DOC = REPO / "docs" / "yiddish_phoneme_set.md"

RULE_HEAD = re.compile(r"^###\s+(R\d+)\s+—\s+(.*)$")
# | `input` | `expected` | note |
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|(.*)\|\s*$")
# inventory rows: | `phone` | desc | word | `ipa` |
INV_ROW = re.compile(r"^\|\s*`[^`]+`\s*\|[^|]*\|\s*([֐-׿][^|]*?)\s*\|\s*`([^`]+)`\s*\|\s*$")


def main() -> int:
    text = DOC.read_text().splitlines()
    rule: str | None = None
    title = ""
    cases: dict[str, list[tuple[str, str]]] = {}
    titles: dict[str, str] = {}
    inv_cases: list[tuple[str, str]] = []

    for line in text:
        m = RULE_HEAD.match(line)
        if m:
            rule, title = m.group(1), m.group(2)
            titles[rule] = title
            cases.setdefault(rule, [])
            continue
        m = INV_ROW.match(line)
        if m and rule is None:
            inv_cases.append((m.group(1).strip(), m.group(2).strip()))
            continue
        m = ROW.match(line)
        if m and rule is not None:
            cases[rule].append((m.group(1).strip(), m.group(2).strip()))

    passed = failed = 0
    failures: list[str] = []

    def check(label: str, word: str, want: str) -> None:
        nonlocal passed, failed
        got = hebrew_to_ipa(word, stress=True)
        expect = "" if want == "∅" else want
        if got == expect:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {label}  {word!r}: want {want!r}, got {got!r}")

    for word, ipa in inv_cases:
        check("INV", word, ipa)

    empty_rules = []
    for rid in sorted(cases):
        if not cases[rid]:
            empty_rules.append(rid)
            continue
        before = failed
        for word, want in cases[rid]:
            check(rid, word, want)
        status = "PASS" if failed == before else "FAIL"
        print(f"{status}  {rid}  {titles[rid][:70]:70s} ({len(cases[rid])} examples)")

    print(f"\ninventory examples: {len(inv_cases)}; rules: {len(cases)}; "
          f"{passed} example(s) passed, {failed} FAILED")
    if empty_rules:
        print(f"RULES WITH NO EXAMPLES (doc bug): {', '.join(empty_rules)}")
    if failures:
        print("\nfailures:")
        print("\n".join(failures))
    return 1 if (failed or empty_rules) else 0


if __name__ == "__main__":
    raise SystemExit(main())
