#!/usr/bin/env python
"""Test suite for yiddish_g2p.py — run: .venv/bin/python scripts/test_g2p.py

Covers the three input modes (unpointed Hasidic, YIVO, fully pointed) plus the
known-bug cases, marked XFAIL so a fix flips them to PASS visibly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yiddish_g2p import hebrew_to_ipa  # noqa: E402

CASES: list[tuple[str, str, str]] = [
    # (input, expected IPA, note)
    # --- unpointed Hasidic (regression: pre-rewrite behavior) ---
    ("וואס", "vus", "mid-word alef = o (Central u)"),
    ("היינט", "haːnt", "unpointed yy = ay in lexicon"),
    ("איך וויל גיין", "ix vil ɡaɪn", "basic Germanic"),
    ("שבת", "ʃubɛs", "LK lexicon swap"),
    ("אויף", "af", "Hasidic af"),
    # --- pointed loshn-koydesh (the new capability) ---
    ("כְּלַל", "klal", "sheva+patah resolved"),
    ("בֵּית", "baɪs", "tsere + matres yud"),
    ("בְּרָכָה", "bruxə", "feminine -e ending"),
    ("שִׂמְחָה", "simxə", "sin dot"),
    ("פְּשַׁט", "pʃat", "dagesh pe = p"),
    ("מִצְוָה", "miʦvə", "vav as consonant /v/"),
    ("כִּי", "ki", "dagesh kaf = k"),
    ("יוֹם", "jɔɪm", "holam = oy"),
    # --- pointed Yiddish (redundant Hasidic pointing) ---
    ("דֶער", "dɛr", "consonant point + restating ayin stays single"),
    ("זָאגְט", "zuɡt", "komets alef digraph"),
    ("ווֶען", "vɛn", "pointed vov-vov"),
    # --- known bugs (XFAIL until fixed) ---
    ("וַוייל", "vaːl", "XFAIL: pasekh on vav should make yy=ay"),
    ("צְבִי", "ʦvi", "XFAIL: bare bet in pointed LK should be /v/"),
]


def main() -> int:
    passed = failed = xfailed = fixed = 0
    for text, want, note in CASES:
        got = hebrew_to_ipa(text)
        xfail = note.startswith("XFAIL")
        ok = got == want
        if ok and not xfail:
            passed += 1
            status = "PASS "
        elif ok and xfail:
            fixed += 1
            status = "FIXED"
        elif xfail:
            xfailed += 1
            status = "xfail"
        else:
            failed += 1
            status = "FAIL "
        print(f"{status}  {text!r:30s} -> {got!r:18s} (want {want!r})  {note}")
    print(f"\n{passed} passed, {failed} FAILED, {xfailed} known-bugs, {fixed} newly fixed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
