#!/usr/bin/env python3
"""Run this FIRST on the training box: python3 selftest.py

Proves the bundle is wired correctly before any data is generated from it.
Exits non-zero on any failure. Needs onnxruntime + numpy.
"""
from __future__ import annotations

import sys
import unicodedata as ud
from pathlib import Path

# yiddish_labels forces the import order (this directory ahead of the engine
# directory, which carries an older yiddish_nikud.py); import it before
# anything else so that ordering is in effect for the rest of this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

fails: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(label)


# 1. engine + tables ---------------------------------------------------------
try:
    from yiddish_labels import verify, text_to_ipa
    report = verify(strict=False)
    for name, n in report["sizes"].items():
        check(n > 0, f"table {name} loaded", f"{n} entries")
    check(not report["problems"], "engine canary readings",
          "; ".join(report["problems"]) or "all correct")
except Exception as e:  # noqa: BLE001
    check(False, "import yiddish_labels", repr(e))
    text_to_ipa = None

# 2. the classes that were wrong in the shipped voice ------------------------
if text_to_ipa:
    for text, want in [
        ("מיט א פאר יאר צוריק", "mit a pˈur jur ʦirˈik"),  # not "far"
        ("ער דאוונט פאר די קהילה", None),                    # far, as preposition
        ("וואס האט ער געזאגט", "vus hut ɛr ɡəzˈuɡt"),        # komets = u, never o/a
    ]:
        got = text_to_ipa(text)
        if want is None:
            check(" far " in f" {got} ", f"preposition פאר stays far", got)
        else:
            check(got == want, f"{text}", got)

# 3. nikud model -------------------------------------------------------------
try:
    from yiddish_nikud import YiddishNikud
    nk = YiddishNikud()
    check(True, "v5 ONNX loaded", f"providers={nk.providers}")
    out = nk.add("מיט א פאר יאר צוריק אין שפיטאל")
    check("פּאָר" in out, "v5 points אַ פּאָר (not פֿאַר)", out)
    long_text = "און ".join(["דער מענטש איז געגאנגען אין דער גאס"] * 40)
    long_out = nk.add(long_text)
    strip = lambda s: "".join(c for c in ud.normalize("NFD", s)
                              if ud.category(c) != "Mn")  # noqa: E731
    check(strip(long_out) == strip(long_text),
          "long row (>512 chars) points without truncation",
          f"{len(long_text)} chars")
except Exception as e:  # noqa: BLE001
    check(False, "nikud model", repr(e))

# 4. end-to-end --------------------------------------------------------------
try:
    from yiddish_labels import text_to_nikud
    src = "מיט א פאר יאר צוריק"
    print(f"\n  text : {src}\n  nikud: {text_to_nikud(src)}\n  ipa  : {text_to_ipa(src)}")
except Exception as e:  # noqa: BLE001
    check(False, "end-to-end", repr(e))

print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILURE(S): ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
