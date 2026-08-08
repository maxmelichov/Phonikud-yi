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
    ("היינט", "haːnt", "ay flattens to long aː (haant), distinct from ey->aj"),  # v3
    ("איך וויל גיין", "ix vil ɡajn", "basic Germanic"),  # v3: aɪ -> aj
    ("שבת", "ʃabɛs", "LK lexicon swap"),
    ("אויף", "oʊf", "v3 §9: oʊf standalone; afn is the fused/reduced form"),  # v3
    # --- pointed loshn-koydesh (the new capability) ---
    ("כְּלַל", "klal", "sheva+patah resolved"),
    ("בֵּית", "bajs", "tsere + matres yud"),  # v3: aɪ -> aj
    ("בְּרָכָה", "bruxə", "feminine -e ending"),
    ("שִׂמְחָה", "simxə", "sin dot"),
    ("פְּשַׁט", "pʃat", "dagesh pe = p"),
    ("מִצְוָה", "miʦvə", "vav as consonant /v/"),
    ("כִּי", "ki", "dagesh kaf = k"),
    ("יוֹם", "jɔjm", "holam = oy"),  # v3: ɔɪ -> ɔj
    # --- pointed Yiddish (redundant Hasidic pointing) ---
    ("דֶער", "dɛr", "consonant point + restating ayin stays single"),
    ("זָאגְט", "zuɡt", "komets alef digraph; v3 §10.2 devoicing is OFF, so zuɡt"),  # v3
    ("ווֶען", "vɛn", "pointed vov-vov"),
    # --- known bugs (XFAIL until fixed) ---
    ("וַוייל", "vaːl", "pasekh on vav makes yy=ay -> aː"),
    ("צְבִי", "ʦvi", "bare bet in pointed LK is /v/ (was XFAIL, now fixed)"),
]


# Stress marking (hebrew_to_ipa(..., stress=True)).
STRESS_CASES: list[tuple[str, str, str]] = [
    ("שבת", "ʃˈabəs", "LK penultimate"),
    ("משפּחה", "miʃpˈuxə", "LK penultimate, 3 syllables"),
    ("חתונה", "xˈasənə", "LK penultimate"),
    ("געקומען", "ɡəkˈimən", "unstressed prefix ge-"),
    ("פארשטיין", "farʃtˈajn", "unstressed prefix far-"),  # v3: aɪ -> aj
    ("ארבעטן", "ˈarbətn", "v3 §1: syllabic -n takes no epenthetic ə; ע is a written vowel so the word IS marked"),  # v3
    ("אונטערגיין", "ˈintərɡajn", "separable prefix IS stressed"),  # v3: aɪ -> aj
    ("גיין", "ɡajn", "monosyllable stays unmarked"),  # v3: aɪ -> aj
    # --- _STRESS_OVERRIDE entries confirmed against audio (scripts/stress_eval.py) ---
    ("אזוי", "azˈɔj", "override: unstressed initial a-, not *Azoy"),  # v3: ɔɪ -> ɔj
    ("חנוכה", "xˈanikə", "override: LK with initial, not penultimate, stress"),
    ("אינטערעסאנט", "intərəsˈant", "override: loanword keeps donor stress"),
    # --- v3 core-phonology regressions ------------------------------------
    ("מאכן", "maxn", "v3: syllabic -n, one written nucleus -> monosyllable, NO mark"),
    ("זאגן", "zuɡn", "v3 §1/§11.2: zuɡn, no ə, no stress mark"),
    ("פארן", "farn", "v3: farn is a monosyllable by the written-nucleus count"),
    ("מענטשן", "mɛnʧn", "v3 §11.2: one written nucleus (ע) -> monosyllable, no mark; gold agrees"),
    ("איז", "iz", "v3 §10.2: no final devoicing"),
    ("אויב", "ɔjb", "v3 §10.2: ɔjb, not ɔjp"),
    ("וועג", "vejɡ", "v3 §5/§10.2: ej away from r, and no final devoicing"),
    ("מער", "mɛr", "v3 §5: the ־ער default is ɛr; ejr never occurs before r"),
    ("ווער", "vɛr", "v3 §5: ɛr default"),
    ("שווער", "ʃvir", "v3 §5: the closed ir-list survives"),
    ("לערנען", "lˈirnən", "v3 §5 ir-list spelling is lirnen, not leernen"),
    ("מסביר", "mazbˈir", "v3 §10.1: voicing-ward assimilation stays ON"),
    ("צוויי", "ʦvaj", "v3 §10.1: /v/ is not a voicing trigger"),
    ("זעהן", "zejn", "v3 §5: ה silent after a vowel before a consonant"),
    ("ייד", "jid", "v3 §4: word-initial יי = ji"),
    ("וואו", "vi", "v3 §5: א silent before a vowel-ו"),
    ("אונז", "inz", "v3 §5 + §10.2"),
    ("אייביג", "ˈajbiɡ", "v3 §5: suffix ־יג = iɡ, no devoicing"),
    ("ארויס", "arˈoʊs", "v3 §11.4: directional a(r)- stresses the second nucleus"),
    ("אראפ", "arˈup", "v3 §11.4"),
    ("אהיים", "ahˈajm", "v3 §11.4"),
    ("אוועק", "avˈɛk", "v3 §11.4"),
    ("אריין", "arˈaːn", "v3 §11.4"),
]


def main() -> int:
    passed = failed = xfailed = fixed = 0
    for text, want, note in CASES + [(t, w, "stress: " + n) for t, w, n in STRESS_CASES]:
        got = hebrew_to_ipa(text, stress=note.startswith("stress:"))
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
