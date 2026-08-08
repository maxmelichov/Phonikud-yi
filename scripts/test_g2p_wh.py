#!/usr/bin/env python
"""Whole-Hebrew register tests — run: .venv/bin/python scripts/test_g2p_wh.py

Covers read_pointed_wh(), the opt-in WH reader for VERIFIED pointed quotations
(spec v2 §7.1). Two invariants are asserted here beyond the readings themselves:
the closed v3 phone inventory, and the fact that the MERGED register is
untouched (the same inputs still give their old hebrew_to_ipa answers).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yiddish_g2p import (  # noqa: E402
    hebrew_to_ipa,
    ipa_phone_violations,
    read_pointed_wh,
)

WH_CASES: list[tuple[str, str, str]] = [
    # --- the two bugs this register exists to fix ---
    ("וּבֵרַכְתִּי", "uvajrˈaxti", "(a) shuruk stays [u]: no Yiddish u->i shift"),
    ("לַחְמְךָ", "lˈaxməxu", "(b) shva-na [ə] (rule 2, adjacent shevas)"),
    # --- (c) final komets-hey ---
    ("תּוֹרָה", "tˈɔjru", "(c) final komets-hey = [u] (Toyru), not merged [ə]"),
    ("חׇכְמָה", "xˈuxmu", "komets katan [u] and SHORT: the sheva after it is nach"),
    # --- shva-na heuristic, rules 1 and 3 ---
    ("בְּרֵאשִׁית", "bərˈajʃis", "rule 1: sheva on the first consonant is na"),
    ("שׁוֹמְרִים", "ʃˈɔjmərim", "rule 3: sheva after a long vowel (cholam) is na"),
    ("מֶלֶךְ", "mˈɛlɛx", "rule 4: word-final sheva is silent"),
    # --- (d) the ordinary Ashkenazi point values ---
    ("וַיֹּאמֶר", "vajˈɔjmɛr", "(d) pasekh, cholam on a consonantal yud, segol"),
    ("שַׁבָּת", "ʃˈabus", "(d) pasekh + komets [u]; dagesh chazak not doubled"),
    ("כִּי", "ki", "(d) chirik + matres yud; monosyllable takes no mark"),
    ("אֱלֹהִים", "ɛlˈɔjhim", "hataf segol; silent alef carrier"),
    ("בָּרוּךְ", "bˈurux", "dagesh bet = b, bare kaf = x, shuruk [u]"),
    # --- word-initial begadkefat is a plosive even with the dot omitted ---
    ("כְנֶסֶת", "kənˈɛsɛs", "initial kaf: dagesh lene is obligatory, dot or not"),
    ("פֵאוֹת", "pˈajɔjs", "initial pe likewise; medial bare tav stays [s]"),
    ("וְכָל", "vəxˈul", "NOT initial: the kaf follows a vowel, stays fricative"),
    # --- a mater vav must keep its point or it reads as a consonant ---
    ("אוֹתוֹ", "ˈɔjsɔj", "final cholam male, not a consonantal /v/"),
    ("אֲנַחְנוּ", "anˈaxnu", "final shuruk, not /v/"),
    ("תּוֹרָתֶךָ", "tɔjrusˈɛxu", "2nd-person ךָ keeps its komets [u]"),
    ("אַחֲרָיו", "axˈarujv", "…ָיו really is a consonantal vav"),
    # --- whole posuk, multiword ---
    ("בְּרֵאשִׁית בָּרָא אֱלֹהִים",
     "bərˈajʃis bˈuru ɛlˈɔjhim",
     "multiword; maqaf and whitespace both split"),
]

# The merged register must be bit-for-bit what it was before WH existed.
MERGED_UNCHANGED: list[tuple[str, str]] = [
    ("בְּרָכָה", "brˈuxə"),
    ("שִׂמְחָה", "sˈimxə"),
    ("בֵּית", "bajs"),
    ("יוֹם", "jɔjm"),
]

INVENTORY_PROBES = [
    "וּבֵרַכְתִּי", "לַחְמְךָ", "בְּרֵאשִׁית", "וַיֹּאמֶר", "תּוֹרָה",
    "וְאֵת הָאָרֶץ", "הַשָּׁמַיִם", "יִשְׂרָאֵל", "אֲשֶׁר", "מִצְוֺת",
]


def main() -> int:
    passed = failed = 0
    for text, want, note in WH_CASES:
        got = read_pointed_wh(text)
        ok = got == want
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"{'PASS ' if ok else 'FAIL '}  {text!r:22s} -> {got!r:22s} "
              f"(want {want!r})  {note}")

    for text, want in MERGED_UNCHANGED:
        got = hebrew_to_ipa(text)
        ok = got == want
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"{'PASS ' if ok else 'FAIL '}  merged {text!r:15s} -> {got!r:22s} "
              f"(want {want!r})  merged register untouched")

    for text in INVENTORY_PROBES:
        bad = ipa_phone_violations(read_pointed_wh(text))
        ok = not bad
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"{'PASS ' if ok else 'FAIL '}  inventory {text!r:18s} -> "
              f"{read_pointed_wh(text)!r:22s} {bad or ''}")

    print(f"\n{passed} passed, {failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
