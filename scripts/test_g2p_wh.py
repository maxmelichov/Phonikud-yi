#!/usr/bin/env python
"""Register tests — run: .venv/bin/python scripts/test_g2p_wh.py

Covers BOTH pointed readers, because the point of either one is the contrast
with the other:

  read_pointed_wh()      the opt-in WH reader for VERIFIED pointed QUOTATIONS
                         (spec v2 §7.1) — shuruk [u], shva-na [ə], final
                         komets-hey [u]
  read_pointed_merged()  the same pointing read as EMBEDDED loshn-koydesh
                         (spec v2 §5/§7) — shuruk [i], final komets-hey [ə]

Two invariants are asserted beyond the readings themselves: the closed v3 phone
inventory, and the fact that the MERGED register is untouched (the same inputs
still give their old hebrew_to_ipa answers).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yiddish_g2p import (  # noqa: E402
    hebrew_to_ipa,
    ipa_phone_violations,
    read_pointed_merged,
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
    # --- furtive patah (pasekh genuvah) ---
    ("רוּחַ", "rˈuax", "furtive patah before guttural: [a] before final ches"),
    ("כֹּחַ", "kˈɔjax", "furtive patah after cholam: kˈɔjax"),
    ("מַשְׁגִּיחַ", "maʃɡˈiax", "furtive patah after chirik: maʃɡˈiax"),
    ("תַּפּוּחַ", "tapˈuax", "furtive patah after shuruk: tapˈuax"),
    ("מַפְתֵּחַ", "maftˈajax", "furtive patah after tsere: maftˈajax"),
    # --- whole posuk, multiword ---
    ("בְּרֵאשִׁית בָּרָא אֱלֹהִים",
     "bərˈajʃis bˈuru ɛlˈɔjhim",
     "multiword; maqaf and whitespace both split"),
]

# read_pointed_merged(): the SAME pointings, read as embedded loshn-koydesh.
# Every ``want`` here is a form the native informant gave in the gold CSV, so
# this table is what pins the merged register's two signature shifts.
MERGED_CASES: list[tuple[str, str, str]] = [
    # --- final komets-hey -> [ə], the Yiddish feminine ending (WH says [u]) ---
    ("תּוֹרָה", "tˈɔjrə", "gold tɔjrə — WH reads the same pointing tˈɔjru"),
    ("בְּרָכָה", "brˈuxə", "gold brˈuxə; note NO initial schwa (WH: bərˈuxu)"),
    ("יְשִׁיבָה", "jəʃˈivə", "gold jəʃˈivə — here the sheva IS pronounced"),
    ("מִשְׁפָּחָה", "miʃpˈuxə", "gold miʃpˈuxə"),
    ("שִׂמְחָה", "sˈimxə", "gold sˈimxə"),
    # --- shuruk / kubuts -> [i], the near-exceptionless u->i shift ---
    ("חִדּוּשׁ", "xˈidiʃ", "gold xˈidiʃ — WH would say xˈiduʃ"),
    ("שִׁדּוּךְ", "ʃˈidix", "gold ʃˈidəx/ʃˈidix; shuruk shifts, WH ʃˈidux"),
    ("תְּרוּמָה", "trˈimə", "both shifts at once (WH: tərˈumu)"),
    ("תְּשׁוּבָה", "ʧˈivə", "gold tshive — תּשׁ affricates, shuruk shifts"),
    # --- begadkefat is the HEBREW rule, not the Yiddish one -----------------
    # A bare ב in Yiddish orthography is /b/ (האָבן, אָבער); in a book pointing
    # it is /v/. read_pointed_merged() knows its input is Hebrew and marks the
    # letters explicitly, so these do not come back *abrˈuhum / *ˈubɔjs.
    ("אַבְרָהָם", "avrˈuhum", "bare ב under a sheva is /v/: Avrohom, not Abrohom"),
    ("אָבוֹת", "ˈuvɔjs", "bare ב whose vowel is a mater is still /v/"),
    ("נָבִיא", "nˈuvi", "the case the engine's own soft-bet guard already had"),
    ("כְנֶסֶת", "knˈɛsəs", "word-INITIAL kaf takes dagesh lene: [k], not [x]"),
    ("פְּנֵי", "pnaj", "initial pe likewise [p]"),
    # --- furtive patah (pasekh genuvah) in merged register ---
    ("רוּחַ", "rˈiəx", "furtive patah + u->i shift: riəx"),
    ("כֹּחַ", "kˈɔjəx", "furtive patah: kˈɔjəx"),
    ("מַשְׁגִּיחַ", "maʒɡˈiəx", "furtive patah: maʒɡˈiəx"),
    ("תַּפּוּחַ", "tapˈiəx", "furtive patah + u->i shift: tapˈiəx"),
    ("לוּחַ", "lˈiəx", "furtive patah + u->i shift: lˈiəx"),
    ("מַפְתֵּחַ", "maftˈajəx", "furtive patah after tsere: maftˈajəx"),
    # --- everything else stays the ordinary merged §5 table ---
    ("שַׁבָּת", "ʃˈabəs", "komets [u] -> reduced [ə] unstressed; gold ʃˈabəs"),
    ("חׇכְמָה", "xˈuxmə", "komets katan [u], final hey [ə]"),
    ("בְּרֵאשִׁית בָּרָא אֱלֹהִים", "brˈajʃis bˈuru əlˈɔjhim",
     "multiword; maqaf and whitespace both split, as in WH"),
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

    for text, want, note in MERGED_CASES:
        got = read_pointed_merged(text)
        ok = got == want
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"{'PASS ' if ok else 'FAIL '}  merged-reg {text!r:22s} -> {got!r:22s} "
              f"(want {want!r})  {note}")

    # The two registers must actually DIFFER wherever a shuruk/kubuts or a final
    # komets-hey is present — a merged reader that silently equalled the WH one
    # would pass every case above that has neither feature.
    for text, _, _ in MERGED_CASES:
        if not any(c in text for c in ("ֻ", "ּ")) and not text.endswith("ה"):
            continue
        ok = read_pointed_merged(text) != read_pointed_wh(text)
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"{'PASS ' if ok else 'FAIL '}  registers differ {text!r:18s} "
              f"merged {read_pointed_merged(text)!r} vs wh {read_pointed_wh(text)!r}")

    for text in INVENTORY_PROBES:
        bad = ipa_phone_violations(read_pointed_merged(text))
        ok = not bad
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"{'PASS ' if ok else 'FAIL '}  inventory(merged) {text!r:18s} -> "
              f"{read_pointed_merged(text)!r:22s} {bad or ''}")

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
