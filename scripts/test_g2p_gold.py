#!/usr/bin/env python
"""GOLD reproduction suite — QA gate (d) of spec v3 §12.

The gold CSV ("g2p_gold_v3 - g2p_gold_v3.csv.csv", 500 native-verified rows) is
authority #1 for this engine. Every primary in it must come back BYTE-IDENTICAL
from hebrew_to_ipa(word, stress=True). A single mismatch fails the run.

Also asserted here, because they are cheap and they guard the same contract:
  * every gold primary stays inside the §1 closed phone inventory
  * every gold primary satisfies the §1 vowel/consonant ratio
  * the §2 normalization and §8/§9 routing behaviours the lexicon depends on

Run:  .venv/bin/python scripts/test_g2p_gold.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yiddish_g2p import (  # noqa: E402
    GOLD_LEXICON,
    devoiced_final,
    g2p_token,
    hebrew_to_ipa,
    ipa_phone_violations,
    lexicon_key,
    violates_vowel_ratio,
)

GOLD_CSV = ROOT / "g2p_gold_v3 - g2p_gold_v3.csv.csv"


def gold_rows() -> list[dict]:
    return list(csv.DictReader(GOLD_CSV.open(encoding="utf-8")))


def primary_of(cell: str) -> str:
    return cell.split("|")[0].split("[")[0].strip()


# (input, expected, note) — routing/normalization behaviour the gold layer owns.
ROUTING_CASES: list[tuple[str, str, str]] = [
    # §2.6 mid-word gershayim -> abbreviation table, never the rule path
    ('שליט"א', "ʃlˈitə", "§8 abbreviation"),
    ('זצ"ל', "zaʦˈal", "§8 abbreviation"),
    ('ז"ל', "zal", "§8 abbreviation"),
    ('זי"ע', "zxisˈɔj jˈuɡin ulˈajni", "§8 multiword abbreviation"),
    ('יו"ט', "jˈɔntəf", "§8 abbreviation"),
    ('ב"ה', "bˈurəx haʃˈɛm", "§8 multiword abbreviation"),
    ("ר'", "rɛb", "§8 ר' clitic-abbreviation"),
    ("ר׳", "rɛb", "§2.2 geresh unified to ' first"),
    ("ה'", "haʃˈɛm", "§8 ה'"),
    # §2.5 clitics — the gold rows first, then the productive split
    ("ס'איז", "siz", "§2.5 gold clitic row"),
    ("סאיז", "siz", "§2.5 apostrophe-less clitic + known word"),
    ("מ'קען", "mˈɛkən", "§2.5 gold clitic row (primary, not mkɛn)"),
    ("כ'האב", "xɔb", "§2.5 gold clitic row"),
    ("ס'האט", "səhˈut", "§2.5 gold clitic row"),
    ("כ'וויל", "xvil", "§2.5 productive clitic detach"),
    # §8 multiword vs the bare word
    ("בית", "bajs", "§9 בית alone is bajs"),
    ("בית מדרש", "bis-mˈɛdrəʃ", "§8 multiword"),
    ("בית־מדרש", "bis-mˈɛdrəʃ", "§8 multiword, makef spelling"),
    # §9 homographs: the primary is emitted, the token rule fires for אויפן
    ("אויף", "oʊf", "§9 primary oʊf standalone"),
    ("אויפן", "afn", "§9 token rule אויפן -> afn"),
    ("שטייט", "ʃtajt", "§9 primary ʃtajt"),
    ("נעמען", "nˈɛmən", "§9 primary nɛmən"),
    ("געוואלט", "ɡəvˈɔlt", "§9 primary ɡəvˈɔlt"),
    # §5 no cross-word sandhi, and punctuation survives the router
    ("דאס איז גוט.", "dus iz ɡit.", "sentence: punctuation preserved, no sandhi"),
]


def main() -> int:
    rows = gold_rows()
    failed = 0

    # --- gate (d): byte-identical reproduction -------------------------------
    for r in rows:
        want = primary_of(r["gold_ipa"])
        got = hebrew_to_ipa(r["word"], stress=True)
        if got != want:
            failed += 1
            print(f"FAIL  gold {r['word']!r}: got {got!r}, want {want!r}")
    print(f"gold primaries: {len(rows) - failed}/{len(rows)} byte-identical")

    # --- the lexicon module itself -------------------------------------------
    missing = [r["word"] for r in rows if lexicon_key(r["word"]) not in GOLD_LEXICON]
    if missing:
        failed += len(missing)
        print(f"FAIL  {len(missing)} gold rows absent from data/gold_lexicon.py: {missing[:5]}")

    for r in rows:
        want = primary_of(r["gold_ipa"])
        bad = ipa_phone_violations(want)
        if bad:
            failed += 1
            print(f"FAIL  gold {r['word']!r} primary {want!r} leaves the §1 inventory: {bad}")
        if violates_vowel_ratio(want):
            failed += 1
            print(f"FAIL  gold {r['word']!r} primary {want!r} breaks the §1 vowel ratio")
        # Yiddish has no word-final /h/. This is the invariant behind the
        # unpointed LK feminine -ה rule (לברכה, עבודה, אמונה, שירה were
        # surfacing as lbrxh / ˈɛbidh / ˈaminh / ʃirh, 25,798 corpus tokens);
        # asserting it on the gold keeps the licence for that rule visible.
        if any(part.endswith("h") for part in want.split()):
            failed += 1
            print(f"FAIL  gold {r['word']!r} primary {want!r} ends in a word-final h")

    # --- routing / normalization ---------------------------------------------
    for text, want, note in ROUTING_CASES:
        got = hebrew_to_ipa(text, stress=True)
        if got == want:
            print(f"PASS  {text!r:16s} -> {got!r:24s} {note}")
        else:
            failed += 1
            print(f"FAIL  {text!r:16s} -> {got!r:24s} (want {want!r})  {note}")

    # --- §2.2 surrounding quotes are stripped for the LOOKUP -------------------
    # (they stay in the output string: punctuation passes through the router
    # untouched, it is only the lexicon key that is cleaned)
    for quoted in ('ישראל"', "„ישראל", "»ישראל«", "'ישראל'"):
        rec = g2p_token(quoted)
        if rec["ipa_primary"] != "jisrˈuəl" or rec["route"] != "lexicon":
            failed += 1
            print(f"FAIL  quoted {quoted!r} did not reach the gold entry: {rec}")

    # --- the §12 record shape -------------------------------------------------
    rec = g2p_token("אויף")
    for field in ("word", "ipa_primary", "variants", "layer", "route", "confidence"):
        if field not in rec:
            failed += 1
            print(f"FAIL  g2p_token record is missing {field!r}")
    if rec.get("route") != "lexicon" or rec.get("confidence") != "HIGH":
        failed += 1
        print(f"FAIL  g2p_token('אויף') should be a HIGH lexicon hit: {rec}")
    if "af" not in rec.get("variants", []):
        failed += 1
        print(f"FAIL  g2p_token('אויף') should carry 'af' as a variant: {rec}")

    # --- word-final devoicing VARIANTS ----------------------------------------
    # Audio (episode 100313) devoices word-final voiced obstruents; the primary
    # stays voiced per the native reviewer, the devoiced reading ships as an
    # auto-appended variant. These assertions pin that it is ADDITIVE.

    # (1) auto-variant present where the primary ends voiced
    for word, want_variant in (("איז", "is"), ("בריוו", "brif"), ("ליב", "lip"),
                               ("וואס", None)):
        rec = g2p_token(word)
        variants = rec.get("variants", [])
        if want_variant is None:
            # (2) no auto-variant on a voiceless final
            if rec.get("auto_variants"):
                failed += 1
                print(f"FAIL  {word!r} has a voiceless final but gained "
                      f"auto-variants {rec['auto_variants']}")
        elif want_variant not in variants:
            failed += 1
            print(f"FAIL  {word!r} should carry {want_variant!r} as a variant: {rec}")

    # (3) gold-listed devoiced variants keep their place FIRST and are not
    #     duplicated by the auto pass
    for word, first in (("זאגט", "zukt"), ("טאג", "tuk"), ("ביז", "bis"),
                        ("יעקב", "jˈankəf")):
        rec = g2p_token(word)
        variants = rec.get("variants", [])
        if not variants or variants[0] != first:
            failed += 1
            print(f"FAIL  gold variant order lost for {word!r}: {variants}")
        if rec.get("auto_variants"):
            failed += 1
            print(f"FAIL  {word!r} already lists {first!r}; auto pass should add "
                  f"nothing, got {rec['auto_variants']}")
        if len(variants) != len(set(variants)):
            failed += 1
            print(f"FAIL  duplicate variants for {word!r}: {variants}")

    # (4) every gold primary is byte-identical WITH the variant machinery live,
    #     and no variant leaves the §1 closed inventory or repeats the primary
    gained = 0
    for r in rows:
        rec = g2p_token(r["word"])
        want = primary_of(r["gold_ipa"])
        if rec["ipa_primary"] != want:
            failed += 1
            print(f"FAIL  variant pass moved the primary of {r['word']!r}: "
                  f"{rec['ipa_primary']!r} != {want!r}")
        if rec["route"] != "lexicon" or rec["confidence"] != "HIGH":
            failed += 1
            print(f"FAIL  variant pass changed routing for {r['word']!r}: {rec}")
        for v in rec.get("variants", []):
            bad = ipa_phone_violations(v)
            if bad:
                failed += 1
                print(f"FAIL  variant {v!r} of {r['word']!r} leaves the §1 "
                      f"inventory: {bad}")
            if v == rec["ipa_primary"]:
                failed += 1
                print(f"FAIL  variant of {r['word']!r} equals its primary: {v!r}")
        gained += 1 if rec.get("auto_variants") else 0
    print(f"auto devoiced-final variants added to {gained}/{len(rows)} gold words")

    # (5) the generator itself: each part-final obstruent devoices, and a form
    #     with no voiced final yields "" (no variant at all)
    for src, want in (("biz", "bis"), ("tuɡ", "tuk"), ("jˈankəv", "jˈankəf"),
                      ("bis-mˈɛdrəʃ", ""), ("ɡit", ""), ("ʃˈabəs", "")):
        got = devoiced_final(src)
        if got != want:
            failed += 1
            print(f"FAIL  devoiced_final({src!r}) -> {got!r}, want {want!r}")
    if devoiced_final("rɛb ʤɔz") != "rɛp ʤɔs":
        failed += 1
        print("FAIL  devoiced_final should devoice EVERY part-final obstruent")

    print(f"\n{failed} FAILED" if failed else "\nall gold checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
