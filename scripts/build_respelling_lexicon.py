#!/usr/bin/env python3
"""Build data/lexicons/printed_respelling_lk.py — rescue #3 for the LK quarantine.

Source: data/kodesh_index/singles.tsv, the review-gated staging output of
scripts/ingest_printed_index.py over the phonetic index of a printed dictionary
of loshn-koydesh-origin words. Each row carries the pronunciation the index
respells in Yiddish letters, read through the YIVO letter table and shifted
into the Central/Hasidic vowel system, stressed by the engine's own §11.5
assigner.

Rank: BELOW data/lexicons/sefaria_pointed_lk.py, ABOVE the model-guess table. A
printed phonemic respelling is attested evidence the way book pointing is, but
Sefaria's pointing feeds the register reader directly while this source needed
a dialect shift whose one lossy rule (o>u: YIVO אָ covers two vowel classes) is
measured net-negative against gold — so where both speak, the pointing wins.
Emitted at LOW confidence, reason 'printed-respelling': evidence, not a native
verdict; these stay in the verification queue.

Taken: rows the ingest marked `clean` only — every `needs_review` row (the
disputed shift rules, unresolved skeleton joins) stays in the staging TSV until
a human clears it. Dropped here on top of that: keys already owned by any
higher tier (gold, abbreviation, multiword, legacy merged-LK, the Latin
exception list, the audio tables, the homograph table, Sefaria), and readings
that fail the §1 gates. A key the index lists more than once keeps the FIRST
reading as primary — the dictionary's own ¹-before-² ordering — and the rest as
variants; the unshifted Standard-Yiddish reading rides along as a variant too,
so a reviewer or an aligner can still choose it.

    .venv/bin/python scripts/build_respelling_lexicon.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yiddish_g2p import (  # noqa: E402
    GOLD_LEXICON,
    _ABBREVIATIONS,
    _AUDIO_ENDORSED,
    _AUDIO_PE,
    _AUDIO_VOWEL,
    _HOMOGRAPH_LK,
    _LK_BARE,
    _MULTIWORD,
    _MULTIWORD_LEGACY,
    _SEFARIA_POINTED,
    _WORD_LATIN,
    _strip_points,
    ipa_phone_violations,
    lexicon_key,
    normalize_surface,
    violates_vowel_ratio,
)

SRC = ROOT / "data" / "kodesh_index" / "singles.tsv"
OUT = ROOT / "data" / "lexicons" / "printed_respelling_lk.py"

# Keys whose reading the executable rules doc pins to the model tier
# (docs/yiddish_phoneme_set.md R22): the index disagrees with those rows
# (tˈilim vs təhˈilim, mˈaxlɔjkəs vs maxalˈɔjkɛs), and adjudicating a
# native-facing dispute is Chezky's call, not a build script's. Excluded
# here so the doc stays true; queue both for review.
DOC_PINNED: frozenset[str] = frozenset({"מחלוקת", "תהילים"})


def readable(ipa: str) -> bool:
    """The §1 gate a reading must pass to be emitted at all."""
    return bool(ipa) and not ipa_phone_violations(ipa) and not violates_vowel_ratio(ipa)


def owned(word: str) -> bool:
    """Does a higher-authority tier already answer for this key?"""
    key = lexicon_key(word)
    bare = _strip_points(normalize_surface(word))
    return (
        key in GOLD_LEXICON or key in _MULTIWORD or key in _MULTIWORD_LEGACY
        or bare in _LK_BARE or bare in _WORD_LATIN
        or key in _AUDIO_ENDORSED or key in _AUDIO_PE or key in _AUDIO_VOWEL
        or key in _HOMOGRAPH_LK or key in _SEFARIA_POINTED
        or key in _ABBREVIATIONS
    )


def main() -> int:
    if not SRC.exists():
        raise SystemExit(
            f"{SRC} missing — run scripts/ingest_printed_index.py first "
            "(it needs kodesh_words.pdf; the staging dir is deliberately "
            "not committed)"
        )
    rows = list(csv.DictReader(SRC.open(encoding="utf-8"), delimiter="\t"))

    entries: dict[str, dict] = {}
    counts = {"rows": len(rows), "not_clean": 0, "doc_pinned": 0, "owned": 0,
              "unreadable": 0, "merged_as_variant": 0, "kept": 0}
    for row in rows:
        if row["status"] != "clean":
            counts["not_clean"] += 1
            continue
        word = row["word_key"]
        ipa = row["hasidic_ipa"]
        if word in DOC_PINNED or lexicon_key(word) in {lexicon_key(w) for w in DOC_PINNED}:
            counts["doc_pinned"] += 1
            continue
        if owned(word):
            counts["owned"] += 1
            continue
        if not readable(ipa):
            counts["unreadable"] += 1
            continue
        key = lexicon_key(word)
        variants: list[str] = []
        standard = row["standard_ipa"]
        if standard and standard != ipa and readable(standard):
            variants.append(standard)
        if key in entries:
            # The dictionary's own ¹-before-² order: first reading stays
            # primary, later ones become variants.
            prior = entries[key]
            for v in [ipa, *variants]:
                if v != prior["ipa"] and v not in prior["variants"]:
                    prior["variants"].append(v)
            counts["merged_as_variant"] += 1
            continue
        entries[key] = {
            "word": word,
            "ipa": ipa,
            "variants": variants,
            "printed": row["phonetic_as_printed"],
            "rules": row["shift_rules_fired"],
            "join": row["join_tier"],
        }
        counts["kept"] += 1

    lines = [
        '"""GENERATED — printed-respelling loshn-koydesh readings.',
        "",
        "Source: the phonetic index of a printed dictionary of loshn-koydesh-",
        "shtamike verter, via the review-gated staging TSVs of",
        "scripts/ingest_printed_index.py (clean rows only). Rescue #3 for the LK",
        "quarantine: ranks BELOW sefaria_pointed_lk.py and ABOVE the model-guess",
        "table; emitted LOW with reason 'printed-respelling'. 'printed' is the",
        "index's own Yiddish-letter respelling; the unshifted Standard-Yiddish",
        "reading rides in 'variants'.",
        "",
        f"{len(entries)} entries."
        " Regenerate: python scripts/build_respelling_lexicon.py",
        '"""',
        "",
        "# word (normalized key) -> entry",
        "PRINTED_RESPELLING_LK: dict[str, dict] = {",
    ]
    for key in sorted(entries):
        e = entries[key]
        lines.append(
            f"    {key!r}: {{\"word\": {e['word']!r}, \"ipa\": {e['ipa']!r}, "
            f"\"variants\": {e['variants']!r}, \"printed\": {e['printed']!r}, "
            f"\"rules\": {e['rules']!r}, \"join\": {e['join']!r}}},"
        )
    lines.append("}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(entries)} entries")
    for name, n in counts.items():
        print(f"  {name:20s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
