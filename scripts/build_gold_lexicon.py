#!/usr/bin/env python
"""Generate data/gold_lexicon.py from the native-verified gold_v3 CSV.

The CSV ("g2p_gold_v3 - g2p_gold_v3.csv.csv", 500 rows) is authority #1 for this
engine: where it disagrees with the spec, the rules or any other lexicon, it
wins. This script freezes it into a committed Python module so the engine has no
CSV dependency at import time and so the seed lexicon is reviewable in diffs.

Run:  .venv/bin/python scripts/build_gold_lexicon.py

Columns consumed:
    word        Hebrew-script surface form (the lookup key, after normalization)
    freq        corpus frequency, kept for LOW_CONF/triage sorting
    layer       G/L/E/A/N/X
    gold_ipa    "|"-separated variants; the FIRST is the primary the engine must
                emit byte-identically. Bracketed annotations are notes, not IPA
                ("bajs [bis- in ...]" -> primary "bajs").

Emitted per entry: ipa_primary (stress-carrying IPA), variants (all readings,
primary first), layer, freq, note.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yiddish_g2p import (  # noqa: E402
    PHONE_INVENTORY,
    lexicon_key,
    normalize_surface,
)

GOLD_CSV = ROOT / "g2p_gold_v3 - g2p_gold_v3.csv.csv"
OUT_PATH = ROOT / "data" / "gold_lexicon.py"

HEADER = '''"""GENERATED FILE -- do not edit by hand.

Seed lexicon built from "g2p_gold_v3 - g2p_gold_v3.csv.csv" by
scripts/build_gold_lexicon.py. Regenerate with:

    .venv/bin/python scripts/build_gold_lexicon.py

Keys are normalized lookup keys (NFC, nikud stripped, geresh/gershayim unified,
final letters folded to their base forms) -- see yiddish_g2p.lexicon_key. Each
value carries the stress-bearing IPA primary that hebrew_to_ipa must reproduce
byte-identically, plus every listed variant, the layer and the corpus frequency.
"""

# word (normalized key) -> entry
GOLD_LEXICON: dict[str, dict] = {
'''

FOOTER = "}\n"


def split_variants(cell: str) -> list[str]:
    """Variants of a gold_ipa cell, primary first, bracketed notes removed."""
    out: list[str] = []
    for chunk in cell.split("|"):
        ipa = chunk.split("[")[0].strip()
        if ipa and ipa not in out:
            out.append(ipa)
    return out


def main() -> int:
    rows = list(csv.DictReader(GOLD_CSV.open(encoding="utf-8")))
    entries: dict[str, dict] = {}
    problems: list[str] = []

    for row in rows:
        word = normalize_surface(row["word"])
        key = lexicon_key(word)
        variants = split_variants(row["gold_ipa"])
        if not variants:
            problems.append(f"{row['word']!r}: empty gold_ipa")
            continue
        primary = variants[0]
        bad = sorted(
            {c for c in primary if c not in PHONE_INVENTORY and c not in " -"}
        )
        if bad:
            problems.append(f"{row['word']!r}: primary {primary!r} uses {bad}")
        if key in entries and entries[key]["ipa_primary"] != primary:
            problems.append(
                f"key collision {key!r}: {entries[key]['ipa_primary']!r} vs {primary!r}"
            )
        entries[key] = {
            "word": word,
            "ipa_primary": primary,
            "variants": variants,
            "layer": row["layer"].strip() or "X",
            "freq": int(row["freq"] or 0),
            "note": (row.get("review_note") or "").strip(),
        }

    lines = [HEADER]
    for key, e in entries.items():
        lines.append(
            "    {k!r}: {{\"word\": {w!r}, \"ipa_primary\": {p!r}, "
            "\"variants\": {v!r}, \"layer\": {l!r}, \"freq\": {f!r}, \"note\": {n!r}}},\n".format(
                k=key, w=e["word"], p=e["ipa_primary"], v=e["variants"],
                l=e["layer"], f=e["freq"], n=e["note"],
            )
        )
    lines.append(FOOTER)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(lines), encoding="utf-8")

    print(f"rows read:      {len(rows)}")
    print(f"entries written:{len(entries):5d}  -> {OUT_PATH}")
    for p in problems:
        print("  WARN", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
