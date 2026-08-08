#!/usr/bin/env python
"""Mine multi-word loshn-koydesh expressions (MWEs) from the corpus.

An MWE is worth a multiword-table entry only when it is *lexicalized*: the
phrase has a frozen reading that per-word routing cannot reconstruct
(בעל הבית -> baləbˈus, not baːl habˈajis). This script finds the candidates;
a human promotes them into ``_MULTIWORD`` in yiddish_g2p.py.

Selection (all three must hold):

  a. at least one part fires the §3 LK detector (or is already an LK-table word),
  b. the n-gram occurs >= --min-count times in data/yiddish_tts_dataset.tsv,
  c. the n-gram exists as a phrase key (n >= 2 words) in
     data/pointed_sources/pointed_index.jsonl -- i.e. the books write it as a
     unit -- OR it is listed in KNOWN_COLLOCATIONS below.

Output: data/lk_mwe_candidates.tsv, ranked by corpus frequency, with the
current engine reading (what hebrew_to_ipa emits today) so a curator can see
at a glance whether a lexicalized entry would change anything.

Usage:
  .venv/bin/python scripts/mine_lk_mwe.py
  .venv/bin/python scripts/mine_lk_mwe.py --min-count 10 --max-n 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yiddish_g2p import (  # noqa: E402
    _LK_BARE,
    _MULTIWORD,
    _MULTIWORD_LEGACY,
    _lk_detector,
    _strip_points,
    g2p_token,
    hebrew_to_ipa,
    lexicon_key,
    normalize_surface,
    split_affixes,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CORPUS = DATA / "yiddish_tts_dataset.tsv"
POINTED_INDEX = DATA / "pointed_sources" / "pointed_index.jsonl"
OUT_TSV = DATA / "lk_mwe_candidates.tsv"

HEB_WORD = re.compile(r"^[֐-׿'\"-]+$")

# Collocations that are lexicalized in *Yiddish* and therefore may be absent
# from a Hebrew-book phrase index (or written solid there). Bare keys.
KNOWN_COLLOCATIONS = {
    "בעל הבית", "בעל הבתים", "בעלי בתים", "בעלת הבית",
    "ראש השנה", "ראש חודש", "ראש ישיבה", "ראשי ישיבות",
    "יום כיפור", "יום טוב", "יום הדין", "ימים נוראים",
    "שבת קודש", "מוצאי שבת", "ערב שבת", "חול המועד",
    "בית המדרש", "בית מדרש", "בית דין", "בית הכנסת", "בית עולם",
    "ארץ ישראל", "כלל ישראל", "עם ישראל", "מדינת ישראל",
    "תלמוד תורה", "בן תורה", "בני תורה", "דעת תורה",
    "בר מצוה", "בת מצוה", "ברית מילה",
    "ברוך השם", "אם ירצה השם", "בעזרת השם", "קידוש השם", "חילול השם",
    "זכרונו לברכה", "זכרונם לברכה", "זכר צדיק לברכה",
    "ריבונו של עולם", "רבונו של עולם", "עולם הבא", "עולם הזה",
    "בורא עולם", "מלך מלכי המלכים",
    "לשון הרע", "שלום בית", "אהבת ישראל", "מסירת נפש",
    "תורה ומצוות", "גמילות חסדים", "כבוד התורה",
    "סעודה שלישית", "מלוה מלכה", "שבע ברכות",
    "פסח שני", "לג בעומר", "תשעה באב", "יום כיפור קטן",
    "חדר אוכל", "כולל אברכים", "זאת אומרת",
}


def load_phrase_index(min_words: int = 2) -> set[str]:
    """Bare phrase keys (>= min_words words) the pointed book index knows."""
    keys: set[str] = set()
    with POINTED_INDEX.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("n", 1) >= min_words:
                keys.add(lexicon_key(row["k"]))
    return keys


def iter_corpus_texts(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            text = (row.get("text") or "").strip()
            if text:
                yield text


def tokenize(text: str) -> list[str]:
    out = []
    for tok in normalize_surface(text).split():
        core = split_affixes(tok)[1]
        out.append(core if core and HEB_WORD.match(core) else "")
    return out


def is_lk_part(core: str) -> bool:
    """LK by the §3 detector, by LK-table membership, or by the engine's layer.

    The layer column is the engine's own verdict (gold layer for gold words,
    detector otherwise), so it catches LK words the letter test misses --
    ישיבה has no ח/ת/שׂ/כּ but the gold marks it L.
    """
    bare = _strip_points(core)
    if bare in _LK_BARE or _lk_detector(bare):
        return True
    try:
        return g2p_token(core)["layer"] == "L"
    except Exception:
        return False


def is_lk_phrase(parts: list[str]) -> bool:
    """LK when any part is LK, or when the LK table already knows the compound.

    יום טוב / יום כיפור carry no LK-diagnostic letter in either half, but the
    hyphenated compound (יום-טובֿ -> יאנטעוו) is an LK-table entry, which is the
    same evidence: the phrase is Hebrew-component, only its spacing varies.
    """
    if any(is_lk_part(p) for p in parts):
        return True
    joined = _strip_points("-".join(parts))
    return joined in _LK_BARE or lexicon_key(joined) in {
        lexicon_key(k) for k in _LK_BARE
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=15)
    ap.add_argument("--max-n", type=int, default=3)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT_TSV)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    phrase_keys = load_phrase_index()
    counts: Counter[str] = Counter()
    lines = 0
    for text in iter_corpus_texts(args.corpus):
        lines += 1
        toks = tokenize(text)
        for n in range(2, args.max_n + 1):
            for i in range(len(toks) - n + 1):
                window = toks[i:i + n]
                if not all(window):
                    continue
                counts[" ".join(window)] += 1

    rows = []
    for phrase, count in counts.items():
        if count < args.min_count:
            continue
        parts = phrase.split()
        if not is_lk_phrase(parts):
            continue
        key = lexicon_key(phrase)
        in_books = key in phrase_keys
        known = key in {lexicon_key(k) for k in KNOWN_COLLOCATIONS}
        if not (in_books or known):
            continue
        rows.append({
            "count": count,
            "phrase": phrase,
            "key": key,
            "words": len(parts),
            "source": "books+known" if in_books and known else
                      ("books" if in_books else "known"),
            "in_engine": "yes" if (key in _MULTIWORD or key in _MULTIWORD_LEGACY)
                         else "no",
            "current_ipa": hebrew_to_ipa(phrase, quarantine=False),
        })

    rows.sort(key=lambda r: (-r["count"], r["phrase"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, delimiter="\t",
            fieldnames=["count", "phrase", "key", "words", "source",
                        "in_engine", "current_ipa"])
        w.writeheader()
        w.writerows(rows)

    covered = sum(r["count"] for r in rows)
    print(f"{lines} corpus lines, {len(counts)} distinct n-grams")
    print(f"{len(rows)} MWE candidates (>= {args.min_count}), "
          f"{covered} phrase occurrences -> {args.out}")
    for r in rows[:25]:
        print(f"  {r['count']:6d}  {r['phrase']:<28} {r['source']:<11} "
              f"{r['in_engine']:<3} {r['current_ipa']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
