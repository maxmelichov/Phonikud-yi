#!/usr/bin/env python3
"""
Build train/val files for fine-tuning the Phonikud diacritizer on Hasidic Yiddish.

Input : data/annotations/<episode_id>.jsonl  (fields: chunk_idx, text_yi,
        text_yi_pointed, ipa, confidence)
Output: data/diacritics/{train,val}.txt   -- one *pointed* line per record, in the
        format the phonikud model trainer expects (it derives the unvocalized input
        by stripping the diacritics).

A record is kept only if:
  * both text_yi and text_yi_pointed are non-empty,
  * their whitespace token counts match,
  * stripping diacritics from text_yi_pointed reproduces text_yi word-for-word
    (>= --min-word-agree of the words), after normalization.

Normalization folds the Yiddish ligatures the two fields disagree on
(U+05F2 -> יי, U+05F1 -> וי, U+05F0 -> וו) and gershayim/geresh punctuation, drops
cantillation accents / meteg / masora circle (noise from the annotator), and
rewrites maqaf as a plain hyphen so the char tokenizer sees a stable inventory.

Split is 95/5 *by episode* so no episode leaks across train and val.

Usage:
    python scripts/prepare_diacritics_dataset.py
    python scripts/prepare_diacritics_dataset.py --stats-only
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- inventories

# Marks we keep: the 12 vowel points + dagesh + rafe + shin/sin dots.
KEEP_MARKS = set(
    "ְֱֲֳִֵֶַָֹֺֻ"
    "ּ"  # dagesh / mapiq
    "ֿ"  # rafe  <-- Yiddish-specific, פֿ בֿ כֿ תֿ
    "ׁׂ"  # shin dot / sin dot
    "ׇ"  # qamats qatan
)

# Everything in the Hebrew combining block that we drop: cantillation accents
# (U+0591-U+05AF), meteg (U+05BD, phonikud reserves it for vocal shva), lower dot,
# masora circle, etc.
ALL_MARKS = re.compile(r"[֑-ֽֿ-ׂׄ-ׇ]")
DROP_MARKS = re.compile(
    "[" + "".join(chr(c) for c in range(0x0591, 0x05C8) if chr(c) not in KEEP_MARKS) + "]"
)

# Ligature / punctuation folding applied to BOTH fields.
FOLD = {
    "װ": "וו",  # װ -> וו
    "ױ": "וי",  # ױ -> וי
    "ײ": "יי",  # ײ -> יי
    "׳": "'",  # ׳ geresh
    "״": '"',  # ״ gershayim
    "־": "-",  # maqaf -> hyphen
    "‎": "",
    "‏": "",
    "‍": "",
    "‌": "",
    "﻿": "",
}

HEB_LETTER = re.compile(r"[א-ת]")


def fold(text: str) -> str:
    """NFC + ligature folding + accent stripping + whitespace squeeze."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in FOLD.items():
        text = text.replace(src, dst)
    text = DROP_MARKS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_marks(text: str) -> str:
    return ALL_MARKS.sub("", text)


# ---------------------------------------------------------------- record check


def check(text_yi: str, pointed: str, min_word_agree: float):
    """Return (ok, reason, pointed_clean, word_agreement)."""
    if not text_yi.strip() or not pointed.strip():
        return False, "empty", "", 0.0

    plain = fold(text_yi)
    ptd = fold(pointed)

    a, b = plain.split(), ptd.split()
    if len(a) != len(b):
        return False, "token_count", ptd, 0.0

    if not a:
        return False, "empty", ptd, 0.0

    # Compare the *unpointed* skeleton word by word. text_yi itself sometimes
    # carries partial nikud, so strip both sides.
    same = sum(1 for x, y in zip(a, b) if strip_marks(x) == strip_marks(y))
    agree = same / len(a)
    if agree < min_word_agree:
        return False, "word_mismatch", ptd, agree

    if not HEB_LETTER.search(ptd):
        return False, "no_hebrew", ptd, agree

    # Must actually carry diacritics, otherwise it is not a training target.
    n_marks = sum(1 for c in ptd if c in KEEP_MARKS)
    if n_marks < 0.15 * len(HEB_LETTER.findall(ptd)):
        return False, "undiacritized", ptd, agree

    return True, "ok", ptd, agree


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=REPO / "data/annotations")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/diacritics")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--min-word-agree", type=float, default=0.90)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    files = sorted(args.annotations.glob("*.jsonl"))
    if not files:
        print(f"no jsonl under {args.annotations}", file=sys.stderr)
        return 1

    per_episode: dict[str, list[str]] = {}
    reasons = collections.Counter()
    mark_counts = collections.Counter()
    n_records = 0

    for path in files:
        ep = path.stem
        kept: list[str] = []
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                reasons["bad_json"] += 1
                continue
            n_records += 1
            if float(rec.get("confidence") or 0.0) < args.min_confidence:
                reasons["low_confidence"] += 1
                continue
            ok, why, ptd, _ = check(
                rec.get("text_yi") or "", rec.get("text_yi_pointed") or "", args.min_word_agree
            )
            reasons[why] += 1
            if ok:
                kept.append(ptd)
                for c in ptd:
                    if c in KEEP_MARKS:
                        mark_counts[c] += 1
        if kept:
            per_episode[ep] = kept

    episodes = sorted(per_episode)
    random.Random(args.seed).shuffle(episodes)
    n_val = max(1, round(len(episodes) * args.val_frac))
    n_test = max(1, round(len(episodes) * args.test_frac))
    val_eps = set(episodes[:n_val])
    test_eps = set(episodes[n_val : n_val + n_test])
    train_eps = set(episodes[n_val + n_test :])
    assert not (val_eps & test_eps) and not (train_eps & val_eps) and not (train_eps & test_eps)

    def collect(eps):
        out = []
        for ep in sorted(eps):
            out.extend(per_episode[ep])
        return out

    train, val, test = collect(train_eps), collect(val_eps), collect(test_eps)

    def words(lines):
        return sum(len(x.split()) for x in lines)

    def chars(lines):
        return sum(len(HEB_LETTER.findall(x)) for x in lines)

    print(f"episodes with data : {len(per_episode)} / {len(files)}")
    print(f"records scanned    : {n_records}")
    print("filter outcome     :")
    for why, n in reasons.most_common():
        print(f"    {why:16s} {n:7d}  ({100 * n / max(n_records, 1):5.1f}%)")
    print()
    for nm, ls, es in (("TRAIN", train, train_eps), ("VAL", val, val_eps), ("TEST", test, test_eps)):
        print(f"{nm:5s}: {len(es):3d} eps  {len(ls):6d} lines  {words(ls):8d} words  {chars(ls):9d} heb chars")
    print()
    print("diacritic inventory kept:")
    for c, n in mark_counts.most_common():
        print(f"    U+{ord(c):04X} {unicodedata.name(c, '?'):34s} {n:8d}")

    if args.stats_only:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines, eps in (("train", train, train_eps), ("val", val, val_eps), ("test", test, test_eps)):
        p = args.out_dir / f"{name}.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {p}  ({len(lines)} lines)")
        (args.out_dir / f"{name}_episodes.txt").write_text(
            "\n".join(sorted(eps)) + "\n", encoding="utf-8"
        )

    stats = {
        "n_records_scanned": n_records,
        "filter_reasons": dict(reasons),
        "train": {"episodes": len(train_eps), "lines": len(train), "words": words(train), "heb_chars": chars(train)},
        "val": {"episodes": len(val_eps), "lines": len(val), "words": words(val), "heb_chars": chars(val)},
        "test": {"episodes": len(test_eps), "lines": len(test), "words": words(test), "heb_chars": chars(test)},
        "marks": {f"U+{ord(c):04X}": n for c, n in mark_counts.most_common()},
        "params": vars(args) | {"annotations": str(args.annotations), "out_dir": str(args.out_dir)},
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
