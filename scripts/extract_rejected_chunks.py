#!/usr/bin/env python3
"""
Dump the plain (unpointed) Yiddish text of every annotation chunk that the
alignment filter REJECTED, so the round-4 teacher can pseudo-label it.

These are the ~13k chunks whose text_yi / text_yi_pointed pair failed the token
count or word agreement check, or that were effectively undiacritized. The
*pointing* is what was unreliable there -- the plain `text_yi` is still perfectly
good input text, so we keep it and let the teacher supply the labels.

Chunks are assigned to the split of their episode, so a rejected chunk from a
val/test episode never leaks into student training.

Usage:
    python scripts/extract_rejected_chunks.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from prepare_diacritics_dataset import HEB_LETTER, check, fold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=REPO / "data/annotations")
    ap.add_argument("--splits", type=Path, default=REPO / "data/diacritics_r3c")
    ap.add_argument("--out", type=Path, default=REPO / "data/pseudo")
    ap.add_argument("--min-word-agree", type=float, default=0.90)
    ap.add_argument("--min-words", type=int, default=3)
    args = ap.parse_args()

    split_of: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for ep in (args.splits / f"{name}_episodes.txt").read_text().split():
            split_of[ep] = name

    kept = collections.Counter()
    reasons = collections.Counter()
    out_lines: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()

    for path in sorted(args.annotations.glob("*.jsonl")):
        split = split_of.get(path.stem)
        if split is None:
            continue
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text_yi = rec.get("text_yi") or ""
            ok, why, _, _ = check(text_yi, rec.get("text_yi_pointed") or "", args.min_word_agree)
            if ok:
                continue  # already in the gold set
            reasons[why] += 1

            # The plain side still has to be usable text.
            plain = fold(text_yi)
            if not plain or not HEB_LETTER.search(plain):
                continue
            if len(plain.split()) < args.min_words:
                continue
            if plain in seen:
                continue
            seen.add(plain)
            out_lines[split].append(plain)
            kept[split] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, lines in out_lines.items():
        p = args.out / f"rejected_{name}.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        words = sum(len(x.split()) for x in lines)
        print(f"{name:5s}: {len(lines):6d} chunks  {words:9,d} words -> {p}")

    print("\nrejection reasons across all episodes:")
    for why, n in reasons.most_common():
        print(f"  {why:16s} {n:6d}")
    (args.out / "extract_stats.json").write_text(
        json.dumps({"kept": dict(kept), "reasons": dict(reasons)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
