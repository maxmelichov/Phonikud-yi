#!/usr/bin/env python
"""Aggregate PhoneticXeus word tags: how does the audio agree with the G2P?

Reads data/audio_lexicon/xeus_tags_*.jsonl (from scripts/xeus_tag.py), joins
against the gold CSV, and reports:
  - overall phone-level agreement (micro) and per-word distribution
  - per gold-word stats: is the audio voting with the gold primary?
  - the phone-confusion table (G2P phone vs aligned heard phone), which
    separates recognizer noise from systematic G2P/dialect mismatches

Usage: .venv/bin/python scripts/xeus_report.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from xeus_tag import align  # noqa: E402

GOLD = REPO / "g2p_gold_v3 - g2p_gold_v3.csv.csv"


def main() -> int:
    recs = []
    for path in glob.glob(str(REPO / "data" / "audio_lexicon" / "xeus_tags_*.jsonl")):
        with open(path) as f:
            recs.extend(json.loads(l) for l in f)
    if not recs:
        sys.exit("no tag files found — run scripts/xeus_tag.py first")

    gold = {}
    for r in csv.DictReader(open(GOLD)):
        variants = [v.split("[")[0].strip() for v in r["gold_ipa"].split("|")]
        gold[r["word"]] = variants

    tot_ph = sum(r["n_phones"] for r in recs)
    tot_hit = sum(r["n_match"] for r in recs)
    print(f"tags: {len(recs)} word tokens, {tot_ph} G2P phones")
    print(f"phone-level agreement (heard == G2P): {tot_hit}/{tot_ph} = {tot_hit/tot_ph:.1%}")

    buckets = Counter()
    for r in recs:
        a = r["agreement"]
        buckets["1.0"] += a == 1.0
        buckets[">=0.75"] += 0.75 <= a < 1.0
        buckets["0.5-0.75"] += 0.5 <= a < 0.75
        buckets["<0.5"] += a < 0.5
    n = len(recs)
    print("word-agreement distribution: " + "  ".join(f"{k}: {v} ({v/n:.0%})" for k, v in buckets.items()))

    # ---- confusion between aligned phone pairs -----------------------------
    conf = Counter()
    for r in recs:
        pred = r["g2p"].split()
        heard = r["heard"].split()
        if not pred or not heard:
            continue
        for pi, hj in align(pred, heard):
            if pi is not None and hj is not None and pred[pi] != heard[hj]:
                conf[(pred[pi], heard[hj])] += 1
    print("\ntop phone confusions (G2P -> heard):")
    for (p, h), c in conf.most_common(15):
        print(f"  {p:3s} -> {h:3s}  {c}")

    # ---- per gold word -----------------------------------------------------
    by_word = defaultdict(list)
    for r in recs:
        by_word[r["word"]].append(r)
    rows = []
    for w, rs in by_word.items():
        if w not in gold or sum(r["n_phones"] for r in rs) == 0:
            continue
        ph = sum(r["n_phones"] for r in rs)
        hit = sum(r["n_match"] for r in rs)
        rows.append((len(rs), hit / ph, w, gold[w][0], rs[0]["g2p"],
                     Counter(r["heard"] for r in rs).most_common(1)[0][0]))
    rows.sort(reverse=True)
    print(f"\ngold words heard in audio: {len(rows)} types")
    print(f"{'n':>3} {'agr':>5}  {'word':14s} {'gold primary':16s} {'g2p phones':22s} most-heard")
    for cnt, agr, w, gp, g2p_ph, heard in rows[:30]:
        print(f"{cnt:3d} {agr:5.0%}  {w:14s} {gp:16s} {g2p_ph:22s} {heard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
