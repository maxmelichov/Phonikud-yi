#!/usr/bin/env python
"""Turn PhoneticXeus word tags into lexicon votes.

Two vote types:
1. VARIANT VOTES — for words whose gold row lists 2+ variants: each audio
   occurrence scores every variant by aligned phone agreement; the variant
   with the highest score gets that clip's vote. Reported when >=3 clips.
2. VOWEL VOTES — for every tagged word: at each G2P vowel position, the modal
   heard phone across clips. Reported when >=3 clips agree >=60% on a phone
   that DIFFERS from the G2P — the raw material for lexicon triage
   (alef-default a vs u vs ɔ, etc.).

Output: data/audio_lexicon/xeus_votes_variants.tsv, xeus_votes_vowels.tsv
and a printed summary.

Usage: .venv/bin/python scripts/xeus_vote.py
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
from xeus_map import VOWELS, tokenize_g2p_ipa  # noqa: E402
from xeus_tag import align  # noqa: E402

GOLD = REPO / "data/gold/g2p_gold_v3.csv"
MIN_CLIPS = 3


def variant_tokens(cell: str) -> list[list[str]]:
    return [tokenize_g2p_ipa(v.split("[")[0].strip().replace(" ", ""))
            for v in cell.split("|")]


def score(variant: list[str], heard: list[str]) -> float:
    if not variant:
        return 0.0
    hits = sum(
        1 for pi, hj in align(variant, heard)
        if pi is not None and hj is not None and variant[pi] == heard[hj]
    )
    return hits / len(variant)


def main() -> int:
    recs = []
    for path in glob.glob(str(REPO / "data" / "audio_lexicon" / "xeus_tags_*.jsonl")):
        with open(path) as f:
            recs.extend(json.loads(l) for l in f)
    by_word = defaultdict(list)
    for r in recs:
        if r["heard"]:
            by_word[r["word"]].append(r)

    gold_rows = {r["word"]: r for r in csv.DictReader(open(GOLD))}

    # ---- 1. variant votes ---------------------------------------------------
    out1 = []
    for w, row in gold_rows.items():
        cell = row["gold_ipa"]
        if "|" not in cell or w not in by_word:
            continue
        clips = by_word[w]
        if len(clips) < MIN_CLIPS:
            continue
        variants = [v.split("[")[0].strip() for v in cell.split("|")]
        vtoks = variant_tokens(cell)
        votes = Counter()
        for r in clips:
            heard = r["heard"].split()
            scores = [score(vt, heard) for vt in vtoks]
            best = max(scores)
            winners = [i for i, s in enumerate(scores) if s == best]
            if len(winners) == 1 and best > 0:
                votes[winners[0]] += 1
        if not votes:
            continue
        top, n_top = votes.most_common(1)[0]
        total = sum(votes.values())
        out1.append({
            "word": w, "clips": len(clips), "decided": total,
            "winner": variants[top], "winner_votes": n_top,
            "primary": variants[0],
            "agrees_with_primary": variants[top] == variants[0],
            "tally": "; ".join(f"{variants[i]}:{c}" for i, c in votes.most_common()),
        })
    out1.sort(key=lambda d: -d["clips"])
    with open(REPO / "data" / "audio_lexicon" / "xeus_votes_variants.tsv", "w") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out1[0]), delimiter="\t")
        wtr.writeheader()
        wtr.writerows(out1)

    n_agree = sum(1 for d in out1 if d["agrees_with_primary"])
    print(f"VARIANT VOTES: {len(out1)} contested gold words with >={MIN_CLIPS} clips")
    print(f"  audio agrees with the gold primary: {n_agree}/{len(out1)} = {n_agree/len(out1):.0%}")
    print("  words where audio prefers the ALTERNATE (top 20 by clips):")
    for d in [d for d in out1 if not d["agrees_with_primary"]][:20]:
        print(f"    {d['word']:12s} primary={d['primary']:12s} audio-> {d['winner']:12s} ({d['tally']})")

    # ---- 2. vowel votes -----------------------------------------------------
    out2 = []
    for w, clips in by_word.items():
        if len(clips) < MIN_CLIPS:
            continue
        g2p = clips[0]["g2p"].split()
        pos_votes: dict[int, Counter] = defaultdict(Counter)
        for r in clips:
            if r["g2p"].split() != g2p:
                continue
            heard = r["heard"].split()
            for pi, hj in align(g2p, heard):
                if pi is not None and hj is not None and g2p[pi] in VOWELS:
                    pos_votes[pi][heard[hj]] += 1
        for pi, votes in pos_votes.items():
            phone, cnt = votes.most_common(1)[0]
            total = sum(votes.values())
            if (phone != g2p[pi] and phone in VOWELS and total >= MIN_CLIPS
                    and cnt / total >= 0.6 and phone != "ə"):
                out2.append({
                    "word": w, "clips": total, "pos": pi,
                    "g2p": " ".join(g2p), "g2p_vowel": g2p[pi],
                    "heard_vowel": phone, "consistency": round(cnt / total, 2),
                    "in_gold": w in gold_rows,
                })
    out2.sort(key=lambda d: (-d["clips"], -d["consistency"]))
    with open(REPO / "data" / "audio_lexicon" / "xeus_votes_vowels.tsv", "w") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out2[0]), delimiter="\t")
        wtr.writeheader()
        wtr.writerows(out2)
    nongold = [d for d in out2 if not d["in_gold"]]
    print(f"\nVOWEL VOTES: {len(out2)} consistent audio-vs-G2P vowel disagreements "
          f"({len(nongold)} on non-gold words — lexicon triage candidates)")
    print("  top 20 non-gold (word, clips, g2p, position vote):")
    for d in nongold[:20]:
        print(f"    {d['word']:14s} n={d['clips']:3d}  {d['g2p']:24s} "
              f"[{d['pos']}] {d['g2p_vowel']} -> {d['heard_vowel']} ({d['consistency']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
