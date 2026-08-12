#!/usr/bin/env python
"""Separate real pronunciation facts from recognizer bias, statistically.

THE PROBLEM WITH RAW AUDIO VOTES. Across the tagged corpus the recognizer
disagrees with the engine on 20% of strongly-voted word-slots, but most of
that is the recognizer, not the engine: it merges sibilants (ʃ->s, ʦ->s,
z->s) and reduces unstressed vowels to ə everywhere it goes. Folding those
verdicts would teach a sibilant-free, schwa-only Yiddish. The previous
folders dodged this with a hand-written allow-list (vowels may only be
corrected TO /u/, letters only on a unanimous vote), which is safe but
throws away most of the evidence and cannot grow without more hand-written
rules.

THE FIX. Ask a different question. Not "what did the recognizer hear?" but
"did it hear something SURPRISING for this word?".

  1. Base rate: over every tagged slot, measure P(heard = h | engine = e).
     ʃ->s at 31% is what this recognizer does to every ʃ; that is the null
     hypothesis, not evidence about any particular word.
  2. Per word-slot, test the observed votes against that base rate with a
     binomial tail probability. A word whose ʃ is heard as s in 9/10 clips
     is surprising under a 31% base rate; one at 4/10 is not.
  3. Fold only the surprising ones -- in ANY direction, for ANY phone, with
     no allow-list. The threshold is on evidence strength, not on which
     substitution I happen to trust.

This is the same logic a linguist applies by ear ("that speaker always says
it that way, and it is not just my hearing"), made explicit so it scales to
every phone instead of the two I hand-picked.

Outputs:
  data/audio_lexicon/confusion.tsv     P(heard|engine), the recognizer's profile
  data/audio_lexicon/calibrated.tsv    per-slot verdicts with p-values
Both are inputs to build_audio_vowel_lexicon.py's successor and to the
training-target builder; neither is loaded by the engine directly.

Usage: .venv/bin/python scripts/audio_calibrate.py [--max-p 0.002]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_map import tokenize_g2p_ipa  # noqa: E402
from yiddish_g2p import lexicon_key  # noqa: E402

TAGS = REPO / "data" / "audio_lexicon" / "pe_sweep_tags.jsonl"
CONF_OUT = REPO / "data" / "audio_lexicon" / "confusion.tsv"
CAL_OUT = REPO / "data" / "audio_lexicon" / "calibrated.tsv"

MIN_VOTES = 4
PSEUDO = 0.5  # Laplace-ish smoothing on the base rate


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p).

    Summed in log space: a frequent word can be heard hundreds of times, and
    math.comb(n, k) for n in the hundreds overflows a float long before the
    term itself underflows."""
    if p >= 1.0:
        return 1.0
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * lp + (n - i) * lq for i in range(k, n + 1)]
    hi = max(terms)
    return min(1.0, math.exp(hi + math.log(sum(math.exp(t - hi) for t in terms))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-p", type=float, default=0.002,
                    help="tail probability below which a verdict is surprising")
    ap.add_argument("--min-votes", type=int, default=MIN_VOTES)
    args = ap.parse_args()

    slots: dict[tuple[str, int], Counter] = defaultdict(Counter)
    shape: dict[str, str] = {}
    surface: dict[str, str] = {}
    with TAGS.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            key = lexicon_key(rec["word"])
            joined = "".join(rec["g2p"])
            if key in shape and shape[key] != joined:
                continue  # a variant spelling with a different phone shape
            shape.setdefault(key, joined)
            surface.setdefault(key, rec["word"])
            for i, (_, heard) in enumerate(zip(rec["g2p"], rec["heard_at"])):
                if heard != "∅":
                    slots[(key, i)][heard] += 1

    # --- 1. the recognizer's own profile, pooled over every slot ------------
    conf: dict[str, Counter] = defaultdict(Counter)
    toks_cache: dict[str, list[str]] = {}
    for (key, i), votes in slots.items():
        toks = toks_cache.setdefault(key, tokenize_g2p_ipa(shape[key]))
        if i >= len(toks):
            continue
        for heard, n in votes.items():
            conf[toks[i]][heard] += n

    with CONF_OUT.open("w", encoding="utf-8") as fh:
        fh.write("engine\theard\tcount\tshare\n")
        for eng in sorted(conf):
            tot = sum(conf[eng].values())
            for heard, n in conf[eng].most_common():
                fh.write(f"{eng}\t{heard}\t{n}\t{n/tot:.4f}\n")

    def base_rate(eng: str, heard: str) -> float:
        tot = sum(conf[eng].values())
        return (conf[eng][heard] + PSEUDO) / (tot + PSEUDO * max(len(conf[eng]), 1))

    # --- 2./3. per-slot surprise -------------------------------------------
    rows = []
    kept = Counter()
    for (key, i), votes in sorted(slots.items()):
        toks = toks_cache.get(key) or tokenize_g2p_ipa(shape[key])
        if i >= len(toks):
            continue
        eng = toks[i]
        n = sum(votes.values())
        if n < args.min_votes:
            continue
        heard, k = votes.most_common(1)[0]
        if heard == eng:
            continue
        p0 = base_rate(eng, heard)
        p = binom_tail(k, n, p0)
        verdict = "FOLD" if p < args.max_p else "bias-explained"
        kept[verdict] += 1
        rows.append((surface[key], key, i, eng, heard, k, n, p0, p, verdict))

    rows.sort(key=lambda r: r[8])
    with CAL_OUT.open("w", encoding="utf-8") as fh:
        fh.write("word\tkey\tslot\tengine\theard\tk\tn\tbase_rate\tp\tverdict\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\t{r[6]}"
                     f"\t{r[7]:.4f}\t{r[8]:.3e}\t{r[9]}\n")

    print(f"recognizer profile -> {CONF_OUT}")
    for eng in ("ʃ", "ʦ", "z", "ej", "aj", "ɔj", "u", "a", "f"):
        if eng in conf:
            tot = sum(conf[eng].values())
            top = ", ".join(f"{h} {n/tot:.0%}" for h, n in conf[eng].most_common(3))
            print(f"   {eng:3s} heard as: {top}")
    print(f"\n{len(rows):,} disagreeing slots with >= {args.min_votes} votes")
    print(f"   FOLD (surprising, p < {args.max_p}): {kept['FOLD']:,}")
    print(f"   bias-explained (ignore):            {kept['bias-explained']:,}")
    print(f"-> {CAL_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
