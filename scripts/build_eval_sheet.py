#!/usr/bin/env python
"""Build the native-speaker evaluation sheet — the instrument that finally
gives this project a measured accuracy number instead of an estimate.

WHY STRATIFIED. A uniform random sample of running text is ~60% high-frequency
Germanic function words the gold lexicon already settled; scoring it would
report a flattering number about the part of the system that was never in
doubt. The sample is therefore drawn in three tiers, each answering a
different question:

  A  baseline    ordinary sentences, mostly lexicon routes
                 -> what a listener actually hears most of the time
  B  ambiguous   dense in open graphemes the writing leaves undecided
                 (א פ יי וי, shuruk-ו) at LOW/MED confidence
                 -> does the evidence chain pick the right reading?
  C  loshn-koydesh  dense in embedded Hebrew, especially rescue-chain
                 readings (audio-endorsed / book-pointed / model-guessed)
                 -> does the no-drop chain produce real words?

Tier membership is recorded per row, so the score can be reported per tier
AND pooled — the pooled number is the honest headline, the per-tier numbers
say where the remaining error lives.

WHAT CHEZKY SEES. Not IPA. The `our_reading` column is the respelling the
gold CSV already uses (gaas, upshatzen), with the STRESSED VOWEL UPPERCASED
so stress can be checked without phonetic training. He fills `correction`
ONLY when a reading is wrong — a blank means approved, which is what makes
the task finishable and what makes WER computable without transcribing
anything.

Two open questions are prepended as sentence 0 so the whole ask is one
message: זוכט (audio 47/51 contradicts gold zixt) and לערנען (two conflicting
native verdicts; decides the whole ־ער class).

Outputs (data/eval/):
  eval_sheet_<date>.tsv   the sheet to send (one row per word)
  eval_sheet_<date>.md    the same thing readable in WhatsApp
  eval_meta_<date>.tsv    route/confidence/reason per row, for scoring by
                          source; kept OUT of the sheet so it cannot bias
                          the annotator

Usage: .venv/bin/python scripts/build_eval_sheet.py [--per-tier 10] [--seed 20260814]
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from romanize import romanize  # noqa: E402
from yiddish_g2p import g2p_token  # noqa: E402

DATASET = REPO / "data" / "yiddish_tts_dataset_v2.tsv"
TOKENS = REPO / "data" / "phonemized" / "v3" / "tokens.tsv"
OUTDIR = REPO / "data" / "eval"

# The corpus text is machine transcription: a minority of lines contain
# mis-heard non-words (דעטרערן, כאפציק). Scoring those would measure the
# transcriber, not the G2P, and would waste the scarce resource this sheet
# spends — a native's attention. Require every word in a sampled sentence to
# be attested often enough that it is certainly a real word.
MIN_WORD_FREQ = 8

HEB = re.compile(r"^[א-תװ-ײ֑-ׇ'\"׳״־-]+$")
OPEN_GRAPHEMES = ("א", "פ", "יי", "וי", "ו")
MIN_WORDS, MAX_WORDS = 7, 16


def classify(text: str, freq: dict[str, int]) -> tuple[str, dict] | None:
    """Tier for one sentence, plus its per-word records."""
    words = [w for w in text.split() if HEB.match(w)]
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return None
    if len(words) < len(text.split()) * 0.9:
        return None  # digits/Latin in the line: not a clean read-aloud item
    if any(freq.get(w, 0) < MIN_WORD_FREQ for w in words):
        return None  # likely a transcription error, see MIN_WORD_FREQ

    recs = [g2p_token(w) for w in words]
    if any(not r["ipa_primary"] for r in recs):
        return None

    n = len(recs)
    low_med = sum(1 for r in recs if r["confidence"] in ("LOW", "MED"))
    lk = sum(1 for r in recs if r["layer"] == "L"
             or "pointed" in r["reason"] or "lk-" in r["reason"])
    ambiguous = sum(1 for w, r in zip(words, recs)
                    if r["confidence"] in ("LOW", "MED")
                    and any(g in w for g in OPEN_GRAPHEMES))

    if lk / n >= 0.30:
        tier = "C-loshn-koydesh"
    elif ambiguous / n >= 0.45:
        tier = "B-ambiguous"
    elif low_med / n <= 0.35:
        tier = "A-baseline"
    else:
        return None  # in between: not a clean example of anything
    return tier, {"words": words, "recs": recs}


# The two standing questions, asked in the same notation as the rest.
QUESTIONS = [
    ("זוכט", "gezIkht / zikht",
     "Our lexicon says ZIKHT (from your earlier verdict). But in 47 of 51 "
     "recordings we hear ZUKHT. Which is right — or are both used?"),
    ("לערנען", "lIrnen",
     "Your sheet said LERNEN, an earlier verdict said LIRNEN. This one word "
     "decides how we read every ־ער word (shver, hern, ver, mer), so it is "
     "the most valuable answer on this page."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--date", default="2026-08-14")
    args = ap.parse_args()

    freq: dict[str, int] = {}
    with TOKENS.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            freq[r["word"]] = max(freq.get(r["word"], 0), int(r["freq"]))

    with DATASET.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["ipa"].strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    buckets: dict[str, list] = {"A-baseline": [], "B-ambiguous": [],
                                "C-loshn-koydesh": []}
    for row in rows:
        if all(len(v) >= args.per_tier for v in buckets.values()):
            break
        got = classify(row["text"], freq)
        if got is None:
            continue
        tier, payload = got
        if len(buckets[tier]) >= args.per_tier:
            continue
        buckets[tier].append((row, payload))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    sheet = OUTDIR / f"eval_sheet_{args.date}.tsv"
    meta = OUTDIR / f"eval_meta_{args.date}.tsv"
    read = OUTDIR / f"eval_sheet_{args.date}.md"

    with sheet.open("w", encoding="utf-8", newline="") as fh, \
         meta.open("w", encoding="utf-8", newline="") as mh:
        sw = csv.writer(fh, delimiter="\t")
        mw = csv.writer(mh, delimiter="\t")
        sw.writerow(["sentence", "word_no", "word", "our_reading",
                     "correction", "comment"])
        mw.writerow(["sentence", "word_no", "word", "ipa", "tier", "route",
                     "confidence", "reason", "episode", "start_s"])

        for i, (word, reading, ask) in enumerate(QUESTIONS, 1):
            sw.writerow([f"Q{i}", "", word, reading, "", ask])
            mw.writerow([f"Q{i}", "", word, "", "question", "", "", "", "", ""])

        sid = 0
        for tier in ("A-baseline", "B-ambiguous", "C-loshn-koydesh"):
            for row, payload in buckets[tier]:
                sid += 1
                sw.writerow([f"S{sid:02d}", "", " ".join(payload["words"]),
                             "", "", "<- full sentence, for context"])
                for k, (w, rec) in enumerate(zip(payload["words"],
                                                 payload["recs"]), 1):
                    sw.writerow([f"S{sid:02d}", k, w,
                                 romanize(rec["ipa_primary"]), "", ""])
                    mw.writerow([f"S{sid:02d}", k, w, rec["ipa_primary"],
                                 tier, rec["route"], rec["confidence"],
                                 rec["reason"], row["episode"], row["start_s"]])

    total = sum(len(p["words"]) for v in buckets.values() for _, p in v)
    with read.open("w", encoding="utf-8") as fh:
        fh.write(f"# Pronunciation check — {args.date}\n\n")
        fh.write(
            "Below is how our system reads each word. **Only write something "
            "when a reading is wrong** — a blank line means it is right, and "
            "that is most of them.\n\n"
            "The CAPITAL letters show which syllable we stress "
            "(`gebEYtn` = ge-BEY-tn). If the sounds are right but the stress "
            "is on the wrong syllable, that counts as wrong too — please say "
            "so.\n\n"
            "**Important:** read each word the way you would say it *slowly, "
            "on its own* — the dictionary form. In fast speech many words get "
            "shortened (הָאט sounds like *hat*), but we want the full form "
            "(*hut*). Do not correct a word just because fast speech sounds "
            "different.\n\n")
        fh.write("## Two questions first\n\n")
        for i, (word, reading, ask) in enumerate(QUESTIONS, 1):
            fh.write(f"**Q{i}. {word}** — we say `{reading}`\n\n> {ask}\n\n")
        fh.write(f"## {sid} sentences ({total} words)\n\n")
        sid2 = 0
        for tier in ("A-baseline", "B-ambiguous", "C-loshn-koydesh"):
            for _, payload in buckets[tier]:
                sid2 += 1
                fh.write(f"### S{sid2:02d}\n\n{' '.join(payload['words'])}\n\n")
                for k, (w, rec) in enumerate(zip(payload["words"],
                                                 payload["recs"]), 1):
                    fh.write(f"{k}. {w} — `{romanize(rec['ipa_primary'])}`\n")
                fh.write("\n")

    for tier, v in buckets.items():
        print(f"{tier}: {len(v)} sentences, "
              f"{sum(len(p['words']) for _, p in v)} words")
    print(f"{sid} sentences / {total} words + {len(QUESTIONS)} questions")
    print(f"-> {sheet}\n-> {read}\n-> {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
