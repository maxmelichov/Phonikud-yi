#!/usr/bin/env python
"""Audio-verify the rescued Hebrew readings (the no-drop chain) with PhoneticXeus.

Every Hebrew word the rules could not voice ships a LOW-confidence rescued
reading (audio-endorsed -> homograph -> Sefaria -> model guess), and the
vowel-less residue ships a quarantined rule approximation. This sweep tests
those readings against the corpus audio, word by word:

  1. targets = dataset tokens whose g2p_token verdict marks them rescued or
     fallback: reasons {sefaria-pointed, model-pointed-guess, lk-fallback,
     pointed-audio-endorsed, audio-homograph, mwe-mined} or route=fallback
  2. coverage credit is read from data/audio_lexicon/pe_sweep_tags.jsonl
     (the pe sweep tags every word of its chunks, LK words included); only
     uncovered types trigger new chunk transcriptions, appended to the same
     tags file so all downstream reports share one evidence pool
  3. per type: clips heard, mean per-phone agreement under the recognizer's
     known bias folds, and the modal heard phone at each disagreeing slot

Verdicts (>=MIN_CLIPS clips):
  AUDIO-OK   mean forgiving agreement >= 0.65  -> reading endorsed by audio
  SUSPECT    mean forgiving agreement <= 0.40  -> reading likely wrong; the
             per-slot modal phones say what the audio heard instead
  mid / thin otherwise -> stays in the human queue unchanged

Output: data/audio_lexicon/lk_sweep_votes.tsv (freq-sorted)

Usage:
  .venv/bin/python scripts/xeus_lk_sweep.py --max-chunks 300
  .venv/bin/python scripts/xeus_lk_sweep.py --report-only
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_tag import load_model  # noqa: E402
from xeus_pe_sweep import (AUDIO, DATASET, TAGS, _HEB, load_rows,  # noqa: E402
                           tag_positional)
from yiddish_g2p import g2p_token, lexicon_key  # noqa: E402

VOTES = REPO / "data" / "audio_lexicon" / "lk_sweep_votes.tsv"

RESCUE_REASONS = ("sefaria-pointed", "model-pointed-guess", "lk-fallback",
                  "pointed-audio-endorsed", "audio-homograph", "mwe-mined")
MIN_CLIPS = 2
OK_BAR = 0.65
SUSPECT_BAR = 0.40

# Recognizer-systematic confusions forgiven when scoring agreement (matches
# the bias analysis in docs/xeus_to_yiddish_map.md).
BIAS = {("ej", "ɛ"), ("aː", "a"), ("ɔ", "a"), ("ə", "ɛ"), ("ə", "a"),
        ("ə", "i"), ("u", "ɔ"), ("z", "s"), ("ɡ", "k"), ("d", "t"),
        ("b", "p"), ("v", "f"), ("ʒ", "ʃ"), ("ʤ", "ʧ")}


def forgiven(g: str, h: str) -> bool:
    return g == h or (g, h) in BIAS or (h, g) in BIAS


def classify(word: str, cache: dict) -> str:
    """'' if not a target, else the rescue reason / 'fallback'."""
    key = lexicon_key(word)
    if key not in cache:
        rec = g2p_token(word)
        why = ""
        for r in RESCUE_REASONS:
            if r in rec["reason"]:
                why = r
                break
        if not why and rec["route"] == "fallback":
            why = "fallback"
        cache[key] = why
    return cache[key]


def build_report(targets: dict[str, str], freq: Counter) -> list[dict]:
    per_type: dict[str, list[tuple[list[str], list[str]]]] = defaultdict(list)
    surface: dict[str, str] = {}
    if TAGS.exists():
        with TAGS.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                key = lexicon_key(rec["word"])
                if key in targets:
                    per_type[key].append((rec["g2p"], rec["heard_at"]))
                    surface.setdefault(key, rec["word"])
    report = []
    for key, clips in per_type.items():
        # score each clip against the type's modal g2p length
        scores, slot_heard = [], defaultdict(Counter)
        for g2p, heard in clips:
            n = len(g2p)
            hit = sum(1 for g, h in zip(g2p, heard) if h != "∅" and forgiven(g, h))
            scores.append(hit / n if n else 0.0)
            for i, (g, h) in enumerate(zip(g2p, heard)):
                if h != "∅" and not forgiven(g, h):
                    slot_heard[(i, g)][h] += 1
        mean = sum(scores) / len(scores)
        n = len(scores)
        if n < MIN_CLIPS:
            verdict = "thin"
        elif mean >= OK_BAR:
            verdict = "AUDIO-OK"
        elif mean <= SUSPECT_BAR:
            verdict = "SUSPECT"
        else:
            verdict = "mid"
        diffs = ";".join(
            f"{i}:{g}->{ctr.most_common(1)[0][0]}x{ctr.most_common(1)[0][1]}"
            for (i, g), ctr in sorted(slot_heard.items())
            if ctr.most_common(1)[0][1] >= 2)
        report.append({
            "word": surface[key], "key": key, "reason": targets[key],
            "freq": freq.get(key, 0), "clips": n,
            "g2p": "".join(clips[0][0]), "agree": round(mean, 2),
            "verdict": verdict, "audio_diffs": diffs,
        })
    report.sort(key=lambda r: -r["freq"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=300)
    ap.add_argument("--clips-per-type", type=int, default=3)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    cache: dict[str, str] = {}
    freq: Counter = Counter()
    where: dict[str, list[int]] = defaultdict(list)
    for ri, row in enumerate(rows):
        seen = set()
        for w in _HEB.findall(row["text"]):
            why = classify(w, cache)
            if why:
                key = lexicon_key(w)
                freq[key] += 1
                if key not in seen:
                    where[key].append(ri)
                    seen.add(key)
    targets = {k: v for k, v in
               ((k, cache.get(k, "")) for k in freq if freq[k] >= args.min_freq)
               if v}
    by_reason = Counter(targets.values())
    print(f"{len(targets)} rescued/fallback LK types (freq>={args.min_freq}): "
          f"{dict(by_reason)}", flush=True)

    if not args.report_only:
        covered: Counter = Counter()
        done: set[tuple[str, int]] = set()
        if TAGS.exists():
            with TAGS.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    done.add((str(rec["episode"]), int(rec["chunk_idx"])))
                    key = lexicon_key(rec["word"])
                    if key in targets:
                        covered[key] += 1
        print(f"prior tags cover {sum(1 for k in targets if covered[k] >= args.clips_per_type)}"
              f"/{len(targets)} types fully", flush=True)
        picked: list[int] = []
        candidates = sorted(
            {ri for k in targets for ri in where[k]},
            key=lambda ri: -sum(1 for w in set(_HEB.findall(rows[ri]["text"]))
                                if lexicon_key(w) in targets))
        for ri in candidates:
            row = rows[ri]
            if (str(row["episode"]), int(row["chunk_idx"])) in done:
                continue
            gain = sum(1 for w in set(_HEB.findall(row["text"]))
                       if lexicon_key(w) in targets
                       and covered[lexicon_key(w)] < args.clips_per_type)
            if gain == 0 or not (AUDIO / f"{row['episode']}.mp3").exists():
                continue
            picked.append(ri)
            for w in set(_HEB.findall(row["text"])):
                k = lexicon_key(w)
                if k in targets:
                    covered[k] += 1
            if len(picked) >= args.max_chunks:
                break
        print(f"{len(picked)} new chunks selected", flush=True)
        if picked:
            model, device = load_model()
            print(f"model on {device}", flush=True)
            with TAGS.open("a", encoding="utf-8") as out:
                for n, ri in enumerate(picked, 1):
                    row = rows[ri]
                    try:
                        recs = tag_positional(model, device,
                                              AUDIO / f"{row['episode']}.mp3", row)
                    except Exception as e:  # noqa: BLE001
                        print(f"chunk {row['episode']}/{row['chunk_idx']} "
                              f"failed: {e}", flush=True)
                        continue
                    for rec in recs:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    if n % 10 == 0 or n == len(picked):
                        print(f"progress {n}/{len(picked)} chunks", flush=True)

    report = build_report(targets, freq)
    with VOTES.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["word", "key", "reason", "freq",
                                            "clips", "g2p", "agree", "verdict",
                                            "audio_diffs"], delimiter="\t")
        wr.writeheader()
        wr.writerows(report)
    counts = Counter(r["verdict"] for r in report)
    print(f"\nreport: {len(report)} types -> {VOTES}  {dict(counts)}", flush=True)
    for r in [x for x in report if x["verdict"] == "SUSPECT"][:25]:
        print(f"  SUSPECT {r['word']} ({r['freq']}x, {r['clips']} clips, "
              f"agree {r['agree']}) g2p={r['g2p']} [{r['reason']}] "
              f"diffs {r['audio_diffs']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
