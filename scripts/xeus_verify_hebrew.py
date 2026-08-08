#!/usr/bin/env python
"""Audio-verify the UNVERIFIED Hebrew pointing with PhoneticXeus.

Quarantined loshn-koydesh words (no verified reading, §6.3) DO have an
unverified reading available: the corpus's model/Gemini-era `text_pointed`
column. This script tests that tier against the audio:

  1. find chunks whose text contains quarantined-LK tokens AND whose pointed
     column aligns token-for-token with the text
  2. build the chunk's expected phone sequence: normal engine output for
     ordinary words, but for quarantined words the reading of their POINTED
     form (quarantine off)
  3. PhoneticXeus-transcribe the audio slice, align, and score the pointed-
     Hebrew words separately from the control words in the same chunks

If pointed-Hebrew agreement ~= control agreement, the unverified pointing is
as trustworthy as the rest of the system and can be promoted (LOW confidence)
instead of quarantined. If it scores far below control, the distrust stands.

Usage: .venv/bin/python scripts/xeus_verify_hebrew.py --limit 120
Output: data/audio_lexicon/hebrew_verify.jsonl + printed summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_map import tokenize_g2p_ipa  # noqa: E402
from xeus_tag import SAMPLE_RATE, align, load_model, slice_audio, transcribe  # noqa: E402
from yiddish_g2p import g2p_token, hebrew_to_ipa  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״־]*")
_NIKUD = re.compile(r"[֑-ׇ]")


def strip_marks(s: str) -> str:
    return _NIKUD.sub("", s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120, help="max chunks")
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--sweep", action="store_true",
                    help="greedy chunk selection for >=--clips-per-type audio "
                         "clips per quarantined type (ignores --per-episode)")
    ap.add_argument("--clips-per-type", type=int, default=3)
    ap.add_argument("--append", action="store_true",
                    help="append to hebrew_verify.jsonl instead of overwriting")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(REPO / "data" / "yiddish_tts_dataset.tsv"), delimiter="\t"))
    candidates = []
    for r in rows:
        if not (REPO / "data" / "audio" / f"{r['episode']}.mp3").exists():
            continue
        toks = _HEB.findall(r["text"])
        ptoks = _HEB.findall(r.get("text_pointed", ""))
        if len(toks) != len(ptoks) or not toks:
            continue
        if any(strip_marks(p) != t.replace("־", "-") and strip_marks(p) != t
               for t, p in zip(toks, ptoks)):
            continue
        # quarantined-LK tokens with usable pointing
        targets = []
        for idx, (t, p) in enumerate(zip(toks, ptoks)):
            rec = g2p_token(t)
            if rec["route"] != "fallback":
                continue
            pointed_read = hebrew_to_ipa(p, stress=True, quarantine=False)
            ph = tokenize_g2p_ipa(pointed_read.replace(" ", ""))
            vowels = [x for x in ph if x in ("a", "aː", "ɛ", "ə", "i", "u", "ɔ", "ej", "aj", "ɔj", "oʊ")]
            if len(ph) >= 2 and vowels:
                targets.append((idx, t, p, ph))
        if targets:
            candidates.append((r, targets))

    if args.sweep:
        # Greedy set cover: pick the chunk adding the most clips to types still
        # under --clips-per-type, until nothing helps or --limit is hit.
        coverage: dict[str, int] = defaultdict(int)
        jobs = []
        remaining = candidates[:]
        while remaining and len(jobs) < args.limit:
            def gain(item):
                _, targets = item
                return sum(1 for _, t, _, _ in targets
                           if coverage[t] < args.clips_per_type)
            remaining.sort(key=gain, reverse=True)
            best = remaining.pop(0)
            if gain(best) == 0:
                break
            jobs.append(best)
            for _, t, _, _ in best[1]:
                coverage[t] += 1
        types_covered = sum(1 for v in coverage.values() if v >= args.clips_per_type)
        print(f"sweep: {len(jobs)} chunks; {len(coverage)} types touched, "
              f"{types_covered} at >={args.clips_per_type} clips", file=sys.stderr)
    else:
        jobs = []
        per_ep = defaultdict(int)
        for r, targets in candidates:
            if per_ep[r["episode"]] >= args.per_episode:
                continue
            jobs.append((r, targets))
            per_ep[r["episode"]] += 1
            if len(jobs) >= args.limit:
                break

    print(f"{len(jobs)} chunks with pointed-Hebrew targets", file=sys.stderr)
    model, device = load_model()
    print(f"device {device}", file=sys.stderr)

    out_path = REPO / "data" / "audio_lexicon" / "hebrew_verify.jsonl"
    t_ph = t_hit = c_ph = c_hit = 0
    n_t = n_c = 0
    with open(out_path, "a" if args.append else "w") as out:
        for r, targets in jobs:
            toks = _HEB.findall(r["text"])
            tmap = {idx: ph for idx, _, _, ph in targets}
            words: list[tuple[str, list[str], bool]] = []
            for idx, t in enumerate(toks):
                if idx in tmap:
                    words.append((t, tmap[idx], True))
                else:
                    ipa = hebrew_to_ipa(t, stress=True)
                    ph = tokenize_g2p_ipa(ipa.replace(" ", ""))
                    if ph:
                        words.append((t, ph, False))
            if not words:
                continue
            try:
                heard = transcribe(model, device, slice_audio(
                    REPO / "data" / "audio" / f"{r['episode']}.mp3",
                    float(r["start_s"]), float(r["end_s"])))
            except Exception as e:  # noqa: BLE001
                print(f"  {r['id']}: {type(e).__name__}", file=sys.stderr)
                continue
            flat, owner = [], []
            for wi, (_, ph, _) in enumerate(words):
                flat.extend(ph)
                owner.extend([wi] * len(ph))
            hits = defaultdict(int)
            heard_by_word = defaultdict(list)
            for pi, hj in align(flat, heard):
                if pi is None:
                    continue
                if hj is not None:
                    heard_by_word[owner[pi]].append(heard[hj])
                    if heard[hj] == flat[pi]:
                        hits[owner[pi]] += 1
            for wi, (w, ph, is_target) in enumerate(words):
                agr = hits[wi] / len(ph)
                if is_target:
                    t_ph += len(ph); t_hit += hits[wi]; n_t += 1
                    out.write(json.dumps({
                        "id": r["id"], "word": w,
                        "pointed_read": " ".join(ph),
                        "heard": " ".join(heard_by_word[wi]),
                        "agreement": round(agr, 3)}, ensure_ascii=False) + "\n")
                elif len(ph) >= 4:  # content-word control, same chunks
                    c_ph += len(ph); c_hit += hits[wi]; n_c += 1
            print(f"  {r['id']}: {len(targets)} targets", file=sys.stderr)

    print(f"\nTARGET (pointed Hebrew, unverified): {n_t} words, "
          f"phone agreement {t_hit}/{t_ph} = {t_hit/max(t_ph,1):.1%}")
    print(f"CONTROL (verified/rule words >=4ph, same chunks): {n_c} words, "
          f"phone agreement {c_hit}/{c_ph} = {c_hit/max(c_ph,1):.1%}")
    print(f"details -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
