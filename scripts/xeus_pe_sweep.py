#!/usr/bin/env python
"""Audio-confirm every pe-default word (the far/par class) with PhoneticXeus.

The engine's §4 rule reads an unpointed פ as /f/ unless a lexical list says
/p/, and flags the guess LOW with reason ``pe-default``. p vs f is a binary,
acoustically robust contrast, so the corpus audio can vote on each type
directly:

  1. every token of data/yiddish_tts_dataset.tsv whose g2p_token reason
     contains ``pe-default`` is a target; types are ranked by corpus count
  2. chunks are picked greedily so each target type is heard in up to
     --clips-per-type different clips (a chunk covers many targets at once)
  3. each chunk is sliced, PhoneticXeus-transcribed, folded to the v3
     inventory and Needleman-Wunsch-aligned with POSITIONS KEPT, so the
     heard phone at each engine f/p slot is known exactly
  4. votes aggregate per (type, pe-position): p-ish = {p, b}, f-ish = {f, v}
     (voicing is the recognizer's known confusion axis, labial place is not)

Verdict per type, mirroring xeus_vote.py's reporting bar: >=3 voting clips
and >=67% on the side that DIFFERS from the engine -> FLIP candidate;
>=67% agreeing with the engine -> CONFIRMED; anything else -> CONTESTED.

Output (append-safe, resumable):
  data/audio_lexicon/pe_sweep_tags.jsonl   raw per-word positional alignments
  data/audio_lexicon/pe_sweep_votes.tsv    per-type verdicts, freq-sorted

Usage:
  .venv/bin/python scripts/xeus_pe_sweep.py --max-chunks 400
  .venv/bin/python scripts/xeus_pe_sweep.py --report-only
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

from xeus_map import tokenize_g2p_ipa  # noqa: E402
from xeus_tag import align, load_model, slice_audio, transcribe  # noqa: E402
from yiddish_g2p import g2p_token, hebrew_to_ipa, lexicon_key  # noqa: E402

DATASET = REPO / "data" / "yiddish_tts_dataset.tsv"
AUDIO = REPO / "data" / "audio"
OUT_DIR = REPO / "data" / "audio_lexicon"
TAGS = OUT_DIR / "pe_sweep_tags.jsonl"
VOTES = OUT_DIR / "pe_sweep_votes.tsv"

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
P_ISH = {"p", "b"}
F_ISH = {"f", "v"}
MIN_CLIPS = 3
MAJORITY = 2 / 3


def is_target(word: str, cache: dict) -> bool:
    key = lexicon_key(word)
    if key not in cache:
        cache[key] = "pe-default" in g2p_token(word)["reason"]
    return cache[key]


def chunk_words(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for w in _HEB.findall(text):
        ipa = hebrew_to_ipa(w, stress=True)
        toks = tokenize_g2p_ipa(ipa.replace(" ", ""))
        if toks:
            out.append((w, toks))
    return out


def tag_positional(model, device, mp3: Path, row: dict) -> list[dict]:
    """Per-word records with the aligned heard phone at every engine slot."""
    words = chunk_words(row["text"])
    if not words:
        return []
    heard = transcribe(model, device,
                       slice_audio(mp3, float(row["start_s"]), float(row["end_s"])))
    flat, owner = [], []
    for wi, (_, toks) in enumerate(words):
        flat.extend(toks)
        owner.extend([wi] * len(toks))
    slots: dict[int, list[str]] = {wi: ["∅"] * len(t) for wi, (_, t) in enumerate(words)}
    base = 0
    starts = []
    for _, toks in words:
        starts.append(base)
        base += len(toks)
    for pi, hj in align(flat, heard):
        if pi is None or hj is None:
            continue
        wi = owner[pi]
        slots[wi][pi - starts[wi]] = heard[hj]
    recs = []
    for wi, (w, toks) in enumerate(words):
        recs.append({
            "episode": row["episode"], "chunk_idx": int(row["chunk_idx"]),
            "word": w, "g2p": toks, "heard_at": slots[wi],
        })
    return recs


def load_rows() -> list[dict]:
    with DATASET.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def build_report(target_keys: set[str], freq: Counter) -> list[dict]:
    """Aggregate pe_sweep_tags.jsonl into per-type verdicts."""
    votes: dict[str, list[Counter]] = {}
    clips: Counter = Counter()
    surface: dict[str, str] = {}
    g2p_of: dict[str, list[str]] = {}
    if not TAGS.exists():
        return []
    with TAGS.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            key = lexicon_key(rec["word"])
            if key not in target_keys:
                continue
            g2p = rec["g2p"]
            pe_pos = [i for i, p in enumerate(g2p) if p in ("f", "p")]
            if not pe_pos:
                continue
            if key not in votes:
                votes[key] = [Counter() for _ in pe_pos]
                surface[key] = rec["word"]
                g2p_of[key] = g2p
            elif len(pe_pos) != len(votes[key]):
                continue  # different tokenization of a variant spelling; skip
            voted = False
            for slot, i in enumerate(pe_pos):
                h = rec["heard_at"][i]
                if h in P_ISH | F_ISH:
                    votes[key][slot][h] += 1
                    voted = True
            if voted:
                clips[key] += 1
    report = []
    for key, slot_votes in votes.items():
        n = clips[key]
        verdicts = []
        for slot, ctr in enumerate(slot_votes):
            p = sum(ctr[c] for c in P_ISH)
            f = sum(ctr[c] for c in F_ISH)
            tot = p + f
            engine = g2p_of[key][[i for i, ph in enumerate(g2p_of[key])
                                  if ph in ("f", "p")][slot]]
            if tot < MIN_CLIPS:
                verdicts.append("thin")
            elif p / tot >= MAJORITY:
                verdicts.append("CONFIRMED" if engine == "p" else "FLIP->p")
            elif f / tot >= MAJORITY:
                verdicts.append("CONFIRMED" if engine == "f" else "FLIP->f")
            else:
                verdicts.append("contested")
        report.append({
            "word": surface[key], "key": key, "freq": freq.get(key, 0),
            "clips": n, "g2p": "".join(g2p_of[key]),
            "votes": ";".join(f"p={sum(c[x] for x in P_ISH)}/f={sum(c[x] for x in F_ISH)}"
                              for c in slot_votes),
            "verdict": "|".join(verdicts),
        })
    report.sort(key=lambda r: -r["freq"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=400)
    ap.add_argument("--clips-per-type", type=int, default=3)
    ap.add_argument("--min-freq", type=int, default=2,
                    help="skip types with fewer dataset occurrences")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    cache: dict[str, bool] = {}
    freq: Counter = Counter()
    where: dict[str, list[int]] = defaultdict(list)  # key -> row indices
    for ri, row in enumerate(rows):
        seen_here = set()
        for w in _HEB.findall(row["text"]):
            if is_target(w, cache):
                key = lexicon_key(w)
                freq[key] += 1
                if key not in seen_here:
                    where[key].append(ri)
                    seen_here.add(key)
    targets = {k for k, c in freq.items() if c >= args.min_freq}
    print(f"{len(targets)} target types (freq>={args.min_freq}), "
          f"{sum(freq[k] for k in targets)} tokens", flush=True)

    if not args.report_only:
        done: set[tuple[str, int]] = set()
        if TAGS.exists():
            with TAGS.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    done.add((str(rec["episode"]), int(rec["chunk_idx"])))
        need: Counter = Counter({k: args.clips_per_type for k in targets})
        for key in targets:  # credit chunks already tagged
            pass
        # greedy chunk pick: most unmet-need coverage first
        picked: list[int] = []
        covered: Counter = Counter()
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
            if gain == 0:
                continue
            if not (AUDIO / f"{row['episode']}.mp3").exists():
                continue
            picked.append(ri)
            for w in set(_HEB.findall(row["text"])):
                k = lexicon_key(w)
                if k in targets:
                    covered[k] += 1
            if len(picked) >= args.max_chunks:
                break
        print(f"{len(picked)} chunks selected "
              f"({sum(1 for k in targets if covered[k] >= args.clips_per_type)}"
              f"/{len(targets)} types fully covered)", flush=True)

        model, device = load_model()
        print(f"model on {device}", flush=True)
        with TAGS.open("a", encoding="utf-8") as out:
            for n, ri in enumerate(picked, 1):
                row = rows[ri]
                mp3 = AUDIO / f"{row['episode']}.mp3"
                try:
                    recs = tag_positional(model, device, mp3, row)
                except Exception as e:  # noqa: BLE001
                    print(f"chunk {row['episode']}/{row['chunk_idx']} failed: {e}",
                          flush=True)
                    continue
                for rec in recs:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if n % 10 == 0 or n == len(picked):
                    print(f"progress {n}/{len(picked)} chunks", flush=True)

    report = build_report(targets, freq)
    with VOTES.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["word", "key", "freq", "clips",
                                            "g2p", "votes", "verdict"],
                            delimiter="\t")
        wr.writeheader()
        wr.writerows(report)
    flips = [r for r in report if "FLIP" in r["verdict"]]
    conf = [r for r in report if "FLIP" not in r["verdict"]
            and "CONFIRMED" in r["verdict"]]
    print(f"\nreport: {len(report)} types voted, {len(conf)} confirmed, "
          f"{len(flips)} FLIP candidates -> {VOTES}", flush=True)
    for r in flips[:30]:
        print(f"  FLIP {r['word']} ({r['freq']}x, {r['clips']} clips) "
              f"g2p={r['g2p']} votes={r['votes']} -> {r['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
