#!/usr/bin/env python
"""Select targeted audio chunks for PhoneticXeus voting.

Given a target word list, scan data/yiddish_tts_dataset.tsv and pick, per word,
up to N chunks that contain it, preferring distinct episodes and only episodes
whose MP3 is present in data/audio/.  Chunks are then deduplicated: one output
row per (episode, chunk_idx) carrying every target word it covers.

Default target list = union of three groups (spec section 12 voting material):
  (a) top --top-low-conf LOW_CONF words by freq that actually have audio coverage
  (b) every gold word whose gold_ipa lists 2+ variants (the contested ones)
  (c) the section 9 homographs

Usage:
  .venv/bin/python scripts/xeus_select_jobs.py
  .venv/bin/python scripts/xeus_select_jobs.py --words-file my_words.txt --per-word 6
Output: data/audio_lexicon/xeus_jobs.tsv  (episode, chunk_idx, words_of_interest)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "data" / "yiddish_tts_dataset.tsv"
AUDIO_DIR = REPO / "data" / "audio"
LOW_CONF = REPO / "data" / "phonemized" / "v3" / "low_conf.tsv"
GOLD = REPO / "g2p_gold_v3 - g2p_gold_v3.csv.csv"
OUT = REPO / "data" / "audio_lexicon" / "xeus_jobs.tsv"

# same word tokenizer as scripts/xeus_tag.py so the tagger sees what we selected
_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")

HOMOGRAPHS = ["שטייט", "נעמען", "בעל", "געוואלט", "עם", "אויף"]

csv.field_size_limit(10**7)


def available_episodes() -> set[str]:
    return {p.stem for p in AUDIO_DIR.glob("*.mp3")}


def load_index(episodes: set[str]) -> tuple[dict[str, list[tuple[str, int]]], int]:
    """word -> [(episode, chunk_idx), ...] over chunks whose episode has audio."""
    index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    n_rows = 0
    with open(DATASET, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ep = row["episode"]
            if ep not in episodes:
                continue
            n_rows += 1
            key = (ep, int(row["chunk_idx"]))
            for w in set(_HEB.findall(row["text"])):
                index[w].append(key)
    return index, n_rows


def low_conf_words(limit: int, index: dict) -> list[str]:
    rows = []
    with open(LOW_CONF, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append((int(r["freq"]), r["word"]))
    rows.sort(key=lambda t: -t[0])
    out = []
    for _freq, w in rows:
        if w in index:  # "actually have audio coverage"
            out.append(w)
            if len(out) >= limit:
                break
    return out


def contested_gold_words() -> list[str]:
    out = []
    with open(GOLD, newline="") as fh:
        for r in csv.DictReader(fh):
            variants = [v.strip() for v in (r.get("gold_ipa") or "").split("|")]
            variants = [v for v in variants if v]
            if len(variants) >= 2:
                out.append(r["word"])
    return out


def select(index: dict, targets: list[str], per_word: int) -> dict[tuple[str, int], list[str]]:
    """Per word pick up to per_word chunks, preferring unseen episodes."""
    chunks: dict[tuple[str, int], list[str]] = defaultdict(list)
    for w in targets:
        cands = index.get(w, [])
        if not cands:
            continue
        by_ep: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for key in cands:
            by_ep[key[0]].append(key)
        picked: list[tuple[str, int]] = []
        # round-robin across episodes -> maximal episode diversity
        eps = sorted(by_ep, key=lambda e: (-len(by_ep[e]), e))
        depth = 0
        while len(picked) < per_word:
            added = False
            for ep in eps:
                if depth < len(by_ep[ep]):
                    picked.append(sorted(by_ep[ep])[depth])
                    added = True
                    if len(picked) >= per_word:
                        break
            if not added:
                break
            depth += 1
        for key in picked:
            chunks[key].append(w)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-word", type=int, default=4)
    ap.add_argument("--top-low-conf", type=int, default=150)
    ap.add_argument("--words-file", help="newline-separated target words (replaces default groups)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    episodes = available_episodes()
    index, n_rows = load_index(episodes)
    print(f"{len(episodes)} episodes with audio; {n_rows} chunks indexed; "
          f"{len(index)} distinct word types", file=sys.stderr)

    if args.words_file:
        targets = [w.strip() for w in open(args.words_file) if w.strip()]
        groups = {"file": targets}
    else:
        groups = {
            "low_conf": low_conf_words(args.top_low_conf, index),
            "gold_contested": contested_gold_words(),
            "homographs": HOMOGRAPHS,
        }

    seen: set[str] = set()
    targets = []
    for name, words in groups.items():
        covered = [w for w in words if w in index]
        new = [w for w in covered if w not in seen]
        seen.update(new)
        targets.extend(new)
        print(f"group {name}: {len(words)} words, {len(covered)} with audio, "
              f"{len(new)} new", file=sys.stderr)
    print(f"target list: {len(targets)} unique words", file=sys.stderr)

    chunks = select(index, targets, args.per_word)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["episode", "chunk_idx", "words_of_interest"])
        for (ep, ci) in sorted(chunks, key=lambda k: (k[0], k[1])):
            w.writerow([ep, ci, " ".join(sorted(set(chunks[(ep, ci)])))])
    n_eps = len({ep for ep, _ in chunks})
    print(f"wrote {len(chunks)} unique chunks across {n_eps} episodes -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
