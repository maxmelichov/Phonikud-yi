#!/usr/bin/env python
"""Corpus-wide audio sweep over EVERY unresolved reading, token-weighted.

The earlier sweeps each targeted one class (pe-default, rescued LK). This one
targets the whole LOW-confidence set -- alef-default vowels, pe letters,
model-guessed and book-pointed Hebrew, lk-fallback -- and picks chunks by how
many LOW *tokens* they resolve, not how many types. That ordering matters:
LOW is 48k types but only 256k tokens, so a type-greedy sweep spends its
compute on hapaxes while the words a listener actually hears go unchecked.
Token-weighted, 903 chunks already cover 49% of LOW tokens and another 1,500
reach ~67%; type coverage over the same run is 7%.

Selection is greedy with a lazy heap: a chunk's gain is recomputed when it
reaches the top, so gains stay exact as coverage fills in, without rescoring
every candidate each round.

Tags land in the SHARED pool data/audio_lexicon/pe_sweep_tags.jsonl, so the
existing folders (build_audio_pe_lexicon.py, build_audio_vowel_lexicon.py,
xeus_lk_sweep.py --report-only) all improve from one run. Resumable: chunks
already in the pool are skipped.

Usage:
  .venv/bin/python scripts/xeus_sweep_all.py --max-chunks 1500
  .venv/bin/python scripts/xeus_sweep_all.py --plan-only          # projection
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_pe_sweep import AUDIO, DATASET, TAGS, tag_positional  # noqa: E402
from xeus_tag import load_model  # noqa: E402
from yiddish_g2p import lexicon_key  # noqa: E402

LOW = REPO / "data" / "phonemized" / "v3" / "low_conf.tsv"
_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
MIN_CLIPS = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=1500)
    ap.add_argument("--clips-per-type", type=int, default=MIN_CLIPS)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    if not LOW.exists():
        raise SystemExit(f"{LOW} missing -- run scripts/run_corpus_v3.py --limit 0")
    low: dict[str, int] = {}
    with LOW.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            low[lexicon_key(row["word"])] = int(row["freq"])
    total_tokens = sum(low.values())

    covered: Counter = Counter()
    done_chunks: set[tuple[str, int]] = set()
    if TAGS.exists():
        with TAGS.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                done_chunks.add((str(rec["episode"]), int(rec["chunk_idx"])))
                key = lexicon_key(rec["word"])
                if key in low:
                    covered[key] += 1

    def resolved_tokens() -> int:
        return sum(v for k, v in low.items() if covered[k] >= args.clips_per_type)

    rows = list(csv.DictReader(DATASET.open(encoding="utf-8"), delimiter="\t"))
    keys_of: list[set[str]] = [
        {lexicon_key(w) for w in set(_HEB.findall(r["text"]))} for r in rows
    ]
    print(f"LOW: {len(low):,} types / {total_tokens:,} tokens; "
          f"{len(done_chunks):,} chunks already tagged, "
          f"{resolved_tokens()/total_tokens:.1%} of LOW tokens resolved",
          flush=True)

    def gain(i: int) -> int:
        return sum(low[k] for k in keys_of[i]
                   if k in low and covered[k] < args.clips_per_type)

    usable = [
        i for i, r in enumerate(rows)
        if (str(r["episode"]), int(r["chunk_idx"])) not in done_chunks
        and (AUDIO / f"{r['episode']}.mp3").exists()
    ]
    heap = [(-g, i) for i in usable if (g := gain(i)) > 0]
    heapq.heapify(heap)

    picked: list[int] = []
    while heap and len(picked) < args.max_chunks:
        neg, i = heapq.heappop(heap)
        g = gain(i)
        if g == 0:
            continue
        if -neg != g:  # stale priority -> reinsert with the true gain
            heapq.heappush(heap, (-g, i))
            continue
        picked.append(i)
        for k in keys_of[i]:
            if k in low:
                covered[k] += 1
    print(f"selected {len(picked):,} chunks -> projected "
          f"{resolved_tokens()/total_tokens:.1%} of LOW tokens resolved",
          flush=True)
    if args.plan_only:
        return 0

    # coverage counters above were advanced for planning; tagging is what makes
    # them real, so re-run in order and write the tags out.
    model, device = load_model()
    print(f"model on {device}", flush=True)
    with TAGS.open("a", encoding="utf-8") as out:
        for n, i in enumerate(picked, 1):
            row = rows[i]
            try:
                recs = tag_positional(model, device,
                                      AUDIO / f"{row['episode']}.mp3", row)
            except Exception as e:  # noqa: BLE001
                print(f"chunk {row['episode']}/{row['chunk_idx']} failed: {e}",
                      flush=True)
                continue
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(picked):
                print(f"progress {n}/{len(picked)} chunks", flush=True)
    print("done -- now rerun the folders:\n"
          "  scripts/build_audio_pe_lexicon.py\n"
          "  scripts/build_audio_vowel_lexicon.py\n"
          "  scripts/xeus_lk_sweep.py --report-only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
