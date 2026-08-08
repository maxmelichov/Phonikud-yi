#!/usr/bin/env python
"""Run the targeted tagging job list from data/audio_lexicon/xeus_jobs.tsv.

Groups jobs by episode, loads PhoneticXeus ONCE, and appends word tags to the
per-episode files data/audio_lexicon/xeus_tags_<episode>.jsonl (same records as
scripts/xeus_tag.py, plus "targets": the words_of_interest for that chunk).

Already-tagged (episode, chunk_idx) pairs are skipped unless --redo, so the job
list can be resumed after an interruption.

Usage:
  .venv/bin/python scripts/xeus_run_jobs.py --limit 2      # smoke run
  .venv/bin/python scripts/xeus_run_jobs.py                # full list
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_tag import load_model, tag_chunk  # noqa: E402

JOBS = REPO / "data" / "audio_lexicon" / "xeus_jobs.tsv"
DATASET = REPO / "data" / "yiddish_tts_dataset.tsv"
AUDIO_DIR = REPO / "data" / "audio"
OUT_DIR = REPO / "data" / "audio_lexicon"

csv.field_size_limit(10**7)


def load_jobs(path: Path) -> dict[str, dict[int, str]]:
    jobs: dict[str, dict[int, str]] = defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            jobs[r["episode"]][int(r["chunk_idx"])] = r["words_of_interest"]
    return jobs


def load_rows(wanted: dict[str, dict[int, str]]) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    with open(DATASET, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ep = r["episode"]
            if ep in wanted and int(r["chunk_idx"]) in wanted[ep]:
                rows[(ep, int(r["chunk_idx"]))] = r
    return rows


def already_done(episode: str) -> set[int]:
    path = OUT_DIR / f"xeus_tags_{episode}.jsonl"
    done: set[int] = set()
    if not path.exists():
        return done
    with open(path) as fh:
        for line in fh:
            try:
                done.add(int(json.loads(line)["chunk_idx"]))
            except Exception:  # noqa: BLE001 — tolerate a partial trailing line
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(JOBS))
    ap.add_argument("--limit", type=int, help="max chunks to process (smoke run)")
    ap.add_argument("--episode", help="restrict to one episode")
    ap.add_argument("--redo", action="store_true", help="re-tag chunks already present")
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    jobs = load_jobs(Path(args.jobs))
    if args.episode:
        jobs = {args.episode: jobs.get(args.episode, {})}

    # order: episode by episode so each MP3 is touched once
    todo: list[tuple[str, int, str]] = []
    for ep in sorted(jobs):
        if not (AUDIO_DIR / f"{ep}.mp3").exists():
            print(f"skip {ep}: no audio", file=sys.stderr)
            continue
        done = set() if args.redo else already_done(ep)
        for ci in sorted(jobs[ep]):
            if ci in done:
                continue
            todo.append((ep, ci, jobs[ep][ci]))
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do", file=sys.stderr)
        return 0

    wanted: dict[str, dict[int, str]] = defaultdict(dict)
    for ep, ci, tw in todo:
        wanted[ep][ci] = tw
    rows = load_rows(wanted)

    print(f"{len(todo)} chunk(s) across {len(wanted)} episode(s); loading PhoneticXeus...",
          file=sys.stderr)
    model, device = load_model()
    print(f"device: {device}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}
    n_rec = n_chunk = 0
    try:
        for ep, ci, targets in todo:
            row = rows.get((ep, ci))
            if row is None:
                print(f"{ep}/{ci}: not in dataset", file=sys.stderr)
                continue
            try:
                recs = tag_chunk(model, device, AUDIO_DIR / f"{ep}.mp3", row)
            except Exception as e:  # noqa: BLE001 — keep the batch alive
                print(f"{ep}/{ci}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            tset = set(targets.split())
            fh = handles.get(ep)
            if fh is None:
                fh = handles[ep] = open(OUT_DIR / f"xeus_tags_{ep}.jsonl", "a")
            for rec in recs:
                rec["targets"] = targets
                rec["is_target"] = rec["word"] in tset
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1
                if args.do_print and rec["is_target"]:
                    print(f"{rec['agreement']:5.2f}  {rec['word']:15s} "
                          f"g2p={rec['g2p']:28s} heard={rec['heard']}")
            fh.flush()
            n_chunk += 1
            print(f"{ep}/{ci}: {len(recs)} words ({len(tset)} target)", file=sys.stderr)
    finally:
        for fh in handles.values():
            fh.close()
    print(f"wrote {n_rec} word tags from {n_chunk} chunk(s) -> {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
