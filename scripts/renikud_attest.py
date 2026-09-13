#!/usr/bin/env python3
"""Second witness for the audio attestation: ReNikud-yi + graph on every attested token.

The v8 pointing tier (prepare_retrain_dataset_v8.py) supervises a rule-path
token when the EAR alone decides its reading with a margin of >= 2 nats.
ReNikud-yi with the same spelling graph reads those words 94% right against
the audio (docs §19, §26) from TEXT context alone, so it is an independent
second witness. This script reruns the production decode
(src/yiddish_renikud.py, the shipped int8 ONNX) over every attested chunk
and rewrites attest.jsonl for a v9 tier:

  agree  (ear's chosen == ReNikud-yi's best)  -> margin kept; if the ear was
         only mildly sure (0.5 <= margin < 2) the agreement lifts it to 2.0
         ("two witnesses"): the occurrence becomes decidable
  differ                                      -> margin negated: the
         occurrence decides nothing and votes for no type reading

Every original field survives; ``ear_margin`` keeps the ear's number,
``renikud`` / ``renikud_margin`` / ``agree`` record the second witness.

Usage: .venv/bin/python scripts/renikud_attest.py [--out data/xeus_ft/attest_v9.jsonl] [--limit N] [--workers 5]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

_MODEL = None


def _init(model_dir: str, threads: int):
    global _MODEL
    import yiddish_renikud as R
    _MODEL = R.ReNikudYi(model_dir, threads=threads)


def _chunk(args):
    """One chunk: (rows) -> list of (key, renikud_best, renikud_margin)."""
    import yiddish_renikud as R
    words, targets = args
    base_words = [R._base_text(w) for w in words]
    base = " ".join(base_words)
    starts = []
    pos = 0
    for bw in base_words:
        starts.append(pos)
        pos += len(bw) + 1
    try:
        lc, lv = _MODEL.logprobs(base)
    except Exception as e:  # noqa: BLE001
        return [(k, None, 0.0, repr(e)) for k, _, _ in targets]
    out = []
    for key, wi, eng in targets:
        w = base_words[wi]
        st = starts[wi]
        if not w:
            out.append((key, None, 0.0, "empty"))
            continue
        cands = R.graph_candidates(eng)
        own = _MODEL.free_reading(st, st + len(w), lc, lv)
        if own and own not in cands:
            cands.append(own)
        scored = sorted(((_MODEL.score(w, c, st, lc, lv), c) for c in cands), key=lambda x: -x[0])
        if not scored or scored[0][0] == float("-inf"):
            out.append((key, None, 0.0, "unalignable"))
            continue
        best_s, best = scored[0]
        second = next((s for s, _ in scored[1:] if s != float("-inf")), None)
        margin = (best_s - second) if second is not None else 99.0
        out.append((key, best, round(float(margin), 3), ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attest", default=str(REPO / "data/xeus_ft/attest.jsonl"))
    ap.add_argument("--targets", default=str(REPO / "data/xeus_ft/attest_targets.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/xeus_ft/attest_v9.jsonl"))
    ap.add_argument("--model", default=str(REPO / "models/renikud_yi_audio/onnx_int8"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lift", type=float, default=0.5, help="ear margin from which agreement lifts to 2.0")
    args = ap.parse_args()

    from xeus_ft_common import read_jsonl, write_jsonl
    ear = {}
    for r in read_jsonl(args.attest):
        ear[(r["episode"], r["chunk_idx"], r["wi"])] = r
    jobs = []
    for i, row in enumerate(read_jsonl(args.targets)):
        if args.limit and i >= args.limit:
            break
        words = [w["w"] for w in row["words"]]
        targets = []
        for wi, w in enumerate(row["words"]):
            key = (row["episode"], row["chunk_idx"], wi)
            r = ear.get(key)
            if r is not None:
                targets.append((key, wi, r["engine"]))
        if targets:
            jobs.append((words, targets))
    print(f"chunks {len(jobs)}  attested occurrences {sum(len(t) for _, t in jobs)}  model {args.model}", flush=True)

    import multiprocessing as mp
    t0 = time.time()
    stats = collections.Counter()
    results = {}
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.model, args.threads)) as pool:
        for n, res in enumerate(pool.imap_unordered(_chunk, jobs, chunksize=4), 1):
            for key, best, margin, err in res:
                results[key] = (best, margin, err)
            if n % 500 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} chunks  {el/60:.1f} min  eta {(len(jobs)-n)*el/n/60:.1f} min", flush=True)

    out_rows = []
    for key, r in ear.items():
        best, rmargin, err = results.get(key, (None, 0.0, "no chunk"))
        new = dict(r)
        new["ear_margin"] = r["margin"]
        new["renikud"] = best
        new["renikud_margin"] = rmargin
        if best is None:
            new["agree"] = None
            stats["no_reading:" + err] += 1
        else:
            agree = best == r["chosen"]
            new["agree"] = agree
            if agree:
                stats["agree"] += 1
                if args.lift <= r["margin"] < 2.0:
                    new["margin"] = 2.0
                    stats["lifted"] += 1
            else:
                stats["differ"] += 1
                new["margin"] = -abs(r["margin"])
                if r["margin"] >= 2.0:
                    stats["differ_was_decided"] += 1
                if best == r["engine"]:
                    stats["differ_renikud_sides_with_engine"] += 1
        out_rows.append(new)
    write_jsonl(args.out, out_rows)
    before = sum(1 for r in ear.values() if r["margin"] >= 2.0)
    after = sum(1 for r in out_rows if r["margin"] >= 2.0)
    print(f"wrote {args.out}: {len(out_rows)} rows in {(time.time()-t0)/60:.1f} min")
    for k, v in sorted(stats.items()):
        print(f"  {k:40s} {v}")
    print(f"  decided occurrences (margin >= 2): ear alone {before}  ->  ear ∧ ReNikud-yi {after}")


if __name__ == "__main__":
    main()
