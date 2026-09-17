#!/usr/bin/env python3
"""A SYNTHETIC attest_lattice.jsonl in the new ear's row format, for developing
renikud_yi_prepare_v2.py before the real file exists.

Derived from the OLD ear's decisions (attest_v9.jsonl, stressless) and the
engine's per-word readings (attest_targets.jsonl): one row per word of every
--every-th chunk (default 12th -> ~2,000 chunks across all episodes).

  chosen      = the old ear's chosen phones where it attested the word at
                margin >= 0 (rule-path words), else the engine's phones
  chosen_ipa  = chosen with the ENGINE's stress ordinal placed on it
  margin      = the old ear's margin where attested, else 0.0
  posterior   = null (the old ear did not report one)
  synthetic   = true on every row

Nothing here is a real decision of the new ear: the file only has the
SHAPE of the real one, so the numbers prepare_v2 prints on it are format
checks, not label counts anyone should train on.

Usage: .venv/bin/python scripts/renikud_v2_synth_attest.py \
           [--out data/scratch/attest_lattice_synth.jsonl] [--every 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from renikud_yi_prepare_v2 import place_stress, stress_ordinal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(REPO / "data/xeus_ft/attest_targets.jsonl"))
    ap.add_argument("--old-attest", default=str(REPO / "data/xeus_ft/attest_v9.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/scratch/attest_lattice_synth.jsonl"))
    ap.add_argument("--every", type=int, default=12, help="take every k-th chunk of attest_targets")
    args = ap.parse_args()

    old: dict[tuple[str, int, int], dict] = {}
    for line in open(args.old_attest, encoding="utf-8"):
        r = json.loads(line)
        old[(r["episode"], int(r["chunk_idx"]), int(r["wi"]))] = r
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_chunks = n_old = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, line in enumerate(open(args.targets, encoding="utf-8")):
            if i % args.every:
                continue
            t = json.loads(line)
            n_chunks += 1
            ep, ci = t["episode"], int(t["chunk_idx"])
            for wi, w in enumerate(t["words"]):
                rec = old.get((ep, ci, wi))
                k = stress_ordinal(w["ipa"])
                if rec is not None and rec["key"] == w["key"] and rec.get("margin") is not None:
                    chosen, margin = list(rec["chosen"]), float(rec["margin"])
                    n_old += 1
                else:
                    chosen, margin = list(w["ph"]), 0.0
                chosen_ipa = place_stress(chosen, k)
                fh.write(json.dumps({
                    "episode": ep, "chunk_idx": ci, "wi": wi, "w": w["w"], "key": w["key"],
                    "route": w["route"], "conf": w["conf"],
                    "engine": list(w["ph"]), "engine_ipa": w["ipa"],
                    "chosen": chosen, "chosen_ipa": chosen_ipa,
                    "stress_index": stress_ordinal(chosen_ipa),
                    "margin": margin, "posterior": None,
                    "n_cand": rec.get("n_cand") if rec else None,
                    "synthetic": True,
                }, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"wrote {out}: {n_rows:,} rows over {n_chunks:,} chunks; {n_old:,} took the old ear's decision, the rest the engine's")


if __name__ == "__main__":
    main()
