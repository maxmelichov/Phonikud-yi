#!/usr/bin/env python3
"""Text side of audio attestation: which corpus tokens the ear should decide.

Runs locally (needs the engine). For every corpus chunk, every Hebrew token
with the engine's reading, its route/confidence, and whether it is a
candidate for attestation:

  attest = the engine reads it by RULE at LOW or MED confidence, it is not a
           certain (Chezky) word, and it has a reading at all.

Those are the tokens the pointing model has never been supervised on
(prepare_retrain_dataset_v3 supervises only lexicon HIGH/MED): 633k tokens,
83k types, 35% of the corpus. Certain words and lexicon words are still
listed — the forced alignment needs every word's reading to place the
others — but carry attest=false.

Output: data/xeus_ft/attest_targets.jsonl (one row per chunk).
Usage:  .venv/bin/python scripts/xeus_attest_text.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import tokenize_ipa, write_jsonl  # noqa: E402
from yiddish_g2p import g2p_token, lexicon_key  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--out", default=str(REPO / "data/xeus_ft/attest_targets.jsonl"))
    args = ap.parse_args()
    csv.field_size_limit(10_000_000)

    certain = {v["key"] for v in json.loads(Path(args.dictionary).read_text(encoding="utf-8")).values()}
    cache: dict[str, dict] = {}
    n_tok = n_att = 0
    rows = []
    for r in csv.DictReader(open(args.corpus, encoding="utf-8"), delimiter="\t"):
        words = []
        for w in _HEB.findall(r["text"]):
            key = lexicon_key(w)
            if w not in cache:
                t = g2p_token(w)
                t = t if isinstance(t, dict) else t.__dict__
                cache[w] = {
                    "ph": tokenize_ipa(t["ipa_primary"] or ""),
                    "ipa": t["ipa_primary"] or "",
                    "route": t["route"], "conf": t["confidence"],
                }
            c = cache[w]
            attest = (key not in certain and c["route"] == "rule"
                      and c["conf"] in ("LOW", "MED") and bool(c["ph"]))
            n_tok += 1
            n_att += attest
            words.append({"w": w, "key": key, "ph": c["ph"], "ipa": c["ipa"],
                          "route": c["route"], "conf": c["conf"], "attest": attest})
        rows.append({"episode": r["episode"], "chunk_idx": int(r["chunk_idx"]),
                     "file": f"data/chunks/{r['episode']}/chunk_{int(r['chunk_idx']):05d}.mp3",
                     "start_s": float(r["start_s"]), "words": words})
    n = write_jsonl(args.out, rows)
    print(f"chunks {n:,}  tokens {n_tok:,}  to attest {n_att:,} ({n_att / n_tok:.1%})  -> {args.out}")


if __name__ == "__main__":
    main()
