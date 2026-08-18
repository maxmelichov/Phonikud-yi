#!/usr/bin/env python
"""Segment partial lines into clean training spans (the settled line policy).

An elided (quarantined) token is still spoken in the audio, so a hole inside a
training line corrupts alignment. Policy: split every corpus line at its
quarantined tokens and keep the clean spans, each with its own token range, so
audio can be cut to match. Spans shorter than --min-tokens are dropped.

Reads data/corpus/yiddish_tts_dataset.tsv, routes each line with g2p_tokens (the same
call the corpus runner uses), and writes data/phonemized/v3/segments.tsv:
  id, span_idx, tok_start, tok_end (inclusive, token indices in the line),
  n_tokens, ipa

Run: .venv/bin/python scripts/make_training_segments.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from yiddish_g2p import g2p_tokens, normalize_ipa_spacing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = REPO / "data" / "phonemized" / "v3" / "segments.tsv"
    n_lines = n_spans = n_tok = 0
    with open(REPO / "data" / "corpus" / "yiddish_tts_dataset.tsv", newline="") as fin, \
         open(out_path, "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow(["id", "span_idx", "tok_start", "tok_end", "n_tokens", "ipa"])
        for i, row in enumerate(csv.DictReader(fin, delimiter="\t")):
            if args.limit and i >= args.limit:
                break
            recs = g2p_tokens(row["text"])
            n_lines += 1
            span: list[tuple[int, str]] = []
            spans: list[list[tuple[int, str]]] = []
            for ti, r in enumerate(recs):
                if r["route"] == "fallback" or not r["ipa_primary"]:
                    if span:
                        spans.append(span)
                    span = []
                else:
                    span.append((ti, r["lead"].replace('"', "") + r["ipa_primary"]
                                 + r["trail"].replace('"', "")))
            if span:
                spans.append(span)
            for si, sp in enumerate(s for s in spans if len(s) >= args.min_tokens):
                ipa = normalize_ipa_spacing(" ".join(t for _, t in sp))
                w.writerow([row["id"], si, sp[0][0], sp[-1][0], len(sp), ipa])
                n_spans += 1
                n_tok += len(sp)
    print(f"{n_lines} lines -> {n_spans} clean spans, {n_tok} tokens -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
