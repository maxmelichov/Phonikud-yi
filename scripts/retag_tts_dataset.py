#!/usr/bin/env python
"""Retag the TTS dataset with the system's own verified stack — no Gemini.

Column sources:
  nikud  phonikud-yi v5 ONNX (models/phonikud_yi_v5/v5.onnx) — the model
         finetuned on audio-corrected, conflict-repaired labels; validated
         per row by letter identity (pointing must strip back to the text)
  ipa    the G2P engine's strict-policy corpus run (data/phonemized/v3/
         lines.tsv), which embodies the full authority chain: gold > audio
         tables > book pointing > model guesses, with QA gates a–d passed

The source data/corpus/yiddish_tts_dataset.tsv is READ ONLY; output is the new
data/corpus/yiddish_tts_dataset_v2.tsv. Rows the strict line policy dropped from
the corpus run keep an empty ipa (they were dropped for a reason — mid-line
quarantined tokens); rows whose v5 pointing fails letter identity keep an
empty nikud and are counted. Nothing is guessed at silently.

Usage: .venv/bin/python scripts/retag_tts_dataset.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from infer_onnx import Diacritizer, strip_marks  # noqa: E402

SRC = REPO / "data" / "corpus" / "yiddish_tts_dataset.tsv"
LINES = REPO / "data" / "phonemized" / "v3" / "lines.tsv"
OUT = REPO / "data" / "corpus" / "yiddish_tts_dataset_v2.tsv"
ONNX = REPO / "models" / "phonikud_yi_v5" / "v5.onnx" / "model.onnx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    ipa_by_id: dict[str, str] = {}
    with LINES.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ipa_by_id[row["id"]] = row["ipa"]

    with SRC.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]

    d = Diacritizer(ONNX)
    limit = d.max_len - 2 * d.off - 8

    def point_long(text: str) -> str:
        """Point a row of any length: split on whitespace runs into <=limit
        segments, point each, reassemble byte-exactly (separators pass
        through the model unchanged; only the segmentation is ours)."""
        if len(text) <= limit:
            return d.point_model(text)[0]
        pieces = re.split(r"(\s+)", text)
        segs: list[str] = []
        cur = ""
        for p in pieces:
            if len(cur) + len(p) > limit and cur:
                segs.append(cur)
                cur = p
            else:
                cur += p
        if cur:
            segs.append(cur)
        return "".join(d.point_model(segs))
    n_ok = n_badnikud = n_noipa = 0
    t0 = time.time()
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["id", "episode", "chunk_idx", "start_s", "end_s",
                     "text", "nikud", "ipa"])
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            # the source text column is PARTIALLY pointed; the model points
            # the bare letters and the identity check is bare-vs-bare
            texts = [strip_marks(unicodedata.normalize("NFC", r["text"]))
                     for r in batch]
            pointed = [point_long(t) for t in texts]
            for r, text, pt in zip(batch, texts, pointed):
                if strip_marks(pt) != text:
                    pt = ""  # letter identity broken: never ship a mutation
                    n_badnikud += 1
                ipa = ipa_by_id.get(r["id"], "")
                if not ipa:
                    n_noipa += 1
                if pt and ipa:
                    n_ok += 1
                wr.writerow([r["id"], r["episode"], r["chunk_idx"],
                             r["start_s"], r["end_s"], text, pt, ipa])
            done = min(i + args.batch_size, len(rows))
            if done % 1600 < args.batch_size or done == len(rows):
                rate = done / (time.time() - t0)
                print(f"{done}/{len(rows)} rows ({rate:.0f}/s)", flush=True)
    print(f"complete rows: {n_ok}/{len(rows)}  "
          f"nikud letter-identity failures: {n_badnikud}  "
          f"no strict-policy ipa: {n_noipa}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
