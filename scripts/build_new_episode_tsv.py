#!/usr/bin/env python
"""Build a v2-shaped TTS dataset TSV for newly annotated episodes.

Reads data/annotations/<episode>.jsonl (Gemini verbatim text per ~30 s chunk,
written by scripts/annotate_audio.py) and writes a TSV with the SAME columns as
data/corpus/yiddish_tts_dataset_v2.tsv:

  id  episode  chunk_idx  start_s  end_s  text  nikud  ipa

  id     <episode>-<chunk_idx:05d>   (the v2 scheme)
  text   the Gemini text_yi with any partial pointing stripped (v2 stores the
         bare letters; retag_tts_dataset.py did the same)
  nikud  src/yiddish_labels.text_to_nikud_batch (phonikud-yi v9 + the label
         stack's guards), blanked when the pointing does not strip back to the
         text letter-for-letter -- never ship a mutation
  ipa    src/yiddish_labels.text_to_ipa (engine + ReNikud-yi context reader),
         the production path

Chunks whose text_yi is empty (music / silence) are dropped, as in v2.
The existing corpus TSVs are never read or written.

Usage:
  .venv/bin/python scripts/build_new_episode_tsv.py \
      --episode 166782 --episode 166297 ... \
      --out data/corpus/yiddish_tts_dataset_new_2026-09.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from infer_onnx import strip_marks  # noqa: E402
from yiddish_labels import text_to_ipa, text_to_nikud_batch  # noqa: E402

ANNOT_DIR = REPO / "data" / "annotations"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", action="append", required=True)
    ap.add_argument("--annotations-dir", default=str(ANNOT_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    rows: list[dict] = []
    per_ep: dict[str, int] = {}
    for eid in args.episode:
        p = Path(args.annotations_dir) / f"{eid}.jsonl"
        if not p.exists():
            print(f"[{eid}] no annotation file {p}", file=sys.stderr)
            per_ep[eid] = 0
            continue
        recs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        recs.sort(key=lambda r: r["chunk_idx"])
        n = 0
        for r in recs:
            text = strip_marks(unicodedata.normalize("NFC", (r.get("text_yi") or "").strip()))
            if not text:
                continue
            rows.append({
                "id": f"{eid}-{int(r['chunk_idx']):05d}",
                "episode": eid,
                "chunk_idx": int(r["chunk_idx"]),
                "start_s": r["start_s"],
                "end_s": r["end_s"],
                "text": text,
            })
            n += 1
        per_ep[eid] = n

    n_nikud = n_ipa = n_bad = 0
    t0 = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["id", "episode", "chunk_idx", "start_s", "end_s", "text", "nikud", "ipa"])
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            texts = [r["text"] for r in batch]
            try:
                pointed = text_to_nikud_batch(texts)
            except Exception as exc:  # noqa: BLE001
                print(f"nikud batch failed at row {i}: {exc!r}", file=sys.stderr)
                pointed = [""] * len(texts)
            for r, pt in zip(batch, pointed):
                if pt and strip_marks(pt) != r["text"]:
                    pt = ""
                    n_bad += 1
                try:
                    ipa = text_to_ipa(r["text"])
                except Exception as exc:  # noqa: BLE001
                    print(f"ipa failed for {r['id']}: {exc!r}", file=sys.stderr)
                    ipa = ""
                n_nikud += bool(pt)
                n_ipa += bool(ipa)
                wr.writerow([r["id"], r["episode"], r["chunk_idx"], r["start_s"],
                             r["end_s"], r["text"], pt, ipa])
            done = min(i + args.batch_size, len(rows))
            if done % 160 < args.batch_size or done == len(rows):
                print(f"{done}/{len(rows)} rows ({done / (time.time() - t0):.1f}/s)", flush=True)

    print(json.dumps({"rows": len(rows), "per_episode": per_ep, "nikud": n_nikud,
                      "nikud_identity_failures": n_bad, "ipa": n_ipa,
                      "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
