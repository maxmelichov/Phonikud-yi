#!/usr/bin/env python3
"""Zero-shot probe of ivrit.ai's Yiddish Whisper on our corpus chunks.

Transcribes N held-out-episode chunks with ``ivrit-ai/yi-whisper-large-v3``
and scores word error rate against the corpus transcript (Yiddish Labs),
after both are reduced to bare letters (nikud stripped, punctuation dropped,
finals folded). The question is only whether it is usable as the *which
word* front end for the lattice, and for transcribing the rest of yiddish24.

  python scripts/whisper_yi_probe.py --data data/xeus_ft --root . --n 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import edit_distance, lexicon_key, read_jsonl  # noqa: E402
from xeus_ft_prepare import load_audio  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
MODEL = "ivrit-ai/yi-whisper-large-v3"


def words(text: str) -> list[str]:
    return [lexicon_key(w).strip("'\"-") for w in _HEB.findall(text)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import pipeline
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((Path(args.data) / "split.json").read_text(encoding="utf-8"))
    val_eps = set(split["val_episodes"])
    rows = [r for r in read_jsonl(Path(args.data) / "chunk_targets.jsonl") if r["episode"] in val_eps]
    rows = rows[: args.n]
    # the corpus text is not in chunk_targets; rebuild it from the words
    t0 = time.time()
    asr = pipeline("automatic-speech-recognition", model=MODEL, device=0 if device == "cuda" else -1,
                   torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                   chunk_length_s=30, return_timestamps=False)
    print(f"loaded {MODEL} in {time.time() - t0:.0f}s on {device}", flush=True)
    results = []
    tot_err = tot_ref = 0
    t0 = time.time()
    for r in rows:
        wav = load_audio(Path(args.root) / r["file"])
        hyp = asr({"raw": wav, "sampling_rate": 16000},
                  generate_kwargs={"language": "yi", "task": "transcribe"})["text"]
        ref_w = [w["key"].strip("'\"-") for w in r["words"]]
        hyp_w = words(hyp)
        err = edit_distance(ref_w, hyp_w)
        tot_err += err
        tot_ref += len(ref_w)
        results.append({"episode": r["episode"], "chunk_idx": r["chunk_idx"], "wer": err / max(1, len(ref_w)),
                        "ref": " ".join(ref_w[:14]), "hyp": hyp[:120]})
        print(f"  {r['episode']}-{r['chunk_idx']:05d} WER {err / max(1, len(ref_w)):.2f}  hyp: {hyp[:70]}", flush=True)
    summary = {"model": MODEL, "chunks": len(rows), "wer": tot_err / max(1, tot_ref),
               "seconds_per_chunk": (time.time() - t0) / max(1, len(rows)), "samples": results}
    out = Path(args.out or Path(args.data) / "whisper_yi_probe.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWER over {len(rows)} chunks: {summary['wer']:.3f}   ({summary['seconds_per_chunk']:.1f} s/chunk)")


if __name__ == "__main__":
    main()
