#!/usr/bin/env python
"""Tag corpus words with audio phonemes from PhoneticXeus, folded to Yiddish v3.

Pipeline per chunk (a row of data/corpus/yiddish_tts_dataset.tsv):
  1. slice the episode MP3 to mono 16 kHz with ffmpeg
  2. PhoneticXeus -> universal IPA phone sequence
  3. fold onto the closed v3 inventory (scripts/xeus_map.py)
  4. Needleman-Wunsch align the heard phones against the G2P prediction for the
     chunk text, with word boundaries tracked
  5. emit one JSONL record per word: g2p phones, heard phones, per-phone
     agreement — the raw material for lexicon voting (spec section 12)

Usage:
  .venv/bin/python scripts/xeus_tag.py --episode 100313 --chunks 0 1 2
  .venv/bin/python scripts/xeus_tag.py --episode 100313 --limit 10
  .venv/bin/python scripts/xeus_tag.py --episode 100313 --limit 3 --print

Output: data/audio_lexicon/xeus_tags_<episode>.jsonl
The model (2.3 GB) downloads on first run to the HF cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_map import map_transcript, tokenize_g2p_ipa, VOWELS  # noqa: E402
from yiddish_g2p import hebrew_to_ipa  # noqa: E402

SAMPLE_RATE = 16000
_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")


def load_model():
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "changelinglab/PhoneticXeus", trust_remote_code=True
    ).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = model.to(device)
    except Exception:
        device = "cpu"
    return model, device


def slice_audio(mp3: Path, start: float, end: float) -> "object":
    """ffmpeg-slice [start, end] of mp3 to a mono 16 kHz tensor."""
    import torch
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-to", str(end),
             "-i", str(mp3), "-ac", "1", "-ar", str(SAMPLE_RATE), tmp.name],
            check=True,
        )
        wav, sr = sf.read(tmp.name, dtype="float32")
    assert sr == SAMPLE_RATE
    return torch.from_numpy(wav)


def transcribe(model, device, wav) -> list[str]:
    import torch

    with torch.no_grad():
        res = model.transcribe(wav.to(device), sampling_rate=SAMPLE_RATE)
    return map_transcript(res[0]["predicted_transcript"])


def g2p_words(text: str) -> list[tuple[str, list[str]]]:
    """(hebrew word, v3 phone tokens) for each word in the chunk text."""
    out = []
    for w in _HEB.findall(text):
        ipa = hebrew_to_ipa(w, stress=True)
        toks = tokenize_g2p_ipa(ipa.replace(" ", ""))
        if toks:
            out.append((w, toks))
    return out


def align(pred: list[str], heard: list[str]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch; returns aligned index pairs (pred_i|None, heard_j|None)."""
    n, m = len(pred), len(heard)
    GAP = -1.0

    def sim(a: str, b: str) -> float:
        if a == b:
            return 2.0
        if (a in VOWELS) == (b in VOWELS):
            return 0.5 if (a in VOWELS) else 0.0
        return -1.5

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * GAP
    for j in range(1, m + 1):
        score[0][j] = j * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(
                score[i - 1][j - 1] + sim(pred[i - 1], heard[j - 1]),
                score[i - 1][j] + GAP,
                score[i][j - 1] + GAP,
            )
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + sim(pred[i - 1], heard[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and score[i][j] == score[i - 1][j] + GAP:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def tag_chunk(model, device, mp3: Path, row: dict) -> list[dict]:
    words = g2p_words(row["text"])
    if not words:
        return []
    heard = transcribe(model, device, slice_audio(mp3, float(row["start_s"]), float(row["end_s"])))
    flat: list[str] = []
    owner: list[int] = []  # phone index -> word index
    for wi, (_, toks) in enumerate(words):
        flat.extend(toks)
        owner.extend([wi] * len(toks))

    per_word_heard: dict[int, list[str]] = {wi: [] for wi in range(len(words))}
    per_word_hits: dict[int, int] = {wi: 0 for wi in range(len(words))}
    for pi, hj in align(flat, heard):
        if pi is None:
            continue  # inserted noise phone between words; unassigned
        wi = owner[pi]
        if hj is not None:
            per_word_heard[wi].append(heard[hj])
            if heard[hj] == flat[pi]:
                per_word_hits[wi] += 1

    recs = []
    for wi, (w, toks) in enumerate(words):
        h = per_word_heard[wi]
        recs.append({
            "episode": row["episode"], "chunk_idx": int(row["chunk_idx"]),
            "word": w,
            "g2p": " ".join(toks),
            "heard": " ".join(h),
            "n_phones": len(toks),
            "n_match": per_word_hits[wi],
            "agreement": round(per_word_hits[wi] / len(toks), 3),
        })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--chunks", type=int, nargs="*", help="chunk_idx values; default: all")
    ap.add_argument("--limit", type=int, help="max chunks to process")
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    mp3 = REPO / "data" / "audio" / f"{args.episode}.mp3"
    if not mp3.exists():
        sys.exit(f"no audio: {mp3}")
    rows = [
        r for r in csv.DictReader(open(REPO / "data" / "corpus" / "yiddish_tts_dataset.tsv"), delimiter="\t")
        if r["episode"] == args.episode
        and (args.chunks is None or int(r["chunk_idx"]) in set(args.chunks))
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("no matching rows")

    print(f"loading PhoneticXeus (first run downloads 2.3 GB)...", file=sys.stderr)
    model, device = load_model()
    print(f"device: {device}; {len(rows)} chunk(s)", file=sys.stderr)

    out_dir = REPO / "data" / "audio_lexicon"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"xeus_tags_{args.episode}.jsonl"
    n = 0
    with open(out_path, "a") as f:
        for row in rows:
            try:
                recs = tag_chunk(model, device, mp3, row)
            except Exception as e:  # noqa: BLE001 — keep batch alive, report chunk
                print(f"chunk {row['chunk_idx']}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if args.do_print:
                    print(f"{rec['agreement']:5.2f}  {rec['word']:15s} g2p={rec['g2p']:28s} heard={rec['heard']}")
            print(f"chunk {row['chunk_idx']}: {len(recs)} words", file=sys.stderr)
    print(f"wrote {n} word tags -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
