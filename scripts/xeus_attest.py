#!/usr/bin/env python3
"""Audio attestation: let the ear decide the open slots of every rule-path token.

Runs on the GPU pod with the fine-tuned Yiddish recognizer. For each corpus
chunk:

  1. forced-align the chunk's full phone string (every word's engine reading)
     with the ear, so every token has a frame span;
  2. for each token marked attest (rule LOW/MED, not a certain word), take
     the spelling's legal readings — the graph of docs/xeus_finetune.md §12,
     branched from the engine's reading on the slots the orthography leaves
     open — and score every candidate by CTC likelihood over that token's
     frames, in one batched call;
  3. record the best reading, the margin to the runner-up in nats, and where
     the engine's own reading ranked.

Nothing here is a training label yet. prepare_retrain_dataset_v8.py turns a
decision into supervision only when the margin clears a bar and a corpus
pointing of the type reads back to the decided reading.

Output: data/xeus_ft/attest.jsonl, one record per attested occurrence:
  {episode, chunk_idx, wi, w, key, engine, chosen, margin, engine_gap, n_cand,
   dur_s, align}

Usage:
  python scripts/xeus_attest.py --data data/xeus_ft --root . --ckpt data/xeus_ft/ckpt/best [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import YI_BLANK, read_jsonl, write_jsonl, yi_ids, yi_logits  # noqa: E402
from xeus_ft_prepare import ChunkAudio, word_spans  # noqa: E402
from xeus_lattice import graph_candidates  # noqa: E402

SR = 16000


def score_batch(lp_seg, cands: list[list[str]]) -> list[float]:
    """CTC NLL of every candidate over the same frames, one call."""
    import torch
    import torch.nn.functional as F
    T = lp_seg.shape[0]
    ids = [yi_ids(c) for c in cands]
    L = max(len(x) for x in ids)
    tgt = torch.zeros(len(ids), L, dtype=torch.long, device=lp_seg.device)
    for i, x in enumerate(ids):
        tgt[i, : len(x)] = torch.tensor(x, device=lp_seg.device)
    tlens = torch.tensor([len(x) for x in ids], device=lp_seg.device)
    lp = lp_seg.unsqueeze(1).expand(T, len(ids), lp_seg.shape[1])
    loss = F.ctc_loss(lp, tgt, torch.full((len(ids),), T, device=lp_seg.device), tlens,
                      blank=YI_BLANK, reduction="none", zero_infinity=True)
    out = loss.tolist()
    return [float("inf") if (len(x) > T or v == 0.0 and len(x) > T) else v for x, v in zip(ids, out)]


def ranked_candidates(reading: list[str], cap: int) -> list[list[str]]:
    """Graph candidates, engine reading first, then by number of changed slots."""
    cands = graph_candidates(reading, max_candidates=512)
    cands.sort(key=lambda c: (sum(a != b for a, b in zip(c, reading)), c))
    return cands[:cap]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--ckpt", default=str(REPO / "data/xeus_ft/ckpt/best"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=16)
    ap.add_argument("--pad-frames", type=int, default=3)
    ap.add_argument("--pad-end-frames", type=int, default=6)
    ap.add_argument("--min-seg-s", type=float, default=0.12)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    import torch
    import torchaudio.functional as taf
    from xeus_yi_decode import load_finetuned

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    rows = list(read_jsonl(data / "attest_targets.jsonl"))
    if args.sample:
        random.Random(0).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    inner, head = load_finetuned(Path(args.ckpt), device)
    hop = inner.points_by_frames()
    print(f"device {device}  chunks {len(rows):,}  ear {args.ckpt}", flush=True)

    out_rows: list[dict] = []
    stats = collections.Counter()
    t0 = time.time()
    batch_rows: list[tuple[dict, np.ndarray]] = []

    def process(row: dict, wav: np.ndarray, lp) -> None:
        stats["chunks"] += 1
        ids: list[int] = []
        owner: list[int] = []
        for wi, w in enumerate(row["words"]):
            for t in yi_ids(w["ph"]):
                ids.append(t)
                owner.append(wi)
        T = lp.shape[0]
        if not ids or len(ids) > T:
            stats["skip_targets"] += 1
            return
        try:
            labels, scores = taf.forced_align(
                lp.unsqueeze(0), torch.tensor([ids], device=lp.device),
                torch.tensor([T], device=lp.device), torch.tensor([len(ids)], device=lp.device),
                blank=YI_BLANK)
        except Exception:  # noqa: BLE001
            stats["skip_align"] += 1
            return
        spans = word_spans(labels[0].cpu(), scores[0].exp().cpu(), owner, YI_BLANK)
        if spans is None:
            stats["skip_spans"] += 1
            return
        words = row["words"]
        for wi, w in enumerate(words):
            if not w.get("attest") or wi not in spans:
                continue
            s, e, align = spans[wi]
            prev_end = max((spans[x][1] for x in spans if x < wi), default=0)
            next_start = min((spans[x][0] for x in spans if x > wi), default=T)
            s = max(s - min(args.pad_frames, (s - prev_end) // 2), 0)
            e = min(e + min(args.pad_end_frames, max(next_start - e, 0)), T)
            dur = (e - s) * hop / SR
            if dur < args.min_seg_s:
                stats["skip_short"] += 1
                continue
            cands = ranked_candidates(w["ph"], args.max_candidates)
            if len(cands) < 2:
                stats["skip_no_choice"] += 1
                continue
            nll = score_batch(lp[s:e], cands)
            order = sorted(range(len(cands)), key=lambda k: nll[k])
            best, second = order[0], order[1]
            if nll[best] == float("inf"):
                stats["skip_unscorable"] += 1
                continue
            stats["attested"] += 1
            out_rows.append({
                "episode": row["episode"], "chunk_idx": row["chunk_idx"], "wi": wi,
                "w": w["w"], "key": w["key"],
                "engine": w["ph"], "chosen": cands[best],
                "margin": round(nll[second] - nll[best], 3),
                "engine_gap": round(nll[0] - nll[best], 3),
                "n_cand": len(cands), "dur_s": round(dur, 3), "align": round(align, 4),
            })

    def flush():
        nonlocal batch_rows
        if not batch_rows:
            return
        lens = torch.tensor([len(w) for _, w in batch_rows])
        speech = torch.zeros(len(batch_rows), int(lens.max()))
        for i, (_, w) in enumerate(batch_rows):
            speech[i, : len(w)] = torch.from_numpy(w)
        speech, lens = speech.to(device), lens.to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, flens = yi_logits(inner, head, speech, lens)
        lp_all = torch.log_softmax(logits.float(), -1)
        for i, (row, wav) in enumerate(batch_rows):
            process(row, wav, lp_all[i, : int(flens[i])])
        batch_rows = []
        del logits, lp_all
        if device == "cuda":
            torch.cuda.empty_cache()

    for n, (row, wav) in enumerate(ChunkAudio(rows, Path(args.root), workers=args.workers), 1):
        batch_rows.append((row, wav))
        if len(batch_rows) >= args.batch:
            flush()
        if n % 500 == 0:
            el = time.time() - t0
            print(f"  [{n:,}/{len(rows):,}] {el / 60:.1f} min  attested {stats['attested']:,}  {n / el:.1f} chunk/s", flush=True)
    flush()

    out = Path(args.out or data / "attest.jsonl")
    write_jsonl(out, out_rows)
    margins = sorted(r["margin"] for r in out_rows)
    changed = sum(r["chosen"] != r["engine"] for r in out_rows)
    summary = {
        "chunks": stats["chunks"], "attested": len(out_rows),
        "engine_reading_kept": len(out_rows) - changed, "engine_reading_changed": changed,
        "margin_median": margins[len(margins) // 2] if margins else None,
        "margin_ge_2": sum(m >= 2 for m in margins), "margin_ge_4": sum(m >= 4 for m in margins),
        "skips": {k: v for k, v in stats.items() if k.startswith("skip")},
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (data / "attest_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
