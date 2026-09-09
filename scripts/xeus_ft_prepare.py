#!/usr/bin/env python3
"""Audio side of the PhoneticXeus fine-tune: cut out the certain words.

Runs on the GPU pod (no engine needed; reads chunk_targets.jsonl from
xeus_ft_text.py). For each corpus chunk:

  1. pretrained PhoneticXeus frame log-probs (its own 428-symbol space)
  2. CTC forced alignment against the chunk's full phone string, so every word
     gets a start/end frame — non-certain words only ever serve this step
  3. maximal runs of consecutive CERTAIN words become training segments; the
     audio is cut at the run's edges, padded up to half the gap to the
     neighbouring word so no foreign phone bleeds in
  4. a certain word with several gold variants gets the variant the pretrained
     model finds most likely for that clip (CTC score over the segment's
     frames), so the label matches what was actually said (də vs di)
  5. the pretrained model's greedy reading of the same frames, folded to the
     inventory, is stored as the baseline this fine-tune is measured against

Segments are assigned to train / val_words / val_eps by split.json: any
segment from a held-out episode is val_eps; any other segment containing a
held-out word type is val_words; training never sees either. A per-type quota
keeps frequent words from drowning the rare ones.

Chunks are visited rarest-word-first so the quota fills for the rare types
early, and the pass stops once every type is saturated (or --limit is hit).

Usage:
  python scripts/xeus_ft_prepare.py --out data/xeus_ft [--limit 200] [--batch 6]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import (  # noqa: E402
    load_pretrained, read_jsonl, to_xeus_syms, xeus_vocab, write_jsonl,
)

SR = 16000


# ----------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------

def load_audio(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr == SR:
            return wav
    except Exception:  # noqa: BLE001 - fall through to ffmpeg
        pass
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class ChunkAudio:
    """Decode chunk MP3s in worker threads ahead of the GPU."""

    def __init__(self, rows: list[dict], root: Path, workers: int = 6):
        from concurrent.futures import ThreadPoolExecutor
        self.rows = rows
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def __iter__(self):
        pending = collections.deque()
        it = iter(self.rows)
        ahead = 24

        def submit():
            row = next(it, None)
            if row is None:
                return False
            pending.append((row, self.pool.submit(load_audio, self.root / row["file"])))
            return True

        for _ in range(ahead):
            if not submit():
                break
        while pending:
            row, fut = pending.popleft()
            submit()
            try:
                wav = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  audio failed {row['file']}: {exc}", file=sys.stderr)
                continue
            yield row, wav


# ----------------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------------

def chunk_targets(row: dict, to_ids):
    """Target ids for the whole chunk (in the aligner's space) plus phone -> word ownership."""
    ids: list[int] = []
    owner: list[int] = []
    for wi, w in enumerate(row["words"]):
        for t in to_ids(w["ph"]):
            ids.append(t)
            owner.append(wi)
    return ids, owner


def word_spans(labels, scores, owner: list[int], blank: int):
    """Per-word (start_frame, end_frame_exclusive, mean_prob) from a forced alignment."""
    from torchaudio.functional import merge_tokens
    spans = merge_tokens(labels, scores, blank=blank)
    if len(spans) != len(owner):
        return None
    out: dict[int, list] = {}
    for k, sp in enumerate(spans):
        wi = owner[k]
        rec = out.setdefault(wi, [sp.start, sp.end, [], 0])
        rec[0] = min(rec[0], sp.start)
        rec[1] = max(rec[1], sp.end)
        rec[2].append(float(sp.score))
    return {wi: (s, e, sum(p) / len(p)) for wi, (s, e, p, _) in out.items()}


def ctc_score(lp_seg, ids: list[int], blank: int) -> float:
    """Negative log-likelihood of ``ids`` over the segment's log-probs (T, C)."""
    import torch
    import torch.nn.functional as F
    if not ids or lp_seg.shape[0] < len(ids):
        return float("inf")
    tgt = torch.tensor([ids], device=lp_seg.device)
    loss = F.ctc_loss(
        lp_seg.unsqueeze(1), tgt,
        torch.tensor([lp_seg.shape[0]], device=lp_seg.device),
        torch.tensor([len(ids)], device=lp_seg.device),
        blank=blank, reduction="sum", zero_infinity=True,
    )
    return float(loss)


def choose_variants(lp_seg, words: list[dict], to_ids, blank: int) -> list[int]:
    """Pick, left to right, the gold variant the aligner prefers for this clip."""
    choice = [0] * len(words)

    def ids_for(ch):
        ids = []
        for w, c in zip(words, ch):
            ids.extend(to_ids(w["variants"][c]))
        return ids

    for i, w in enumerate(words):
        if len(w["variants"]) < 2:
            continue
        best, best_c = None, 0
        for c in range(len(w["variants"])):
            trial = list(choice)
            trial[i] = c
            s = ctc_score(lp_seg, ids_for(trial), blank)
            if best is None or s < best:
                best, best_c = s, c
        choice[i] = best_c
    return choice


def greedy_fold(lp_seg, vocab: list[str], blank: int, fold) -> list[str]:
    ids = lp_seg.argmax(-1).tolist()
    syms: list[str] = []
    prev = None
    for t in ids:
        if t != blank and t != prev:
            syms.append(vocab[t])
        prev = t
    out: list[str] = []
    for s in syms:
        if s.startswith("<") and s.endswith(">"):
            continue
        out.extend(fold(s))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--out", default=None, help="defaults to --data")
    ap.add_argument("--root", default=str(REPO), help="where data/chunks lives")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="chunks to process (0 = until saturated)")
    ap.add_argument("--quota", type=int, default=None, help="override split.json quota")
    ap.add_argument("--max-val", type=int, default=4000, help="cap per validation split")
    ap.add_argument("--max-seg-s", type=float, default=8.0)
    ap.add_argument("--min-seg-s", type=float, default=0.2)
    ap.add_argument("--min-align", type=float, default=0.25,
                    help="drop segments whose aligned frames average below this posterior "
                         "(absolute; only meaningful for the pretrained aligner)")
    ap.add_argument("--drop-bottom", type=float, default=None,
                    help="instead of --min-align, drop this fraction of each split's segments with "
                         "the lowest alignment score. A fine-tuned CTC aligner is peaky (its "
                         "per-frame label posteriors are low everywhere), so only a relative "
                         "cut is comparable across models. Implied 0.10 with --align-ckpt.")
    ap.add_argument("--pad-frames", type=int, default=3, help="max context at the start (20 ms frames)")
    ap.add_argument("--pad-end-frames", type=int, default=None,
                    help="max context at the end; may run up to the next word's onset "
                         "(default: same as --pad-frames, capped at half the gap)")
    ap.add_argument("--align-ckpt", default=None,
                    help="fine-tuned checkpoint dir: align and choose variants with it, in the "
                         "Yiddish phone space. The pretrained model still produces the baseline.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--sample", action="store_true",
                    help="random chunk order instead of rarest-first (smoke tests)")
    ap.add_argument("--targets", default=None, help="targets JSONL (default <data>/chunk_targets.jsonl)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="do not load the pretrained model for the baseline column (halves GPU memory; "
                         "needed for 60 s messages with a fine-tuned aligner)")
    ap.add_argument("--split", default=None, help="split JSON (default <data>/split.json)")
    args = ap.parse_args()

    import torch
    import torchaudio.functional as taf
    from xeus_map import fold_phone_string

    if args.align_ckpt and args.drop_bottom is None:
        args.drop_bottom = 0.10
    if args.drop_bottom is not None:
        args.min_align = 0.0
    data = Path(args.data)
    out = Path(args.out or args.data)
    seg_dir = out / "seg"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(args.split or data / "split.json").read_text(encoding="utf-8"))
    quota = args.quota or split["quota"]
    val_eps = set(split["val_episodes"])
    val_types = set(split["val_types"])
    totals = split["type_counts"]

    rows = list(read_jsonl(args.targets or data / "chunk_targets.jsonl"))
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # Rarest-first ordering: a chunk's priority is the corpus count of the
    # rarest certain word in it. Held-out episodes go first so val_eps fills
    # regardless of where the pass stops.
    def priority(row):
        cs = [totals.get(w["key"], 0) for w in row["words"] if w.get("certain")]
        return (0 if row["episode"] in val_eps else 1, min(cs) if cs else 10**9)
    if not args.sample:
        rows.sort(key=priority)
    if args.limit:
        rows = rows[: args.limit]

    _, inner = load_pretrained(device if not (args.no_baseline and args.align_ckpt) else "cpu")
    vocab, x2i = xeus_vocab(inner)
    hop = inner.points_by_frames()

    if args.align_ckpt:
        from xeus_ft_common import YI_BLANK, yi_ids
        from xeus_yi_decode import load_finetuned
        ft_inner, ft_head = load_finetuned(Path(args.align_ckpt), device)
        blank = YI_BLANK
        to_ids = yi_ids
        space = f"yiddish ({args.align_ckpt})"
    else:
        ft_inner = ft_head = None
        blank = inner.blank_id
        to_ids = lambda phones: [x2i[s] for s in to_xeus_syms(phones) if s in x2i]  # noqa: E731
        space = "xeus (pretrained)"
    print(f"device {device}  chunks {len(rows):,}  quota {quota}  hop {hop}  aligner: {space}", flush=True)

    train_counts: collections.Counter = collections.Counter()
    n_split: collections.Counter = collections.Counter()
    stats = collections.Counter()
    segments: list[dict] = []
    for d in ("train", "val_words", "val_eps"):
        (seg_dir / d).mkdir(parents=True, exist_ok=True)

    def saturated() -> bool:
        for key, total in totals.items():
            if key in val_types:
                continue
            if train_counts[key] < min(quota, total):
                return False
        return True

    t0 = time.time()
    batch_rows: list[tuple[dict, np.ndarray]] = []

    def flush():
        nonlocal batch_rows
        if not batch_rows:
            return
        lens = torch.tensor([len(w) for _, w in batch_rows])
        speech = torch.zeros(len(batch_rows), int(lens.max()))
        for i, (_, w) in enumerate(batch_rows):
            speech[i, : len(w)] = torch.from_numpy(w)
        speech, lens = speech.to(device), lens.to(device)
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                if ft_inner is not None and args.no_baseline:
                    from xeus_ft_common import yi_logits
                    ft_logits, ft_flens = yi_logits(ft_inner, ft_head, speech, lens)
                    lp_all = torch.log_softmax(ft_logits.float(), -1)
                    lp_base_all, flens = lp_all, ft_flens
                else:
                    logits, flens = inner.ctc_logits(speech, lens)
                    lp_base_all = torch.log_softmax(logits.float(), -1)
                    if ft_inner is not None:
                        from xeus_ft_common import yi_logits
                        ft_logits, ft_flens = yi_logits(ft_inner, ft_head, speech, lens)
                        lp_all = torch.log_softmax(ft_logits.float(), -1)
                    else:
                        lp_all, ft_flens = lp_base_all, flens
        except torch.OutOfMemoryError:
            # one 60 s message can exceed the card; skip it rather than lose the pass
            stats["skip_oom"] += len(batch_rows)
            batch_rows = []
            torch.cuda.empty_cache()
            return
        for i, (row, wav) in enumerate(batch_rows):
            n = min(int(flens[i]), int(ft_flens[i]))
            process(row, wav, lp_all[i, :n], lp_base_all[i, :n])
        batch_rows = []
        # With two 575M models resident, fragmentation alone can OOM a 24 GB
        # card by the second batch; give the allocator its blocks back.
        del lp_base_all, lp_all
        if device == "cuda":
            torch.cuda.empty_cache()

    def process(row: dict, wav: np.ndarray, lp, lp_base) -> None:
        stats["chunks"] += 1
        ids, owner = chunk_targets(row, to_ids)
        T = lp.shape[0]
        if not ids or len(ids) > T:
            stats["skip_targets"] += 1
            return
        try:
            labels, scores = taf.forced_align(
                lp.unsqueeze(0), torch.tensor([ids], device=lp.device),
                torch.tensor([T], device=lp.device), torch.tensor([len(ids)], device=lp.device),
                blank=blank,
            )
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not stop the pass
            stats["skip_align"] += 1
            if stats["skip_align"] <= 3:
                print(f"  align failed {row['file']}: {exc}", file=sys.stderr)
            return
        spans = word_spans(labels[0].cpu(), scores[0].exp().cpu(), owner, blank)
        if spans is None:
            stats["skip_spans"] += 1
            return
        words = row["words"]
        is_val_ep = row["episode"] in val_eps

        # Maximal runs of certain words that actually got a span.
        runs: list[list[int]] = []
        cur: list[int] = []
        for wi, w in enumerate(words):
            if w.get("certain") and wi in spans:
                cur.append(wi)
            else:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)

        max_frames = int(args.max_seg_s * SR / hop)
        for run in runs:
            # Split long runs at word boundaries.
            pieces: list[list[int]] = []
            piece: list[int] = []
            for wi in run:
                if piece and spans[wi][1] - spans[piece[0]][0] > max_frames:
                    pieces.append(piece)
                    piece = []
                piece.append(wi)
            if piece:
                pieces.append(piece)
            for piece in pieces:
                first, last = piece[0], piece[-1]
                s, e = spans[first][0], spans[last][1]
                # Context: up to pad-frames, never past halfway to the neighbour.
                prev_end = max((spans[w][1] for w in spans if w < first), default=0)
                next_start = min((spans[w][0] for w in spans if w > last), default=T)
                s = max(s - min(args.pad_frames, (s - prev_end) // 2), 0)
                if args.pad_end_frames is None:
                    e = min(e + min(args.pad_frames, (next_start - e) // 2), T)
                else:
                    # A word-final schwa the aligner handed to the next word sits
                    # in this gap; take it, but stop at the neighbour's onset.
                    e = min(e + min(args.pad_end_frames, max(next_start - e, 0)), T)
                dur = (e - s) * hop / SR
                if dur < args.min_seg_s:
                    stats["skip_short"] += 1
                    continue
                align = sum(spans[w][2] for w in piece) / len(piece)
                if align < args.min_align:
                    stats["skip_align_score"] += 1
                    continue
                keys = [words[w]["key"] for w in piece]
                if is_val_ep:
                    which = "val_eps"
                elif any(k in val_types for k in keys):
                    which = "val_words"
                else:
                    which = "train"
                if which == "train":
                    if not any(train_counts[k] < quota for k in keys):
                        stats["skip_quota"] += 1
                        continue
                elif n_split[which] >= args.max_val:
                    stats["skip_valcap"] += 1
                    continue

                lp_seg = lp[s:e]
                seg_words = [words[w] for w in piece]
                choice = choose_variants(lp_seg, seg_words, to_ids, blank)
                target: list[str] = []
                for w, c in zip(seg_words, choice):
                    target.extend(w["variants"][c])
                baseline = [] if args.no_baseline else greedy_fold(lp_base[s:e], vocab, inner.blank_id, fold_phone_string)

                seg_id = f"{row['episode']}-{row['chunk_idx']:05d}-{stats['chunks']:06d}-{len(segments):07d}"
                a, b = s * hop, min(e * hop, len(wav))
                np.save(seg_dir / which / f"{seg_id}.npy", (wav[a:b] * 32767).astype(np.int16))
                segments.append({
                    "id": seg_id, "split": which,
                    "episode": row["episode"], "chunk_idx": row["chunk_idx"],
                    "start_s": round(row["start_s"] + a / SR, 3),
                    "end_s": round(row["start_s"] + b / SR, 3),
                    "dur_s": round((b - a) / SR, 3),
                    "words": [{"w": w["w"], "key": w["key"], "variant": c}
                              for w, c in zip(seg_words, choice)],
                    "target": target,
                    "baseline": baseline,
                    "align_score": round(align, 4),
                })
                n_split[which] += 1
                if which == "train":
                    for k in keys:
                        train_counts[k] += 1
                stats["seg_seconds"] += (b - a) / SR

    audio = ChunkAudio(rows, Path(args.root), workers=args.workers)
    for n, (row, wav) in enumerate(audio, 1):
        batch_rows.append((row, wav))
        if len(batch_rows) >= args.batch:
            flush()
        if n % 200 == 0:
            el = time.time() - t0
            print(f"  [{n:,}/{len(rows):,}] {el / 60:.1f} min  segs train={n_split['train']:,} "
                  f"val_words={n_split['val_words']:,} val_eps={n_split['val_eps']:,}  "
                  f"audio {stats['seg_seconds'] / 3600:.2f} h  {n / el:.1f} chunk/s", flush=True)
            if not args.limit and n % 1000 == 0 and saturated():
                print("  every type saturated — stopping early", flush=True)
                break
    flush()

    dropped_rel = 0
    if args.drop_bottom:
        keep: list[dict] = []
        for which in ("train", "val_words", "val_eps"):
            group = sorted((g for g in segments if g["split"] == which), key=lambda g: g["align_score"])
            cut = int(len(group) * args.drop_bottom)
            for g in group[:cut]:
                (seg_dir / which / f"{g['id']}.npy").unlink(missing_ok=True)
                n_split[which] -= 1
                dropped_rel += 1
            keep.extend(group[cut:])
        segments = keep
        stats["skip_align_relative"] = dropped_rel
    write_jsonl(out / "segments.jsonl", segments)
    unsat = sorted(((k, train_counts[k], totals.get(k, 0)) for k in totals if k not in val_types
                    and train_counts[k] < min(quota, totals.get(k, 0))), key=lambda t: t[1])
    summary = {
        "chunks_seen": stats["chunks"],
        "segments": dict(n_split),
        "segment_hours": round(stats["seg_seconds"] / 3600, 2),
        "skips": {k: v for k, v in stats.items() if k.startswith("skip")},
        "types_in_train": sum(1 for k in train_counts if train_counts[k] > 0),
        "types_under_quota": len(unsat),
        "worst_covered": unsat[:20],
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "hop": hop,
    }
    (out / "prepare_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
