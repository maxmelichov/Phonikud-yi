#!/usr/bin/env python3
"""Fine-tune PhoneticXeus into a Yiddish phone recognizer (GPU pod).

Model surgery, and why it is shaped this way:

  * The pretrained 428-way CTC head stays where it is, frozen. The encoder was
    trained with inter-CTC conditioning — intermediate posteriors from that
    head are fed back into later blocks — so removing or resizing it would
    change what the encoder sees at every layer.
  * A new 35-way Yiddish head sits beside it on the final encoder output and
    carries the training loss. It is warm-started from the pretrained rows
    (xeus_ft_common.build_yi_head), so step 0 is the pretrained model folded
    onto the inventory, and training only has to move it.
  * The CNN frontend is frozen (as in the original recipe) and so are the
    lowest --freeze-blocks encoder blocks: with a few thousand seconds of
    audio the low-level acoustics are not what needs to change.

Only segments from segments.jsonl with split == train are ever used for a
gradient. The two validation splits are scored every epoch with greedy PER;
the checkpoint with the best val_words PER (unseen word types) is kept.

Usage:
  python scripts/xeus_ft_train.py --data data/xeus_ft --epochs 4 [--limit 500]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import (  # noqa: E402
    PerAccumulator, YI_BLANK, YI_VOCAB, build_yi_head, encoder_blocks, greedy_decode,
    load_pretrained, read_jsonl, yi_ids, yi_logits,
)

SR = 16000


class Segments:
    def __init__(self, rows: list[dict], seg_dir: Path):
        self.rows = rows
        self.seg_dir = seg_dir

    def __len__(self) -> int:
        return len(self.rows)

    def audio(self, i: int) -> np.ndarray:
        r = self.rows[i]
        return np.load(self.seg_dir / r["split"] / f"{r['id']}.npy").astype(np.float32) / 32767.0

    def batches(self, seconds: float, shuffle: bool, rng: random.Random) -> list[list[int]]:
        """Length-bucketed batches capped by total padded seconds."""
        order = sorted(range(len(self.rows)), key=lambda i: self.rows[i]["dur_s"])
        out: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0.0
        for i in order:
            d = self.rows[i]["dur_s"]
            if cur and max(cur_max, d) * (len(cur) + 1) > seconds:
                out.append(cur)
                cur, cur_max = [], 0.0
            cur.append(i)
            cur_max = max(cur_max, d)
        if cur:
            out.append(cur)
        if shuffle:
            rng.shuffle(out)
        return out


class ChunkSegments(Segments):
    """Whole 30 s corpus chunks, target = the engine's reading of every word.

    The noisy, full-coverage label set for a pretraining pass: every word has
    a reading (the rule engine's), most of them right, none of them vouched
    for. The fine-tune on the certain clips comes after."""

    def __init__(self, chunk_rows: list[dict], root: Path):
        rows = []
        for r in chunk_rows:
            target = [p for w in r["words"] for p in (w.get("ph") or [])]
            if target:
                rows.append({"id": f"{r['episode']}-{r['chunk_idx']:05d}", "split": "chunk", "target": target,
                             "dur_s": 30.0, "file": r["file"], "words": [], "baseline": []})
        super().__init__(rows, root)
        self.root = root

    def audio(self, i: int) -> np.ndarray:
        from xeus_ft_prepare import load_audio
        return load_audio(self.root / self.rows[i]["file"])


def collate(ds: Segments, idx: list[int], device):
    import torch
    wavs = [ds.audio(i) for i in idx]
    lens = torch.tensor([len(w) for w in wavs])
    speech = torch.zeros(len(wavs), int(lens.max()))
    for k, w in enumerate(wavs):
        speech[k, : len(w)] = torch.from_numpy(w)
    targets = [yi_ids(ds.rows[i]["target"]) for i in idx]
    tlens = torch.tensor([len(t) for t in targets])
    flat = torch.tensor([x for t in targets for x in t], dtype=torch.long)
    return speech.to(device), lens.to(device), flat.to(device), tlens.to(device)


def augment(speech, lens, rng: random.Random):
    """Waveform augmentation for robustness to rooms and microphones the corpus
    does not contain: one speed factor per batch (0.9 / 1.0 / 1.1), then per
    clip a random gain (±6 dB) and, half the time, white noise at 10–40 dB SNR.
    Lengths are updated for the speed change. Runs on the device."""
    import torch
    import torchaudio.functional as taf
    factor = rng.choice((0.9, 1.0, 1.1))
    if factor != 1.0:
        orig = 1000
        new = int(round(orig / factor))
        speech = taf.resample(speech, orig, new)
        lens = torch.clamp((lens.float() * (new / orig)).long(), max=speech.shape[1])
    B = speech.shape[0]
    gain_db = torch.empty(B, 1, device=speech.device).uniform_(-6.0, 6.0)
    speech = speech * (10.0 ** (gain_db / 20.0))
    mask = torch.rand(B, 1, device=speech.device) < 0.5
    snr_db = torch.empty(B, 1, device=speech.device).uniform_(10.0, 40.0)
    rms = speech.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-5)
    noise = torch.randn_like(speech) * rms / (10.0 ** (snr_db / 20.0))
    speech = speech + noise * mask
    return speech.clamp(-1.0, 1.0), lens


def evaluate(inner, head, ds: Segments, device, seconds: float, use_amp: bool) -> PerAccumulator:
    import torch
    acc = PerAccumulator()
    inner.eval()
    head.eval()
    rng = random.Random(0)
    with torch.no_grad():
        for idx in ds.batches(seconds, shuffle=False, rng=rng):
            speech, lens, _, _ = collate(ds, idx, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, flens = yi_logits(inner, head, speech, lens)
            for i, hyp in zip(idx, greedy_decode(logits.float(), flens)):
                acc.add(ds.rows[i]["target"], hyp)
    return acc


def baseline(ds: Segments) -> PerAccumulator:
    acc = PerAccumulator()
    for r in ds.rows:
        acc.add(r["target"], r["baseline"])
    return acc


def save_ckpt(inner, head, path: Path, meta: dict) -> None:
    import torch
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in inner.state_dict().items()},
              str(path / "inner.safetensors"))
    torch.save(head.state_dict(), path / "yi_head.pt")
    (path / "meta.json").write_text(json.dumps(meta | {"vocab": list(YI_VOCAB)}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--out", default=None, help="defaults to <data>/ckpt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--batch-seconds", type=float, default=48.0)
    ap.add_argument("--eval-seconds", type=float, default=96.0)
    ap.add_argument("--freeze-blocks", type=int, default=6)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="training segments (0 = all)")
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="stop after this many minutes")
    ap.add_argument("--augment", action="store_true", help="speed / gain / noise augmentation on training clips")
    ap.add_argument("--chunks", default=None,
                    help="pretraining mode: train on whole corpus chunks (attest_targets.jsonl) with the "
                         "engine's readings as targets, instead of the certain-word clips")
    ap.add_argument("--root", default=str(REPO), help="where data/chunks lives (with --chunks)")
    ap.add_argument("--train-blank-penalty", type=float, default=0.0,
                    help="subtract this from the blank logit inside the training loss only. The model must then "
                         "earn every blank frame against a handicap, which pushes half-believed phones (the "
                         "word-final ə it drops) above blank at plain decode time. Decoding is unchanged.")
    ap.add_argument("--init-ckpt", default=None,
                    help="continue from a fine-tuned checkpoint dir instead of the pretrained weights")
    ap.add_argument("--oversample-schwa", type=int, default=1,
                    help="repeat clips containing a word-final ə this many times per epoch. The model "
                         "drops word-final schwas (docs §11); at decode time a blank penalty only trades "
                         "errors, so the fix has to be in what it trains on.")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda" and not args.no_amp
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data = Path(args.data)
    out = Path(args.out or data / "ckpt")
    out.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(data / "segments.jsonl"))
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val_words", "val_eps")}
    if args.limit:
        rng.shuffle(by["train"])
        by["train"] = by["train"][: args.limit]
    if args.val_limit:
        for s in ("val_words", "val_eps"):
            by[s] = by[s][: args.val_limit]
    if args.oversample_schwa > 1:
        dictionary = json.loads((data / "dictionary.json").read_text(encoding="utf-8"))
        by_key = {v["key"]: v["variants"] for v in dictionary.values()}

        def has_final_schwa(r) -> bool:
            for w in r["words"]:
                vs = by_key.get(w["key"]) or []
                v = vs[w["variant"]] if w["variant"] < len(vs) else (vs[0] if vs else [])
                if v and v[-1] == "ə":
                    return True
            return False
        extra = [r for r in by["train"] if has_final_schwa(r)]
        by["train"] = by["train"] + extra * (args.oversample_schwa - 1)
        print(f"oversampling {len(extra):,} clips with a word-final ə x{args.oversample_schwa}", flush=True)
    ds = {s: Segments(r, data / "seg") for s, r in by.items()}
    if args.chunks:
        chunk_rows = list(read_jsonl(args.chunks))
        if args.limit:
            rng.shuffle(chunk_rows); chunk_rows = chunk_rows[: args.limit]
        ds["train"] = ChunkSegments(chunk_rows, Path(args.root))
        by["train"] = ds["train"].rows
        print(f"pretraining on {len(by['train']):,} whole chunks with engine readings", flush=True)
    hours = {s: round(sum(r["dur_s"] for r in v) / 3600, 2) for s, v in by.items()}
    print(f"segments {({s: len(v) for s, v in by.items()})}  hours {hours}", flush=True)

    if args.init_ckpt:
        from xeus_yi_decode import load_finetuned
        inner, head = load_finetuned(Path(args.init_ckpt), device)
        print(f"warm start from {args.init_ckpt}", flush=True)
    else:
        _, inner = load_pretrained(device)
        head = build_yi_head(inner).to(device)

    # Freeze: frontend, preencoder, original head, lowest encoder blocks.
    for p in inner.parameters():
        p.requires_grad = False
    blocks = encoder_blocks(inner)
    for i, blk in enumerate(blocks):
        if i >= args.freeze_blocks:
            for p in blk.parameters():
                p.requires_grad = True
    # Everything in the encoder that is not a block (final norm, positional
    # conv, conditioning layers) trains too.
    block_ids = {id(p) for blk in blocks for p in blk.parameters()}
    for n, p in inner.encoder.named_parameters():
        if id(p) not in block_ids:
            p.requires_grad = True
    enc_params = [p for p in inner.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in enc_params) + sum(p.numel() for p in head.parameters())
    n_total = sum(p.numel() for p in inner.parameters())
    print(f"trainable {n_train / 1e6:.1f}M of {n_total / 1e6:.1f}M  (blocks {args.freeze_blocks}..{len(blocks) - 1} + head)", flush=True)

    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_enc},
         {"params": list(head.parameters()), "lr": args.lr_head}],
        weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )
    batches_per_epoch = len(ds["train"].batches(args.batch_seconds, shuffle=False, rng=rng))
    total_steps = max(1, batches_per_epoch * args.epochs)

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    # Baseline: the pretrained model + fold map on the very same segments.
    base = {s: baseline(ds[s]).summary() for s in ("val_words", "val_eps")}
    print("baseline  " + "  ".join(f"{s}: PER {v['per']:.3f} exact {v['exact_match']:.3f}" for s, v in base.items()), flush=True)
    # Epoch 0: the warm-started head before any training.
    ep0 = {s: evaluate(inner, head, ds[s], device, args.eval_seconds, use_amp).summary() for s in ("val_words", "val_eps")}
    print("epoch 0   " + "  ".join(f"{s}: PER {v['per']:.3f} exact {v['exact_match']:.3f}" for s, v in ep0.items()), flush=True)

    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    log.write(json.dumps({"event": "baseline", **base}, ensure_ascii=False) + "\n")
    log.write(json.dumps({"event": "epoch", "epoch": 0, "val": ep0}, ensure_ascii=False) + "\n")
    best = ep0["val_words"]["per"]
    save_ckpt(inner, head, out / "best", {"epoch": 0, "val": ep0, "baseline": base, "args": vars(args)})

    step = 0
    t0 = time.time()
    stop = False
    for epoch in range(1, args.epochs + 1):
        inner.train()
        head.train()
        inner.frontend.eval()
        for i, blk in enumerate(blocks):
            if i < args.freeze_blocks:
                blk.eval()
        run_loss, run_n = 0.0, 0
        for bi, idx in enumerate(ds["train"].batches(args.batch_seconds, shuffle=True, rng=rng), 1):
            speech, lens, flat, tlens = collate(ds["train"], idx, device)
            if args.augment:
                speech, lens = augment(speech, lens, rng)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, flens = yi_logits(inner, head, speech, lens)
            logits = logits.float()
            if args.train_blank_penalty:
                logits = logits.clone()
                logits[..., YI_BLANK] -= args.train_blank_penalty
            lp = torch.log_softmax(logits, -1).transpose(0, 1)
            loss = F.ctc_loss(lp, flat, flens, tlens, blank=YI_BLANK, reduction="mean", zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_params + list(head.parameters()), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_n += 1
            if bi % 50 == 0:
                el = time.time() - t0
                print(f"  ep{epoch} [{bi}/{batches_per_epoch}] loss {run_loss / run_n:.3f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {el / 60:.1f} min  {step / el:.2f} step/s", flush=True)
                log.write(json.dumps({"event": "step", "epoch": epoch, "step": step,
                                      "loss": run_loss / run_n, "lr": sched.get_last_lr()[0]}) + "\n")
                run_loss, run_n = 0.0, 0
            if args.time_budget_min and (time.time() - t0) / 60 > args.time_budget_min:
                print("  time budget reached", flush=True)
                stop = True
                break
        val = {s: evaluate(inner, head, ds[s], device, args.eval_seconds, use_amp).summary() for s in ("val_words", "val_eps")}
        print(f"epoch {epoch}   " + "  ".join(f"{s}: PER {v['per']:.3f} exact {v['exact_match']:.3f}" for s, v in val.items())
              + f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)
        hard = val["val_words"]["hard_phones"]
        print("   hard phones (val_words recall): " + "  ".join(f"{p} {h['recall']:.2f}" for p, h in hard.items() if h["n"]), flush=True)
        log.write(json.dumps({"event": "epoch", "epoch": epoch, "val": val, "elapsed_min": (time.time() - t0) / 60},
                             ensure_ascii=False) + "\n")
        log.flush()
        meta = {"epoch": epoch, "val": val, "baseline": base, "args": vars(args)}
        save_ckpt(inner, head, out / "last", meta)
        if val["val_words"]["per"] < best:
            best = val["val_words"]["per"]
            save_ckpt(inner, head, out / "best", meta)
            print(f"   new best val_words PER {best:.3f} -> {out / 'best'}", flush=True)
        if stop:
            break
    log.close()
    print(f"done. best val_words PER {best:.3f} (baseline {base['val_words']['per']:.3f})")


if __name__ == "__main__":
    main()
