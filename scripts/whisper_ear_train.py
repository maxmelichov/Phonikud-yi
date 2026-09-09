#!/usr/bin/env python3
"""A Whisper-encoder ear: yi-whisper's encoder, truncated to N layers, + CTC.

The XEUS ear (xeus_ft_train.py) is 575M parameters. This trains the same
34-phone CTC task on the encoder of ``ivrit-ai/yi-whisper-large-v3`` — a
Whisper large-v3 encoder already fine-tuned on ~97 h of Yiddish — cut to the
first --layers blocks (4 → 22M, 8 → 86M, 32 → 635M), on the same clips, with
the same metrics, so the two can be compared directly.

Whisper's encoder is written for 30-second windows; HF's forward insists on
exactly 3000 mel frames. Padding every 1-second clip to 30 s would waste 97%
of the compute, so the forward here re-implements the encoder's few lines —
two convolutions, the positional table sliced to the clip's length, the
blocks, the final norm — and pads only to the longest clip in the batch.
Log-mel features are Whisper's own (128 bins, 10 ms hop; the convolutions
halve that to 20 ms frames).

Usage:
  python scripts/whisper_ear_train.py --data data/xeus_ft --layers 8 --out data/xeus_ft/ckpt_whisper8
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

from xeus_ft_common import PerAccumulator, YI_BLANK, YI_VOCAB, greedy_decode, read_jsonl, yi_ids  # noqa: E402
from xeus_ft_train import Segments, augment, baseline  # noqa: E402

SR = 16000
MEL_HOP = 160          # 10 ms
FRAME = MEL_HOP * 2    # the encoder's convolutions halve the mel rate: 20 ms


class WhisperEar:
    """Encoder truncated to N layers, mel front end, CTC head."""

    def __init__(self, name: str, layers: int, device: str):
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        full = WhisperModel.from_pretrained(name)
        self.enc = full.encoder
        del full
        self.enc.layers = self.enc.layers[:layers]
        self.enc.config.encoder_layers = layers
        d = self.enc.config.d_model
        self.head = torch.nn.Linear(d, len(YI_VOCAB))
        self.enc.to(device)
        self.head.to(device)
        self.device = device

    def parameters(self):
        yield from self.enc.parameters()
        yield from self.head.parameters()

    def train(self):
        self.enc.train(); self.head.train()

    def eval(self):
        self.enc.eval(); self.head.eval()

    def features(self, speech, lens):
        """Log-mel for each clip, padded to the longest in the batch: (B, 128, T_mel)."""
        import torch
        wavs = [speech[i, : int(lens[i])].detach().cpu().numpy() for i in range(speech.shape[0])]
        max_len = max(len(w) for w in wavs)
        # Whisper's extractor pads/truncates to 30 s; ask for the batch's own length instead.
        feats = self.fe(wavs, sampling_rate=SR, return_tensors="pt", padding="longest",
                        max_length=max_len, truncation=True, return_attention_mask=False)
        # The extractor still pads to its 30 s window; keep only the batch's own frames
        # (never more than the positional table's 1500 encoder frames).
        t_mel = min(math.ceil(max_len / MEL_HOP) + 1, 3000)
        t_mel += t_mel % 2                      # the conv stack halves it; keep it even
        mel = feats.input_features[:, :, :t_mel].to(self.device)
        n_frames = torch.tensor([min(math.ceil(len(w) / FRAME), t_mel // 2) for w in wavs], device=self.device)
        return mel, n_frames

    def logits(self, speech, lens):
        import torch
        import torch.nn.functional as F
        mel, n_frames = self.features(speech, lens)
        enc = self.enc
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x))          # (B, d, T/2)
        x = x.permute(0, 2, 1)             # (B, T, d)
        T = x.shape[1]
        x = x + enc.embed_positions.weight[:T]
        for layer in enc.layers:
            x = layer(x, attention_mask=None, layer_head_mask=None)[0]
        x = enc.layer_norm(x)
        return self.head(x), torch.clamp(n_frames, max=T)


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


def evaluate(ear: WhisperEar, ds: Segments, seconds: float, use_amp: bool) -> PerAccumulator:
    import torch
    acc = PerAccumulator()
    ear.eval()
    with torch.no_grad():
        for idx in ds.batches(seconds, shuffle=False, rng=random.Random(0)):
            speech, lens, _, _ = collate(ds, idx, ear.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, flens = ear.logits(speech, lens)
            for i, hyp in zip(idx, greedy_decode(logits.float(), flens)):
                acc.add(ds.rows[i]["target"], hyp)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ivrit-ai/yi-whisper-large-v3")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--batch-seconds", type=float, default=64.0)
    ap.add_argument("--eval-seconds", type=float, default=128.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda" and not args.no_amp
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(data / "segments.jsonl"))
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val_words", "val_eps")}
    if args.limit:
        rng.shuffle(by["train"]); by["train"] = by["train"][: args.limit]
    if args.val_limit:
        for s in ("val_words", "val_eps"):
            by[s] = by[s][: args.val_limit]
    ds = {s: Segments(r, data / "seg") for s, r in by.items()}
    print(f"segments {({s: len(v) for s, v in by.items()})}", flush=True)

    ear = WhisperEar(args.name, args.layers, device)
    n_params = sum(p.numel() for p in ear.parameters())
    print(f"whisper ear: {args.layers} layers, {n_params / 1e6:.0f}M params", flush=True)
    opt = torch.optim.AdamW([{"params": list(ear.enc.parameters()), "lr": args.lr_enc},
                             {"params": list(ear.head.parameters()), "lr": args.lr_head}],
                            weight_decay=0.01, betas=(0.9, 0.98))
    per_epoch = len(ds["train"].batches(args.batch_seconds, shuffle=False, rng=rng))
    total = max(1, per_epoch * args.epochs)

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    base = {s: baseline(ds[s]).summary() for s in ("val_words", "val_eps")}
    print("baseline (pretrained XEUS + fold)  " + "  ".join(f"{s}: PER {v['per']:.3f}" for s, v in base.items()), flush=True)
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    best = 9.0
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ear.train()
        run, k = 0.0, 0
        for bi, idx in enumerate(ds["train"].batches(args.batch_seconds, shuffle=True, rng=rng), 1):
            speech, lens, flat, tlens = collate(ds["train"], idx, device)
            if args.augment:
                speech, lens = augment(speech, lens, rng)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, flens = ear.logits(speech, lens)
            lp = torch.log_softmax(logits.float(), -1).transpose(0, 1)
            loss = F.ctc_loss(lp, flat, flens, tlens, blank=YI_BLANK, reduction="mean", zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ear.parameters()), 5.0)
            opt.step(); sched.step(); step += 1
            run += loss.item(); k += 1
            if bi % 100 == 0:
                el = time.time() - t0
                print(f"  ep{epoch} [{bi}/{per_epoch}] loss {run / k:.3f} lr {sched.get_last_lr()[0]:.2e} {el / 60:.1f} min {step / el:.2f} step/s", flush=True)
                run, k = 0.0, 0
            if args.time_budget_min and (time.time() - t0) / 60 > args.time_budget_min:
                break
        val = {s: evaluate(ear, ds[s], args.eval_seconds, use_amp).summary() for s in ("val_words", "val_eps")}
        hard = val["val_words"]["hard_phones"]
        print(f"epoch {epoch}   " + "  ".join(f"{s}: PER {v['per']:.3f} exact {v['exact_match']:.3f}" for s, v in val.items())
              + f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)
        print("   hard phones (val_words recall): " + "  ".join(f"{p} {h['recall']:.2f}" for p, h in hard.items() if h["n"]), flush=True)
        log.write(json.dumps({"event": "epoch", "epoch": epoch, "layers": args.layers, "val": val,
                              "elapsed_min": (time.time() - t0) / 60}, ensure_ascii=False) + "\n"); log.flush()
        if val["val_words"]["per"] < best:
            best = val["val_words"]["per"]
            torch.save({"layers": args.layers, "name": args.name, "encoder": ear.enc.state_dict(),
                        "head": ear.head.state_dict(), "val": val, "epoch": epoch}, out / "best.pt")
            print(f"   new best val_words PER {best:.3f}", flush=True)
    log.write(json.dumps({"event": "done", "layers": args.layers, "best_val_words_per": best,
                          "params_m": round(n_params / 1e6), "minutes": (time.time() - t0) / 60}) + "\n")
    log.close()
    print(f"done. whisper ear ({args.layers} layers): best val_words PER {best:.3f}")


if __name__ == "__main__":
    main()
