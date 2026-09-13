#!/usr/bin/env python3
"""Does the ear DISCRIMINATE the open slots? Paired likelihood test on identical clips.

Free-decode recall of oʊ (§25: the canary) is measured on 10 clips in the
episode split. What the lattice / attestation actually ask the ear is
narrower: given a clip and two readings that differ in one open slot
(oʊ vs ɔj, ə vs ɛ), which one gets the higher CTC likelihood? This scores
every val clip whose target holds exactly one phone of the pair under each
checkpoint, for the true reading and the swapped one, and counts wins.

Usage:
  python scripts/xeus_ft_probe_pairs.py --ckpts data/xeus_ft/ckpt/best data/xeus_ft/ear3/ckpt_att/best \
      [--pairs "oʊ:ɔj,ə:ɛ"] [--limit-per-pair 600] [--out data/xeus_ft/ear3/probe_pairs.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import read_jsonl, yi_ids, yi_logits, YI_BLANK  # noqa: E402
from xeus_ft_train import Segments, collate  # noqa: E402
from xeus_yi_decode import load_finetuned  # noqa: E402


def sign_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def nll(lp, targets: list[list[str]]) -> list[float]:
    import torch
    import torch.nn.functional as F
    T = lp.shape[0]
    ids = [yi_ids(t) for t in targets]
    L = max(len(x) for x in ids)
    tgt = torch.zeros(len(ids), L, dtype=torch.long, device=lp.device)
    for i, x in enumerate(ids):
        tgt[i, : len(x)] = torch.tensor(x, device=lp.device)
    tl = torch.tensor([len(x) for x in ids], device=lp.device)
    lpe = lp.unsqueeze(1).expand(T, len(ids), lp.shape[1])
    return F.ctc_loss(lpe, tgt, torch.full((len(ids),), T, device=lp.device), tl,
                      blank=YI_BLANK, reduction="none", zero_infinity=True).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft/run3"))
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--pairs", default="oʊ:ɔj,ə:ɛ")
    ap.add_argument("--splits", default="val_words,val_eps")
    ap.add_argument("--limit-per-pair", type=int, default=600, help="per (split, phone); oʊ clips are all kept")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO / "data/xeus_ft/ear3/probe_pairs.json"))
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    splits = args.splits.split(",")

    rng = random.Random(0)
    rows = []
    for r in read_jsonl(Path(args.data) / "segments.jsonl"):
        if r["split"] not in splits:
            continue
        for a, b in pairs:
            for x, y in ((a, b), (b, a)):
                if r["target"].count(x) == 1 and y not in r["target"]:
                    rows.append({**r, "_pair": f"{a}:{b}", "_true": x, "_swap": y})
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["_true"]), []).append(r)
    kept = []
    for k, rs in sorted(by.items()):
        rng.shuffle(rs)
        kept += rs[: args.limit_per_pair]
        print(f"  {k[0]:10s} {k[1]:3s} clips {len(rs)} kept {min(len(rs), args.limit_per_pair)}")
    ds = Segments(kept, Path(args.data) / "seg")
    print(f"{len(kept)} clips, device {device}")

    results = {}
    wins = {}
    for ck in args.ckpts:
        inner, head = load_finetuned(Path(ck), device)
        inner.eval(); head.eval()
        w = [None] * len(kept)
        with torch.no_grad():
            for idx in ds.batches(args.seconds, shuffle=False, rng=random.Random(0)):
                speech, lens, _, _ = collate(ds, idx, device)
                logits, flens = yi_logits(inner, head, speech, lens)
                lp = torch.log_softmax(logits.float(), -1).cpu()  # ctc_loss has no MPS kernel
                for j, i in enumerate(idx):
                    r = kept[i]
                    swapped = [r["_swap"] if p == r["_true"] else p for p in r["target"]]
                    t, s = nll(lp[j, : int(flens[j])], [r["target"], swapped])
                    w[i] = s - t  # > 0: the true reading wins, in nats
        wins[ck] = w
        summ = {}
        for r, m in zip(kept, w):
            key = f"{r['split']} {r['_true']} (n={sum(1 for q in kept if q['split']==r['split'] and q['_true']==r['_true'])})"
            summ.setdefault(key, [0, 0])
            summ[key][0 if m > 0 else 1] += 1
        results[ck] = {k: {"true_wins": v[0], "swap_wins": v[1], "acc": round(v[0] / (v[0] + v[1]), 3)} for k, v in summ.items()}
        print(f"\n{ck}")
        for k, v in results[ck].items():
            print(f"  {k:28s} true reading wins {v['true_wins']:4d} / {v['true_wins']+v['swap_wins']:4d}  = {v['acc']:.3f}")
        del inner, head
    if len(args.ckpts) == 2:
        a, b = args.ckpts
        paired = {}
        for r, ma, mb in zip(kept, wins[a], wins[b]):
            key = f"{r['split']} {r['_true']}"
            paired.setdefault(key, [0, 0])
            if (ma > 0) != (mb > 0):
                paired[key][0 if ma > 0 else 1] += 1
        print(f"\npaired A={a} vs B={b}: clips only A right / only B right, sign-test p")
        for k, (ao, bo) in paired.items():
            print(f"  {k:20s} A-only {ao:3d}  B-only {bo:3d}  p={sign_p(ao, bo):.3g}")
        results["paired"] = {k: {"a_only": v[0], "b_only": v[1], "p": sign_p(*v)} for k, v in paired.items()}
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
