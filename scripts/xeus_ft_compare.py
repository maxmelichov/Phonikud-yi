#!/usr/bin/env python3
"""Two fine-tuned ears on identical clips: PER, exact match, and a paired sign test.

Aggregate PERs from two training runs are not directly comparable when the
runs scored different clips; this scores both checkpoints on the same rows —
the corpus val_words/val_eps splits plus, with --extra-data, that set's
val_eps clips as val_speakers (speakers neither ear saw in the corpus) — and
reports, per split, how many clips each ear alone gets exactly right, with the
two-sided binomial sign test on those discordant clips.

  python scripts/xeus_ft_compare.py --a data/xeus_ft/ckpt/best --b data/xeus_ft/ckpt_wa/best \
      --extra-data data/xeus_ft_wa --out data/xeus_ft/compare_wa.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import PerAccumulator, greedy_decode, read_jsonl, yi_logits  # noqa: E402
from xeus_ft_train import Segments, collate  # noqa: E402
from xeus_yi_decode import load_finetuned  # noqa: E402


def sign_test(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def score(inner, head, ds: Segments, device: str, seconds: float) -> tuple[PerAccumulator, list[bool]]:
    import torch
    acc, exact = PerAccumulator(), [False] * len(ds)
    inner.eval(); head.eval()
    with torch.no_grad():
        for idx in ds.batches(seconds, shuffle=False, rng=random.Random(0)):
            speech, lens, _, _ = collate(ds, idx, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, flens = yi_logits(inner, head, speech, lens)
            for i, hyp in zip(idx, greedy_decode(logits.float(), flens)):
                acc.add(ds.rows[i]["target"], hyp)
                exact[i] = hyp == ds.rows[i]["target"]
    return acc, exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--extra-data", default=None)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-seconds", type=float, default=96.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    rows = list(read_jsonl(data / "segments.jsonl"))
    splits = {s: Segments([r for r in rows if r["split"] == s][: args.limit or None], data / "seg") for s in ("val_words", "val_eps")}
    if args.extra_data:
        xd = Path(args.extra_data)
        xval = [r for r in read_jsonl(xd / "segments.jsonl") if r["split"] == "val_eps"][: args.limit or None]
        splits["val_speakers"] = Segments(xval, xd / "seg")
    report = {"a": args.a, "b": args.b, "splits": {}}
    results = {}
    for tag, ck in (("a", args.a), ("b", args.b)):
        inner, head = load_finetuned(Path(ck), device)
        results[tag] = {s: score(inner, head, ds, device, args.eval_seconds) for s, ds in splits.items()}
        del inner, head
        torch.cuda.empty_cache()
    for s in splits:
        (acc_a, ex_a), (acc_b, ex_b) = results["a"][s], results["b"][s]
        a_only = sum(x and not y for x, y in zip(ex_a, ex_b))
        b_only = sum(y and not x for x, y in zip(ex_a, ex_b))
        rec = {"clips": len(ex_a), "a": acc_a.summary(), "b": acc_b.summary(),
               "a_only_exact": a_only, "b_only_exact": b_only, "sign_p": sign_test(a_only, b_only)}
        report["splits"][s] = rec
        print(f"{s:13} n={len(ex_a):5}  A: PER {acc_a.per:.3f} exact {rec['a']['exact_match']:.3f}   "
              f"B: PER {acc_b.per:.3f} exact {rec['b']['exact_match']:.3f}   A-only {a_only}  B-only {b_only}  p={rec['sign_p']:.2g}", flush=True)
        ha, hb = rec["a"]["hard_phones"], rec["b"]["hard_phones"]
        print("   recall A/B: " + "  ".join(f"{p} {ha[p]['recall']:.2f}/{hb[p]['recall']:.2f}" for p in ha if ha[p]["n"]), flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
