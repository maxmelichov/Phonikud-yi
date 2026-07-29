#!/usr/bin/env python3
"""
CPU throughput benchmark: small student vs round-4 teacher.

The student and the teacher ONNX exports happen to share an interface
(input_ids + attention_mask -> nikud/shin/rafe logits), so the same harness runs
both; only the character tokenisation differs. Torch teacher is included because
that is what the pipeline uses today.

Usage:
    python scripts/benchmark_models.py --test data/diacritics_r3c/test.txt \
        --out models/phonikud_yi_small/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "phonikud" / "model"))

MARKS = re.compile(r"[֑-ׇ]")


def load_lines(path, limit=None):
    lines = [MARKS.sub("", l.strip()) for l in
             Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[:limit] if limit else lines


def bench(fn, lines, batch_size, warmup=2):
    for i in range(min(warmup, len(lines))):
        fn(lines[i : i + 1])
    n_chars = sum(len(l) for l in lines)
    t0 = time.perf_counter()
    for i in range(0, len(lines), batch_size):
        fn(lines[i : i + batch_size])
    dt = time.perf_counter() - t0
    return {"seconds": round(dt, 2), "chars_per_sec": round(n_chars / dt),
            "lines_per_sec": round(len(lines) / dt, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=REPO / "data/diacritics_r3c/test.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200, help="lines to time")
    ap.add_argument("--batch-sizes", default="1,8")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    lines = load_lines(args.test, args.limit)
    n_chars = sum(len(l) for l in lines)
    print(f"benchmark on {len(lines)} lines / {n_chars:,} chars", flush=True)
    batches = [int(x) for x in args.batch_sizes.split(",")]

    results = {"n_lines": len(lines), "n_chars": n_chars, "models": {}}
    small = REPO / "models/phonikud_yi_small"
    t_onnx = REPO / "models/phonikud_yi/round4/onnx"

    import onnxruntime as ort
    from infer_onnx import Diacritizer

    def size_mb(*paths):
        return round(sum(p.stat().st_size for p in paths if p.exists()) / 1e6, 1)

    # ---- student ONNX (fp32 / int8) -------------------------------------
    for name, path in (("student ONNX fp32", small / "student.onnx"),
                       ("student ONNX int8", small / "student.int8.onnx")):
        if not path.exists():
            continue
        d = Diacritizer(path, None, args.threads)
        entry = {"file_mb": size_mb(path), "params": 18_416_666, "runs": {}}
        for b in batches:
            entry["runs"][f"batch{b}"] = bench(lambda ls: d.point_model(ls), lines, b)
            print(f"  {name} b{b}: {entry['runs'][f'batch{b}']}", flush=True)
        results["models"][name] = entry

    # ---- teacher ONNX (fp32 / int8) -------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO / "models/phonikud_yi/round4/stageB")

    def make_teacher_onnx(path):
        opts = ort.SessionOptions()
        if args.threads:
            opts.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        def run(ls):
            enc = tok(list(ls), padding=True, truncation=True, max_length=512,
                      return_tensors="np")
            return sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                   "attention_mask": enc["attention_mask"].astype(np.int64)})
        return run

    for name, path, extra in (
        ("teacher ONNX fp32", t_onnx / "model.onnx", [t_onnx / "model.onnx.data"]),
        ("teacher ONNX int8", t_onnx / "model.int8.onnx", []),
    ):
        if not path.exists():
            continue
        run = make_teacher_onnx(path)
        entry = {"file_mb": size_mb(path, *extra), "params": 305_800_000, "runs": {}}
        for b in batches:
            entry["runs"][f"batch{b}"] = bench(run, lines, b)
            print(f"  {name} b{b}: {entry['runs'][f'batch{b}']}", flush=True)
        results["models"][name] = entry

    # ---- teacher torch ---------------------------------------------------
    import torch
    import os
    from src.model.phonikud_model import PhonikudModel

    torch.set_num_threads(args.threads or os.cpu_count() or 4)
    m = PhonikudModel.from_pretrained(REPO / "models/phonikud_yi/round4/stageB").eval()

    def run_torch(ls):
        with torch.no_grad():
            enc = tok(list(ls), padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            return m(dict(enc))

    ckpt = REPO / "models/phonikud_yi/round4/stageB/model.safetensors"
    entry = {"file_mb": size_mb(ckpt), "params": 305_800_000, "runs": {}}
    for b in batches:
        entry["runs"][f"batch{b}"] = bench(run_torch, lines, b)
        print(f"  teacher torch fp32 b{b}: {entry['runs'][f'batch{b}']}", flush=True)
    results["models"]["teacher torch fp32"] = entry

    results["threads"] = args.threads or os.cpu_count()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
