#!/usr/bin/env python3
"""Export the best Yiddish diacritizer (round-4 stageB) to ONNX.

Wraps the model so the graph takes (input_ids, attention_mask) and returns the
three mirror-head logits. Dynamic axes on batch and sequence. Verifies the
export by comparing ONNX outputs against torch on real test sentences.

Usage:
    .venv/bin/python scripts/export_onnx.py \
        --model models/phonikud_yi/round4/stageB \
        --out models/phonikud_yi/round4/onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phonikud" / "model"))

from src.model.phonikud_model import PhonikudModel  # noqa: E402


class MirrorHeadsWrapper(torch.nn.Module):
    def __init__(self, model: PhonikudModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model({"input_ids": input_ids, "attention_mask": attention_mask})
        return out.yi_nikud_logits, out.yi_shin_logits, out.yi_rafe_logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phonikud_yi/round4/stageB")
    ap.add_argument("--out", default="models/phonikud_yi/round4/onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = PhonikudModel.from_pretrained(args.model)
    model.eval()

    wrapper = MirrorHeadsWrapper(model)
    sample = tokenizer(["א טעסט זאץ"], return_tensors="pt")

    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["nikud_logits", "shin_logits", "rafe_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "nikud_logits": {0: "batch", 1: "seq"},
            "shin_logits": {0: "batch", 1: "seq"},
            "rafe_logits": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
    )
    size_mb = sum(f.stat().st_size for f in out_dir.glob("model.onnx*")) / 1e6
    print(f"exported: {onnx_path} ({size_mb:.0f} MB)")

    # copy tokenizer + labels so the dir is self-contained
    for name in (
        "tokenizer.json", "tokenizer_config.json", "vocab.txt",
        "special_tokens_map.json", "yi_labels.json",
    ):
        src = Path(args.model) / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())

    # ---- verify against torch on real sentences ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_file = REPO / "data/diacritics_r3c/test.txt"
    sents = []
    if test_file.exists():
        import re
        strip = re.compile(r"[֑-ׇ]")
        with open(test_file, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 5:
                    break
                sents.append(strip.sub("", line.strip())[:200])
    else:
        sents = ["וואס מאכסטו הייnט"]

    max_diff = 0.0
    mismatches = 0
    for s in sents:
        enc = tokenizer([s], return_tensors="pt")
        with torch.no_grad():
            t_nik, t_shin, t_rafe = wrapper(enc["input_ids"], enc["attention_mask"])
        o_nik, o_shin, o_rafe = sess.run(
            None,
            {
                "input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy(),
            },
        )
        for t, o in ((t_nik, o_nik), (t_shin, o_shin), (t_rafe, o_rafe)):
            max_diff = max(max_diff, float(np.abs(t.numpy() - o).max()))
            mismatches += int((t.numpy().argmax(-1) != o.argmax(-1)).sum())

    print(f"verify: max |logit diff| = {max_diff:.2e}, argmax mismatches = {mismatches}")
    if mismatches:
        print("WARNING: predictions differ between torch and ONNX", file=sys.stderr)
        return 1
    print("OK: ONNX predictions identical to torch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
