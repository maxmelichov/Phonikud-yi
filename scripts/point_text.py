#!/usr/bin/env python3
"""
Torch inference for the Yiddish diacritizer: bare text -> fully pointed text.

Same load pattern as `scripts/benchmark_models.py` / `scripts/eval_oov_wordlevel.py`
(`PhonikudModel.from_pretrained(<dir>)` after putting `phonikud/model` on the
path), but the checkpoint directory is a flag, so the retrained model can be
exercised without touching -- or breaking -- the shipping one. The default is
still `models/phonikud_yi/round4/stageB`.

Decoding uses the round-4 "Hebrew-mirror" heads (yi_nikud / yi_shin / yi_rafe)
and emits marks in NFC order, matching `scripts/canonicalize_pointing.py`.

Usage:
    python scripts/point_text.py --text "דער קאזשניצער מגיד פלעגט זיך פירן"
    python scripts/point_text.py --model models/phonikud_yi_v2/best --stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "phonikud" / "model"))

from src.model.phonikud_model import PhonikudModel  # noqa: E402

from phonikud_yi_data import MirrorLabels, is_heb, render, strip_marks  # noqa: E402

DEFAULT_MODEL = REPO / "models/phonikud_yi/round4/stageB"


class Pointer:
    def __init__(self, model_dir=DEFAULT_MODEL, device: str = "auto",
                 max_length: int = 512):
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = PhonikudModel.from_pretrained(model_dir).eval().to(device)
        self.labels = MirrorLabels(self.model.config)

    def _split_long(self, text: str) -> List[str]:
        """Cut at spaces into pieces the char model can see whole.

        BertTokenizerFast would silently truncate anything past `max_length`,
        which drops the tail of a long line instead of pointing it. The pieces
        keep every original character (including the space they were cut at), so
        concatenating the pointed pieces reproduces the input exactly.
        """
        budget = self.max_length - 2
        if len(text) <= budget:
            return [text]
        out, start = [], 0
        while start < len(text):
            end = min(start + budget, len(text))
            if end < len(text):
                sp = text.rfind(" ", start + 1, end)
                if sp > start:
                    end = sp
            out.append(text[start:end])
            start = end
        return out

    @torch.no_grad()
    def point(self, sentences: List[str]) -> List[str]:
        """Point each sentence; long sentences are cut at word boundaries and
        re-joined so nothing is truncated."""
        pieces, owner = [], []
        for i, s in enumerate(sentences):
            for piece in self._split_long(strip_marks(s)):
                pieces.append(piece)
                owner.append(i)
        pointed = ["" for _ in sentences]
        for k in range(0, len(pieces), 8):
            for j, out in enumerate(self._point_batch(pieces[k:k + 8])):
                pointed[owner[k + j]] += out
        return pointed

    @torch.no_grad()
    def _point_batch(self, bare: List[str]) -> List[str]:
        enc = self.tokenizer(bare, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt",
                             return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc.pop("offset_mapping")
        out = self.model({k: v.to(self.device) for k, v in enc.items()})
        pn = out.yi_nikud_logits.argmax(-1).cpu()
        ps = out.yi_shin_logits.argmax(-1).cpu()
        pr = out.yi_rafe_logits.argmax(-1).cpu()

        results = []
        for b, text in enumerate(bare):
            marks = [""] * len(text)
            for t in range(offsets.shape[1]):
                s, e = int(offsets[b, t, 0]), int(offsets[b, t, 1])
                if e - s != 1 or s >= len(text) or not is_heb(text[s]):
                    continue
                marks[s] = self.labels.to_marks(
                    int(pn[b, t]), int(ps[b, t]), int(pr[b, t]), text[s])
            results.append("".join(render(c, m) for c, m in zip(text, marks)))
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--text", default=None)
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    lines = []
    if args.text:
        lines.append(args.text)
    if args.stdin or not lines:
        lines += [l.strip() for l in sys.stdin.read().splitlines() if l.strip()]
    if not lines:
        ap.error("give --text or pipe lines on stdin")

    p = Pointer(args.model, args.device)
    for out in p.point(lines):
        print(out)


if __name__ == "__main__":
    main()
