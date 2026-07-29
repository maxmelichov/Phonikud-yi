#!/usr/bin/env python3
"""
Pure-python inference for the small Yiddish diacritizer.

Dictionary-first: any word whose bare form is in data/canonical_pointing.tsv gets
the canonical pointing straight from the table; only out-of-dictionary words are
pointed by the ONNX model. The model still sees the *whole* sentence, so OOV
words are pointed in context -- the dictionary only overrides the output.

Everything the model needs (char vocab, the 22 nikud classes, shin/rafe tables)
is embedded in the .onnx metadata, so this script needs no sidecar files beyond
the optional dictionary.

Usage:
    python scripts/infer_onnx.py --model models/phonikud_yi_small/student.onnx \\
        --text "דער קאזשניצער מגיד פלעגט זיך פירן"
    echo "..." | python scripts/infer_onnx.py -m ... --stdin
    python scripts/infer_onnx.py -m ... --benchmark data/diacritics_r3c/test.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

HEB = re.compile(r"[א-ת]")
MARKS = re.compile(r"[֑-ׇ]")
DAGESH = "ּ"


def strip_marks(s: str) -> str:
    return MARKS.sub("", s)


def split_token(tok: str):
    """Split leading/trailing punctuation off a token's core.

    Combining marks belong to the letter they follow, so when trimming from the
    right we skip over any marks first and then test the *base* character. The
    naive version treated a word-final sheva as trailing punctuation, which meant
    a canonical replacement got the stolen mark re-appended after it and produced
    a doubled point (זִיךְ -> זִיךְְ)."""
    n = len(tok)
    i = 0
    while i < n and not HEB.match(tok[i]):
        i += 1
    j = n
    while j > i:
        k = j - 1
        while k > i and MARKS.match(tok[k]):
            k -= 1
        if HEB.match(tok[k]):
            break  # real letter: its marks run to j-1, so the core ends at j
        j = k  # punctuation: drop it and keep scanning left
    return tok[:i], tok[i:j], tok[j:]

class Diacritizer:
    def __init__(self, onnx_path, dictionary=None, threads=None):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.itos = json.loads(meta["vocab"])
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.nikud = json.loads(meta["nikud_classes"])
        self.shin = json.loads(meta["shin_classes"])
        self.rafe = json.loads(meta["rafe_classes"])
        self.max_len = int(meta.get("max_len", 512))
        self.dict = self._load_dict(dictionary) if dictionary else {}

    @staticmethod
    def _load_dict(path):
        m = {}
        with Path(path).open(encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0]:
                    m[p[0]] = p[1]
        return m

    # ---------------------------------------------------------------- model

    def _encode(self, text):
        return [self.stoi.get(c, 1) for c in text]

    def point_model(self, texts):
        """Point a batch of strings with the model only (no dictionary)."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [unicodedata.normalize("NFC", t)[: self.max_len] for t in texts]
        T = max(len(t) for t in texts)
        B = len(texts)
        ids = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=np.int64)
        for b, t in enumerate(texts):
            e = self._encode(t)
            ids[b, : len(e)] = e
            mask[b, : len(e)] = 1

        nk, sh, rf = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})
        nk, sh, rf = nk.argmax(-1), sh.argmax(-1), rf.argmax(-1)

        out = []
        for b, t in enumerate(texts):
            buf = []
            for i, ch in enumerate(t):
                if not HEB.match(ch):
                    buf.append(ch)
                    continue
                cls = self.nikud[int(nk[b, i])]
                dag = DAGESH if DAGESH in cls else ""
                vowel = cls.replace(DAGESH, "")
                dot = self.shin[int(sh[b, i])] if ch == "ש" else ""
                raf = self.rafe[int(rf[b, i])]
                buf.append(ch + dag + raf + dot + vowel)
            out.append("".join(buf))
        return out

    def marks_for(self, texts):
        """Per-character canonical mark strings (None on non-Hebrew chars).
        Used by the evaluation harness so the student is scored exactly like the
        teacher."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [unicodedata.normalize("NFC", t)[: self.max_len] for t in texts]
        T = max(len(t) for t in texts)
        B = len(texts)
        ids = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=np.int64)
        for b, t in enumerate(texts):
            e = self._encode(t)
            ids[b, : len(e)] = e
            mask[b, : len(e)] = 1
        nk, sh, rf = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})
        nk, sh, rf = nk.argmax(-1), sh.argmax(-1), rf.argmax(-1)
        out = []
        for b, t in enumerate(texts):
            row = []
            for i, ch in enumerate(t):
                if not HEB.match(ch):
                    row.append(None)
                    continue
                cls = self.nikud[int(nk[b, i])]
                dag = DAGESH if DAGESH in cls else ""
                vowel = cls.replace(DAGESH, "")
                dot = self.shin[int(sh[b, i])] if ch == "ש" else ""
                raf = self.rafe[int(rf[b, i])]
                row.append(dag + raf + dot + vowel)
            out.append(row)
        return out

    # ----------------------------------------------------------- public API

    def point(self, text, use_dict=True):
        """Dictionary-first pointing of a single string."""
        pointed = self.point_model(text)[0]
        if not use_dict or not self.dict:
            return pointed

        # Re-tokenize the model output and swap in dictionary forms.
        src_toks = unicodedata.normalize("NFC", text).split()
        out_toks = pointed.split()
        if len(src_toks) != len(out_toks):  # shouldn't happen; be safe
            return pointed
        merged = []
        for src, got in zip(src_toks, out_toks):
            pre, core, post = split_token(got)
            bare = strip_marks(core)
            merged.append(pre + self.dict.get(bare, core) + post)
        return " ".join(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-d", "--dict", default="data/canonical_pointing.tsv")
    ap.add_argument("--text")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--no-dict", action="store_true")
    ap.add_argument("--benchmark", type=Path)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    dict_path = None if args.no_dict else (args.dict if Path(args.dict).exists() else None)
    d = Diacritizer(args.model, dict_path, args.threads)

    if args.benchmark:
        lines = [strip_marks(l.strip()) for l in
                 args.benchmark.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_chars = sum(len(l) for l in lines)
        t0 = time.time()
        for i in range(0, len(lines), args.batch_size):
            d.point_model(lines[i : i + args.batch_size])
        dt = time.time() - t0
        print(json.dumps({
            "lines": len(lines), "chars": n_chars, "seconds": round(dt, 2),
            "chars_per_sec": round(n_chars / dt), "lines_per_sec": round(len(lines) / dt, 1),
            "batch_size": args.batch_size,
        }, indent=2))
        return

    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if line:
                print(d.point(line, use_dict=not args.no_dict))
        return

    if not args.text:
        ap.error("give --text, --stdin or --benchmark")
    print(d.point(args.text, use_dict=not args.no_dict))


if __name__ == "__main__":
    main()
