#!/usr/bin/env python3
"""
Shared data plumbing for the masked-supervision retrain of phonikud-yi.

The retrain dataset (`data/retrain/{train,val,test}.jsonl`) carries a *per-word
supervision mask* that the upstream `.txt` format could not express:

    {"id":..., "episode":..., "text": <bare>, "pointed": <fully pointed>,
     "supervised": [bool per whitespace token], ...}

`supervised[i] == False` means "the v3 engine does not verify this word's
reading" -- keep whatever pointing it happens to have, but emit **no gradient**
for it. That is done by setting the per-character mark to `None`, which becomes
`IGNORE = -100` in every head's label tensor, exactly the path the upstream
collator already used for non-Hebrew characters.

Label projection is the round-4 "Hebrew-mirror" scheme, read out of the
checkpoint's own config (`yi_nikud_classes` / `shin_classes` / `rafe_classes`)
rather than re-declared here, so the heads stay bit-compatible with
`scripts/export_onnx.py`, `scripts/infer_onnx.py` and the student distillation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

IGNORE = -100

DAGESH = "ּ"
RAFE_CHAR = "ֿ"
SHIN_DOT = "ׁ"
SIN_DOT = "ׂ"

VOWELS = "ְֱֲֳִֵֶַָׇֹֺֻ"
VOWEL_SET = set(VOWELS)
MARKS = VOWEL_SET | {DAGESH, RAFE_CHAR, SHIN_DOT, SIN_DOT}

# Encoding variants folded before labelling, per dicta_model.YI_VOWEL_FOLD.
YI_VOWEL_FOLD = {"ֺ": "ֹ", "ׇ": "ָ"}

HEB_LETTER = re.compile(r"[א-ת]")


def is_heb(ch: str) -> bool:
    return "א" <= ch <= "ת"


def canon(marks: str) -> str:
    """dagesh, rafe, dot, vowel -- upstream `yi_data.canon` order (label identity)."""
    dagesh = DAGESH if DAGESH in marks else ""
    rafe = RAFE_CHAR if RAFE_CHAR in marks else ""
    dot = SHIN_DOT if SHIN_DOT in marks else (SIN_DOT if SIN_DOT in marks else "")
    vowel = next((m for m in marks if m in VOWEL_SET), "")
    return dagesh + rafe + dot + vowel


def strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", s) if c not in MARKS)


# ------------------------------------------------------------------ examples


@dataclass
class YiExample:
    text: str                     # consonant skeleton -- the model input
    marks: List[Optional[str]]    # canonical mark string per char; None = no gradient
    row_id: str = ""


def parse_row(pointed: str, supervised: Optional[List[bool]], row_id: str = "") -> YiExample:
    """Pointed text + per-word mask -> (skeleton, per-char marks).

    Word index advances on whitespace runs in the *pointed* string, which is how
    `supervised` was built (`pointed.split()`).
    """
    pointed = unicodedata.normalize("NFC", pointed).strip()
    chars: List[str] = []
    marks: List[Optional[str]] = []

    word_idx = -1
    in_word = False
    i, n = 0, len(pointed)
    while i < n:
        ch = pointed[i]
        i += 1
        if ch in MARKS:
            continue  # stray mark with no base letter
        if ch.isspace():
            in_word = False
        elif not in_word:
            in_word = True
            word_idx += 1
        got = ""
        while i < n and pointed[i] in MARKS:
            got += pointed[i]
            i += 1
        chars.append(ch)
        if not is_heb(ch):
            marks.append(None)
            continue
        ok = True
        if supervised is not None:
            ok = 0 <= word_idx < len(supervised) and bool(supervised[word_idx])
        marks.append(canon(got) if ok else None)

    return YiExample("".join(chars), marks, row_id)


def _slice(ex: YiExample, lo: int, hi: int) -> YiExample:
    while lo < hi and ex.text[lo].isspace():
        lo += 1
    while hi > lo and ex.text[hi - 1].isspace():
        hi -= 1
    return YiExample(ex.text[lo:hi], ex.marks[lo:hi], ex.row_id)


def chunk(ex: YiExample, max_chars: int) -> List[YiExample]:
    """Split at word boundaries so the char<->mask alignment is never broken."""
    out: List[YiExample] = []
    start = 0
    while start < len(ex.text):
        end = min(start + max_chars, len(ex.text))
        if end < len(ex.text):
            sp = ex.text.rfind(" ", start + 1, end)
            if sp > start:
                end = sp
        piece = _slice(ex, start, end)
        if piece.text and HEB_LETTER.search(piece.text):
            out.append(piece)
        start = end
    return out


def read_jsonl(
    path: Path,
    max_chars: int = 480,
    limit: Optional[int] = None,
    supervised_only: bool = True,
) -> List[YiExample]:
    """Read a retrain split. `limit` counts ROWS (pre-chunking), for smoke runs."""
    out: List[YiExample] = []
    rows = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows += 1
            if limit is not None and rows > limit:
                break
            mask = rec.get("supervised") if supervised_only else None
            ex = parse_row(rec["pointed"], mask, rec.get("id", ""))
            if not HEB_LETTER.search(ex.text):
                continue
            out.extend(chunk(ex, max_chars))
    return out


def read_rows(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


# --------------------------------------------------------------- label space


class MirrorLabels:
    """Round-4 mirror projection, driven by the checkpoint's own config."""

    def __init__(self, config):
        self.yi_nikud_classes: List[str] = list(config.yi_nikud_classes)
        self.shin_classes: List[str] = list(config.shin_classes)
        self.rafe_classes: List[str] = list(config.rafe_classes)
        self._id: Dict[str, int] = {c: i for i, c in enumerate(self.yi_nikud_classes)}
        assert self.yi_nikud_classes[0] == "", "class 0 must be NO_MARK"

    def to_ids(self, marks: Optional[str], char: str):
        if marks is None:
            return IGNORE, IGNORE, IGNORE
        dagesh = DAGESH if DAGESH in marks else ""
        vowel = next((m for m in marks if m in VOWEL_SET), "")
        vowel = YI_VOWEL_FOLD.get(vowel, vowel)
        nikud = self._id.get(dagesh + vowel)
        if nikud is None:
            nikud = self._id.get(vowel, 0)
        shin = (1 if SIN_DOT in marks else 0) if char == "ש" else IGNORE
        rafe = 1 if RAFE_CHAR in marks else 0
        return nikud, shin, rafe

    def to_marks(self, nikud_id: int, shin_id: int, rafe_id: int, char: str) -> str:
        """Inverse of `to_ids`, in canon() order."""
        nk = self.yi_nikud_classes[nikud_id]
        dagesh = DAGESH if DAGESH in nk else ""
        vowel = nk.replace(DAGESH, "")
        dot = ""
        if char == "ש":
            dot = SIN_DOT if shin_id == 1 else SHIN_DOT
        return dagesh + (RAFE_CHAR if rafe_id else "") + dot + vowel


def render(char: str, marks: str) -> str:
    """Emit a base letter plus its marks in NFC (Unicode canonical) order:
    vowel, dagesh, rafe, shin/sin dot -- the order `canonicalize_pointing.py`
    writes and the order the corpus is stored in."""
    if not marks:
        return char
    vowel = next((m for m in marks if m in VOWEL_SET), "")
    dagesh = DAGESH if DAGESH in marks else ""
    rafe = RAFE_CHAR if RAFE_CHAR in marks else ""
    dot = SHIN_DOT if SHIN_DOT in marks else (SIN_DOT if SIN_DOT in marks else "")
    return unicodedata.normalize("NFC", char + vowel + dagesh + rafe + dot)


# ------------------------------------------------------------------ batching


@dataclass
class Batch:
    input: dict
    nikud: torch.Tensor
    shin: torch.Tensor
    rafe: torch.Tensor
    texts: List[str]


class Collator:
    def __init__(self, tokenizer, labels: MirrorLabels, max_length: int = 512):
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length

    def __call__(self, items: List[YiExample]) -> Batch:
        texts = [it.text for it in items]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = enc.pop("offset_mapping")
        bsz, seq = enc["input_ids"].shape
        blank = lambda: torch.full((bsz, seq), IGNORE, dtype=torch.long)  # noqa: E731
        nikud, shin, rafe = blank(), blank(), blank()

        for b, item in enumerate(items):
            for t in range(seq):
                s, e = int(offsets[b, t, 0]), int(offsets[b, t, 1])
                if e - s != 1 or s >= len(item.marks):
                    continue
                m = item.marks[s]
                if m is None:
                    continue
                n_, s_, r_ = self.labels.to_ids(m, item.text[s])
                nikud[b, t], shin[b, t], rafe[b, t] = n_, s_, r_

        return Batch(input=dict(enc), nikud=nikud, shin=shin, rafe=rafe, texts=texts)


class ExampleDataset(Dataset):
    def __init__(self, examples: List[YiExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_loader(examples, tokenizer, labels, batch_size, shuffle,
                max_length=512, num_workers=0):
    return DataLoader(
        ExampleDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collator(tokenizer, labels, max_length),
        num_workers=num_workers,
        pin_memory=False,   # MPS + pinned memory is a known hang source
        drop_last=False,
    )
