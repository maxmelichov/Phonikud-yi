#!/usr/bin/env python3
"""Shared pieces of the PhoneticXeus -> Yiddish fine-tune.

Everything here runs on the GPU pod as well as locally, so it must not import
the G2P engine or anything outside torch / torchaudio / transformers. The
text side of data preparation (which needs the engine) lives in
``xeus_ft_text.py`` and ships its output as JSONL.

The recognizer is trained over the closed v3 inventory of
``docs/yiddish_phoneme_set.md`` minus stress: CTC stress placement is not
reliable and stress stays the G2P's job. Vocabulary = ``<blank>`` + 34 phones.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Iterator

# ----------------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------------

YI_VOWELS: tuple[str, ...] = ("a", "aː", "ɛ", "ə", "i", "u", "ɔ", "ej", "aj", "ɔj", "oʊ")
YI_CONSONANTS: tuple[str, ...] = (
    "b", "d", "f", "ɡ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t",
    "v", "z", "x", "ʃ", "ʒ", "ʦ", "ʧ", "ʤ", "ŋ",
)
YI_PHONES: tuple[str, ...] = YI_VOWELS + YI_CONSONANTS
YI_VOCAB: tuple[str, ...] = ("<blank>",) + YI_PHONES
YI_BLANK = 0
YI2ID: dict[str, int] = {p: i for i, p in enumerate(YI_VOCAB)}
_MULTI = ("aː", "ej", "aj", "ɔj", "oʊ")

#: Yiddish phone -> the PhoneticXeus symbols it is written with. Used to build
#: alignment targets in the pretrained model's own vocabulary and to warm-start
#: the Yiddish head from the matching rows of the pretrained head.
V3_TO_XEUS: dict[str, list[str]] = {
    "aj": ["a", "ɪ"],
    "ej": ["e", "ɪ"],
    "ɔj": ["ɔ", "ɪ"],
    "oʊ": ["o", "ʊ"],
    "ʦ": ["t͡s"],
    "ʧ": ["t͡ʃ"],
    "ʤ": ["d͡ʒ"],
}

#: The phones the pretrained recognizer is known to mishear (docs/audio_evidence.md
#: §4). Reported separately in every evaluation so the fine-tune is judged on
#: exactly the confusions it was meant to fix.
HARD_PHONES: tuple[str, ...] = ("ʦ", "aj", "ej", "z", "ʃ", "u", "f", "ɔj", "oʊ", "ʒ",
                                 # run-1 regressions (docs/xeus_finetune.md §10): tracked from run 2 on
                                 "ɛ", "i", "ə")


# ----------------------------------------------------------------------------
# Lexicon keys
# ----------------------------------------------------------------------------

import re as _re
import unicodedata as _ud

_FINAL_FOLD = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_LIGATURE_FOLD = str.maketrans({"ײ": "יי", "ױ": "וי", "װ": "וו"})
_GERESH = _re.compile(r"[׳ʼ‘’`]")
_GERSHAYIM = _re.compile(r"[״“”]")
_DASH = _re.compile(r"[־‐‑‒–—]")


def lexicon_key(word: str) -> str:
    """The engine's whole-token lookup key (yiddish_g2p.lexicon_key), mirrored
    so the pod and the decoder need no engine: NFC, unified geresh/gershayim
    and dashes, nikud stripped, finals and YIVO ligatures folded. Verified
    identical to the engine over every surface form in the corpus."""
    text = _ud.normalize("NFC", word)
    text = _DASH.sub("-", text)
    text = _GERESH.sub("'", text)
    text = _GERSHAYIM.sub('"', text)
    text = _ud.normalize("NFD", text)
    text = "".join(c for c in text if _ud.category(c) != "Mn")
    text = _ud.normalize("NFC", text)
    return text.translate(_FINAL_FOLD).translate(_LIGATURE_FOLD)


def tokenize_ipa(ipa: str) -> list[str]:
    """Split an engine/gold IPA string into inventory phones, dropping stress."""
    out: list[str] = []
    s = ipa.replace(" ", "")
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "ˈˌ":
            i += 1
            continue
        for m in _MULTI:
            if s.startswith(m, i):
                out.append(m)
                i += len(m)
                break
        else:
            if ch in YI2ID:
                out.append(ch)
            i += 1
    return out


def to_xeus_syms(phones: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in phones:
        out.extend(V3_TO_XEUS.get(p, [p]))
    return out


def yi_ids(phones: Iterable[str]) -> list[int]:
    return [YI2ID[p] for p in phones]


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_pretrained(device: str = "cpu", dtype=None):
    """The HF wrapper and the inner XeusPRModel, on ``device``."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained("changelinglab/PhoneticXeus", trust_remote_code=True)
    model = model.eval().to(device)
    if dtype is not None:
        model = model.to(dtype)
    return model, model.model


def xeus_vocab(inner) -> tuple[list[str], dict[str, int]]:
    vocab = list(inner.token_list)
    return vocab, {s: i for i, s in enumerate(vocab)}


def build_yi_head(inner):
    """A fresh CTC head over YI_VOCAB, warm-started from the pretrained head.

    Each Yiddish phone's row is the mean of the pretrained rows for the symbols
    it is written with (``V3_TO_XEUS``): a diphthong starts like its first
    element and ends like its second, and an affricate row exists as-is. The
    blank row is copied. Nothing is random, so epoch 0 already behaves like
    the pretrained recognizer folded onto the inventory.
    """
    import torch

    old = inner.ctc.ctc_lo
    _, x2i = xeus_vocab(inner)
    head = torch.nn.Linear(old.in_features, len(YI_VOCAB), bias=old.bias is not None)
    with torch.no_grad():
        w = old.weight.detach()
        b = old.bias.detach() if old.bias is not None else None
        rows = [x2i["<blank>"]]
        for p in YI_PHONES:
            rows.append([x2i[s] for s in V3_TO_XEUS.get(p, [p])])
        for k, src in enumerate(rows):
            idx = [src] if isinstance(src, int) else src
            head.weight[k] = w[idx].mean(0)
            if b is not None:
                head.bias[k] = b[idx].mean()
    return head.to(old.weight.device)


def encoder_blocks(inner):
    """The encoder's block list, found by shape rather than by name."""
    import torch

    enc = inner.encoder
    for name in ("encoders", "blocks", "layers"):
        mod = getattr(enc, name, None)
        if mod is not None and hasattr(mod, "__len__") and len(mod) > 4:
            return mod
    for name, mod in enc.named_children():
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)) and len(mod) > 4:
            return mod
    raise RuntimeError("could not locate the encoder block list")


def yi_logits(inner, head, speech, lengths):
    """Frame-level Yiddish logits (B, T, |YI_VOCAB|) and frame lengths."""
    enc, enc_lens = inner.encode(speech, lengths)
    if isinstance(enc, tuple):
        enc = enc[0]
    return head(enc), enc_lens


def greedy_decode(logits, lengths, blank: int = YI_BLANK) -> list[list[str]]:
    """Collapse argmax frames into phone strings, one per batch item."""
    ids = logits.argmax(-1).cpu().tolist()
    out = []
    for row, n in zip(ids, lengths.tolist()):
        phones: list[str] = []
        prev = None
        for t in row[:n]:
            if t != blank and t != prev:
                phones.append(YI_VOCAB[t])
            prev = t
        out.append(phones)
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def align_pairs(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Needleman-Wunsch pairing (same scoring as scripts/xeus_tag.py) for confusion counts."""
    n, m = len(ref), len(hyp)
    gap = -1.0
    vowels = set(YI_VOWELS)

    def sim(a: str, b: str) -> float:
        if a == b:
            return 2.0
        if (a in vowels) == (b in vowels):
            return 0.5 if a in vowels else 0.0
        return -1.5

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
    for j in range(1, m + 1):
        score[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(
                score[i - 1][j - 1] + sim(ref[i - 1], hyp[j - 1]),
                score[i - 1][j] + gap,
                score[i][j - 1] + gap,
            )
    pairs: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and math.isclose(score[i][j], score[i - 1][j - 1] + sim(ref[i - 1], hyp[j - 1])):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and math.isclose(score[i][j], score[i - 1][j] + gap):
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    return pairs[::-1]


class PerAccumulator:
    """Phone error rate plus per-phone confusion, accumulated over segments."""

    def __init__(self) -> None:
        self.edits = 0
        self.ref_len = 0
        self.n = 0
        self.exact = 0
        self.confusion: dict[str, dict[str, int]] = {}

    def add(self, ref: list[str], hyp: list[str]) -> None:
        self.edits += edit_distance(ref, hyp)
        self.ref_len += len(ref)
        self.n += 1
        self.exact += int(ref == hyp)
        for r, h in align_pairs(ref, hyp):
            if r is None:
                continue
            row = self.confusion.setdefault(r, {})
            key = h if h is not None else "∅"
            row[key] = row.get(key, 0) + 1

    @property
    def per(self) -> float:
        return self.edits / self.ref_len if self.ref_len else float("nan")

    def phone_recall(self, phone: str) -> tuple[float, int]:
        row = self.confusion.get(phone, {})
        total = sum(row.values())
        return (row.get(phone, 0) / total if total else float("nan")), total

    def summary(self) -> dict:
        hard = {}
        for p in HARD_PHONES:
            rec, total = self.phone_recall(p)
            row = self.confusion.get(p, {})
            top = sorted(row.items(), key=lambda kv: -kv[1])[:3]
            hard[p] = {"recall": rec, "n": total, "heard_as": top}
        return {
            "per": self.per,
            "segments": self.n,
            "exact_match": self.exact / self.n if self.n else float("nan"),
            "ref_phones": self.ref_len,
            "hard_phones": hard,
        }
