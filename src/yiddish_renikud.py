"""ReNikud-yi in the label stack: context-aware readings for the words no table knows.

WHAT IT IS. The rule engine (yiddish_g2p) reads a word it has no table entry
for by spelling rules, and Hasidic spelling leaves several slots open — א as
a/ɔ/u, פ as p/f, יי as aj/ej/aː, וי as ɔj/oʊ, a reduced ɛ/ə — which the rules
settle by default. ReNikud-yi (docs/xeus_finetune.md §18-19; the ReNikud
frame of Melichov, Kolani & Alper 2026 applied to Yiddish) is a character
BERT, the v6 pointing model's body, with a (consonant, vowel, stress) head on
every letter, trained on the corpus's gold and audio-attested readings. It
sees the whole sentence, so it can tell the same spelling apart by context.

HOW IT IS USED. Lexicon-first and graph-constrained, the combination that
measured best (§19: on 3,748 unlabelled words across six held-out episodes,
agreement with the host's audio 94.5%, the rule engine alone 88.0%):

  * a word any table answers (route 'lexicon') is left exactly as the table
    says — nothing here outranks a native verdict;
  * a rule-path word's engine reading defines the spelling's legal readings:
    every combination of its open slots (the §12 graph), plus the model's own
    free guess; each candidate is aligned letter-by-letter to the spelling
    (yi_align) and scored by the sum of the model's per-letter log-probs; the
    best legal reading replaces the engine's. Stress stays where the engine
    put it.

Installed by yiddish_labels at import through yiddish_g2p.set_context_reader;
PHONIKUD_YI_RENIKUD=0 leaves the engine as it was. Needs onnxruntime + numpy;
the export sits in onnx_renikud_yi/ beside this file in a bundle, or under
models/renikud_yi_audio/onnx in the repo ($PHONIKUD_YI_RENIKUD_MODEL overrides).
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import unicodedata as ud
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "scripts"):  # yi_align: beside us in a bundle, in scripts/ in the repo
    if (_p / "yi_align.py").exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_CANDIDATES = (
    os.environ.get("PHONIKUD_YI_RENIKUD_MODEL", ""),
    str(_HERE / "onnx_renikud_yi"),
    str(_HERE.parent / "models" / "renikud_yi_audio" / "onnx_int8"),
    str(_HERE.parent / "models" / "renikud_yi_audio" / "onnx"),
)

#: Slots the orthography leaves open (spec §4): a phone may be swapped for any
#: other member of its class; the stop/fricative pairs only word-finally.
_OPEN_CLASSES = (
    frozenset({"a", "ɔ", "u", "aː"}),   # א
    frozenset({"f", "p"}),              # פ
    frozenset({"aj", "aː", "ej"}),      # יי
    frozenset({"ɔj", "oʊ"}),            # וי
    frozenset({"i", "u"}),              # ו
    frozenset({"ɛ", "ə"}),              # reduction
)
_FINAL_DEVOICE = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s"}
_MAX_CANDIDATES = 64
_MULTI = ("aː", "ej", "aj", "ɔj", "oʊ")
_WS = re.compile(r"(\s+)")


def model_dir() -> str | None:
    for cand in _CANDIDATES:
        if cand and os.path.exists(os.path.join(cand, "model.onnx")):
            return cand
    return None


def available() -> bool:
    return model_dir() is not None


def _has_marks(word: str) -> bool:
    """True when the word carries any combining mark (nikud the writer supplied)."""
    return any(ud.category(c) == "Mn" for c in ud.normalize("NFKC", word))


def _base_text(text: str) -> str:
    """What the tokenizer sees: NFKC, marks stripped, lowercased."""
    return "".join(c for c in ud.normalize("NFKC", text) if ud.category(c) != "Mn").lower()


def graph_candidates(reading: list[str], max_candidates: int = _MAX_CANDIDATES) -> list[list[str]]:
    """Every legal reading reachable from ``reading`` by the open slots."""
    slots: list[list[str]] = []
    n = len(reading)
    for i, p in enumerate(reading):
        alts = {p}
        for cls in _OPEN_CLASSES:
            if p in cls:
                alts |= cls
        if i == n - 1 and p in _FINAL_DEVOICE:
            alts.add(_FINAL_DEVOICE[p])
        slots.append(sorted(alts, key=lambda x: (x != p, x)))
    out: list[list[str]] = []
    for combo in itertools.product(*slots):
        out.append(list(combo))
        if len(out) >= max_candidates:
            break
    return out


def restress(engine_ipa: str, phones: list[str], inventory: set[str]) -> str:
    """The engine's stress mark put back at the same phone index in ``phones``."""
    idx = None
    seen = 0
    s = engine_ipa.replace(" ", "")
    i = 0
    while i < len(s):
        if s[i] == "ˈ":
            idx = seen
            i += 1
            continue
        for m in _MULTI:
            if s.startswith(m, i):
                seen += 1
                i += len(m)
                break
        else:
            if s[i] in inventory:
                seen += 1
            i += 1
    out = []
    for k, p in enumerate(phones):
        if idx is not None and k == idx:
            out.append("ˈ")
        out.append(p)
    return "".join(out)


class ReNikudYi:
    """The exported model plus the graph-constrained decode."""

    def __init__(self, model_dir_: str | None = None, providers: list[str] | None = None):
        import onnxruntime as ort
        self.model_dir = model_dir_ or model_dir()
        if not self.model_dir:
            raise RuntimeError("no ReNikud-yi export found (model.onnx in: " + ", ".join(c for c in _CANDIDATES if c) + ")")
        self._session = ort.InferenceSession(os.path.join(self.model_dir, "model.onnx"), providers=providers or ["CPUExecutionProvider"])
        meta = self._session.get_modelmeta().custom_metadata_map
        missing = {"vocab", "consonants", "vowels", "cls_id", "sep_id", "pad_id", "unk_id"} - set(meta)
        if missing:
            raise RuntimeError(f"{self.model_dir}/model.onnx lacks runtime metadata {sorted(missing)}; re-export with scripts/export_renikud_onnx.py")
        self._vocab = {c: i for i, c in enumerate(json.loads(meta["vocab"]))}
        self.consonants: list[str] = json.loads(meta["consonants"])
        self.vowels: list[str] = json.loads(meta["vowels"])
        self._c2i = {c: i for i, c in enumerate(self.consonants)}
        self._v2i = {v: i for i, v in enumerate(self.vowels)}
        self.inventory = {p for p in self.consonants + self.vowels if p}
        self._cls, self._sep, self._pad, self._unk = (int(meta[k]) for k in ("cls_id", "sep_id", "pad_id", "unk_id"))
        self._chunk = int(meta.get("max_len", 512)) - 8
        self.providers = self._session.get_providers()

    # ------------------------------------------------------------ model
    def _split(self, base: str) -> list[str]:
        if len(base) <= self._chunk:
            return [base]
        out: list[str] = []
        cur = ""
        for piece in _WS.split(base):
            if cur and len(cur) + len(piece) > self._chunk:
                out.append(cur)
                cur = piece
            else:
                cur += piece
        if cur:
            out.append(cur)
        return out

    def logprobs(self, base: str):
        """Per-character (consonant, vowel) log-probabilities: two arrays (len(base), C) / (len(base), V)."""
        import numpy as np
        segs = self._split(base)
        ids = [[self._cls] + [self._vocab.get(c, self._unk) for c in s] + [self._sep] for s in segs]
        width = max(len(r) for r in ids)
        input_ids = np.full((len(ids), width), self._pad, dtype=np.int64)
        attention = np.zeros((len(ids), width), dtype=np.int64)
        for r, row in enumerate(ids):
            input_ids[r, : len(row)] = row
            attention[r, : len(row)] = 1
        cons, vow, _ = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention})
        lc = np.concatenate([_log_softmax(cons[r, 1: 1 + len(s)]) for r, s in enumerate(segs)])
        lv = np.concatenate([_log_softmax(vow[r, 1: 1 + len(s)]) for r, s in enumerate(segs)])
        return lc, lv

    # ------------------------------------------------------------ decode
    def tokenize(self, ipa: str) -> list[str]:
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
                if ch in self.inventory:
                    out.append(ch)
                i += 1
        return out

    def score(self, word: str, phones: list[str], start: int, lc, lv) -> float:
        """Log-probability of a reading under the model, letter-aligned; -inf if unalignable."""
        from yi_align import align_word, parse_chunk
        aligned = align_word(word, "".join(phones))
        if aligned is None or len(aligned) != len(word):
            return float("-inf")
        total = 0.0
        for k, (_, chunk) in enumerate(aligned):
            cons, vowel, _ = parse_chunk(chunk) if chunk else ("", "", 0)
            total += float(lc[start + k, self._c2i.get(cons, 0)]) + float(lv[start + k, self._v2i.get(vowel, 0)])
        return total

    def free_reading(self, start: int, end: int, lc, lv) -> list[str]:
        out: list[str] = []
        for k in range(start, end):
            c = self.consonants[int(lc[k].argmax())]
            v = self.vowels[int(lv[k].argmax())]
            if c:
                out.append(c)
            if v:
                out.append(v)
        return out

    def read_records(self, text: str, records: list[dict]) -> list[dict]:
        """The context reader yiddish_g2p.g2p_tokens calls: rule-path records rescored in place."""
        base = _base_text(text)
        if not any(r.get("route") == "rule" and r.get("ipa_primary") for r in records):
            return records
        lc, lv = self.logprobs(base)
        cursor = 0
        out = list(records)
        for i, rec in enumerate(records):
            w = _base_text(str(rec.get("word") or ""))
            if not w:
                continue
            pos = base.find(w, cursor)
            if pos < 0:
                continue
            cursor = pos + len(w)
            if rec.get("route") != "rule" or not rec.get("ipa_primary") or " " in w:
                continue
            if _has_marks(str(rec.get("word") or "")):
                continue  # the writer pointed this word: their marks outrank the model
            eng_ipa = str(rec["ipa_primary"])
            eng = self.tokenize(eng_ipa)
            if not eng:
                continue
            cands = graph_candidates(eng)
            own = self.free_reading(pos, pos + len(w), lc, lv)
            if own and own not in cands:
                cands.append(own)
            scored = sorted(((self.score(w, c, pos, lc, lv), c) for c in cands), key=lambda x: -x[0])
            best_score, best = scored[0]
            if best_score == float("-inf") or best == eng:
                continue
            new = dict(rec)  # never mutate the engine's cached record
            new["ipa_primary"] = restress(eng_ipa, best, self.inventory) if len(best) == len(eng) else "".join(best)
            new["layer"] = "R"
            new["reason"] = f"ReNikud-yi, graph-constrained: engine read {eng_ipa}"
            out[i] = new
        return out


def _log_softmax(x):
    import numpy as np
    x = x.astype(np.float32)
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(-1, keepdims=True))


_INSTANCE: ReNikudYi | None = None


def context_reader(text: str, records: list[dict]) -> list[dict]:
    """Process-wide lazily-loaded instance, in the shape set_context_reader wants."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ReNikudYi()
    return _INSTANCE.read_records(text, records)
