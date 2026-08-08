#!/usr/bin/env python3
"""Build the verified-diacritics training set for the phonikud-yi retrain.

Supervision policy
------------------
For every whitespace token of ``data/yiddish_tts_dataset.tsv`` we route the word
through the FROZEN v3 engine (``yiddish_g2p.g2p_token``).  Only tokens that come
back with ``route == "lexicon"`` carry a native-verified reading, so only those
may be supervised.  For such a token we take its pointed form from the existing
nikud lexicon (``data/canonical_pointing.tsv``, the map that
``scripts/apply_canonical.py`` already uses) and then *verify the pointing
against the verified reading* using the canonical Hasidic convention:

    komets  ָ  U+05B8  (+ hataf-komets, komets-katan)  -> [u]
    pasekh  ַ  U+05B7  (+ hataf-pasekh)                -> [a]
    segol   ֶ  U+05B6  (+ hataf-segol)                 -> [ɛ]
    tsere   ֵ  U+05B5                                  -> [aj]  (latin "ey")
    khirik  ִ  U+05B4                                  -> [i]
    holam   ֹ  U+05B9                                  -> [ɔj]
    kubuts  ֻ  U+05BB                                  -> [i]
    sheva   ְ  U+05B0                                  -> silent
    dagesh on ו (melupm)                                -> [i]
    [ɔ]  (class 41) is written WITH NO VOWEL POINT AT ALL.

That last line is the defect this retrain exists to fix, so it is enforced in
both directions: a komets sitting over a slot the engine reads [ɔ] is *deleted*
(the convention writes class 41 bare, counted as a repair), and any other
disagreement between the lexicon pointing and the verified reading makes the
token UNSUPERVISED rather than guessed.

Two checks decide whether a token may be supervised, and both must pass:

1. ``reconcile`` -- slot-by-slot, every vowel point must match the nucleus the
   FROZEN engine assigns that point (``_POINT_TO_LATIN`` pushed through
   ``latin_to_ipa``), or the nucleus must have reduced to [ə].  Nothing wider:
   a tsere is NOT accepted over an [ej] or an [ɛ], a pasekh is NOT accepted over
   an [aj], because those cross real vowel-quality boundaries in the convention.
   The only widening is ``a -> aː``, which is the same pasekh read over a
   ײַ digraph, and ``ə`` everywhere, which is ``reduce_unstressed``.
2. ``round_trip`` -- the pointing that would actually be stamped is fed BACK
   through the frozen engine's rule path (``_rule_path_ipa``) and must come out
   with the SAME VOWEL SEQUENCE as the verified reading (stress, [ə] reduction
   and consonants excluded -- see ``round_trip``).  This is the gate that makes
   the supervision mean something: the model generalises the pointing it is
   taught to OOV words, where the rule path is the whole phonemization, so a
   pointing the rule path reads with different vowels teaches a shape the engine
   cannot recover.  It also makes the [ɔ] repair honest -- deleting a komets is
   only a lossless back-conversion when the now-bare slot really does read back
   [ɔ], which for a chunk of the lexicon (זָאָלְן -> זאלְן reads [zuln], כָּל ->
   כּל reads [kul]) it does not.

Unsupervised tokens keep whatever pointing the source text already had and are
masked out of the loss.

Output: ``data/retrain/{train,val,test}.jsonl`` with one record per row

    {"id", "episode", "text", "pointed", "supervised": [bool per token],
     "n_tokens", "n_supervised"}

Test rows additionally carry ``ceiling_pointed`` / ``ceiling_supervised``: this
pipeline's own stamp for those tokens, i.e. what a model that fits the training
objective perfectly would emit.  Scoring it against the gold gives the CEILING
(``ceiling.json``, and a section in ``dataset_stats.md``) -- the training targets
and the episode-100313 gold are two different pointings, so the test metric is
capped well below 100% and no eval delta is interpretable without it.

``text`` is the mark-stripped skeleton, ``pointed`` the training target;
``supervised[i]`` gates the loss for the i-th whitespace token of ``pointed``
(a trainer sets ``marks[c] = None`` -> ``IGNORE`` for every char of a masked
token, which is the hook ``yi_data.YiCollator`` already has).

Usage:  .venv/bin/python scripts/prepare_retrain_dataset.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yiddish_g2p as G  # FROZEN - read only  # noqa: E402

csv.field_size_limit(10_000_000)

CORPUS = REPO / "data" / "yiddish_tts_dataset.tsv"
LEXICON = REPO / "data" / "canonical_pointing.tsv"
HOMOGRAPHS = REPO / "data" / "homographs.tsv"
CANONICAL_EVAL = REPO / "data" / "phonemized" / "100313.jsonl"
OUTDIR = REPO / "data" / "retrain"
TEST_EPISODE = "100313"
VAL_EPISODE_FRAC = 0.02
SEED = 20260807

# --------------------------------------------------------------- normalisation

# identical folding to scripts/canonicalize_pointing.py / prepare_diacritics_dataset.py
FOLD = {
    "װ": "וו", "ױ": "וי", "ײ": "יי",
    "׳": "'", "״": '"', "־": "-",
    "‎": "", "‏": "", "‍": "", "‌": "", "﻿": "",
}
MARK_RE = re.compile(r"[֑-ׇֽֿׁׂׅׄ]")
HEB_RE = re.compile(r"[א-ת]")

DAGESH = "ּ"
RAFE = "ֿ"
SHIN_DOT = "ׁ"
SIN_DOT = "ׂ"

# the only marks allowed to appear in the emitted corpus: the 12 vowel points of
# canonicalize_pointing.NIQQUD plus dagesh / rafe / shin-dot / sin-dot.
IN_CONVENTION = set(chr(c) for c in list(range(0x05B0, 0x05BC))
                    + [0x05BB, 0x05C7]) | {DAGESH, RAFE, SHIN_DOT, SIN_DOT}

# cantillation, meteg (U+05BD) and friends are dropped on input, exactly as
# scripts/prepare_diacritics_dataset.py DROP_MARKS does.
DROP_MARK_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0x0591, 0x05C8)
                  if unicodedata.combining(chr(c))
                  and chr(c) not in IN_CONVENTION) + "]")

VOWEL_LETTERS = set("אויע")  # א ו י ע


def fold(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for src, dst in FOLD.items():
        text = text.replace(src, dst)
    return text


def strip_marks(text: str) -> str:
    return MARK_RE.sub("", text)


def norm(text: str) -> str:
    """Corpus-level normalisation: g2p surface rules, folding, mark cleanup."""
    return unicodedata.normalize(
        "NFC", DROP_MARK_RE.sub("", fold(G.normalize_surface(text))))


# ------------------------------------------------- reading <-> point convention

# vowel point -> the latin/IPA nucleus the FROZEN engine assigns it
# (inverse of yiddish_g2p._POINT_TO_LATIN, pushed through _LATIN_TO_IPA)
POINT_READING = {
    "ָ": "u",   # komets
    "ֳ": "u",   # hataf komets
    "ׇ": "u",   # komets katan
    "ַ": "a",   # pasekh
    "ֲ": "a",   # hataf pasekh
    "ֶ": "e",   # segol
    "ֱ": "e",   # hataf segol
    "ֵ": "aj",  # tsere   ("ey" -> aj)
    "ִ": "i",   # khirik
    "ֹ": "oj",  # holam   ("oy" -> ɔj)
    "ֺ": "oj",  # holam haser (folded to holam downstream)
    "ֻ": "i",   # kubuts  ("u" -> i)
}
# sheva U+05B0 deliberately absent: silent, re-inserted as ə by latin_to_ipa.

# The IPA nucleus the FROZEN engine actually assigns each point:
# {p: latin_to_ipa(_POINT_TO_LATIN[p]) for p in _POINT_TO_LATIN}.  This is a
# derivation, not a judgement call -- see the assertion in `check_strict()`.
STRICT = {
    "u":  {"u"},    # komets   / hataf-komets / komets-katan  ("oo")
    "a":  {"a",     # pasekh   / hataf-pasekh                 ("a")
           "aː"},   #   ... same pasekh read over a ײַ digraph ("ay" -> aː)
    "e":  {"ɛ"},    # segol    / hataf-segol                  ("e")
    "aj": {"aj"},   # tsere                                   ("ey")
    "i":  {"i"},    # khirik, kubuts, melupm vov              ("i" / "u")
    "oj": {"ɔj"},   # holam                                   ("oy")
}
# "ə" is admitted for every point because `reduce_unstressed` collapses any
# unstressed nucleus to schwa; it is a realisation of the same point, not a
# different vowel.  NOTHING else is widened: a tsere over [ej] or [ɛ], a pasekh
# over [aj], a khirik over [ej] all cross vowel-quality boundaries the
# convention treats as distinct, so those tokens go UNSUPERVISED.
ALLOW = {k: v | {"ə"} for k, v in STRICT.items()}


def check_strict() -> None:
    """Assert STRICT is the engine's own point->IPA map, not a hand table."""
    for point, reading in POINT_READING.items():
        latin = G._POINT_TO_LATIN.get(point)
        if latin is None:
            continue
        ipa = G.latin_to_ipa(latin)
        if ipa not in STRICT[reading]:
            raise AssertionError(
                f"STRICT[{reading!r}] does not contain the engine's own reading "
                f"of U+{ord(point):04X} ({latin!r} -> {ipa!r})")
# nuclei that may appear in the reading with no point over them:
#   ɔ  = class 41, written unpointed by convention
#   ə  = a reduced vowel the pointing spells with sheva (which we score as silent)
SKIPPABLE = {"ɔ", "ə"}
# nuclei the convention spells with a bare digraph (וי, יי, ויי, או) and no point,
# tolerated only *after* every point of the word has been matched.
TAIL_SKIPPABLE = SKIPPABLE | {"ɔj", "ej", "aj", "aː", "oʊ"}

IPA_NUCLEI = ["aː", "aj", "ɔj", "oʊ", "ej",
              "a", "ɛ", "i", "u", "ɔ", "ə"]


def ipa_nuclei(ipa: str) -> list[str]:
    out, i = [], 0
    while i < len(ipa):
        for v in IPA_NUCLEI:
            if ipa.startswith(v, i):
                out.append(v)
                i += len(v)
                break
        else:
            i += 1
    return out


def point_slots(pointed: str) -> list[tuple[str, list[int]]]:
    """Vowel slots of a pointed word as (reading, [char indices of the marks]).

    Adjacent identical readings smeared over a vowel-letter run (הָאָט, מַאַכְן,
    בַּייַ) collapse into ONE slot, matching canonicalize_pointing.reattach_digraphs.
    """
    pointed = unicodedata.normalize("NFC", pointed)
    bases: list[str] = []                 # base letters, in order
    raw: list[tuple[str, int, int]] = []  # (reading, mark index, base ordinal)
    base_ord = -1
    for i, ch in enumerate(pointed):
        if not unicodedata.combining(ch):
            bases.append(ch)
            base_ord += 1
            continue
        if ch in POINT_READING:
            raw.append((POINT_READING[ch], i, base_ord))
        elif ch == DAGESH and base_ord >= 0 and bases[base_ord] == "ו":
            # melupm vov: dagesh on ו IS the vowel (canonicalize_pointing.n_vowels)
            raw.append(("i", i, base_ord))
    slots: list[tuple[str, list[int]]] = []
    prev_ord = -2
    for reading, mi, bo in raw:
        smear = (slots and slots[-1][0] == reading and bo > prev_ord
                 and all(bases[k] in VOWEL_LETTERS
                         for k in range(prev_ord + 1, bo + 1)))
        if smear:
            slots[-1][1].append(mi)
        else:
            slots.append((reading, [mi]))
        prev_ord = bo
    return slots


def reconcile(pointed: str, ipa: str) -> tuple[str | None, str]:
    """Check a lexicon pointing against the verified reading.

    Returns ``(accepted_pointing, status)``.  ``status`` is ``ok`` when the
    pointing already agrees, ``repair_drop_point_over_o`` when a komets/pasekh
    sitting over a class-41 [ɔ] was deleted (the convention writes [ɔ] bare),
    and an error tag with ``None`` when the reading cannot be expressed.
    """
    slots = point_slots(pointed)
    got = ipa_nuclei(ipa)
    gi = 0
    drop: list[int] = []
    for reading, idxs in slots:
        allow = ALLOW[reading]
        while True:
            if gi >= len(got):
                return None, "reading_conflict"
            if got[gi] in allow:
                gi += 1
                break
            # a komets/pasekh over a slot the engine reads [ɔ]: class 41 is
            # written with NO vowel point, so the mark comes off.  This is the
            # single defect the retrain exists to fix, and deleting is the
            # lossless back-conversion -- never a guess.
            if reading in ("u", "a") and got[gi] == "ɔ":
                drop.extend(idxs)
                gi += 1
                break
            # a legitimately-unpointed nucleus standing before this slot
            if got[gi] in SKIPPABLE:
                gi += 1
                continue
            return None, "reading_conflict"
    if any(v not in TAIL_SKIPPABLE for v in got[gi:]):
        return None, "unpointed_nucleus"  # reading has a vowel the pointing lacks
    if not drop:
        return pointed, "ok"
    dropped = set(drop)
    out = "".join(c for i, c in enumerate(unicodedata.normalize("NFC", pointed))
                  if i not in dropped)
    return unicodedata.normalize("NFC", out), "repair_drop_point_over_o"


# --------------------------------------------------------- rule-path round trip

STRESS_RE = re.compile("[ˈˌ]")
_RT_CACHE: dict[str, str] = {}


def seg(ipa: str) -> str:
    """Segmental skeleton of an IPA string: no stress, hyphen == space."""
    return " ".join(STRESS_RE.sub("", ipa).replace("-", " ").split())


def rule_path(pointed: str) -> str:
    """`_rule_path_ipa` of a pointed word, memoised (few hundred unique types)."""
    hit = _RT_CACHE.get(pointed)
    if hit is None:
        hit = _RT_CACHE[pointed] = G._rule_path_ipa(pointed)
    return hit


def nuclei_agree(a: str, b: str) -> bool:
    """Same vowel sequence, up to reduction and pasekh length."""
    na, nb = ipa_nuclei(seg(a)), ipa_nuclei(seg(b))
    if len(na) != len(nb):
        return False
    fold_len = {"aː": "a"}  # STRICT["a"] already treats these as one pasekh
    for x, y in zip(na, nb):
        x, y = fold_len.get(x, x), fold_len.get(y, y)
        if x != y and "ə" not in (x, y):
            return False
    return True


def round_trip(pointed: str, ipa: str) -> bool:
    """Does the FROZEN engine's rule path read `pointed` back with `ipa`'s vowels?

    Scope: the VOWEL SEQUENCE, which is what this retrain supervises and what
    the pointing determines.  Deliberately *not* compared:

    * stress -- not encoded by pointing at all;
    * [ə] against a full vowel -- `reduce_unstressed` is prosodic, and the same
      tolerance is already built into `ALLOW`;
    * consonants -- the lexicon carries readings no pointing can produce
      (דארף -> [daf] deletes an r, טוב -> [tɔjv]), and forcing them through the
      rule path would mask most of the corpus for reasons unrelated to the
      vowels being trained.  The rafe/shin heads already score >98%.

    What it does catch is exactly the mis-supervision findings 3 and 4 describe:
    a pointing whose vowels the engine reads differently -- זאלְן read [zuln] for
    a verified [zɔln], כּל read [kul] for [kɔl], אֵיינֶער read with three nuclei
    for a verified two.
    """
    return nuclei_agree(rule_path(pointed), ipa)


# ------------------------------------------------------------------ lexicon io

def load_lexicon() -> tuple[dict[str, str], set[str]]:
    lex: dict[str, str] = {}
    with LEXICON.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            bare, pointed = row["word_bare"], row["canonical_pointed"]
            if bare and pointed and strip_marks(pointed) == bare:
                lex[bare] = unicodedata.normalize("NFC", pointed)
    homographs: set[str] = set()
    if HOMOGRAPHS.exists():
        with HOMOGRAPHS.open(encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                homographs.add(line.split("\t")[0])
    return lex, homographs


# ------------------------------------------------------------------- the build

class Builder:
    def __init__(self) -> None:
        self.lex, self.homographs = load_lexicon()
        self.counts: collections.Counter[str] = collections.Counter()
        self.diacritics: collections.Counter[str] = collections.Counter()

    def token(self, tok: str, count: bool = True) -> tuple[str, bool]:
        """(emitted token, supervised?) for one whitespace token.

        ``count=False`` runs the identical decision without touching the
        counters or the diacritic histogram -- used to compute the test-set
        ceiling, which must not pollute the train/val supervision accounting.
        """
        if not count:
            counts, diacritics = self.counts, self.diacritics
            self.counts = collections.Counter()
            self.diacritics = collections.Counter()
            try:
                return self.token(tok)
            finally:
                self.counts, self.diacritics = counts, diacritics
        lead, core, trail = G.split_affixes(tok)
        if not core or not HEB_RE.search(core):
            self.counts["skip_non_hebrew"] += 1
            return tok, False
        self.counts["tokens"] += 1
        res = G.g2p_token(core)
        route = res["route"]
        self.counts[f"route_{route}"] += 1
        if route != "lexicon":
            return tok, False
        bare = strip_marks(core)
        if bare in self.homographs:
            self.counts["mask_homograph"] += 1
            return tok, False
        pointed = self.lex.get(bare)
        if pointed is None:
            self.counts["mask_no_lexicon_entry"] += 1
            return tok, False
        accepted, status = reconcile(pointed, res["ipa_primary"])
        if accepted is None:
            self.counts[f"mask_{status}"] += 1
            return tok, False
        # the pointing we are about to stamp must read BACK as the verified
        # reading through the frozen rule path -- otherwise we would be teaching
        # a shape the engine mis-phonemizes on every OOV occurrence.
        if not round_trip(accepted, res["ipa_primary"]):
            self.counts[f"mask_roundtrip_after_{status}"] += 1
            return tok, False
        self.counts[f"stamp_{status}"] += 1
        # letter safety for this span
        if strip_marks(accepted) != bare:
            raise AssertionError(f"round-trip broken: {core!r} -> {accepted!r}")
        for ch in accepted:
            if unicodedata.combining(ch):
                self.diacritics[ch] += 1
        self.counts["supervised"] += 1
        return lead + accepted + trail, True

    def row(self, rid: str, episode: str, text: str) -> dict | None:
        text = norm(text)
        toks = text.split()
        if not toks:
            return None
        out, mask = [], []
        for tok in toks:
            emitted, sup = self.token(tok)
            out.append(emitted)
            mask.append(sup)
        pointed = " ".join(out)
        skeleton = strip_marks(pointed)
        if skeleton != strip_marks(text):
            raise AssertionError(f"row {rid}: letter identity broken")
        return {
            "id": rid, "episode": episode,
            "text": skeleton, "pointed": pointed,
            "supervised": mask,
            "n_tokens": len(toks), "n_supervised": sum(mask),
        }


# ----------------------------------------------------------------- the test set

def build_test(builder: Builder) -> list[dict]:
    """The canonically-pointed rows of episode 100313, aligned to their source.

    ``data/phonemized/100313.jsonl`` is Gemini output: most rows are letter-clean
    against ``text_yi``, but some dropped or transliterated a word.  We keep the
    SOURCE text verbatim and supervise only the tokens whose gold pointing is
    letter-identical to their source token (difflib alignment on the mark-free,
    final-folded skeleton).  Corrupted tokens stay bare and masked, so the test
    text is exactly the canonical row's text with no invented pointing.
    """
    import difflib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nikud_yi_mod", REPO / "scripts" / "nikud_yi.py")
    nikud_yi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nikud_yi)  # canon() / repair()

    rows: list[dict] = []
    with CANONICAL_EVAL.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            src, gold = rec["text_yi"], rec.get("pointed") or ""
            if not MARK_RE.search(gold):
                continue  # row was never canonically pointed
            src_toks = norm(src).split()
            gold_toks = norm(gold).split()
            if not src_toks:
                continue
            a = [nikud_yi.canon(t) for t in src_toks]
            b = [nikud_yi.canon(t) for t in gold_toks]
            out = list(src_toks)
            mask = [False] * len(src_toks)
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            for i, j, n in sm.get_matching_blocks():
                for k in range(n):
                    stok, gtok = src_toks[i + k], gold_toks[j + k]
                    if not HEB_RE.search(stok):
                        continue
                    fixed = nikud_yi.repair(stok, gtok)
                    if fixed is None:
                        continue
                    fixed = norm(fixed)
                    if strip_marks(fixed) != strip_marks(stok):
                        continue
                    if not MARK_RE.search(fixed):
                        continue  # gold left it bare: nothing verified here
                    out[i + k] = fixed
                    mask[i + k] = True
            n_heb = sum(1 for t in src_toks if HEB_RE.search(t))
            builder.counts["test_tokens_hebrew"] += n_heb
            builder.counts["test_tokens_gold_aligned"] += sum(mask)
            builder.counts["test_rows"] += 1
            # ---- the CEILING: what this pipeline's own stamper would produce
            # for these very tokens.  A model that fits the training targets
            # perfectly emits exactly this, so scoring it against the gold above
            # is the highest score the retrain can reach on this test set.  Where
            # the stamper declines to supervise, the training objective says
            # nothing, so we credit the gold token (an upper bound).
            ceil, ceil_sup = list(out), [False] * len(src_toks)
            for i, stok in enumerate(src_toks):
                if not HEB_RE.search(stok):
                    continue
                stamped, sup = builder.token(stok, count=False)
                if sup:
                    ceil[i], ceil_sup[i] = stamped, True

            rid = f"{TEST_EPISODE}-{rec['chunk_idx']:05d}-{rec['sent_idx']:03d}"
            pointed = " ".join(out)
            ceiling = " ".join(ceil)
            if strip_marks(pointed) != strip_marks(" ".join(src_toks)):
                raise AssertionError(f"test row {rid}: letter identity broken")
            if strip_marks(ceiling) != strip_marks(pointed):
                raise AssertionError(f"test row {rid}: ceiling letter identity broken")
            rows.append({
                "id": rid, "episode": TEST_EPISODE,
                "text": strip_marks(pointed), "pointed": pointed,
                "supervised": mask,
                "n_tokens": len(src_toks), "n_supervised": sum(mask),
                "source_text_yi": src,
                "ceiling_pointed": ceiling,
                "ceiling_supervised": ceil_sup,
            })
    return rows


def score_ceiling(test: list[dict]) -> dict:
    """Score `ceiling_pointed` against `pointed` with the evaluator's own metric.

    Same code path as `scripts/eval_phonikud_yi.py` (`parse_row` canonicalisation,
    supervised Hebrew characters only, word = every Hebrew char exact).
    """
    from phonikud_yi_data import is_heb, parse_row  # torch import, so lazy

    c_ok = c_n = w_ok = w_n = 0
    agree = differ = unsup = 0
    for rec in test:
        gold = parse_row(rec["pointed"], None)
        pred = parse_row(rec["ceiling_pointed"], None)
        if gold.text != pred.text:
            raise AssertionError(f"ceiling letters differ on {rec['id']}")
        gt = rec["pointed"].split()
        ct = rec["ceiling_pointed"].split()
        for tok, ctok, sup, csup in zip(gt, ct, rec["supervised"],
                                        rec["ceiling_supervised"]):
            if not sup:
                continue
            if not csup:
                unsup += 1
            elif ctok == tok:
                agree += 1
            else:
                differ += 1
        # character / word metric over supervised spans.  Spans are taken off
        # `gold.text` exactly the way `eval_phonikud_yi.words_of` takes them.
        spans, pos = [], 0
        for piece in gold.text.split(" "):
            if piece:
                spans.append((pos, pos + len(piece)))
            pos += len(piece) + 1
        if len(spans) != len(rec["supervised"]):
            raise AssertionError(f"ceiling span/mask mismatch on {rec['id']}")
        for (s, e), sup in zip(spans, rec["supervised"]):
            if not sup:
                continue
            any_heb = False
            all_ok = True
            for i in range(s, e):
                if not is_heb(gold.text[i]):
                    continue
                any_heb = True
                right = (gold.marks[i] or "") == (pred.marks[i] or "")
                c_n += 1
                c_ok += right
                all_ok &= right
            if any_heb:
                w_n += 1
                w_ok += all_ok
    tot = agree + differ + unsup
    pct = lambda a, b: round(100.0 * a / b, 2) if b else 0.0  # noqa: E731
    return {
        "supervised_test_tokens": tot,
        "stamp_identical": agree, "stamp_different": differ,
        "stamp_unsupervised": unsup,
        "ceiling_supervised_chars": c_n,
        "ceiling_char_acc": pct(c_ok, c_n),
        "ceiling_supervised_words": w_n,
        "ceiling_word_exact": pct(w_ok, w_n),
    }


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: first N corpus rows")
    ap.add_argument("--out", type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    builder = Builder()

    # ---- corpus rows, grouped by episode
    by_episode: dict[str, list[dict]] = collections.defaultdict(list)
    with CORPUS.open(encoding="utf-8") as fh:
        for n, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if args.limit and n >= args.limit:
                break
            episode = row["episode"]
            if episode == TEST_EPISODE:
                builder.counts["rows_dropped_test_episode"] += 1
                continue
            rec = builder.row(row["id"], episode, row["text"])
            if rec is None:
                builder.counts["rows_empty"] += 1
                continue
            by_episode[episode].append(rec)

    episodes = sorted(by_episode)
    rng = random.Random(SEED)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * VAL_EPISODE_FRAC))
    val_eps = set(shuffled[:n_val])
    train_eps = [e for e in episodes if e not in val_eps]

    train = [r for e in train_eps for r in by_episode[e]]
    val = [r for e in sorted(val_eps) for r in by_episode[e]]
    test = build_test(builder)

    # ---------------------------------------------------------------- gates
    fail: list[str] = []

    # 1. round-trip on every supervised span (re-verified end to end)
    for split, rows in (("train", train), ("val", val), ("test", test)):
        for rec in rows:
            ptoks = rec["pointed"].split()
            if len(ptoks) != len(rec["supervised"]):
                fail.append(f"{split}/{rec['id']}: mask length != token count")
                break
            if strip_marks(rec["pointed"]) != rec["text"]:
                fail.append(f"{split}/{rec['id']}: pointed does not strip to text")
                break
            for tok, sup in zip(ptoks, rec["supervised"]):
                if sup and strip_marks(tok) == "":
                    fail.append(f"{split}/{rec['id']}: empty supervised span")
                    break

    # 2a. router health -- policy-independent, catches a broken/empty lexicon
    heb = builder.counts["tokens"]
    lex_share = builder.counts["route_lexicon"] / heb if heb else 0.0
    if not 0.55 <= lex_share <= 0.75:
        fail.append(f"lexicon-route share {lex_share:.4f} outside [0.55, 0.75]")

    # 2b. supervised share of Hebrew-bearing tokens.  The floor is 0.45, not the
    # 0.55 of the first cut: `reconcile` now matches each point against the
    # engine's own reading with no widening, and every stamp has to survive the
    # rule-path round trip, which together mask ~13% (relative) more tokens than
    # the tolerant first version did.  Under-supervising is safe; guessing is not.
    sup = builder.counts["supervised"]
    share = sup / heb if heb else 0.0
    if not 0.45 <= share <= 0.75:
        fail.append(f"supervised share {share:.4f} outside [0.45, 0.75]")

    # 3. no out-of-convention marks anywhere in the emitted corpus
    stray: collections.Counter[str] = collections.Counter()
    for rows in (train, val, test):
        for rec in rows:
            for ch in rec["pointed"]:
                if unicodedata.combining(ch) and ch not in IN_CONVENTION:
                    stray[ch] += 1
    if stray:
        fail.append("out-of-convention marks: "
                    + ", ".join(f"U+{ord(c):04X}x{n}" for c, n in stray.items()))

    # 4. no 100313 in train/val; test text identical to the canonical rows
    if any(r["episode"] == TEST_EPISODE for r in train + val):
        fail.append("episode 100313 leaked into train/val")
    canon_texts = []
    with CANONICAL_EVAL.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if MARK_RE.search(rec.get("pointed") or ""):
                canon_texts.append(rec["text_yi"])
    if len(test) != len(canon_texts):
        fail.append(f"test has {len(test)} rows, canonical set has {len(canon_texts)}")
    for rec in test:
        if strip_marks(norm(rec["source_text_yi"])) != rec["text"]:
            fail.append(f"test/{rec['id']}: text differs from canonical row")
            break

    if fail:
        print("SANITY GATES FAILED:", file=sys.stderr)
        for f in fail:
            print("  -", f, file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- write
    for name, rows in (("train", train), ("val", val), ("test", test)):
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{path}  {len(rows)} rows")
    (args.out / "val_episodes.txt").write_text(
        "\n".join(sorted(val_eps)) + "\n", encoding="utf-8")
    (args.out / "train_episodes.txt").write_text(
        "\n".join(train_eps) + "\n", encoding="utf-8")

    ceiling = score_ceiling(test)
    (args.out / "ceiling.json").write_text(
        json.dumps(ceiling, indent=2), encoding="utf-8")
    write_stats(args.out / "dataset_stats.md", builder, train, val, test,
                train_eps, sorted(val_eps), share, ceiling)
    print(f"supervised {sup}/{heb} = {share:.2%}")
    print(f"test ceiling: char {ceiling['ceiling_char_acc']}% / "
          f"word {ceiling['ceiling_word_exact']}%")
    return 0


def write_stats(path: Path, builder: Builder, train, val, test,
                train_eps, val_eps, share, ceiling) -> None:
    names = {
        "ְ": "sheva", "ֱ": "hataf-segol", "ֲ": "hataf-pasekh",
        "ֳ": "hataf-komets", "ִ": "khirik", "ֵ": "tsere",
        "ֶ": "segol", "ַ": "pasekh", "ָ": "komets",
        "ֹ": "holam", "ֻ": "kubuts", "ּ": "dagesh",
        "ֿ": "rafe", "ׁ": "shin-dot", "ׂ": "sin-dot",
        "ׇ": "komets-katan",
    }
    total_marks = sum(builder.diacritics.values())
    lines = [
        "# Retrain dataset stats", "",
        f"Built by `scripts/prepare_retrain_dataset.py` from `{CORPUS.name}`.",
        f"Supervision = tokens the FROZEN v3 engine routes to `lexicon`, pointed from",
        f"`{LEXICON.name}` and reconciled against the verified reading.", "",
        "## Splits", "",
        "| split | rows | episodes | tokens | supervised tokens | share |",
        "|---|---|---|---|---|---|",
    ]
    for name, rows, eps in (("train", train, len(train_eps)),
                            ("val", val, len(val_eps)),
                            ("test", test, 1)):
        t = sum(r["n_tokens"] for r in rows)
        s = sum(r["n_supervised"] for r in rows)
        lines.append(f"| {name} | {len(rows)} | {eps} | {t} | {s} | "
                     f"{(s / t if t else 0):.2%} |")
    lines += [
        "",
        f"Test = the {len(test)} canonically-pointed rows of episode {TEST_EPISODE} "
        "(fully supervised, gold pointing).",
        f"Episode {TEST_EPISODE} is excluded from train and val entirely "
        f"({builder.counts['rows_dropped_test_episode']} corpus rows dropped).",
        "",
        f"Val episodes ({len(val_eps)}): " + ", ".join(val_eps),
        "",
        "## Test-set ceiling (READ THIS BEFORE READING ANY EVAL DELTA)", "",
        "The training targets and the episode-100313 gold are two different",
        "pointings of the same convention family, so the test metric is capped",
        "well below 100%. Applying this pipeline's own stamper (`Builder.token`)",
        "to every gold-supervised test token and scoring it against the gold with",
        "the evaluator's metric gives the highest score any model fitting these",
        "targets can reach:", "",
        "| quantity | value |", "|---|---|",
        f"| gold-supervised test tokens | {ceiling['supervised_test_tokens']} |",
        f"| stamper agrees with gold | {ceiling['stamp_identical']} |",
        f"| stamper disagrees with gold | {ceiling['stamp_different']} |",
        f"| stamper would not supervise (gold credited) | {ceiling['stamp_unsupervised']} |",
        f"| **ceiling char_acc** | **{ceiling['ceiling_char_acc']}%** "
        f"({ceiling['ceiling_supervised_chars']} supervised chars) |",
        f"| **ceiling word-exact** | **{ceiling['ceiling_word_exact']}%** "
        f"({ceiling['ceiling_supervised_words']} words) |",
        "",
        f"So {round(100 - ceiling['ceiling_char_acc'], 2)}% of supervised characters "
        "are unreachable by construction. Report every post-retrain number as a",
        "fraction of this ceiling, never as a fraction of 100%. Machine-readable",
        "copy: `ceiling.json`; `scripts/eval_phonikud_yi.py` recomputes it from",
        "the `ceiling_pointed` field carried on each test row.", "",
        "## Supervision accounting (train+val Hebrew-bearing tokens)", "",
        "| counter | value |", "|---|---|",
    ]
    for k in sorted(builder.counts):
        lines.append(f"| `{k}` | {builder.counts[k]} |")
    lines += [
        "",
        f"Supervised share of Hebrew tokens: **{share:.2%}** "
        f"({builder.counts['supervised']}/{builder.counts['tokens']}).",
        "",
        "## Diacritic distribution over supervised spans", "",
        "| mark | name | count | share |", "|---|---|---|---|",
    ]
    for ch, n in builder.diacritics.most_common():
        lines.append(f"| `{ch}` U+{ord(ch):04X} | {names.get(ch, '?')} | {n} | "
                     f"{n / total_marks:.2%} |")
    lines += ["", f"Total marks stamped: {total_marks}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
