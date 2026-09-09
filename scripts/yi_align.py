#!/usr/bin/env python3
"""Letter-level alignment of Yiddish words to their IPA — the ReNikud frame.

ReNikud (Melichov, Kolani, Alper 2026) casts Hebrew G2P as per-letter
classification: every letter predicts a (consonant, vowel, stress) triple, and
a constrained DP aligner turns a (word, IPA) pair into one such chunk per
letter for training. This module is that aligner for Yiddish.

What is different from Hebrew, and how it is handled:

  * Yiddish writes its vowels. A vowel letter (א ע י ו and the digraphs יי וי)
    takes consonant ∅ and a vowel; a consonant letter in a Germanic word takes
    a consonant and vowel ∅. Loshn-koydesh words behave like Hebrew — a letter
    may carry both (שבת → ש=ʃa ב=bə ת=s).
  * Digraphs are two letters, one phone: וו→v, יי→aj/ej/aː, וי→ɔj/oʊ, זש→ʒ,
    טש→ʧ, דזש→ʤ. The phone attaches to the FIRST letter; the following
    letter(s) are ∅. The aligner tries the digraph reading before the single
    letter reading at each position.
  * Stress: the ˈ mark precedes the stressed vowel in the engine/gold IPA; a
    letter whose chunk contains ˈ gets stress=1.
  * The pointed forms in the corpus are stripped first; pointing never enters
    the alignment.

Label vocabulary (docs/yiddish_phoneme_set.md §1):
  consonants  ∅ b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ
  vowels      ∅ a aː ɛ ə i u ɔ ej aj ɔj oʊ
  stress      0 / 1

`align_word(word, ipa)` → list of (letter, chunk) or None. `parse_chunk`
splits a chunk into the triple. `python scripts/yi_align.py` scores the
aligner on the gold lexicon and a corpus sample.
"""
from __future__ import annotations

import functools
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

CONSONANTS: tuple[str, ...] = ("b", "d", "f", "ɡ", "h", "j", "k", "l", "m", "n", "p", "r",
                               "s", "t", "v", "z", "x", "ʃ", "ʒ", "ʦ", "ʧ", "ʤ", "ŋ")
VOWELS: tuple[str, ...] = ("aː", "ej", "aj", "ɔj", "oʊ", "a", "ɛ", "ə", "i", "u", "ɔ")  # longest first
STRESS = "ˈ"
MARKERS = ("'", '"', "-")

#: What each single letter may sound like: (consonants, vowels). "" = silent.
#: A letter listed with both may carry a consonant and a vowel in one chunk
#: (the loshn-koydesh case); the aligner tries every combination.
LETTER: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "א": (("",), VOWELS + ("",)),                     # vowel letter; shtumer alef silent
    "ב": (("b", "v", "p"), VOWELS + ("",)),           # p: final devoicing
    "ג": (("ɡ", "k"), VOWELS + ("",)),
    "ד": (("d", "t", ""), VOWELS + ("",)),            # רעדט → rɛt
    "ה": (("h", ""), VOWELS + ("",)),                  # LK final ה → ə
    "ו": (("v", "f", ""), VOWELS + ("",)),
    "ז": (("z", "s", "ʒ"), VOWELS + ("",)),
    "ח": (("x",), VOWELS + ("",)),
    "ט": (("t",), VOWELS + ("",)),
    "י": (("j", ""), VOWELS + ("",)),
    "כ": (("k", "x"), VOWELS + ("",)),
    "ל": (("l", ""), VOWELS + ("",)),
    "מ": (("m",), VOWELS + ("",)),
    "נ": (("n", "ŋ"), VOWELS + ("",)),
    "ס": (("s", "z"), VOWELS + ("",)),                # ארויס-ג- → z: voicing assimilation
    "ע": (("",), VOWELS + ("",)),
    "פ": (("f", "p"), VOWELS + ("",)),
    "צ": (("ʦ", "ʧ"), VOWELS + ("",)),
    "ק": (("k",), VOWELS + ("",)),
    "ר": (("r", ""), VOWELS + ("",)),                  # דארף → daf: r drops before a consonant
    "ש": (("ʃ", "s", "ʒ"), VOWELS + ("",)),
    "ת": (("t", "s"), VOWELS + ("",)),
}
_FINALS = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
_LIGATURES = {"ײ": "יי", "ױ": "וי", "װ": "וו"}

#: Digraphs: letters → the consonant/vowel readings the pair can take.
DIGRAPHS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "דזש": (("ʤ",), ("",)),
    "וו": (("v",), VOWELS + ("",)),
    "יי": (("",), ("aj", "ej", "aː", "i")),
    "וי": (("",), ("ɔj", "oʊ", "u")),
    "זש": (("ʒ",), VOWELS + ("",)),
    "טש": (("ʧ",), VOWELS + ("",)),
    "תש": (("ʧ",), VOWELS + ("",)),                    # תשובה → ʧˈivə
    "טס": (("ʦ",), VOWELS + ("",)),                    # רעכטס → rɛxʦ
    "טצ": (("ʦ",), VOWELS + ("",)),
}


def normalize_word(word: str) -> str:
    w = unicodedata.normalize("NFD", word)
    w = "".join(ch for ch in w if unicodedata.category(ch) != "Mn")
    w = unicodedata.normalize("NFC", w)
    w = w.translate(str.maketrans(_FINALS)).translate(str.maketrans(_LIGATURES))
    return w.replace("׳", "'").replace("״", '"')


def _chunks_for(cons: tuple[str, ...], vows: tuple[str, ...], rest: str):
    """Yield (chunk, length) readings of ``rest``'s prefix as consonant+stress?+vowel."""
    for c in cons:
        if c and not rest.startswith(c):
            continue
        n = len(c)
        for stressed in (True, False):
            s = 0
            if stressed:
                if rest[n:n + 1] != STRESS:
                    continue
                s = 1
            for v in vows:
                if v:
                    if not rest[n + s:].startswith(v):
                        continue
                    L = n + s + len(v)
                elif stressed:
                    continue          # a stress mark must precede a vowel
                else:
                    L = n
                yield rest[:L], L
    # loshn-koydesh furtive pattern: the vowel precedes the consonant (ə x)
    for c in cons:
        if not c:
            continue
        for stressed in (True, False):
            s = 1 if stressed else 0
            if stressed and not rest.startswith(STRESS):
                continue
            for v in vows:
                if not v or not rest[s:].startswith(v):
                    continue
                L = s + len(v)
                if rest[L:].startswith(c):
                    yield rest[:L + len(c)], L + len(c)


def align_word(word: str, ipa: str) -> list[tuple[str, str]] | None:
    """One IPA chunk per letter of ``word`` (nikud stripped), or None."""
    w = normalize_word(word)
    if " " in ipa.strip():
        return None               # a spelled-out abbreviation: letters do not map to it
    ipa = ipa.replace(" ", "").replace("-", "")

    @functools.lru_cache(maxsize=None)
    def search(h: int, i: int):
        if h == len(w) and i == len(ipa):
            return ()
        if h == len(w):
            return None
        ch = w[h]
        if ch in MARKERS:
            r = search(h + 1, i)
            return ((ch, ""),) + r if r is not None else None
        rest = ipa[i:]
        # digraphs first (longest), phone on the first letter, rest silent
        for dg, (cons, vows) in DIGRAPHS.items():
            if w.startswith(dg, h):
                for chunk, L in _chunks_for(cons, vows, rest):
                    r = search(h + len(dg), i + L)
                    if r is not None:
                        return ((dg[0], chunk),) + tuple((c, "") for c in dg[1:]) + r
        if ch not in LETTER:
            return None
        cons, vows = LETTER[ch]
        for chunk, L in _chunks_for(cons, vows, rest):
            r = search(h + 1, i + L)
            if r is not None:
                return ((ch, chunk),) + r
        return None

    if len(w) > 40 or len(ipa) > 80:
        return None                # a URL, a number run, a pasted string: not a word
    try:
        out = search(0, 0)
    except RecursionError:
        return None
    return list(out) if out is not None else None


def parse_chunk(chunk: str) -> tuple[str, str, int]:
    """chunk → (consonant, vowel, stress)."""
    stress = int(STRESS in chunk)
    c = chunk.replace(STRESS, "")
    cons = ""
    for k in sorted(CONSONANTS, key=len, reverse=True):
        if c.startswith(k):
            cons = k
            c = c[len(k):]
            break
    if not cons:
        # vowel-first (furtive) chunk: vowel then consonant
        for k in sorted(CONSONANTS, key=len, reverse=True):
            if c.endswith(k) and c[: -len(k)] in VOWELS:
                cons = k
                c = c[: -len(k)]
                break
    vowel = c
    if vowel and vowel not in VOWELS:
        raise ValueError(f"bad chunk {chunk!r}")
    return cons, vowel, stress


def _demo() -> None:
    import csv
    import json
    import re as _re
    sys.path.insert(0, str(REPO))
    gold = list(csv.DictReader(open(REPO / "data/gold/g2p_gold_v3.csv", encoding="utf-8")))
    ok = fail = 0
    failures = []
    for r in gold:
        word = r["word"]
        for v in r["gold_ipa"].split("|"):
            v = _re.sub(r"\[.*?\]", "", v).strip()
            if " " in v or not v:
                continue          # multiword expansion of an abbreviation
            a = align_word(word, v)
            if a is None:
                fail += 1
                failures.append((word, v))
            else:
                ok += 1
    print(f"gold readings aligned: {ok}/{ok + fail} ({ok / (ok + fail):.1%})")
    for w, v in failures[:25]:
        print(f"  FAIL {w:12} {v}")
    for w, v in [("שבת", "ʃˈabəs"), ("מיט", "mit"), ("שוין", "ʃɔjn"), ("צוריק", "ʦirˈik"), ("וואס", "vus"), ("היינט", "haːnt"), ("ביינאכט", "baːnˈaxt")]:
        print(f"  {w:10} {v:10} -> {align_word(w, v)}")
    # corpus sample: engine readings
    from yiddish_g2p import g2p_token  # noqa: E402
    import re
    HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
    rows = list(csv.DictReader(open(REPO / "data/corpus/yiddish_tts_dataset.tsv", encoding="utf-8"), delimiter="\t"))
    seen = set(); ok = fail = 0; cf = []
    for r in rows[:400]:
        for w in HEB.findall(r["text"]):
            if w in seen:
                continue
            seen.add(w)
            t = g2p_token(w); t = t if isinstance(t, dict) else t.__dict__
            ipa = t["ipa_primary"] or ""
            if not ipa:
                continue
            if align_word(w, ipa) is None:
                fail += 1
                cf.append((w, ipa, t["route"]))
            else:
                ok += 1
    print(f"corpus types (first 400 chunks) aligned: {ok}/{ok + fail} ({ok / (ok + fail):.1%})")
    import collections
    print("  failures by route:", collections.Counter(r for _, _, r in cf).most_common())
    for w, v, rt in cf[:25]:
        print(f"  FAIL {w:12} {v:14} {rt}")


if __name__ == "__main__":
    _demo()
