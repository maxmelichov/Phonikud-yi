"""Map PhoneticXeus universal-IPA phones onto the closed Yiddish v3 inventory.

PhoneticXeus (changelinglab/PhoneticXeus) emits phones from a 428-symbol
universal IPA vocabulary. The Yiddish pipeline (data/g2p_spec_v3.md section 1)
allows exactly:

    vowels      a aː ɛ ə i u ɔ ej aj ɔj oʊ
    consonants  b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ

Anything the recognizer hears is folded onto its nearest member of that set so
recognizer output and G2P output become directly comparable. The folding is
deliberately many-to-one and conservative: a monophthong [e] becomes ɛ (not ej)
and [o] becomes ɔ (not oʊ), so only clear diphthongs vote for the diphthong
classes. Symbols with no sensible target (clicks, ʔ, tones) are dropped.
"""
from __future__ import annotations

import re
import unicodedata

# Diacritics / suprasegmentals the recognizer attaches that carry no vote for
# our phoneme classes: length is handled explicitly, stress from CTC output is
# unreliable, and secondary articulations are folded into the base phone.
_STRIP = re.compile(
    "[ʰʲʷʱˠˤ"   # ʰ ʲ ʷ ʱ ˠ ˤ
    "̩̯̥̤̰̼̪̻̟̠̃̊"
    "̹̜̝̞̘̙̬̈̆͜͡"
    "ˈˌˑ﻿‍.'ˑ˞]"
)

# Multi-character sequences first, longest match wins.
_SEQ: list[tuple[str, str]] = [
    # affricates
    ("ʈʂ", "ʧ"), ("dʑ", "ʤ"), ("tɕ", "ʧ"), ("tʃ", "ʧ"), ("dʒ", "ʤ"),
    ("ts", "ʦ"), ("dz", "z"),
    # diphthongs -> the four licensed ones
    ("aɪ", "aj"), ("ai", "aj"), ("ɑɪ", "aj"), ("ʌɪ", "aj"),
    ("eɪ", "ej"), ("ei", "ej"), ("ɛɪ", "ej"),
    ("ɔɪ", "ɔj"), ("ɔi", "ɔj"), ("oɪ", "ɔj"), ("oi", "ɔj"),
    ("oʊ", "oʊ"), ("ou", "oʊ"), ("aʊ", "oʊ"), ("au", "oʊ"), ("əʊ", "oʊ"),
    # long vowels
    ("aː", "aː"), ("ɑː", "aː"),
    ("iː", "i"), ("uː", "u"), ("eː", "ej"), ("oː", "oʊ"),
    ("ɛː", "ɛ"), ("ɔː", "ɔ"), ("əː", "ə"), ("yː", "i"), ("øː", "ɛ"),
]

_ONE: dict[str, str] = {
    # vowels
    "a": "a", "ä": "a", "ɑ": "a", "ɐ": "a", "ʌ": "a", "æ": "a",
    "e": "ɛ", "ɛ": "ɛ", "ø": "ɛ", "œ": "ɛ",
    "ə": "ə", "ɜ": "ə", "ɘ": "ə", "ɤ": "ə",
    "i": "i", "ɪ": "i", "y": "i", "ɨ": "i", "ʏ": "i",
    "u": "u", "ʊ": "u", "ɯ": "u", "ʉ": "u",
    "o": "ɔ", "ɔ": "ɔ", "ɒ": "ɔ",
    # consonants
    "b": "b", "β": "v", "p": "p",
    "d": "d", "ɖ": "d", "t": "t", "ʈ": "t",
    "ɡ": "ɡ", "g": "ɡ", "k": "k", "q": "k", "c": "k", "ɟ": "ɡ",
    "f": "f", "v": "v", "ʋ": "v", "w": "v",
    "h": "h", "ɦ": "h",
    "j": "j", "ʝ": "j",
    "l": "l", "ɫ": "l", "ɭ": "l", "ʎ": "l",
    "m": "m", "ɱ": "m",
    "n": "n", "ɳ": "n", "ɲ": "n",
    "ŋ": "ŋ",
    "r": "r", "ɹ": "r", "ɾ": "r", "ʀ": "r", "ʁ": "r", "ɻ": "r", "ɽ": "r",
    "s": "s", "ʂ": "ʃ", "ɕ": "ʃ", "ʃ": "ʃ",
    "z": "z", "ʐ": "ʒ", "ʑ": "ʒ", "ʒ": "ʒ",
    "x": "x", "χ": "x", "ç": "x", "ɣ": "x", "ʕ": "",
    "θ": "s", "ð": "z",
    "ʔ": "", "ʜ": "h", "ʢ": "",
    # rare/exotic symbols the recognizer can emit on non-target speech; folded
    # to the nearest Yiddish phone so a stray frame still casts a sensible vote
    "ɸ": "f", "ʍ": "v", "ɰ": "v", "ɥ": "j",
    "ɓ": "b", "ɗ": "d", "ɠ": "ɡ", "ɢ": "ɡ", "ʄ": "ʤ", "ʙ": "b",
    "ħ": "x", "ɧ": "x", "ɴ": "ŋ", "ɬ": "l", "ɮ": "l",
    "ɞ": "ə", "ɵ": "ə", "ɶ": "a",
}

# Symbols deliberately dropped (no Yiddish vote): glottal/pharyngeal stops and
# clicks. Documented so the coverage test can tell intent from accident.
DELIBERATE_DROPS = {"ʔ", "ʕ", "ʢ", "ǃ", "ǀ", "ǁ", "ʘ", "ǂ"}

VOWELS = {"a", "aː", "ɛ", "ə", "i", "u", "ɔ", "ej", "aj", "ɔj", "oʊ"}
INVENTORY = VOWELS | {
    "b", "d", "f", "ɡ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t",
    "v", "z", "x", "ʃ", "ʒ", "ʦ", "ʧ", "ʤ", "ŋ",
}


def fold_phone_string(s: str) -> list[str]:
    """Fold one recognizer phone (or phone run) into 0+ inventory phones."""
    s = unicodedata.normalize("NFD", s)
    s = _STRIP.sub("", s)
    s = unicodedata.normalize("NFC", s)
    out: list[str] = []
    i = 0
    while i < len(s):
        for src, dst in _SEQ:
            if s.startswith(src, i):
                if dst:
                    out.append(dst)
                i += len(src)
                break
        else:
            ch = s[i]
            mapped = _ONE.get(ch)
            if mapped:
                out.append(mapped)
            elif ch == "ː" and out and out[-1] == "a":
                out[-1] = "aː"
            # anything else (tones, clicks, junk) is dropped
            i += 1
    return out


def map_transcript(predicted: str) -> list[str]:
    """Map a slash-separated PhoneticXeus transcript to inventory phones.

    `predicted_transcript` looks like "a/ʃ/t/eɪ/..." with <...> special tokens.
    """
    phones: list[str] = []
    for tok in predicted.split("/"):
        tok = tok.strip()
        if not tok or (tok.startswith("<") and tok.endswith(">")):
            continue
        phones.extend(fold_phone_string(tok))
    return phones


def tokenize_g2p_ipa(ipa: str) -> list[str]:
    """Split an engine IPA word (no spaces) into inventory phone tokens."""
    out: list[str] = []
    i = 0
    multis = ("aː", "ej", "aj", "ɔj", "oʊ")
    while i < len(ipa):
        ch = ipa[i]
        if ch == "ˈ":
            i += 1
            continue
        for m in multis:
            if ipa.startswith(m, i):
                out.append(m)
                i += len(m)
                break
        else:
            if ch in INVENTORY:
                out.append(ch)
            i += 1
    return out
