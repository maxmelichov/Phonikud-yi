"""IPA -> the Latin respelling Chezky reads and writes.

Not a transliteration standard: it is the notation the gold CSV's `gold`
column already uses (gaas, upshatzen, dernukh), with the STRESSED SYLLABLE
UPPERCASED so a native can check stress placement without knowing IPA.

Extracted from scripts/build_verification_batch_v4.py (which keeps its own
copy: it is a finished one-shot whose output is already folded into gold, so
it is left untouched). New tooling should import from here.
"""
from __future__ import annotations

CONS = [("ʦ", "ts"), ("ʧ", "tsh"), ("ʤ", "dzh"), ("ʃ", "sh"), ("ʒ", "zh"),
        ("x", "kh"), ("ŋ", "ng"), ("j", "y"), ("ɡ", "g")]
VOW = [("ej", "ey"), ("aj", "ay"), ("ɔj", "oy"), ("oʊ", "ou"), ("aː", "aa"),
       ("ɛ", "e"), ("ə", "e"), ("ɔ", "o")]
V = set("aeiouy")


def romanize(ipa: str) -> str:
    """IPA -> respelling, stressed syllable uppercased ('a pUR yur')."""
    if not ipa:
        return ""
    s = ipa
    for a, b in VOW:
        s = s.replace(a, b)
    for a, b in CONS:
        s = s.replace(a, b)
    s = s.replace("ː", "")
    # Uppercase the stressed VOWEL only (dernUkh, geBEYtn). The older builder
    # uppercased the whole syllable including its onset cluster, which reads
    # as noise on clusters: deRNUkh.
    res: list[str] = []
    i = 0
    while i < len(s):
        if s[i] != "ˈ":
            res.append(s[i])
            i += 1
            continue
        i += 1  # skip the mark; the vowel run that follows is the stress
        while i < len(s) and s[i] not in V:
            res.append(s[i])
            i += 1
        while i < len(s) and s[i] in V:
            res.append(s[i].upper())
            i += 1
    return "".join(res)
