#!/usr/bin/env python
"""Generate data/audio_vowel_lk.py from the PhoneticXeus positional tags.

The §2 alef-default reads an unpointed א as /a/ and flags the word LOW; the
corpus audio can correct the vowel the same way the pe sweep corrected the
letter. The fold policy is deliberately asymmetric:

  * a slot qualifies with >= MIN_VOTES aligned vowel votes and a >= 2/3
    modal phone that differs from the engine;
  * the modal phone must be in CLEAN_TARGETS — phones the recognizer is
    never observed to produce spuriously (docs/xeus_to_yiddish_map.md: its
    biases run ej->ɛ, aː->a, ɔ->a, u->ɔ, ə->ɛ/a/i — every *output* of a bias
    is contaminated as evidence, every phone absent from the output side is
    trustworthy). Today that set is {u}: exactly the komets vowel the
    a-default misses, and the one direction recognizer noise cannot fake;
  * the word must still be on the rule path at LOW with an alef-default
    reason — gold, legacy, audio-pe and MWE verdicts are never touched;
  * STRESS IS NEVER MOVED: the substitution edits vowel characters in the
    engine's own stressed primary, in place. The ˈ mark keeps its position
    (it precedes the vowel; jˈarʦajt -> jˈurʦajt).

Slots that clear the vote bar but fail the CLEAN_TARGETS test (modal a<->ɔ
etc., where recognizer bias could be the whole story) are written to
data/audio_lexicon/vowel_queue.tsv for the native-verification batch
instead of being folded.

Usage: .venv/bin/python scripts/build_audio_vowel_lexicon.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

TAGS = REPO / "data" / "audio_lexicon" / "pe_sweep_tags.jsonl"
OUT = REPO / "data" / "audio_vowel_lk.py"
QUEUE = REPO / "data" / "audio_lexicon" / "vowel_queue.tsv"

MIN_VOTES = 3
MAJORITY = 2 / 3
CLEAN_TARGETS = {"u"}

MULTIS = ("aː", "ej", "aj", "ɔj", "oʊ")

HEADER = '''"""GENERATED — audio-confirmed vowel corrections for alef-default words.

Source: PhoneticXeus positional tags (scripts/xeus_pe_sweep.py /
xeus_lk_sweep.py) folded by scripts/build_audio_vowel_lexicon.py. Each entry
is the engine's own stressed rule-path reading with >= {min_votes}-vote,
>= 2/3-majority vowel slots substituted — and only toward CLEAN_TARGETS
({targets}), phones the recognizer never produces spuriously, so its known
biases cannot inject a vowel. Stress marks are untouched by construction.

Consulted after every gold/legacy lexicon and after data/audio_pe_lk.py,
before the rule path; emitted at MED confidence, reason 'audio-vowel'.
Regenerate: .venv/bin/python scripts/build_audio_vowel_lexicon.py
Never hand-edit.
"""

AUDIO_VOWEL_LK = {{
'''


def sub_slots(stressed: str, repl: dict[int, str]) -> str:
    """Replace phone-slot -> phone in a stressed IPA string, keeping ˈ."""
    out: list[str] = []
    i = 0
    slot = 0
    while i < len(stressed):
        ch = stressed[i]
        if ch == "ˈ":
            out.append(ch)
            i += 1
            continue
        tok = None
        for m in MULTIS:
            if stressed.startswith(m, i):
                tok = m
                break
        if tok is None:
            tok = ch
        out.append(repl.get(slot, tok))
        i += len(tok)
        slot += 1
    return "".join(out)


def main() -> int:
    if OUT.exists():
        OUT.rename(OUT.with_suffix(".py.bak"))
    try:
        import importlib
        import yiddish_g2p
        importlib.reload(yiddish_g2p)
        from yiddish_g2p import g2p_token, lexicon_key
        from xeus_map import VOWELS, tokenize_g2p_ipa

        cache: dict[str, dict | None] = {}

        def target_rec(word: str) -> dict | None:
            key = lexicon_key(word)
            if key not in cache:
                rec = g2p_token(word)
                ok = (rec["route"] == "rule" and rec["confidence"] == "LOW"
                      and "alef-default" in rec["reason"])
                cache[key] = rec if ok else None
            return cache[key]

        votes: dict[tuple[str, int], Counter] = defaultdict(Counter)
        seen_g2p: dict[str, str] = {}
        surface: dict[str, str] = {}
        with TAGS.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                word = rec["word"]
                if target_rec(word) is None:
                    continue
                key = lexicon_key(word)
                g2p = rec["g2p"]
                joined = "".join(g2p)
                if key in seen_g2p and seen_g2p[key] != joined:
                    continue  # variant spelling with a different phone shape
                seen_g2p.setdefault(key, joined)
                surface.setdefault(key, word)
                for i, (g, h) in enumerate(zip(g2p, rec["heard_at"])):
                    if g in VOWELS and h != "∅":
                        votes[(key, i)][h] += 1

        folded: dict[str, dict[int, tuple[str, str, str]]] = defaultdict(dict)
        queued: list[tuple[str, int, str, str, str]] = []
        for (key, i), ctr in votes.items():
            tot = sum(ctr.values())
            modal, n = ctr.most_common(1)[0]
            engine = tokenize_g2p_ipa(seen_g2p[key])[i]
            if modal == engine or tot < MIN_VOTES or n / tot < MAJORITY:
                continue
            if modal in CLEAN_TARGETS:
                folded[key][i] = (engine, modal, f"{n}/{tot}")
            else:
                queued.append((surface[key], i, engine, modal, f"{n}/{tot}"))

        entries = []
        for key, slots in folded.items():
            rec = g2p_token(surface[key])
            new_ipa = sub_slots(rec["ipa_primary"],
                                {i: m for i, (_, m, _) in slots.items()})
            detail = ";".join(f"{i}:{e}->{m}({v})"
                              for i, (e, m, v) in sorted(slots.items()))
            entries.append((key, surface[key], new_ipa, detail))
        entries.sort(key=lambda t: t[0])
    finally:
        bak = OUT.with_suffix(".py.bak")
        if bak.exists() and not OUT.exists():
            bak.rename(OUT)

    with OUT.open("w", encoding="utf-8") as fh:
        fh.write(HEADER.format(min_votes=MIN_VOTES,
                               targets=sorted(CLEAN_TARGETS)))
        for key, word, ipa, detail in entries:
            # repr(), not an f-string quote: Yiddish keys carry an apostrophe
            # (מורא'דיקע, אויפ'ן) that closes a hand-written literal early and
            # makes the whole generated module unparsable -- which the engine's
            # loader then swallows into an empty table, silently.
            fh.write(f"    {key!r}: {{\"ipa\": {ipa!r},"
                     f" \"slots\": {detail!r}}},  # {word}\n")
        fh.write("}\n")
    bak = OUT.with_suffix(".py.bak")
    if bak.exists():
        bak.unlink()

    queued.sort()
    with QUEUE.open("w", encoding="utf-8") as fh:
        fh.write("word\tslot\tengine\theard_modal\tvotes\n")
        for row in queued:
            fh.write("\t".join(str(x) for x in row) + "\n")
    print(f"{len(entries)} entries -> {OUT}")
    print(f"{len(queued)} contested/unsafe slots -> {QUEUE}")
    for key, word, ipa, detail in entries[:15]:
        print(f"  {word}: {ipa}  [{detail}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
