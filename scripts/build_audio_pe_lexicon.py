#!/usr/bin/env python
"""Generate data/audio_pe_lk.py from the PhoneticXeus פ-sweep votes.

Input: data/audio_lexicon/pe_sweep_votes.tsv (scripts/xeus_pe_sweep.py).
A type is folded when the audio contradicts the engine's f-default hard:

    every voted pe-slot says FLIP->p, with >= MIN_P p-ish votes and
    p >= RATIO * f  (3-0, 8-1, 12-1 fold; 2-1 stays in the queue)

The entry's IPA is the engine's own reading with the voted slots flipped
f -> p: the vowels and stress were never in question, only the letter the
writer left unpointed. Entries are consulted by the router AFTER every
gold/legacy lexicon (audio never outranks a native or published verdict —
by construction these words have neither) and BEFORE the rule path, at MED
confidence with reason 'audio-pe' so they leave the LOW queue but stay
visible.

Usage: .venv/bin/python scripts/build_audio_pe_lexicon.py
Then:  .venv/bin/python scripts/test_g2p.py && scripts/test_g2p_gold.py ...
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

VOTES = REPO / "data" / "audio_lexicon" / "pe_sweep_votes.tsv"
OUT = REPO / "data" / "audio_pe_lk.py"

MIN_P = 3
RATIO = 3.0

HEADER = '''"""GENERATED — audio-confirmed /p/ readings for pe-default words.

Source: scripts/xeus_pe_sweep.py votes (PhoneticXeus over data/audio/,
positional alignment at every engine f/p slot). Each entry's every voted
pe-slot heard p-ish (p,b) with >= {min_p} votes and p >= {ratio}x f. The IPA
is the engine's rule-path reading with those slots flipped f -> p.

Consulted after all gold/legacy lexicons, before the rule path; emitted at
MED confidence, reason 'audio-pe'. Regenerate:
    .venv/bin/python scripts/xeus_pe_sweep.py --report-only
    .venv/bin/python scripts/build_audio_pe_lexicon.py
Never hand-edit.
"""

AUDIO_PE_LK = {{
'''


def main() -> int:
    # Import AFTER path setup; the engine must not see a half-written table.
    if OUT.exists():
        OUT.rename(OUT.with_suffix(".py.bak"))
    try:
        import importlib
        import yiddish_g2p
        importlib.reload(yiddish_g2p)
        from yiddish_g2p import (g2p_token, lexicon_key, _strip_points,
                                 normalize_surface)

        rows = list(csv.DictReader(VOTES.open(encoding="utf-8"), delimiter="\t"))
        folded, skipped = [], []
        for r in rows:
            if "FLIP" not in r["verdict"]:
                continue
            slots = r["votes"].split(";")
            verds = r["verdict"].split("|")
            ok = True
            flip_idx = []
            for si, (v, verd) in enumerate(zip(slots, verds)):
                p = int(v.split("/")[0].split("=")[1])
                f = int(v.split("/")[1].split("=")[1])
                if verd == "FLIP->p":
                    if p >= MIN_P and (f == 0 or p >= RATIO * f):
                        flip_idx.append(si)
                    else:
                        ok = False
                elif verd == "CONFIRMED":
                    continue
                else:  # thin / contested / FLIP->f on another slot
                    ok = False
            if not (ok and flip_idx):
                skipped.append((r["word"], r["votes"], r["verdict"]))
                continue
            # read the BARE form: pointed tokens never consult this table (the
            # router's point guard), so the entry must be the bare reading —
            # a vote-row surface that happens to carry a point (פאטייטאָ) must
            # not fold that vowel into the bare key's entry.
            rec = g2p_token(_strip_points(normalize_surface(r["word"])))
            if "pe-default" not in rec["reason"]:
                skipped.append((r["word"], r["votes"], "no-longer-pe-default"))
                continue
            ipa = rec["ipa_primary"]
            # flip the si-th f/p slot (engine emits f on defaulted slots)
            out, slot = [], 0
            for ch in ipa:
                if ch in ("f", "p"):
                    out.append("p" if slot in flip_idx else ch)
                    slot += 1
                else:
                    out.append(ch)
            folded.append((lexicon_key(r["word"]), r["word"], "".join(out),
                           r["votes"], int(r["freq"])))
    finally:
        bak = OUT.with_suffix(".py.bak")
        if bak.exists() and not OUT.exists():
            bak.rename(OUT)

    folded.sort(key=lambda t: -t[4])
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write(HEADER.format(min_p=MIN_P, ratio=RATIO))
        for key, word, ipa, votes, freq in folded:
            # repr(), not an f-string quote -- see build_audio_vowel_lexicon.py:
            # an apostrophe in a Yiddish key breaks the generated module and the
            # engine's loader turns that into an empty table without complaining.
            fh.write(f"    {key!r}: {{\"ipa\": {ipa!r}, \"votes\": {votes!r},"
                     f" \"freq\": {freq}}},  # {word}\n")
        fh.write("}\n")
    bak = OUT.with_suffix(".py.bak")
    if bak.exists():
        bak.unlink()
    print(f"{len(folded)} entries -> {OUT}; {len(skipped)} flips left queued")
    for w, v, why in skipped:
        print(f"  queued: {w} ({v}) [{why}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
