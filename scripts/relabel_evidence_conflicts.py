#!/usr/bin/env python
"""Repair training labels that CONTRADICT a higher-authority verdict.

The evidence stampers only fill unsupervised tokens; a v1 label that is
already supervised — and wrong — survives every stamping pass and teaches the
model the error forever (מיט אַ פֿאַר יאָר kept far because the Gemini-era
label said so). This pass closes that hole for the classes where the system
now holds hard evidence:

  1. the א פאר bigram: the following פאר is the noun פּאָר (gold אפאר,
     Chezky-verified) — the token is re-pointed פּאָר outright;
  2. audio-pe words: the corpus audio voted the פ a /p/ unanimously — the
     label's rafe is swapped for a dagesh (slot-mapped);
  3. audio-vowel words: the audio voted a komets vowel — the label's pasekh
     on the א becomes a komets (or a bare א gains one).

Every repair is VALIDATED, never assumed: the candidate re-pointing must
(a) strip to the identical letters and (b) read back — through the engine,
stress ignored — as the evidence IPA. A token where no candidate validates
is left alone and counted, not guessed at.

Runs in place on data/retrain2/{train,val}.jsonl (the post-stamp dataset).

Usage: .venv/bin/python scripts/relabel_evidence_conflicts.py [--dry-run]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yiddish_g2p as G  # noqa: E402
from yiddish_g2p import lexicon_key  # noqa: E402

try:
    from data.audio_pe_lk import AUDIO_PE_LK
except ImportError:
    AUDIO_PE_LK = {}
try:
    from data.audio_vowel_lk import AUDIO_VOWEL_LK
except ImportError:
    AUDIO_VOWEL_LK = {}

DATA = REPO / "data" / "retrain2"
MARKS = re.compile(r"[֑-ׇ]")
PASEKH, KOMETS, DAGESH, RAFE = "ַ", "ָ", "ּ", "ֿ"
PUR = "פּאָר"


def seg(ipa: str) -> str:
    return ipa.replace("ˈ", "").replace(" ", "")


def strip_marks(s: str) -> str:
    return MARKS.sub("", s)


def read(pointed: str) -> str:
    try:
        return seg(G.hebrew_to_ipa(pointed, stress=True, quarantine=False))
    except Exception:  # noqa: BLE001
        return ""


def units(tok: str) -> list[str]:
    """Letter units: base char + its combining marks."""
    out: list[str] = []
    for ch in tok:
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def candidates(core: str, want: str) -> str | None:
    """A minimally-edited re-pointing of ``core`` that reads as ``want``.

    Edit moves per letter unit: on פ/ף swap RAFE->DAGESH / add DAGESH; on א
    swap PASEKH->KOMETS / add KOMETS. All subsets of applicable moves are
    tried smallest-first; the first candidate that keeps the letters and
    reads back as the evidence IPA wins."""
    us = units(core)
    moves: list[tuple[int, str]] = []
    for i, u in enumerate(us):
        base = u[0]
        if base in "פף":
            if RAFE in u:
                moves.append((i, u.replace(RAFE, DAGESH)))
            elif DAGESH not in u:
                moves.append((i, u[0] + DAGESH + u[1:]))
        if base == "א":
            if PASEKH in u:
                moves.append((i, u.replace(PASEKH, KOMETS)))
            elif KOMETS not in u:
                moves.append((i, u[0] + KOMETS + u[1:]))
    for r in range(1, min(len(moves), 3) + 1):
        for combo in itertools.combinations(moves, r):
            if len({i for i, _ in combo}) != len(combo):
                continue
            cand = list(us)
            for i, repl in combo:
                cand[i] = repl
            joined = "".join(cand)
            if strip_marks(joined) != strip_marks(core):
                continue
            if read(joined) == want:
                return joined
    return None


def split_token(tok: str) -> tuple[str, str, str]:
    heb = re.compile(r"[א-ת]")
    n = len(tok)
    i = 0
    while i < n and not heb.match(tok[i]):
        i += 1
    j = n
    while j > i:
        k = j - 1
        while k > i and (unicodedata.combining(tok[k]) or MARKS.match(tok[k])):
            k -= 1
        if heb.match(tok[k]):
            break
        j = k
    return tok[:i], tok[i:j], tok[j:]


def relabel_row(row: dict, counts: Counter) -> dict:
    ptoks = row["pointed"].split()
    mask = list(row["supervised"])
    changed = False
    prev_bare = ""
    for pos, ptok in enumerate(ptoks):
        lead, core, trail = split_token(ptok)
        bare = strip_marks(core)
        this_prev, prev_bare = prev_bare, bare
        if not bare:
            continue
        key = lexicon_key(bare)
        want = None
        why = None
        if bare == "פאר" and this_prev == "א":
            # gold evidence: stamp even a previously-unsupervised token
            want, why = "pur", "far-par"
        elif not mask[pos]:
            continue  # audio classes repair only wrong SUPERVISED labels;
            # unsupervised ones were already handled by the stamping pass
        elif key in AUDIO_PE_LK:
            want, why = seg(AUDIO_PE_LK[key]["ipa"]), "audio-pe"
        elif key in AUDIO_VOWEL_LK:
            want, why = seg(AUDIO_VOWEL_LK[key]["ipa"]), "audio-vowel"
        if want is None:
            continue
        counts[f"seen:{why}"] += 1
        if read(core) == want:
            counts[f"already-correct:{why}"] += 1
            continue
        fixed = PUR if why == "far-par" else candidates(core, want)
        if fixed is None or strip_marks(fixed) != bare or read(fixed) != want:
            counts[f"unfixable:{why}"] += 1
            continue
        ptoks[pos] = lead + fixed + trail
        if not mask[pos]:
            mask[pos] = True
            counts[f"newly-supervised:{why}"] += 1
        counts[f"fixed:{why}"] += 1
        changed = True
    if changed:
        row = dict(row)
        row["pointed"] = " ".join(ptoks)
        row["supervised"] = mask
        row["n_supervised"] = sum(mask)
        if strip_marks(row["pointed"]) != row["text"]:
            raise AssertionError(f"row {row['id']}: letter identity broken")
        counts["rows_changed"] += 1
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    counts: Counter = Counter()
    for name in ("train.jsonl", "val.jsonl"):
        path = DATA / name
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        out = [relabel_row(r, counts) for r in rows]
        if not args.dry_run:
            with path.open("w", encoding="utf-8") as fh:
                for r in out:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
