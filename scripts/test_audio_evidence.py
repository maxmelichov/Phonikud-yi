#!/usr/bin/env python
"""Gate for the audio-evidence layer — run: .venv/bin/python scripts/test_audio_evidence.py

Four suites:
  A. data/audio_pe_lk.py integrity: every generated entry stays inside the
     closed §1 inventory, differs from the engine's own rule-path reading
     ONLY by f->p flips, introduces at least one p, never shadows a gold or
     legacy verdict, and actually routes (route=lexicon, reason=audio-pe,
     confidence=MED).
  B. Authority order: no audio verdict — pe flip or LK SUSPECT — may touch a
     word the gold lexicon pins. Audio is tier 2; Chezky is tier 1.
  C. Vote-file discipline: every pe FLIP row in pe_sweep_votes.tsv really
     clears the majority bar its verdict claims (the folder re-derives the
     numbers rather than trusting the label).
  D. Sweep verdict logic: xeus_pe_sweep.build_report on synthetic tags —
     unanimous flip, engine-confirmed, thin, and contested inputs each get
     the verdict the folding step assumes.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from yiddish_g2p import (GOLD_LEXICON, g2p_token, lexicon_key,  # noqa: E402
                         _rule_path_ipa, _AUDIO_PE)

INVENTORY = set("aːɛəiuɔejbdfɡhjklmnprstvzxʃʒʦʧʤŋˈ" + "oʊ")

passed = failed = 0


def check(ok: bool, label: str) -> None:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {label}")


# --- A. audio_pe table integrity --------------------------------------------
from data.audio_pe_lk import AUDIO_PE_LK  # noqa: E402

check(len(AUDIO_PE_LK) > 0, "A: table is non-empty")
for word, entry in AUDIO_PE_LK.items():
    ipa = entry["ipa"]
    bad = {ch for ch in ipa if ch not in INVENTORY}
    check(not bad, f"A: {word} ipa {ipa!r} outside inventory: {bad}")

    rule = _rule_path_ipa(word, stress=True)
    check(len(rule) == len(ipa), f"A: {word} length differs from rule path "
                                 f"({rule!r} vs {ipa!r})")
    if len(rule) == len(ipa):
        diffs = [(a, b) for a, b in zip(rule, ipa) if a != b]
        check(all(a == "f" and b == "p" for a, b in diffs),
              f"A: {word} differs from rule path beyond f->p: {diffs}")
        check(any(b == "p" for _, b in diffs),
              f"A: {word} entry flips nothing ({ipa!r})")

    check(lexicon_key(word) == word,
          f"A: {word} key not in lexicon_key form")
    check(word not in GOLD_LEXICON,
          f"A: {word} shadows a gold row — builder must never emit it")

    rec = g2p_token(word)
    check(rec["ipa_primary"] == ipa and rec["reason"] == "audio-pe"
          and rec["route"] == "lexicon" and rec["confidence"] == "MED",
          f"A: {word} does not route to its entry "
          f"(got {rec['ipa_primary']!r} {rec['route']}/{rec['reason']})")

# the loaded engine table is exactly the file (no key lost in normalization)
check(set(_AUDIO_PE) == {lexicon_key(w) for w in AUDIO_PE_LK},
      "A: engine-loaded table differs from data/audio_pe_lk.py")

# --- A2. audio_vowel table integrity ----------------------------------------
from data.audio_vowel_lk import AUDIO_VOWEL_LK  # noqa: E402
from xeus_map import VOWELS, tokenize_g2p_ipa  # noqa: E402

CLEAN_TARGETS = {"u"}

check(len(AUDIO_VOWEL_LK) > 0, "A2: vowel table is non-empty")
for word, entry in AUDIO_VOWEL_LK.items():
    ipa = entry["ipa"]
    bad = {ch for ch in ipa if ch not in INVENTORY}
    check(not bad, f"A2: {word} ipa {ipa!r} outside inventory: {bad}")

    # The builder substitutes into the engine's own baseline with this table
    # absent — the rule path OR a gold-anchored prefix-stem rescue. Diffs the
    # rescue contributed (f->p from a gold stem, aj->aː) are legitimate there
    # and must not be charged to the audio fold, so reconstruct that baseline.
    import yiddish_g2p as _m
    _saved = _m._AUDIO_VOWEL.pop(word, None)
    _m._ROUTE_CACHE.clear()
    rule = _m.g2p_token(word)["ipa_primary"]
    if _saved is not None:
        _m._AUDIO_VOWEL[word] = _saved
    _m._ROUTE_CACHE.clear()
    # STRESS NEVER MOVES: identical ˈ count and identical prefix-of-phones
    # position — strip vowels of both and the skeletons must match exactly.
    check(rule.count("ˈ") == ipa.count("ˈ"),
          f"A2: {word} stress count changed ({rule!r} -> {ipa!r})")
    r_toks, e_toks = tokenize_g2p_ipa(rule), tokenize_g2p_ipa(ipa)
    check(len(r_toks) == len(e_toks),
          f"A2: {word} phone count changed ({rule!r} -> {ipa!r})")
    if len(r_toks) == len(e_toks):
        diffs = [(a, b) for a, b in zip(r_toks, e_toks) if a != b]
        check(diffs and all(a in VOWELS and b in CLEAN_TARGETS
                            for a, b in diffs),
              f"A2: {word} diffs beyond vowel->clean-target: {diffs}")
    # the ˈ mark must sit immediately before a vowel in the new reading too
    for i, ch in enumerate(ipa):
        if ch == "ˈ":
            rest = ipa[i + 1:]
            check(any(rest.startswith(v) for v in
                      ("aː", "ej", "aj", "ɔj", "oʊ", "a", "ɛ", "ə", "i", "u", "ɔ")),
                  f"A2: {word} stress mark not before a vowel in {ipa!r}")

    check(lexicon_key(word) == word, f"A2: {word} key not in lexicon_key form")
    check(word not in GOLD_LEXICON, f"A2: {word} shadows a gold row")
    check(word not in AUDIO_PE_LK, f"A2: {word} collides with the audio-pe table")

    rec = g2p_token(word)
    check(rec["ipa_primary"] == ipa and rec["reason"] == "audio-vowel"
          and rec["route"] == "lexicon" and rec["confidence"] == "MED",
          f"A2: {word} does not route to its entry "
          f"(got {rec['ipa_primary']!r} {rec['route']}/{rec['reason']})")

# --- B. authority order ------------------------------------------------------
votes_path = REPO / "data" / "audio_lexicon" / "lk_sweep_votes.tsv"
if votes_path.exists():
    with votes_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["verdict"] == "SUSPECT":
                check(row["key"] not in GOLD_LEXICON,
                      f"B: SUSPECT {row['word']} is a gold word — the sweep "
                      f"must never target tier 1")
else:
    print("skip  B: lk_sweep_votes.tsv absent")

# --- C. pe vote-file discipline ---------------------------------------------
pe_votes = REPO / "data" / "audio_lexicon" / "pe_sweep_votes.tsv"
with pe_votes.open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        for v, verd in zip(row["votes"].split(";"), row["verdict"].split("|")):
            p = int(v.split("/")[0].split("=")[1])
            f = int(v.split("/")[1].split("=")[1])
            if verd in ("FLIP->p", "FLIP->f"):
                win = p if verd == "FLIP->p" else f
                check(p + f >= 3 and win / (p + f) >= 2 / 3,
                      f"C: {row['word']} verdict {verd} not supported by {v}")
            elif verd == "thin":
                check(p + f < 3, f"C: {row['word']} thin but has {p + f} votes")

# --- D. sweep verdict logic on synthetic tags -------------------------------
import xeus_pe_sweep as sweep  # noqa: E402

CASES = [  # (g2p, clips' heard_at, expected verdict)
    (["f", "a"], [["p", "a"], ["p", "a"], ["b", "a"]], "FLIP->p"),
    (["f", "a"], [["f", "a"], ["f", "a"], ["v", "a"]], "CONFIRMED"),
    (["f", "a"], [["p", "a"], ["f", "a"]], "thin"),
    (["f", "a"], [["p", "a"], ["p", "a"], ["f", "a"], ["f", "a"]], "contested"),
]
with tempfile.TemporaryDirectory() as td:
    orig = sweep.TAGS
    try:
        sweep.TAGS = Path(td) / "tags.jsonl"
        with sweep.TAGS.open("w", encoding="utf-8") as fh:
            for i, (g2p, clips, _) in enumerate(CASES):
                for ci, heard in enumerate(clips):
                    fh.write(json.dumps({
                        "episode": "t", "chunk_idx": i * 100 + ci,
                        "word": f"w{i}", "g2p": g2p, "heard_at": heard,
                    }) + "\n")
        report = sweep.build_report({f"w{i}" for i in range(len(CASES))},
                                    Counter())
        got = {r["key"]: r["verdict"] for r in report}
        for i, (_, _, want) in enumerate(CASES):
            check(got.get(f"w{i}") == want,
                  f"D: case {i} verdict {got.get(f'w{i}')!r}, want {want!r}")
    finally:
        sweep.TAGS = orig

print(f"\n{passed} passed, {failed} FAILED")
sys.exit(1 if failed else 0)
