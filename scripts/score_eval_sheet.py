#!/usr/bin/env python
"""Score a returned native-speaker evaluation sheet: WER, PER, stress.

TWO METRICS, DELIBERATELY SEPARATE.

  WER  word error rate — the fraction of words the native corrected. Needs
       nothing but the blanks, so it is available the moment the sheet comes
       back. This is the headline number and the one that goes in the paper.

  PER  phoneme error rate — Levenshtein distance between our phones and the
       corrected phones, over the corrected words only, normalised by the
       total phones of the reference. Requires each correction to be turned
       into IPA first (add a `corrected_ipa` column); that transcription is a
       judgement call, so it is a deliberate manual step, not a guess this
       script makes. PER matters because it separates "wrong neighbourhood"
       from "one vowel off": maxalˈɔjkɛs vs maxˈalɔjkəs is 100% WER but ~18%
       PER, and only the second number tells you the engine nearly had it.

STRESS IS SCORED SEPARATELY, not as a phone. Counting ˈ as a token would
charge a misplaced stress twice (one deletion + one insertion) and mix a
prosodic error into a segmental metric.

Distances are computed over PHONE TOKENS from the engine's own tokenizer, so
aː/ej/ʦ count as one symbol each; a naive character distance would inflate
every diphthong error into two edits.

Usage:
  .venv/bin/python scripts/score_eval_sheet.py data/eval/eval_sheet_<date>.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_map import tokenize_g2p_ipa  # noqa: E402

STRESS = "ˈ"


def levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance over phone tokens."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def phones(ipa: str) -> list[str]:
    return tokenize_g2p_ipa(ipa.replace(STRESS, ""))


def stress_slot(ipa: str) -> int | None:
    """Index of the stressed phone, or None if unmarked."""
    if STRESS not in ipa:
        return None
    return len(tokenize_g2p_ipa(ipa.split(STRESS)[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", type=Path)
    ap.add_argument("--meta", type=Path,
                    help="eval_meta_<date>.tsv (default: alongside the sheet)")
    args = ap.parse_args()

    meta_path = args.meta or Path(str(args.sheet).replace("eval_sheet_",
                                                          "eval_meta_"))
    meta: dict[tuple[str, str], dict] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                meta[(r["sentence"], r["word_no"])] = r
    else:
        print(f"note: {meta_path} absent — no per-route breakdown", flush=True)

    rows = []
    with args.sheet.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not r["word_no"] or r["sentence"].startswith("Q"):
                continue  # context line or standing question, not a scored word
            rows.append(r)

    if not rows:
        print("no scored word rows found")
        return 1

    n = len(rows)
    wrong = [r for r in rows if r["correction"].strip()]
    print(f"WORDS SCORED: {n}")
    print(f"WER: {100 * len(wrong) / n:.2f}%  "
          f"({len(wrong)} corrected / {n})")
    print(f"word accuracy: {100 * (n - len(wrong)) / n:.2f}%\n")

    # --- breakdowns, only as informative as the meta file --------------------
    for field in ("tier", "route", "confidence"):
        if not meta:
            break
        tot: Counter = Counter()
        bad: Counter = Counter()
        for r in rows:
            m = meta.get((r["sentence"], r["word_no"]))
            if not m:
                continue
            tot[m[field]] += 1
            if r["correction"].strip():
                bad[m[field]] += 1
        if not tot:
            continue
        print(f"WER by {field}:")
        for k in sorted(tot, key=lambda k: -tot[k]):
            print(f"  {k:22s} {100 * bad[k] / tot[k]:6.2f}%  "
                  f"({bad[k]}/{tot[k]})")
        print()

    # --- PER, over whatever has been transcribed to IPA ----------------------
    has_ipa = [r for r in wrong if r.get("corrected_ipa", "").strip()]
    if not has_ipa:
        print(f"PER: not computable yet — {len(wrong)} corrections need a "
              f"`corrected_ipa` column before phoneme distance means anything.")
        if wrong:
            print("\ncorrections awaiting transcription:")
            for r in wrong[:40]:
                m = meta.get((r["sentence"], r["word_no"]), {})
                print(f"  {r['word']}\t{r['our_reading']} -> "
                      f"{r['correction']}\t[{m.get('ipa', '')}]")
        return 0

    edits = ref_len = 0
    stress_wrong = stress_tot = 0
    by_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in has_ipa:
        m = meta.get((r["sentence"], r["word_no"]), {})
        ours, theirs = m.get("ipa", ""), r["corrected_ipa"].strip()
        if not ours:
            continue
        d = levenshtein(phones(ours), phones(theirs))
        edits += d
        ref_len += len(phones(theirs))
        tier = m.get("tier", "?")
        by_tier[tier][0] += d
        by_tier[tier][1] += len(phones(theirs))
        if stress_slot(theirs) is not None:
            stress_tot += 1
            stress_wrong += stress_slot(ours) != stress_slot(theirs)

    # words the native approved contribute their phones as correct matches
    approved_phones = sum(
        len(phones(meta.get((r["sentence"], r["word_no"]), {}).get("ipa", "")))
        for r in rows if not r["correction"].strip())
    total_ref = ref_len + approved_phones
    print(f"PER: {100 * edits / total_ref:.2f}%  "
          f"({edits} edits / {total_ref} phones)")
    print(f"  (over corrected words alone: "
          f"{100 * edits / ref_len:.2f}%, {edits}/{ref_len})")
    if stress_tot:
        print(f"stress misplaced: {stress_wrong}/{stress_tot} of corrected "
              f"polysyllables")
    print("\nPER by tier:")
    for t, (d, ln) in sorted(by_tier.items()):
        print(f"  {t:22s} {100 * d / ln:6.2f}%  ({d}/{ln})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
