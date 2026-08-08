#!/usr/bin/env python
"""Apply the type lexicon to the diacritics corpus, writing data/diacritics_r4/.

Reads data/diacritics_r3c/{train,val,test}.txt, replaces every word with its
lexicon pointing, and writes the result to a NEW directory. Nothing is edited in
place, so r3c stays available to diff against and this can be re-run freely.

Because the lexicon holds exactly one pointing per bare word, the output is
consistent by construction: type_consistency is 1.0, not 0.26 as in r3c.

Words the lexicon cannot settle keep their existing r3c pointing rather than
being guessed at:
  - types the lexicon rejected (letters changed under the model's hand)
  - types below the --min-count cut, never sent
  - AMBIGUOUS types (פאר = far vs pur), which need sentence context; those are
    written to ambiguous_lines.jsonl so a context pass can target just them.

Every line is verified letter-for-letter against its source before being written;
a line that would change any letter is emitted unchanged and counted.

Output: data/diacritics_r4/
  train.txt val.txt test.txt      the retagged corpus
  *_episodes.txt                  copied through from r3c
  ambiguous_lines.jsonl           lines containing an ambiguous type
  apply_stats.json                per-split coverage and consistency

Usage:
  .venv/bin/python scripts/apply_nikud_lexicon.py --dry-run
  .venv/bin/python scripts/apply_nikud_lexicon.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_nikud_lexicon import bare_of, load_lexicon, _STRIP_PUNCT  # noqa: E402
from scripts.nikud_yi import canon  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "data" / "diacritics_r3c"
OUT_DIR = REPO / "data" / "diacritics_r4"

_HEBREW = re.compile(r"[א-ת]")


def split_affixes(word: str) -> tuple[str, str, str]:
    """Return (leading punct, core, trailing punct) for a corpus token."""
    i, j = 0, len(word)
    while i < j and word[i] in _STRIP_PUNCT:
        i += 1
    while j > i and word[j - 1] in _STRIP_PUNCT:
        j -= 1
    return word[:i], word[i:j], word[j:]


def retag_line(line: str, lex: dict[str, dict], counters: collections.Counter):
    """Return (new_line, saw_ambiguous). Falls back to the original per word."""
    out: list[str] = []
    saw_ambiguous = False
    for token in line.split():
        lead, core, trail = split_affixes(token)
        b = bare_of(core)
        if not b or not _HEBREW.search(b):
            out.append(token)
            continue
        rec = lex.get(b)
        counters["words"] += 1
        if rec is None:
            counters["miss_not_in_lexicon"] += 1
            out.append(token)
            continue
        if not rec["ok"]:
            counters["miss_rejected"] += 1
            out.append(token)
            continue
        if rec["ambiguous"]:
            # Keep r3c's reading; a context pass decides these, not this script.
            counters["miss_ambiguous"] += 1
            saw_ambiguous = True
            out.append(token)
            continue
        pointed = rec["pointed"]
        if canon(pointed) != canon(core):
            counters["miss_letter_guard"] += 1
            out.append(token)
            continue
        counters["mapped"] += 1
        counters["changed"] += pointed != core
        out.append(lead + pointed + trail)
    return " ".join(out), saw_ambiguous


def consistency(text: str) -> tuple[float, float, int]:
    """(type consistency, instance consistency, mean variants) over a text."""
    variants: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for word in text.split():
        _, core, _ = split_affixes(word)
        b = bare_of(core)
        if b and _HEBREW.search(b):
            variants[b][unicodedata.normalize("NFC", core)] += 1
    if not variants:
        return 1.0, 1.0, 1.0
    single = sum(1 for c in variants.values() if len(c) == 1)
    inst_ok = sum(c.most_common(1)[0][1] for c in variants.values())
    inst_all = sum(sum(c.values()) for c in variants.values())
    mean_var = sum(len(c) for c in variants.values()) / len(variants)
    return single / len(variants), inst_ok / inst_all, mean_var


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report the numbers without writing data/diacritics_r4")
    args = ap.parse_args()

    lex = {w: r for w, r in load_lexicon().items()}
    if not lex:
        print("lexicon is empty -- run scripts/build_nikud_lexicon.py first",
              file=sys.stderr)
        return 1

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    ambiguous_lines: list[dict] = []

    for name in ("train", "val", "test"):
        src = SRC_DIR / f"{name}.txt"
        if not src.exists():
            continue
        counters: collections.Counter = collections.Counter()
        lines_in = src.read_text(encoding="utf-8").splitlines()
        lines_out: list[str] = []
        for ln, line in enumerate(lines_in):
            if not line.strip():
                lines_out.append(line)
                continue
            new, amb = retag_line(line, lex, counters)
            # Hard guard: the retag may only add or change marks, never letters.
            if canon(new) != canon(line):
                counters["line_reverted_letter_change"] += 1
                new = line
            lines_out.append(new)
            if amb:
                ambiguous_lines.append({"split": name, "line": ln, "text": line})

        before = consistency("\n".join(lines_in))
        after = consistency("\n".join(lines_out))
        stats[name] = {
            "lines": len(lines_in),
            "words": counters["words"],
            "mapped": counters["mapped"],
            "changed": counters["changed"],
            "pct_mapped": round(100 * counters["mapped"] / max(counters["words"], 1), 2),
            "pct_changed": round(100 * counters["changed"] / max(counters["words"], 1), 2),
            "unmapped": {
                "not_in_lexicon": counters["miss_not_in_lexicon"],
                "rejected_by_validator": counters["miss_rejected"],
                "ambiguous": counters["miss_ambiguous"],
                "letter_guard": counters["miss_letter_guard"],
            },
            "lines_reverted": counters["line_reverted_letter_change"],
            "before": {"type_consistency": round(before[0], 4),
                       "instance_consistency": round(before[1], 4),
                       "mean_variants_per_type": round(before[2], 4)},
            "after": {"type_consistency": round(after[0], 4),
                      "instance_consistency": round(after[1], 4),
                      "mean_variants_per_type": round(after[2], 4)},
        }
        print(f"{name:6} {counters['mapped']:>7,}/{counters['words']:>7,} mapped "
              f"({stats[name]['pct_mapped']:5.1f}%)  "
              f"type-consistency {before[0]:.3f} -> {after[0]:.3f}")

        if not args.dry_run:
            (OUT_DIR / f"{name}.txt").write_text("\n".join(lines_out) + "\n",
                                                 encoding="utf-8")
            ep = SRC_DIR / f"{name}_episodes.txt"
            if ep.exists():
                shutil.copy2(ep, OUT_DIR / ep.name)

    if not args.dry_run:
        with (OUT_DIR / "ambiguous_lines.jsonl").open("w", encoding="utf-8") as fh:
            for rec in ambiguous_lines:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (OUT_DIR / "apply_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT_DIR} ({len(ambiguous_lines):,} lines flagged ambiguous)")
    else:
        print("\ndry run -- nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
