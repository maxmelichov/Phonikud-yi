#!/usr/bin/env python3
"""
Apply data/canonical_pointing.tsv to the diacritics corpus.

Reads data/diacritics_r2/{train,val,test}.txt (and the matching *_episodes.txt,
which are copied through unchanged so the split stays identical) and writes
data/diacritics_r3c/ with every mapped word replaced by its canonical pointed
form.  Matching is on the *bare* (mark-stripped) core of each whitespace token;
leading/trailing punctuation is preserved verbatim.

The map is derived from train only, but it is applied to all three splits --
val/test targets become consistent with what the model is trained to emit.
Word types the adjudicator flagged as homographs are absent from the map and
therefore keep their original pointing.

Usage:
    python scripts/apply_canonical.py
    python scripts/apply_canonical.py --dry-run
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from canonicalize_pointing import (  # noqa: E402
    collect,
    consistency,
    fold,
    has_hebrew,
    mechanical,
    split_token,
    strip_marks,
)


def load_map(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        header = next(fh)
        assert header.startswith("word_bare"), header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            m[parts[0]] = parts[1]
    return m


def apply_line(line: str, cmap: dict[str, str], counter: collections.Counter) -> str:
    out = []
    for tok in fold(line).split():
        pre, core, post = split_token(tok)
        if core and has_hebrew(core):
            counter["words"] += 1
            core_m = mechanical(core)
            canon = cmap.get(strip_marks(core_m))
            if canon is None:
                counter["unmapped"] += 1
                new = core_m
            else:
                new = canon
                counter["mapped"] += 1
            if new != core:
                counter["changed"] += 1
            core = new
        out.append(pre + core + post)
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO / "data/diacritics_r2")
    ap.add_argument("--dst", type=Path, default=REPO / "data/diacritics_r3c")
    ap.add_argument("--map", type=Path, default=REPO / "data/canonical_pointing.tsv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cmap = load_map(args.map)
    print(f"canonical map: {len(cmap):,} word types")
    if not args.dry_run:
        args.dst.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        src = args.src / f"{split}.txt"
        counter: collections.Counter = collections.Counter()
        lines = [apply_line(ln, cmap, counter) for ln in src.read_text(encoding="utf-8").splitlines()]
        if not args.dry_run:
            (args.dst / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            shutil.copyfile(args.src / f"{split}_episodes.txt", args.dst / f"{split}_episodes.txt")
        n = counter["words"]
        report[split] = {
            "lines": len(lines),
            "words": n,
            "mapped": counter["mapped"],
            "unmapped": counter["unmapped"],
            "changed": counter["changed"],
            "pct_mapped": round(100 * counter["mapped"] / max(n, 1), 2),
            "pct_changed": round(100 * counter["changed"] / max(n, 1), 2),
        }
        r = report[split]
        print(f"{split:5s}: {r['lines']:6,} lines  {n:8,} words  "
              f"mapped {r['pct_mapped']:5.1f}%  changed {r['changed']:8,} ({r['pct_changed']:5.1f}%)  "
              f"unmapped {r['unmapped']:,}")

    if args.dry_run:
        return 0

    # ---- before/after consistency, recomputed from the files themselves
    print("\nconsistency (word types seen >=3x in the split):")
    print(f"{'split':6s} {'mean vars/type':>16s} {'types':>10s} {'instances':>12s}")
    for split in ("train", "val", "test"):
        b = consistency(collect(args.src / f"{split}.txt"))
        a = consistency(collect(args.dst / f"{split}.txt"))
        report[split]["before"] = b
        report[split]["after"] = a
        print(f"{split:6s} {b['mean_variants_all_types']:7.3f} -> {a['mean_variants_all_types']:5.3f}  "
              f"{100 * b['type_consistency']:6.1f}% -> {100 * a['type_consistency']:5.1f}%  "
              f"{100 * b['instance_consistency']:6.1f}% -> {100 * a['instance_consistency']:5.1f}%")

    (args.dst / "apply_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
