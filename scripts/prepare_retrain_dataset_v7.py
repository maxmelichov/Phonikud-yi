#!/usr/bin/env python3
"""Retrain dataset v7 = retrain3 + homograph unmask at the scorer decide bar.

Merger over v3 (``data/retrain3/``), not a rebuild: v3's rows are the input,
nothing v3 already supervised is touched, and ``test.jsonl`` / ``val.jsonl``
stay byte-for-byte copies of v3's (asserted).

v6 trained on retrain3, which carried through homograph occurrences unmasked
at MARGIN_MIN=0.08. v7 lowers that bar to 0.05 — the same threshold
``scripts/xeus_score_homographs.py`` already used to mark an occurrence
``decided``. Type-level homograph stamps stay forbidden; only per-occurrence
audio decisions are applied, with the same letter-identity rails as
``scripts/unmask_homographs.py``.

וי IPA is not rewritten here. The pointing model trains on nikud, not IPA;
``yiddish_g2p`` defaults Germanic וי to ɔj and keeps oʊ lexical
(class 54 + אויס־/ארויס־/אויף). Gold already has גרויס/טויט/בלויז as ɔj
and אויך as ɔjx (sibling lexicon fix). Eval uses that live engine.

Output: data/retrain7/{train,val,test}.jsonl + dataset_stats.md.

Usage:  .venv/bin/python scripts/prepare_retrain_dataset_v7.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import unmask_homographs as U  # noqa: E402

V3DIR = REPO / "data" / "retrain3"
OUTDIR = REPO / "data" / "retrain7"
VOTES = REPO / "data" / "homographs" / "votes.jsonl"
V6_MARGIN = 0.08
V7_MARGIN = 0.05


def copy_split(name: str) -> None:
    src = V3DIR / name
    dst = OUTDIR / name
    shutil.copy2(src, dst)
    a = src.read_bytes()
    b = dst.read_bytes()
    if a != b:
        raise AssertionError(f"{name} copy is not byte-identical")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margin", type=float, default=V7_MARGIN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (V3DIR / "train.jsonl").exists():
        sys.exit(f"missing {V3DIR / 'train.jsonl'}")

    print(f"v6-comparable dry-run at margin={V6_MARGIN} on retrain3:", flush=True)
    v6_stats, v6_words = U.unmask_file(
        V3DIR / "train.jsonl", VOTES, None, V6_MARGIN, dry_run=True,
    )
    U.report(v6_stats, v6_words)

    print(f"\nv7 unmask at margin={args.margin} on retrain3:", flush=True)
    out_train = None if args.dry_run else OUTDIR / "train.jsonl"
    v7_stats, v7_words = U.unmask_file(
        V3DIR / "train.jsonl", VOTES, out_train, args.margin, dry_run=args.dry_run,
    )
    U.report(v7_stats, v7_words)

    if not args.dry_run:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        copy_split("val.jsonl")
        copy_split("test.jsonl")
        for extra in ("train_episodes.txt", "val_episodes.txt"):
            src = V3DIR / extra
            if src.exists():
                shutil.copy2(src, OUTDIR / extra)

    lines = [
        "# Retrain dataset v7 — homograph unmask at the scorer decide bar",
        "",
        "Merger over v3 (`data/retrain3/`): v3 train rows are the input;",
        "already-supervised tokens are never restamped; `val.jsonl` and",
        "`test.jsonl` are byte-for-byte copies of v3's (asserted).",
        "",
        "v6 unmasked audio-decided homograph occurrences at `MARGIN_MIN=0.08`.",
        "v7 uses `0.05`, the scorer's own decide bar in",
        "`scripts/xeus_score_homographs.py`. Type-level homograph stamps stay",
        "forbidden.",
        "",
        "## Headline",
        "",
        "| metric | v6 (0.08) | v7 (0.05) |",
        "| --- | ---: | ---: |",
        f"| confident votes (cleared the bar) | {v6_stats['confident']} | {v7_stats['confident']} |",
        f"| skip_low_margin | {v6_stats['skip_low_margin']} | {v7_stats['skip_low_margin']} |",
        f"| newly unmasked on retrain3 | {v6_stats['unmasked']} | {v7_stats['unmasked']} |",
        f"| skip_already_supervised | {v6_stats['skip_already_supervised']} | {v7_stats['skip_already_supervised']} |",
        f"| rows changed | {v6_stats['rows_changed']} | {v7_stats['rows_changed']} |",
        "",
        "Newly unmasked on retrain3 is the *additional* supervision this pass",
        "adds. Occurrences already unmasked at 0.08 (and carried through v2/v3)",
        "count as `skip_already_supervised`.",
        "",
        "## וי IPA (not rewritten)",
        "",
        "Phonikud trains on pointing, not IPA. `yiddish_g2p` defaults וי to ɔj;",
        "oʊ is lexical (class 54 + אויס־/ארויס־/אויף). Gold already has",
        "גרויס / טויט / בלויז as ɔj. אויך remains gold `oʊx` (sibling lexicon",
        "work); not guessed here. אנגעהויבן is not touched.",
        "",
        "## Top newly unmasked types (v7)",
        "",
        "| type | tokens |",
        "| --- | ---: |",
    ]
    for w, n in v7_words.most_common(20):
        lines.append(f"| {w} | {n} |")
    lines.append("")

    if not args.dry_run:
        (OUTDIR / "dataset_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {OUTDIR / 'dataset_stats.md'}")
        # machine-readable counts for the training log
        summary = {
            "v6_margin": V6_MARGIN,
            "v7_margin": args.margin,
            "v6": dict(v6_stats),
            "v7": dict(v7_stats),
            "v7_unmasked_by_word": dict(v7_words.most_common()),
        }
        (OUTDIR / "unmask_counts.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
