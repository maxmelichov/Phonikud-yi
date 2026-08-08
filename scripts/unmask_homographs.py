#!/usr/bin/env python3
"""Un-mask audio-decided homograph occurrences in the retrain training set.

scripts/prepare_retrain_dataset.py masks every homograph token it meets
(counter 'mask_homograph', 11,789 tokens): the lexicon knows the word has two
readings but not which one THIS sentence means, and stamping the wrong pointing
teaches the model a lie. The audio decider answers exactly that question, per
occurrence — so wherever it decided confidently, the mask can come off.

The bar here is deliberately NOT the promotion bar in
scripts/build_homograph_lexicon.py. That one asks "does this TYPE have one
reading we can put in a lexicon?" and needs a majority across occurrences.
This one asks "was THIS occurrence heard clearly?" and needs only that the
winning candidate beat the runner-up by MARGIN_MIN — a type that is genuinely
split still yields good supervision at the occurrences where the audio was
unambiguous, which is the whole point of per-token supervision.

MARGIN_MIN is on the scorer's ``margin`` field and therefore on ITS scale:
(best_fit - runner_fit) * clamp(best_fit, 0, 1), where both fits are
length-normalised to <= 1.0. It is held at 1.67x the scorer's own decide bar
(0.05), the same ratio the old ratio-margin scale used — the scale changed when
xeus_score_homographs.py stopped dividing the lead by the winner's fit (which
made the WORST-fitting windows report the highest confidence), so votes.jsonl
must be rescored, not reused, when this number is compared against it.

Safety rails, all of them hard failures rather than silent skips:

  * the vote's word must sit at the vote's position in the row's text;
  * the winning pointing must be letter-identical to the token it replaces
    (marks stripped) — the dataset's invariant is that ``text`` is exactly
    ``pointed`` with the diacritics removed, and a broken row would corrupt
    training silently;
  * only tokens currently masked are touched; an already-supervised token keeps
    the pointing the canonical lexicon gave it.

The originals are never overwritten: output goes to train_unmasked.jsonl beside
them.

    python scripts/unmask_homographs.py [--margin 0.08] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yiddish_g2p as G  # noqa: E402

VOTES = ROOT / "data" / "homographs" / "votes.jsonl"
TRAIN = ROOT / "data" / "retrain" / "train.jsonl"
OUT = ROOT / "data" / "retrain" / "train_unmasked.jsonl"

MARGIN_MIN = 0.08


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_marks(s: str) -> str:
    return "".join(ch for ch in norm(s) if not unicodedata.combining(ch))


def load_votes(path: Path, margin_min: float) -> tuple[dict, Counter]:
    """(row id, position) -> winning pointed form, for the confident decisions."""
    stats: Counter[str] = Counter()
    picked: dict[tuple[str, int], dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            v = json.loads(line)
            stats["votes"] += 1
            if v.get("decision") != "decided" or not v.get("winner_pointed"):
                stats["skip_undecided"] += 1
                continue
            if float(v.get("margin") or 0.0) < margin_min:
                stats["skip_low_margin"] += 1
                continue
            stats["confident"] += 1
            picked[(v["id"], int(v["position"]))] = v
    return picked, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margin", type=float, default=MARGIN_MIN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    picked, stats = load_votes(VOTES, args.margin)
    used: set[tuple[str, int]] = set()
    out_lines: list[str] = []
    words: Counter[str] = Counter()

    with TRAIN.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            toks = row["text"].split()
            ptoks = row["pointed"].split()
            mask = list(row["supervised"])
            changed = False
            for pos, tok in enumerate(toks):
                v = picked.get((row["id"], pos))
                if v is None:
                    continue
                used.add((row["id"], pos))
                lead, core, trail = G.split_affixes(tok)
                if strip_marks(core) != strip_marks(v["word"]):
                    stats["skip_word_mismatch"] += 1
                    continue
                if mask[pos]:
                    stats["skip_already_supervised"] += 1
                    continue
                pointed = norm(v["winner_pointed"])
                if strip_marks(pointed) != strip_marks(core):
                    stats["skip_letters_differ"] += 1
                    continue
                ptoks[pos] = lead + pointed + trail
                mask[pos] = True
                words[v["word"]] += 1
                changed = True
                stats["unmasked"] += 1
            if changed:
                row["pointed"] = " ".join(ptoks)
                row["supervised"] = mask
                row["n_supervised"] = sum(mask)
                if strip_marks(row["pointed"]) != strip_marks(row["text"]):
                    raise AssertionError(f"row {row['id']}: letter identity broken")
                stats["rows_changed"] += 1
            stats["rows"] += 1
            stats["supervised_after"] += sum(mask)
            out_lines.append(json.dumps(row, ensure_ascii=False))

    stats["votes_no_matching_row"] = len(set(picked) - used)
    if not args.dry_run:
        OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
    for k in sorted(stats):
        print(f"{k:28s} {stats[k]}")
    print("\nunmasked by word:")
    for w, n in words.most_common(20):
        print(f"  {w:14s} {n}")


if __name__ == "__main__":
    main()
