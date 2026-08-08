#!/usr/bin/env python3
"""Build candidate readings for the homograph-conflict quarantine bucket.

The Sefaria rescue (scripts/build_sefaria_lexicon.py) skips a type when its
unpointed form has no >= 80%-dominant vocalization in the verified pointed
sources — 1,223 types / 18,225 tokens tagged 'homograph-conflict' in
data/verification/skipped_hebrew_full.tsv. Those words are not unknown: the
book sources tell us exactly WHICH readings are in play, they just cannot say
which one a given Yiddish sentence means. This script turns each of them into
an explicit candidate set for the audio decider to choose from.

Method, per type:

  * pull the unigram pointings from data/pointed_sources/pointed_index.jsonl;
  * phonemic_fold() them into groups, so the editions' cosmetic disagreements
    (qamats qatan, holam over a male vav, gemination dagesh) never masquerade
    as a homograph;
  * pick each group's MAXIMALLY-MARKED printed form (build_sefaria_lexicon's
    _reader_marks tie-break: the word-initial dagesh lene and the point on a
    mater vav are the marks read_pointed_wh() actually needs);
  * read it in the Whole-Hebrew register — the register for pointed loshn
    koydesh. The merged register is used only as a FALLBACK for a pointing
    read_pointed_wh() cannot read: run on an already-pointed form it does not
    give a second reading of the word, it gives a degraded one (schwa
    deletion, u->i, a lost mater), and those degradations were creating
    phantom homographs out of a single pointing (118 types before this).

Every candidate is checked against the closed v3 phone inventory and the §1
vowel-shape rule; failures are dropped, not emitted.

TWO SEPARATE QUESTIONS, and conflating them is a bug this script had:

  "does every ATTESTED pointing read the same?"  — asked over ALL fold groups,
      unfiltered. Only a yes here is a fold collapse.
  "which readings deserve a vote?"               — asked over the LIVE groups,
      those holding >= 5% of occurrences AND >= 2 occurrences; the long tail
      below that is edition typos.

A type whose rivals exist but are thin is NOT collapsed: dropping them and
shipping the survivor as a free single-reading rescue is how חתם came to be
emitted as xˈɔjsum ('seal') for 527 tokens of חתם סופר. Such a type goes to
the audio decider with its full unfiltered candidate set, and if the audio
cannot decide it stays quarantined — which is the correct answer.

Output, both keyed by the quarantined surface form:

  data/homographs/candidates.json  types with >= 2 DISTINCT phone strings —
                                   real homographs, they need an audio verdict;
  data/homographs/collapsed.json   types where EVERY attested pointing reads to
                                   one phone string — the print differs, the
                                   reading does not, so these are free rescues.

    python scripts/build_homograph_candidates.py
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_pointed_index import phonemic_fold  # noqa: E402
from scripts.build_sefaria_lexicon import _reader_marks, skeleton  # noqa: E402
from scripts.xeus_map import tokenize_g2p_ipa  # noqa: E402
from yiddish_g2p import (  # noqa: E402
    hebrew_to_ipa,
    ipa_phone_violations,
    read_pointed_wh,
    violates_vowel_ratio,
)

SKIPPED = ROOT / "data" / "verification" / "skipped_hebrew_full.tsv"
INDEX = ROOT / "data" / "pointed_sources" / "pointed_index.jsonl"
OUTDIR = ROOT / "data" / "homographs"
CANDIDATES = OUTDIR / "candidates.json"
COLLAPSED = OUTDIR / "collapsed.json"

CATEGORY = "homograph-conflict"
SHARE_MIN = 0.05
COUNT_MIN = 2


def load_conflicts(path: Path) -> list[tuple[str, int]]:
    """(word, corpus freq) for the homograph-conflict rows, highest freq first."""
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        fi, wi, ci = (header.index("freq"), header.index("word"),
                      header.index("category"))
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= max(fi, wi, ci) or cells[ci] != CATEGORY:
                continue
            rows.append((cells[wi], int(cells[fi])))
    rows.sort(key=lambda t: (-t[1], t[0]))
    return rows


def load_index(path: Path) -> dict[str, list[list]]:
    """Unigram entries only: unpointed key -> [[pointed, count], ...]."""
    out: dict[str, list[list]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("n") == 1:
                out[unicodedata.normalize("NFC", rec["k"])] = rec["p"]
    return out


def group_pointings(pointings: list[list],
                    thin_filter: bool = True) -> list[tuple[str, int, float]]:
    """(maximally-marked form, group count, share) per fold group.

    With ``thin_filter`` (the default) groups below SHARE_MIN of occurrences or
    below COUNT_MIN occurrences are dropped as edition noise; shares are of the
    GRAND total, so they still sum to less than 1 when a tail was cut, which is
    the honest number. Pass thin_filter=False to see every attested reading —
    that unfiltered view, and only it, may be used to ask whether the fold
    groups collapsed, because a rival deleted for thin evidence is still a
    rival.
    """
    groups: dict[str, list[list]] = defaultdict(list)
    for form, count in pointings:
        groups[phonemic_fold(form)].append([form, count])
    grand = sum(c for _, c in pointings)
    if not grand:
        return []
    kept = []
    for members in groups.values():
        total = sum(c for _, c in members)
        share = total / grand
        if thin_filter and (total < COUNT_MIN or share < SHARE_MIN):
            continue
        rep = max(members, key=lambda fc: (
            _reader_marks(fc[0]), fc[1],
            sum(1 for ch in fc[0] if unicodedata.combining(ch))))[0]
        kept.append((rep, total, round(share, 4)))
    kept.sort(key=lambda t: (-t[1], t[0]))
    return kept


def read_safely(reader, form: str) -> str | None:
    """Reading of ``form``, or None if it errors or leaves the v3 inventory."""
    try:
        ipa = reader(form)
    except Exception:  # noqa: BLE001
        return None
    if not ipa or ipa_phone_violations(ipa) or violates_vowel_ratio(ipa):
        return None
    return ipa


def phones_of(ipa: str) -> str:
    """Space-separated inventory tokens; stress marks dropped, words joined."""
    return " ".join(tok for word in ipa.split()
                    for tok in tokenize_g2p_ipa(word))


def candidates_for(word: str, pointings: list[list],
                   thin_filter: bool = True) -> list[dict]:
    """Distinct-sounding candidate readings, best-attested first.

    One candidate per fold group: the Whole-Hebrew reading of the group's
    maximally-marked form, falling back to the merged register only when
    read_pointed_wh() cannot read that form at all. The merged register is a
    reader of UNPOINTED Yiddish; handed a pointed loshn-koydesh form it returns
    a degraded version of the same reading (מְחֻלָּל -> mxˈilul beside
    məxˈulul, כְּחוּט -> kxit), not a second reading, and offering that to the
    audio decider both invents homographs and lets a mangling win a vote.

    Two candidates that tokenize to the same phone string are the SAME reading
    however differently they are printed, so the later one is folded away.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for form, count, share in group_pointings(pointings, thin_filter):
        if skeleton(form) != skeleton(word):
            continue  # pointing is of a different word, not this one
        for register, reader in (("wh", read_pointed_wh),
                                 ("merged", hebrew_to_ipa)):
            ipa = read_safely(reader, form)
            if ipa is None:
                continue
            phones = phones_of(ipa)
            if not phones:
                continue
            if phones not in seen:
                seen.add(phones)
                out.append({
                    "pointed": form,
                    "register": register,
                    "ipa": ipa,
                    "phones": phones,
                    "source_count": count,
                    "share": share,
                })
            break  # this fold group has been read; the merged read is a backup
    return out


def build() -> tuple[dict, dict, dict]:
    index = load_index(INDEX)
    stats: dict[str, int] = defaultdict(int)
    true_h: dict[str, dict] = {}
    collapsed: dict[str, dict] = {}

    for word, freq in load_conflicts(SKIPPED):
        stats["conflict_types"] += 1
        stats["conflict_tokens"] += freq
        entry = index.get(skeleton(word))
        if not entry:
            stats["no_index_hit"] += 1
            continue
        # The collapse question is asked over EVERY attested pointing; the
        # thin filter may only choose which readings get a vote, never make a
        # rival disappear so the survivor looks unambiguous.
        every = candidates_for(word, entry, thin_filter=False)
        if not every:
            stats["no_readable_candidate"] += 1
            continue
        if len(every) == 1:
            collapsed[word] = {"freq": freq, **every[0]}
            stats["collapsed_types"] += 1
            stats["collapsed_tokens"] += freq
        else:
            live = candidates_for(word, entry, thin_filter=True)
            # A single live reading with thin rivals is not a free rescue: put
            # the full set to the audio decider and let it stay quarantined if
            # the audio cannot separate them.
            cands = live if len(live) >= 2 else every
            if len(live) < 2:
                stats["thin_rivals_to_audio"] += 1
                stats["thin_rivals_to_audio_tokens"] += freq
            true_h[word] = {"freq": freq, "candidates": cands}
            stats["true_homograph_types"] += 1
            stats["true_homograph_tokens"] += freq
            stats["candidate_readings"] += len(cands)
    return true_h, collapsed, dict(stats)


def sanity(true_h: dict, collapsed: dict) -> list[str]:
    """Re-assert the closed inventory / vowel-shape invariants on the output."""
    bad = []
    rows = [(w, c) for w, r in true_h.items() for c in r["candidates"]]
    rows += [(w, r) for w, r in collapsed.items()]
    for word, cand in rows:
        if ipa_phone_violations(cand["ipa"]):
            bad.append(f"{word}: off-inventory {cand['ipa']!r}")
        if violates_vowel_ratio(cand["ipa"]):
            bad.append(f"{word}: vowel-shape {cand['ipa']!r}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, write nothing")
    args = ap.parse_args()

    true_h, collapsed, stats = build()
    bad = sanity(true_h, collapsed)

    if not args.dry_run:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        CANDIDATES.write_text(
            json.dumps(true_h, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
        COLLAPSED.write_text(
            json.dumps(collapsed, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")

    for k in sorted(stats):
        print(f"{k:24s} {stats[k]:,}")
    print(f"{'sanity_violations':24s} {len(bad):,}")
    for line in bad[:20]:
        print("  !", line)

    print("\ntop true homographs")
    top = sorted(true_h.items(), key=lambda kv: (-kv[1]["freq"], kv[0]))[:15]
    for word, rec in top:
        readings = "  |  ".join(
            f"{c['pointed']} [{c['register']}] {c['ipa']} ({c['share']:.0%})"
            for c in rec["candidates"])
        print(f"{rec['freq']:6,}  {word}: {readings}")


if __name__ == "__main__":
    main()
