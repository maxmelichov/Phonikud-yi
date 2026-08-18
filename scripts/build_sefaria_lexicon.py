#!/usr/bin/env python3
"""Build data/lexicons/sefaria_pointed_lk.py — rescue #2 for the loshn-koydesh quarantine.

For every quarantined type, look its unpointed form up in the verified pointed
index (data/pointed_sources/pointed_index.jsonl, built from Sefaria's MAM Tanakh
and Torat Emet Mishnah/Siddur). Accept the pointing when the sources agree —
exactly one distinct vocalization, or a dominant one at >= 80% of occurrences —
and read it in the register the word is actually being used in
(scripts/register_policy.py): MERGED by default, since a loshn-koydesh word in
a Yiddish sentence takes the Yiddish shifts (shuruk -> [i], final komets-hey ->
[ə]); Whole-Hebrew only where audio or corpus usage says the word is QUOTED.
The losing register ships as a variant.

Rejected: readings that leave the closed v3 phone inventory, readings that fail
the §1 vowel-shape rule, pointings whose letter skeleton differs from the
quarantined spelling, words already covered by data/lexicons/audio_endorsed_lk.py (audio
evidence outranks book pointing) and words whose lexicon_key already collides
with a gold/abbreviation/multiword/legacy lexicon entry.

    python scripts/build_sefaria_lexicon.py
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
from scripts.register_policy import (  # noqa: E402
    SPAN_MIN,
    WH_SHARE_MIN,
    decide,
    quoted_shares,
)
from yiddish_g2p import (  # noqa: E402
    GOLD_LEXICON,
    _ABBREVIATIONS,
    _AUDIO_ENDORSED,
    _LK_BARE,
    _MULTIWORD,
    _MULTIWORD_LEGACY,
    _WORD_LATIN,
    _strip_points,
    ipa_phone_violations,
    lexicon_key,
    normalize_surface,
    violates_vowel_ratio,
)


def readable(ipa: str) -> bool:
    """The §1 gate a reading must pass to be emitted at all."""
    return bool(ipa) and not ipa_phone_violations(ipa) and not violates_vowel_ratio(ipa)

QUARANTINE = ROOT / "data" / "pointed_sources" / "quarantine_full_snapshot.tsv"
INDEX = ROOT / "data" / "pointed_sources" / "pointed_index.jsonl"
OUT = ROOT / "data" / "lexicons" / "sefaria_pointed_lk.py"
DOMINANCE_MIN = 0.80


_DAGESH = "ּ"
_HOLAMS = ("ֹ", "ֺ")
_VAV = "ו"
_BEGADKEFAT = set("בכפת")  # bet kaf pe tav
_VOWEL_POINT = set("ְֱֲֳִֵֶַָ"
                   "ׇֹֺֻ")


def _reader_marks(form: str) -> int:
    """How many marks this printed form carries that phonemic_fold() drops but
    read_pointed_wh() actually needs.

    phonemic_fold() collapses three things. Two of them are inert for the
    reader — a gemination dagesh outside bet/kaf/pe/tav (never realised) and a
    qamats qatan (read [u] either way). The other two are decisive:

      * the dagesh lene on a word-INITIAL bet/kaf/pe/tav, which is the only
        thing selecting b/k/p/t over v/x/f/s (בָּתִּים bˈutim vs בָתִּים vˈutim);
      * the holam or shuruk dagesh on a MATER vav, without which the reader
        sees a bare vav and says /v/ (אוֹתוֹ ˈɔjsɔj vs אוֹתו ɔjsv).

    Counting only these keeps the tie-break from preferring a scribal stray
    dagesh (מֶּלֶךְ over מֶלֶךְ) just because it has one more combining char.
    """
    n = 0
    base = ""
    own: list[str] = []
    seen: list[tuple[str, str]] = []
    for ch in unicodedata.normalize("NFC", form):
        if unicodedata.combining(ch):
            own.append(ch)
        else:
            if base:
                seen.append((base, "".join(own)))
            base, own = ch, []
    if base:
        seen.append((base, "".join(own)))
    for i, (ltr, mk) in enumerate(seen):
        if i == 0 and ltr in _BEGADKEFAT and _DAGESH in mk:
            n += 1
        if ltr == _VAV and not any(m in _VOWEL_POINT for m in mk) and (
                _DAGESH in mk or any(h in mk for h in _HOLAMS)):
            n += 1
    return n


def skeleton(word: str) -> str:
    """Letters only, NFC — what must survive the trip through the pointing."""
    return unicodedata.normalize("NFC", _strip_points(normalize_surface(word)))


def load_quarantine(path: Path) -> list[tuple[str, int]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        wi, fi = header.index("word"), header.index("freq")
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= max(wi, fi):
                continue
            rows.append((cells[wi], int(cells[fi])))
    return rows


def load_index(path: Path) -> dict[str, list[list]]:
    """Unigram entries only: key -> [[pointed, count], ...]."""
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("n") == 1:
                out[unicodedata.normalize("NFC", rec["k"])] = rec["p"]
    return out


def choose(pointings: list[list]) -> tuple[str, int, float, int] | None:
    """(pointed form, n_sources, dominance, n_variants) for an agreed pointing.

    Candidates are grouped by phonemic_fold(), so the two editions' cosmetic
    disagreements (qamats qatan, holam over a male vav, gemination dagesh) do
    not count as a conflict.

    Within the winning group the MOST FULLY MARKED printed form wins, ties
    broken by occurrence count -- NOT the other way round. Two forms only land
    in one group when they differ by a mark that phonemic_fold() drops, and
    every one of those marks is either inert for read_pointed_wh() (gemination
    dagesh, qamats qatan spelled U+05C7) or is exactly the mark the reader
    needs (the word-initial dagesh that selects b/k/p over v/x/f, the holam or
    shuruk over a male vav). Picking by raw count instead handed the reader the
    DEFECTIVE spelling whenever the under-marked edition happened to be more
    frequent: בָתִּים (n=35) beat בָּתִּים (n=26) and produced vˈutim.
    """
    groups: dict[str, list[list]] = defaultdict(list)
    for form, count in pointings:
        groups[phonemic_fold(form)].append([form, count])
    totals = {k: sum(c for _, c in v) for k, v in groups.items()}
    grand = sum(totals.values())
    if not grand:
        return None
    best = max(totals, key=lambda k: totals[k])
    dominance = totals[best] / grand
    if len(groups) > 1 and dominance < DOMINANCE_MIN:
        return None
    form = max(groups[best], key=lambda fc: (
        _reader_marks(fc[0]), fc[1],
        sum(1 for ch in fc[0] if unicodedata.combining(ch))))[0]
    return form, totals[best], round(dominance, 3), len(groups)


def already_routed(word: str) -> bool:
    """True if some lexicon already owns this key — never shadow one."""
    key = lexicon_key(word)
    bare = skeleton(word)
    return (key in _AUDIO_ENDORSED or key in GOLD_LEXICON or key in _ABBREVIATIONS
            or key in _MULTIWORD or key in _MULTIWORD_LEGACY
            or bare in _LK_BARE or bare in _WORD_LATIN)


def build() -> tuple[list[tuple[str, dict, int]], dict[str, int],
                     list[tuple[str, int]], list[tuple[str, int, dict]]]:
    quarantine = sorted(load_quarantine(QUARANTINE), key=lambda t: (-t[1], t[0]))
    index = load_index(INDEX)
    shares = quoted_shares()
    stats: dict[str, int] = defaultdict(int)
    accepted: list[tuple[str, dict, int]] = []
    rejected: list[tuple[str, int]] = []
    flips: list[tuple[str, int, dict]] = []
    by_key: dict[str, str] = {}

    for word, freq in quarantine:
        bare = skeleton(word)
        if already_routed(word):
            stats["skip_already_routed"] += 1
            rejected.append((word, freq))
            continue
        entry = index.get(bare)
        if not entry:
            stats["no_hit"] += 1
            rejected.append((word, freq))
            continue
        picked = choose(entry)
        if picked is None:
            stats["conflicting"] += 1
            rejected.append((word, freq))
            continue
        form, n_sources, dominance, n_variants = picked
        if skeleton(form) != bare:
            stats["reject_letters_lost"] += 1
            rejected.append((word, freq))
            continue
        try:
            verdict = decide(word, form, shares, validate=readable)
        except Exception:  # noqa: BLE001
            stats["reject_read_error"] += 1
            rejected.append((word, freq))
            continue
        if verdict is None:
            # NEITHER register produced a speakable reading — the old
            # empty/bad-phone/vowel-shape rejections, now collapsed into one
            # because a reading is only rejected when both of them fail.
            stats["reject_unreadable"] += 1
            rejected.append((word, freq))
            continue
        ipa = verdict["ipa"]
        stats[f"register_{verdict['register']}"] += 1
        stats[f"why_{verdict['why']}"] += 1
        # The engine keys this table by lexicon_key, which folds final letters,
        # so two spellings can land on one key. Keep the first (highest-freq)
        # and drop a later one that disagrees, rather than letting dict order
        # decide which reading the engine ends up with.
        key = lexicon_key(word)
        if key in by_key and by_key[key] != ipa:
            stats["reject_key_collision"] += 1
            rejected.append((word, freq))
            continue
        by_key[key] = ipa
        stats["accepted"] += 1
        stats["accepted_tokens"] += freq
        if verdict["merged"] != verdict["wh"]:
            stats["register_distinguishable"] += 1
            if verdict["register"] == "merged":
                stats["flipped_to_merged"] += 1
                stats["flipped_tokens"] += freq
                flips.append((word, freq, verdict))
        accepted.append((word, {
            "ipa": ipa,
            "variants": verdict["variants"],
            "pointed": form,
            "register": verdict["register"],
            "why": verdict["why"],
            "n_sources": n_sources,
            "dominance": dominance,
        }, freq))

    stats["quarantine_types"] = len(quarantine)
    stats["quarantine_tokens"] = sum(f for _, f in quarantine)
    accepted.sort(key=lambda t: (-t[2], t[0]))
    rejected.sort(key=lambda t: (-t[1], t[0]))
    flips.sort(key=lambda t: (-t[1], t[0]))
    return accepted, dict(stats), rejected, flips


HEADER = '''"""GENERATED — Sefaria-pointed loshn-koydesh readings.

Source: VERIFIED pointed editions published on Sefaria — *Miqra according to
the Masorah* (Tanakh, CC-BY-SA) and *Torat Emet 357* (Mishnah, Siddur Ashkenaz,
Public Domain); see data/pointed_sources/README.md for the full attribution and
the CC-BY-SA obligations that ride along with this file. Each quarantined type
whose unpointed form has exactly one vocalization in those sources, or a
dominant one at >= {dom:.0%} of occurrences, is read in the register that type is
actually used in (scripts/register_policy.py).

REGISTER. The default is MERGED — read_pointed_merged(), the §5 nikud table as
the engine reads embedded loshn-koydesh: shuruk/kubuts take the near-
exceptionless Yiddish u->i shift, a final komets-hey is [ə]. That is what the
native informant's gold readings show for a Hebrew word sitting inside a
Yiddish sentence (תורה tɔjrə, ברכה brˈuxə, חידוש xˈidiʃ), and it is what these
words are doing here. The Whole-Hebrew register (read_pointed_wh(), spec v2
§7.1: shuruk [u], shva-na [ə], final komets-hey [u]) is the QUOTATION register
and is used only where the evidence says the word is quoted:

  * audio — the clips in data/audio_lexicon/ fit the WH reading better; or
  * usage — >= {wh:.0%} of the type's corpus tokens sit inside a run of >= {span}
    consecutive loshn-koydesh tokens, which is what a cited posuk looks like
    and what running Yiddish does not.

Each entry records which register won ('register') and why ('why'). The LOSING
register is kept as a 'variants' entry rather than thrown away, so a forced
aligner or a reviewer can still choose it.

Rescue #2 for the quarantine, and it ranks BELOW data/lexicons/audio_endorsed_lk.py:
audio evidence outranks book pointing, so words endorsed there are excluded
here. Also excluded: readings outside the closed v3 phone inventory or failing
the §1 vowel-shape rule, pointings whose letters differ from the quarantined
spelling, and keys already owned by a gold/abbreviation/multiword/legacy entry.
Emitted at LOW confidence with reason 'sefaria-pointed' — book pointing of a
quoted posuk is evidence, not a native verdict on how the word is said inside a
Yiddish sentence, so these stay in the verification queue.

{n} entries / {tok:,} quarantined tokens, sorted by corpus frequency.
Regenerate: python scripts/build_sefaria_lexicon.py
"""

SEFARIA_POINTED_LK = {{
'''


def emit(accepted, stats) -> str:
    lines = [HEADER.format(dom=DOMINANCE_MIN, n=len(accepted),
                           wh=WH_SHARE_MIN, span=SPAN_MIN,
                           tok=stats.get("accepted_tokens", 0))]
    for word, rec, freq in accepted:
        lines.append(
            "    %r: {\"ipa\": %r, \"variants\": %r, \"pointed\": %r, "
            "\"register\": %r, \"why\": %r, "
            "\"n_sources\": %d, \"dominance\": %r},  # freq %d\n"
            % (word, rec["ipa"], rec["variants"], rec["pointed"],
               rec["register"], rec["why"], rec["n_sources"],
               rec["dominance"], freq)
        )
    lines.append("}\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--flips", type=int, default=15,
                    help="how many register flips to list")
    args = ap.parse_args()

    accepted, stats, rejected, flips = build()
    if not args.dry_run:
        OUT.write_text(emit(accepted, stats), encoding="utf-8")
        print(f"wrote {OUT}")
    for k in sorted(stats):
        print(f"{k:28s} {stats[k]}")
    tok = stats["accepted_tokens"] / max(stats["quarantine_tokens"], 1)
    print(f"token coverage               {tok:.1%}")
    print("\ntop rescued:")
    for word, rec, freq in accepted[:args.top]:
        print(f"  {word:14s} {freq:6d}  {rec['pointed']:18s} {rec['ipa']}")

    # The point of the change: how many types stopped being read as quotations.
    print(f"\nREGISTER FLIPS (WH -> merged): {len(flips)} types, "
          f"{stats.get('flipped_tokens', 0)} tokens; "
          f"{stats.get('register_distinguishable', 0)} types where the two "
          f"registers differ at all")
    print(f"  {'word':14s} {'freq':>6s}  {'was (WH)':22s} {'now (merged)':22s} why")
    for word, freq, v in flips[:args.flips]:
        print(f"  {word:14s} {freq:6d}  {v['wh']:22s} {v['merged']:22s} {v['why']}")
    kept = [(w, f, r) for w, r, f in accepted
            if r["register"] == "wh" and r["variants"]]
    print(f"\nWH KEPT (quoted register wins): {len(kept)} types")
    for word, freq, rec in kept[:args.flips]:
        print(f"  {word:14s} {freq:6d}  {rec['ipa']:22s} {rec['why']}")

    print("\ntop still quarantined:")
    for word, freq in rejected[:args.top]:
        print(f"  {word:14s} {freq:6d}")


if __name__ == "__main__":
    main()
