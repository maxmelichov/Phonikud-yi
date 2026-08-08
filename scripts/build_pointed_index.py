#!/usr/bin/env python3
"""Build a pointed-Hebrew index from verified published texts (Sefaria).

Reads the raw JSON downloaded into data/pointed_sources/raw/ (Tanakh = "Miqra
according to the Masorah", Mishnah + Siddur Ashkenaz = "Torat Emet 357") and
emits data/pointed_sources/pointed_index.jsonl:

    {"k": <unpointed form>, "n": 1, "t": <total>, "p": [[<pointed>, count], ...]}

`k` is the NFC, nikud/te'amim-stripped surface form (final letters exactly as
printed).  `p` is sorted by descending count.  Phrase entries (n = 2, 3) are
verse-internal n-grams restricted to n-grams all of whose words occur in the
quarantine type list, which keeps the index small.

Usage:
    python scripts/build_pointed_index.py            # build index
    python scripts/build_pointed_index.py --coverage # + coverage report
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "pointed_sources" / "raw"
OUT_DIR = ROOT / "data" / "pointed_sources"
INDEX_PATH = OUT_DIR / "pointed_index.jsonl"
COVERAGE_PATH = OUT_DIR / "coverage_report.md"
# The full-corpus quarantine (14,732 types / 100,827 tokens) snapshotted from
# `run_corpus_v3.py --limit 0`.  data/phonemized/v3/quarantine.tsv is whatever the
# last (possibly --limit'ed) run left behind, so it is not a stable reference set.
QUARANTINE = OUT_DIR / "quarantine_full_snapshot.tsv"

# Combining marks to drop when producing the unpointed key: te'amim (U+0591-05AF),
# nikud (U+05B0-05BD), rafe (U+05BF), shin/sin dots (U+05C1-05C2), and the
# remaining point-like marks (U+05C4-05C7, incl. qamats qatan).
_STRIP = re.compile("[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4-\u05c7]")
# Marks dropped from the *pointed* value: te'amim (U+0591-05AF), meteg/siluq
# (U+05BD), rafe (U+05BF), upper/lower dot (U+05C4-05C5), nun hafukha (U+05C6).
# Kept: nikud + dagesh (U+05B0-05BC), shin/sin dots (U+05C1-05C2) and qamats
# qatan (U+05C7), which MAM marks explicitly and which is real vowel information.
_TEAMIM = re.compile("[\u0591-\u05af\u05bd\u05bf\u05c4-\u05c6]")
_HAS_NIKUD = re.compile("[\u05b0-\u05bc\u05c1\u05c2\u05c7]")
# Separators inside a verse: maqaf, paseq, sof pasuk, nun hafukha, and ASCII/HTML noise.
_SEP = re.compile("[־׀׃׆\\s]+")
_TAG = re.compile(r"<[^>]+>")
_HEB_LETTER = re.compile("[א-ת]")
# Combining marks that belong to the word: nikud + dagesh (U+05B0-05BC), meteg
# (U+05BD), rafe (U+05BF), shin/sin dots (U+05C1-05C2), upper/lower dot
# (U+05C4-05C5), qamats qatan (U+05C7) and the te'amim (U+0591-05AF). Excluded
# on purpose: maqaf (U+05BE), paseq (U+05C0), sof pasuq (U+05C3) and nun
# hafukha (U+05C6), which are punctuation and are handled by _SEP.
_MARK = "\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7"
# Trailing/leading punctuation that is not part of a word form. The trailing
# class MUST spare _MARK: a point on the LAST letter of a word (the holam of
# אוֹתוֹ, the shuruk dagesh of אֲנַחְנוּ, the komets of תּוֹרָתֶךָ) sits at the end
# of the string, and stripping it silently turned a mater vav into a
# consonantal /v/ and dropped the 2nd-person suffix's vowel.
_EDGE = re.compile(r"^[^א-ת]+|[^א-ת׳״'\"" + _MARK + r"]+$")


def strip_points(s: str) -> str:
    """NFC form with all nikud / cantillation removed."""
    return unicodedata.normalize("NFC", _STRIP.sub("", unicodedata.normalize("NFC", s)))


_DAGESH, _QAMATS_QATAN, _QAMATS, _VAV = "\u05bc", "\u05c7", "\u05b8", "\u05d5"
# Holam (U+05B9) and holam haser for vav (U+05BA) — editions disagree on which.
_HOLAMS = ("\u05b9", "\u05ba")
# Letters where a dagesh changes the consonant in the Ashkenazi reading.
_DAGESH_PHONEMIC = set("\u05d1\u05db\u05e4\u05ea")  # bet kaf pe tav


def phonemic_fold(pointed: str) -> str:
    """Collapse pointing differences that are edition convention, not sound.

    MAM and Torat Emet disagree on marks that do not change the Ashkenazi
    reading, which otherwise splits one vocalization across several index
    entries.  Folded here: gemination dagesh outside bet/kaf/pe/tav (not
    realized), qamats qatan (written only by MAM), and the holam dot over a
    male vav (Torat Emet often omits it).  Marks that *do* change the sound —
    shin vs sin dot, dagesh in bet/kaf/pe/tav — are preserved.

    This is a comparison key only; the index stores the forms as printed.
    """
    out = []
    base = ""
    n_letters = 0
    for ch in unicodedata.normalize("NFC", pointed):
        if ch == _DAGESH and (n_letters == 1 or base not in _DAGESH_PHONEMIC):
            # Word-initial dagesh is always lene, so it is non-contrastive; the
            # editions write it inconsistently (כֹּהֵן / כֹהֵן).
            continue
        if ch == _QAMATS_QATAN:
            out.append(_QAMATS)
            continue
        if ch in _HOLAMS and base == _VAV:
            continue
        if not unicodedata.combining(ch):
            base = ch
            n_letters += 1
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def iter_strings(node):
    if isinstance(node, str):
        if node.strip():
            yield node
    elif isinstance(node, list):
        for x in node:
            yield from iter_strings(x)
    elif isinstance(node, dict):
        for x in node.values():
            yield from iter_strings(x)


def load_segments():
    """Yield (source_label, segment_text) for every raw file."""
    for path in sorted(RAW.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.stem == "Siddur_Ashkenaz":
            label = "Siddur Ashkenaz (Torat Emet 357)"
            for node in data.values():
                for seg in iter_strings(node.get("text", [])):
                    yield label, seg
            continue
        versions = data.get("versions") or []
        if not versions:
            continue
        v = versions[0]
        label = f"{data.get('book', path.stem)} ({v.get('versionTitle')})"
        for seg in iter_strings(v.get("text", [])):
            yield label, seg


def tokenize(segment: str):
    """Split a segment into pointed word tokens, te'amim removed.

    Cantillation carries no vowel information for the G2P but fragments the
    index badly (הַמֶּ֔לֶך / הַמֶּ֖לֶך / הַמֶּֽלֶך are one vocalization), so it is
    stripped from the stored pointed form.
    """
    seg = _TAG.sub(" ", segment)
    words = []
    for raw in _SEP.split(seg):
        w = _EDGE.sub("", unicodedata.normalize("NFC", raw))
        w = unicodedata.normalize("NFC", _TEAMIM.sub("", w))
        if w and _HEB_LETTER.search(w):
            words.append(w)
    return words


def load_quarantine_types(path: Path = QUARANTINE):
    types = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            w = unicodedata.normalize("NFC", row["word"])
            types[w] = int(row["freq"])
    return types


def build(verbose=True):
    quar = load_quarantine_types()
    quar_keys = {strip_points(w) for w in quar}

    uni = defaultdict(Counter)   # unpointed -> Counter(pointed)
    big = defaultdict(Counter)
    tri = defaultdict(Counter)
    n_seg = 0
    sources = Counter()

    for label, seg in load_segments():
        toks = tokenize(seg)
        if not toks:
            continue
        n_seg += 1
        sources[label.split(" (")[-1]] += 1
        keys = [strip_points(t) for t in toks]
        # A token with no nikud at all carries no pointing information (some
        # Siddur/Mishnah segments are unvocalized); it must not become a candidate.
        pointed = [bool(_HAS_NIKUD.search(t)) for t in toks]
        for k, t, ok in zip(keys, toks, pointed):
            if k and ok:
                uni[k][t] += 1
        for n, store in ((2, big), (3, tri)):
            for i in range(len(toks) - n + 1):
                ks = keys[i:i + n]
                if all(ks) and all(pointed[i:i + n]) and all(k in quar_keys for k in ks):
                    store[" ".join(ks)][" ".join(toks[i:i + n])] += 1
        if verbose and n_seg % 20000 == 0:
            print(f"  {n_seg} segments, {len(uni)} types", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with INDEX_PATH.open("w", encoding="utf-8") as fh:
        for n, store in ((1, uni), (2, big), (3, tri)):
            for k in sorted(store):
                c = store[k]
                rec = {
                    "k": k,
                    "n": n,
                    "t": sum(c.values()),
                    "p": [[p, n_] for p, n_ in c.most_common()],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    if verbose:
        print(
            f"segments={n_seg} unigram_types={len(uni)} bigrams={len(big)} "
            f"trigrams={len(tri)} rows={written} "
            f"size={INDEX_PATH.stat().st_size / 1e6:.1f}MB",
            file=sys.stderr,
        )
    return uni, quar


# Geresh / gershayim (and their ASCII stand-ins) mark an abbreviation, not a word.
_ABBREV = re.compile(r"""[\u05f3\u05f4'"]""")
# Yiddish/Hebrew clitic prefixes: vav, bet, kaf, lamed, mem, shin, he.
_PREFIXES = "\u05d5\u05d1\u05db\u05dc\u05de\u05e9\u05d4"


def _strip_prefix_hit(uni, word: str) -> bool:
    """True if the word hits the index once one or two clitic prefixes come off."""
    k = strip_points(word)
    for n in (1, 2):
        if len(k) > n + 1 and all(c in _PREFIXES for c in k[:n]) and k[n:] in uni:
            return True
    return False


def _bucketize(uni, quar, fold):
    """fold=False: compare pointed forms as printed. fold=True: phonemic_fold."""
    buckets = {"single": [], "dominant": [], "conflict": [], "miss": []}
    for word, freq in quar.items():
        c = uni.get(strip_points(word))
        if c and fold:
            merged = Counter()
            for p, n in c.items():
                merged[phonemic_fold(p)] += n
            c = merged
        if not c:
            b = "miss"
        elif len(c) == 1:
            b = "single"
        else:
            top = c.most_common(1)[0][1]
            b = "dominant" if top / sum(c.values()) >= 0.80 else "conflict"
        buckets[b].append((word, freq))
    return buckets


def _table(buckets, n_types, n_toks):
    rows = []
    for key, title in (
        ("single", "(a) exactly one pointing"),
        ("dominant", "(b) dominant pointing (>=80%)"),
        ("conflict", "(c) multiple conflicting pointings"),
        ("miss", "(d) no hit in index"),
    ):
        items = buckets[key]
        tk = sum(f for _, f in items)
        rows.append((title, len(items), len(items) / n_types, tk, tk / n_toks))
    lines = [
        "| bucket | types | % types | tokens | % tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for title, ct, pt, tk, ptk in rows:
        lines.append(f"| {title} | {ct:,} | {pt:6.1%} | {tk:,} | {ptk:6.1%} |")
    return rows, lines


def coverage(uni, quar):
    buckets = _bucketize(uni, quar, fold=True)
    strict = _bucketize(uni, quar, fold=False)

    n_types = len(quar)
    n_toks = sum(quar.values())
    rows, main_tbl = _table(buckets, n_types, n_toks)
    _, strict_tbl = _table(strict, n_types, n_toks)

    lines = [
        "# Pointed-source coverage of the v3 quarantine",
        "",
        "Source: Sefaria verified pointed texts (Tanakh = *Miqra according to the",
        "Masorah*, CC-BY-SA; Mishnah + Siddur Ashkenaz = *Torat Emet 357*, Public",
        "Domain). Index: `data/pointed_sources/pointed_index.jsonl`.",
        "",
        f"Quarantine: **{n_types:,} types / {n_toks:,} tokens** "
        "(`data/pointed_sources/quarantine_full_snapshot.tsv`, produced by "
        "`run_corpus_v3.py --limit 0`).",
        "",
        "## Coverage (phonemic fold)",
        "",
        "Candidate pointings are compared after `phonemic_fold()`, which collapses",
        "differences that do not change the Ashkenazi reading (gemination dagesh",
        "outside bet/kaf/pe/tav, qamats qatan, the holam dot over a male vav) and",
        "which the two editions write inconsistently. Cantillation is already",
        "stripped at index time.",
        "",
    ] + main_tbl + [
        "",
        "### Same buckets, comparing pointings exactly as printed",
        "",
        "The gap between the two tables is pure edition convention, i.e. conflicts",
        "that need no linguistic decision.",
        "",
    ] + strict_tbl

    hit_t = n_types - len(buckets["miss"])
    hit_k = n_toks - sum(f for _, f in buckets["miss"])
    usable_t = len(buckets["single"]) + len(buckets["dominant"])
    usable_k = sum(f for _, f in buckets["single"]) + sum(f for _, f in buckets["dominant"])
    lines += [
        "",
        f"**Any hit:** {hit_t:,} types ({hit_t / n_types:.1%}) / "
        f"{hit_k:,} tokens ({hit_k / n_toks:.1%}).",
        "",
        f"**Directly usable (a + b):** {usable_t:,} types ({usable_t / n_types:.1%}) / "
        f"{usable_k:,} tokens ({usable_k / n_toks:.1%}).",
        "",
        "## Top 40 unresolved conflicts (bucket c), by quarantine token frequency",
        "",
        "Genuine ambiguity: these need a homograph decision (usually context), not",
        "just a source. Alternatives shown folded.",
        "",
        "| word | freq | pointings (count) |",
        "|---|---:|---|",
    ]
    for w, f in sorted(buckets["conflict"], key=lambda x: -x[1])[:40]:
        merged = Counter()
        for p, n in uni[strip_points(w)].items():
            merged[phonemic_fold(p)] += n
        alts = ", ".join(f"{p} ({n})" for p, n in merged.most_common(4))
        lines.append(f"| {w} | {f:,} | {alts} |")

    lines += [
        "",
        "## What the misses are (bucket d)",
        "",
    ]
    miss = buckets["miss"]
    non_heb = [(w, f) for w, f in miss if not _HEB_LETTER.search(w)]
    abbrev = [(w, f) for w, f in miss if _HEB_LETTER.search(w) and _ABBREV.search(w)]
    rest = [(w, f) for w, f in miss
            if _HEB_LETTER.search(w) and not _ABBREV.search(w)]
    prefixed = [(w, f) for w, f in rest if _strip_prefix_hit(uni, w)]
    plain = [(w, f) for w, f in rest if not _strip_prefix_hit(uni, w)]
    mt = sum(f for _, f in miss)
    lines += [
        "| kind | types | tokens | note |",
        "|---|---:|---:|---|",
        f"| no Hebrew letter (Latin, digits, phone numbers) | {len(non_heb):,} | "
        f"{sum(f for _, f in non_heb):,} | out of scope for a Hebrew source |",
        f"| abbreviation (geresh / gershayim) | {len(abbrev):,} | "
        f"{sum(f for _, f in abbrev):,} | needs an expansion table, not pointing |",
        f"| Hebrew word, hit after stripping a clitic prefix | {len(prefixed):,} | "
        f"{sum(f for _, f in prefixed):,} | reachable by prefix-aware lookup |",
        f"| Hebrew word, genuinely absent | {len(plain):,} | "
        f"{sum(f for _, f in plain):,} | mostly Talmudic/Aramaic and modern coinages |",
        "",
        f"Total bucket (d): {len(miss):,} types / {mt:,} tokens.",
        "",
        "Prefix-aware lookup (strip one or two of "
        "ו/ב/כ/ל/מ/ש/ה and retry) is the single "
        "biggest available gain and is left to the consumer, since the prefix must "
        "be re-vocalized by the engine rather than read off the source.",
        "",
        "## Top 40 misses (bucket d), by quarantine token frequency",
        "",
        "| word | freq |",
        "|---|---:|",
    ]
    for w, f in sorted(miss, key=lambda x: -x[1])[:40]:
        lines.append(f"| {w} | {f:,} |")
    lines.append("")

    COVERAGE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return rows, (hit_t, hit_k, usable_t, usable_k, n_types, n_toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()
    uni, quar = build()
    if args.coverage:
        rows, tot = coverage(uni, quar)
        for title, ct, pt, tk, ptk in rows:
            print(f"{title:40s} {ct:6,} types ({pt:6.1%})  {tk:7,} tokens ({ptk:6.1%})")
        print(f"wrote {COVERAGE_PATH}")


if __name__ == "__main__":
    main()
