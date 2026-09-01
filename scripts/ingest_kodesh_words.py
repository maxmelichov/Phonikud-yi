#!/usr/bin/env python3
"""Stage the phonetic index of Niborski's loshn-koydesh dictionary for review.

The index (kodesh_words.pdf, 79 pages) is the only large machine-readable source
that pairs an ETYMOLOGICAL loshn-koydesh spelling with an explicit respelling of
how the word is actually said -- exactly the pairing the LK path (spec v3 §6)
cannot recover from orthography and currently has to buy one entry at a time
from audio or from pointed books. 7.5k pairs is roughly twenty times the whole
L-layer plus the Sefaria rescue.

Two things stop it from being a lexicon drop-in, and this script exists to make
both of them visible instead of silent:

  * REGISTER. The respelling is standard (Litvish/YIVO) Yiddish, not the Central
    Yiddish this engine targets. Every row therefore carries BOTH readings: the
    literal standard one, and the Hasidic one obtained by the vowel shift
    (o->u, u->i, ey->ay, ay->aa), with the rules that fired named per row so a
    reviewer can filter on them rather than trust them.
  * PROVENANCE. The source is a copyrighted dictionary index, so nothing here
    writes into data/lexicons/. The output is four review-gated TSVs plus a
    quarantine file; a human decides what, if anything, ships.

Nothing is ever patched to make it pass: a reading that leaves the §1 closed
inventory or fails the §1 vowel-shape rule is REJECTED, and a key already owned
by gold / abbreviations / multiwords / audio / Sefaria is never re-stated -- it
either agrees (and is dropped) or it disagrees (and becomes a conflict row).

    .venv/bin/python scripts/ingest_kodesh_words.py

Re-extracts from the PDF by per-page column crops. pdftotext's whole-page
-layout dump interleaves the two column blocks and loses the ':' source notes,
so the crops are the source of truth; --pairs re-uses a cached JSON dump for a
faster rerun.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yiddish_g2p import (  # noqa: E402
    GOLD_LEXICON,
    STRESS,
    _ABBREVIATIONS,
    _AUDIO_ENDORSED,
    _AUDIO_PE,
    _AUDIO_VOWEL,
    _MULTIWORD,
    _MULTIWORD_LEGACY,
    _SEFARIA_POINTED,
    add_stress,
    ipa_phone_violations,
    latin_to_ipa,
    _nuclei_spans,
    lexicon_key,
    reduce_unstressed,
    violates_vowel_ratio,
)

REPO = Path(__file__).resolve().parent.parent
PDF = REPO.parent / "kodesh_words.pdf"
OUT = REPO / "data" / "kodesh_index"
PAGES = 79
# The index is set in two column blocks per page; the right-hand block is read
# first (RTL). Within a block pdftotext emits <etymological>  <phonetic>.
COLUMN_CROPS = ((300, 295), (0, 300))

# =====================================================================
# EXTRACTION
# =====================================================================
_BIDI = re.compile(r"[‪-‮‎‏]")


def _crop(page: int, x: int, width: int) -> str:
    argv = ["pdftotext", "-f", str(page), "-l", str(page), "-x", str(x),
            "-y", "0", "-W", str(width), "-H", "842", "-layout", str(PDF), "-"]
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout


def extract_records() -> list[list[str]]:
    """Every printed line as its list of whitespace-separated cells."""
    records: list[list[str]] = []
    for page in range(1, PAGES + 1):
        for x, width in COLUMN_CROPS:
            for line in _crop(page, x, width).splitlines():
                line = _BIDI.sub("", line).strip()
                if not line:
                    continue
                records.append([c for c in re.split(r"\s{2,}", line) if c])
    return records


# =====================================================================
# ROW ASSEMBLY (spec of the printed page, not of the engine)
# =====================================================================
# A printed line is normally <hebrew> <phonetic>. The exceptions, all verified
# against the page images:
#   <note> :<hebrew> <phonetic>   -- ':' introduces the form the RIGHT column
#                                    actually respells (source citation, or an
#                                    acronym's expansion)
#   <hebrew> /<alt> <phonetic>    -- '/' offers an alternative spelling; a
#                                    "- /x" alternative replaces only the last
#                                    hyphen component
#   a bare numeral on its own line -- the printed homograph superscript, which
#                                    lands BETWEEN the two columns
#   a bare word on its own line    -- a wrapped continuation of the line above
#   a single cell holding both halves, single-spaced, phonetic first -- long
#                                    multiword entries the column could not hold
_SECTION_HEADS = set("אבגדהוזחטיכלמנסעפצקרשת")
_HEB_ONLY = set("תחשׂ")


class Row:
    __slots__ = ("heb", "phon", "note", "homograph", "variant_of", "page")

    def __init__(self, heb: str, phon: str, note: str = "", homograph: str = "",
                 variant_of: str = "") -> None:
        self.heb = heb
        self.phon = phon
        self.note = note
        self.homograph = homograph
        self.variant_of = variant_of


def _split_alternatives(cell: str) -> tuple[str, list[str]]:
    """(primary form, alternative spellings) for a cell carrying '/' variants."""
    parts = [p.strip(" -–") for p in cell.split("/")]
    primary = parts[0].strip()
    alts: list[str] = []
    for alt in parts[1:]:
        if not alt:
            continue
        if "–" in primary and "–" not in alt:
            # "A–B /C" replaces the final component only.
            alts.append(primary.rsplit("–", 1)[0] + "–" + alt)
        else:
            alts.append(alt)
    return primary, alts


def _merged_cell(cell: str) -> tuple[str, str] | None:
    """Split a single cell that holds phonetic and hebrew halves, or None.

    These lines are always a multiword entry whose two halves have the same
    number of words, so the seam is the middle space; the phonetic half comes
    first and never uses the LK-only letters ת ח שׂ.
    """
    words = cell.split()
    if len(words) < 2 or len(words) % 2:
        return None
    half = len(words) // 2
    phon, heb = " ".join(words[:half]), " ".join(words[half:])
    if _HEB_ONLY & set(phon) or not (_HEB_ONLY & set(heb) or "–" in heb or "'" in heb):
        return None
    return heb, phon


def assemble(records: list[list[str]]) -> tuple[list[Row], list[list[str]]]:
    """Printed lines -> Rows, plus the cells no rule could place."""
    rows: list[Row] = []
    dropped: list[list[str]] = []
    # A homograph superscript arrives between its two columns, so the hebrew
    # half is already parked here when the numeral shows up.
    pending_heb: str | None = None
    pending_hom = ""
    pending_note = ""

    def emit(heb: str, phon: str, note: str = "", hom: str = "") -> None:
        heb, heb_alts = _split_alternatives(heb)
        phon, phon_alts = _split_alternatives(phon)
        rows.append(Row(heb, phon, note, hom))
        for alt in heb_alts:
            rows.append(Row(alt, phon, note, hom, variant_of=heb))
        for alt in phon_alts:
            rows.append(Row(heb, alt, note, hom, variant_of=heb))

    for cells in records:
        if pending_heb is not None and len(cells) == 1 and cells[0].strip().isdigit():
            pending_hom = cells[0].strip()
            continue
        if pending_heb is not None and len(cells) == 1 and not cells[0].strip().isdigit():
            emit(pending_heb, cells[0], pending_note, pending_hom)
            pending_heb, pending_hom, pending_note = None, "", ""
            continue

        note = ""
        colon = [i for i, c in enumerate(cells) if c.startswith(":")]
        if colon:
            i = colon[0]
            note = " ".join(cells[:i])
            cells = [cells[i][1:].strip()] + cells[i + 1:]

        if len(cells) >= 2:
            heb = cells[0]
            phon = " ".join(cells[1:])
            if phon.strip().isdigit():
                pending_heb, pending_hom, pending_note = heb, phon.strip(), note
                continue
            emit(heb, phon, note)
            continue

        cell = cells[0].strip()
        if cell in _SECTION_HEADS or cell.isdigit():
            continue
        merged = _merged_cell(cell)
        if merged:
            emit(merged[0], merged[1], note)
            continue
        if rows and "–" not in cell and " " not in cell:
            # A wrapped tail. It belongs to whichever half of the previous row
            # was left hanging on a dash, else to the (longer) hebrew half.
            prev = rows[-1]
            if prev.phon.endswith("–"):
                prev.phon += cell
            elif prev.heb.endswith("–"):
                prev.heb += cell
            else:
                prev.heb += " " + cell
            continue
        dropped.append(cells)
    return rows, dropped


# =====================================================================
# READING THE PHONETIC COLUMN
# =====================================================================
# The right column is standard YIVO orthography. Every symbol below is scanned
# longest-first; the two Latin renderings differ ONLY in the vowels, so the
# nuclei line up one-for-one and a stress index computed on one transfers to
# the other.
#
#   symbol   standard   Central/Hasidic   shift rule
#   ------   --------   ---------------   ----------
#   אַ        a          a                 -
#   אָ        ɔ          u                 o>u
#   ע        ɛ          ɛ                  -
#   י / יִ    i          i                 -
#   ו / וּ    u          i                 u>i
#   יי       ej         aj                 ey>ay
#   ײַ        aj         aː                ay>aa
#   וי       ɔj         ɔj                 -
#
# The engine's Latin labels are Central-Yiddish-valued (see the LATIN LABEL KEY
# in yiddish_g2p), which is why the standard column needs its own labels: the
# engine writes 'oo' for /u/ and 'ey' for /aj/.
# YIVO's yud-with-khirik (ייִד, מציאות -> מעציִעס) is the one vowel the font does
# not round-trip: pdftotext emits the yud followed by an asterisk. It is folded
# to a private-use character up front so the plain "יי" digraph can never eat
# the first half of a "ייִ" sequence.
HIRIQ_YOD = "\ue000"
_HIRIQ_YOD_PRINTED = "י*"
# Concatenating two Latin pieces can forge a digraph the source never wrote:
# ע+ע would read as "ee" (/ej/) and ט+ס as "ts". A geresh is spliced in between
# -- latin_to_ipa and _nuclei_spans both skip it, so it costs nothing.
_FORGED_DIGRAPHS = frozenset({"ee", "ey", "ay", "oy", "ou", "oo",
                              "ts", "sh", "zh", "kh"})
_VOWELS: list[tuple[str, str, str, str]] = [
    # (source symbol, standard latin, hasidic latin, shift rule name)
    ("אָ", "o", "oo", "o>u"),
    ("אַ", "a", "a", ""),
    ("ײַ", "ey", "ay", "ay>aa"),
    ("וּ", "oo", "u", "u>i"),
    ("וי", "oy", "oy", ""),
    ("יי", "ee", "ey", "ey>ay"),
    (HIRIQ_YOD, "i", "i", ""),  # see _HIRIQ_YOD_PRINTED
    ("ע", "e", "e", ""),
    ("ו", "oo", "u", "u>i"),
    ("י", "i", "i", ""),
]
# The consonant DIGRAPHS have to be scanned before the vowels: וו is /v/, and
# the vowel table would otherwise read it as two vov-vowels (אַבֿרהם's respelling
# אַווראָם came out *auurom).
_CONSONANT_DIGRAPHS: list[tuple[str, str]] = [
    ("דזש", "dzh"), ("טש", "tsh"), ("זש", "zh"), ("וו", "v"),
]
_CONSONANTS: list[tuple[str, str]] = [
    ("ב", "b"), ("ג", "g"), ("ד", "d"), ("ה", "h"), ("ז", "z"), ("ט", "t"),
    ("כ", "kh"), ("ך", "kh"), ("ל", "l"), ("מ", "m"), ("ם", "m"),
    ("נ", "n"), ("ן", "n"), ("ס", "s"), ("פּ", "p"), ("פֿ", "f"), ("ף", "f"),
    ("צ", "ts"), ("ץ", "ts"), ("ק", "k"), ("ר", "r"), ("ש", "sh"),
]
_NFC = lambda t: unicodedata.normalize("NFC", t)  # noqa: E731
_VOWELS = [(_NFC(s), a, b, c) for s, a, b, c in _VOWELS]
_CONSONANT_DIGRAPHS = [(_NFC(s), a) for s, a in _CONSONANT_DIGRAPHS]
_CONSONANTS = [(_NFC(s), a) for s, a in _CONSONANTS]
_PHON_ALLOWED = set("".join(s for s, *_ in _VOWELS)
                    + "".join(s for s, _ in _CONSONANT_DIGRAPHS + _CONSONANTS)
                    + "א־–- ()" + _HIRIQ_YOD_PRINTED)


class ReadError(Exception):
    """The phonetic cell holds something the YIVO letter table cannot read."""


def read_phonetic(cell: str) -> tuple[str, str, set[str]]:
    """(standard latin, hasidic latin, shift rules fired) for one respelling.

    A bare (unpointed) א is YIVO's silent vowel carrier and is dropped; a י that
    opens a syllable in front of another vowel is the consonant j.
    """
    cell = unicodedata.normalize("NFC", cell).replace("(", "").replace(")", "")
    cell = cell.replace(_HIRIQ_YOD_PRINTED, HIRIQ_YOD)
    std: list[str] = []
    has: list[str] = []
    fired: set[str] = set()

    def push(a: str, b: str) -> None:
        for buf, piece in ((std, a), (has, b)):
            if buf and buf[-1] and (buf[-1][-1] + piece[:1]) in _FORGED_DIGRAPHS:
                buf.append("'")
            buf.append(piece)

    i, n = 0, len(cell)
    while i < n:
        ch = cell[i]
        if ch == " ":
            push(" ", " ")
            i += 1
            continue
        if ch in "-–־":
            push("=", "=")
            i += 1
            continue
        if ch == "א" and _is_carrier(cell, i + 1):
            i += 1
            continue
        for src, lat in _CONSONANT_DIGRAPHS:
            if cell.startswith(src, i):
                push(lat, lat)
                i += len(src)
                break
        else:
            for src, s_lat, h_lat, rule in _VOWELS:
                if cell.startswith(src, i):
                    if src == "י" and (not std or std[-1] in " =") and \
                            _starts_vowel(cell, i + 1):
                        push("y", "y")
                    elif src == HIRIQ_YOD and std and std[-1][-1] in "aeiou":
                        # A khirik-yud after a vowel is [ji], not a bare [i]:
                        # חיים is xˈajim and ירושלים jəruʃulˈajim, both gold.
                        push("yi", "yi")
                    else:
                        push(s_lat, h_lat)
                        if rule:
                            fired.add(rule)
                    i += len(src)
                    break
            else:
                for src, lat in _CONSONANTS:
                    if cell.startswith(src, i):
                        push(lat, lat)
                        i += len(src)
                        break
                else:
                    raise ReadError(
                        f"unreadable {ch!r} ({unicodedata.name(ch, '?')})")
    return "".join(std), "".join(has), fired


def _starts_vowel(cell: str, at: int) -> bool:
    """Whether a vowel symbol other than a plain yud begins at ``at``."""
    return any(cell.startswith(src, at) for src, *_ in _VOWELS if src != "י")


def _is_carrier(cell: str, at: int) -> bool:
    """Whether the א before ``at`` is a silent carrier rather than a consonant.

    It is one in front of any vowel symbol, and in front of the bare vowel
    letters י and ו (איבעד, אומען) -- but not in front of וו, which is /v/.
    """
    if cell.startswith("וו", at):
        return False
    return cell[at:at + 1] in ("י", "ו") or _starts_vowel(cell, at)


# =====================================================================
# RULE CLASSES ON TOP OF THE LETTER-BY-LETTER READING
# =====================================================================
# Both rules below are asserted by the ingest brief and CONTRADICTED by the
# native-verified gold (מוסדות mˈɔjzdəs keeps -əs; כח kˈɔjəx, פסח pˈajsəx,
# רוח rˈiəx, שכח ʃkˈɔjəx all keep the ...əx shape). They are applied so the
# staging file reflects the brief, they name themselves in shift_rules_fired,
# and every row they touch is marked needs_review -- that is what makes them
# one grep away from being reverted.
DISPUTED_RULES = {"sfx-ות>ojs", "furtive-patakh"}


def apply_rule_classes(heb: str, std: str, has: str) -> tuple[str, str, set[str]]:
    """Suffix ־ות and the final-ח furtive patakh, applied to the Latin forms."""
    fired: set[str] = set()
    heb_tail = heb.split("–")[-1].split()[-1] if heb.strip() else ""
    if heb_tail.endswith("ות") and has.endswith("es"):
        std, has = (f[:-2] + ("'oys" if f[-3:-2] == "o" else "oys") for f in (std, has))
        fired.add("sfx-ות>ojs")
    if heb_tail.endswith("ח") and has.endswith("ekh"):
        # Hebrew's furtive patah is an /a/, not the schwa the respelling writes.
        std = std[:-3] + "akh"
        has = has[:-3] + "akh"
        fired.add("furtive-patakh")
    return std, has, fired


def stress(latin: str) -> str:
    """§11.5 penult retraction, applied per space token and per compound head.

    A hyphenated LK compound carries ONE stress, on its final element -- the
    shape the engine's own multiword table uses (bis-mˈɛdrəʃ, ʃˈabəs kˈɔjdəʃ).
    """
    out = []
    for token in latin.split(" "):
        if not token:
            continue
        parts = token.split("=")
        parts[-1] = add_stress(parts[-1], penult=True)
        out.append("=".join(parts))
    return " ".join(out)


def _stress_index(marked: str) -> list[int | None]:
    """Which nucleus of each space token carries the mark (None = unmarked)."""
    idx: list[int | None] = []
    for token in marked.split(" "):
        if STRESS not in token:
            idx.append(None)
            continue
        before = token[: token.index(STRESS)].replace("=", "")
        idx.append(len(_nuclei_spans(before)))
    return idx


def _apply_stress_index(latin: str, idx: list[int | None]) -> str:
    """Put the mark on the same nucleus number of a parallel Latin string."""
    out = []
    for token, want in zip(latin.split(" "), idx):
        if want is None:
            out.append(token)
            continue
        flat = token.replace("=", "")
        spans = _nuclei_spans(flat)
        if want >= len(spans):
            out.append(token)
            continue
        at = spans[want][0]
        # Re-inflate the hyphens the nucleus count ignored.
        seen = 0
        for pos, ch in enumerate(token):
            if ch != "=":
                if seen == at:
                    out.append(token[:pos] + STRESS + token[pos:])
                    break
                seen += 1
        else:
            out.append(token)
    return " ".join(out)


def to_ipa(std_latin: str, has_latin: str) -> tuple[str, str]:
    marked = stress(has_latin)
    std_marked = _apply_stress_index(std_latin, _stress_index(marked))
    return tuple(reduce_unstressed(latin_to_ipa(m)).replace("=", "-")
                 for m in (std_marked, marked))


# =====================================================================
# PLURAL-UNDER-SINGULAR DETECTION
# =====================================================================
# ~1/6 of the index gives the PLURAL pronunciation under a singular headword
# (נס -> ניסים, בעל–נס -> באַלע–ניסים). Ingesting those as the headword's reading
# would poison the lexicon, so they are quarantined, never merged.
_MATRES = set("אוהי")
_PLURAL_HEADS = {"בעל": "באַלע"}


def plural_reason(heb: str, phon: str) -> str:
    """Why this row is a plural under a singular headword, or ''."""
    heb_parts = re.split(r"[–\s]+", heb.strip())
    phon_parts = re.split(r"[–\s]+", phon.replace("*", "").strip())
    if not heb_parts or not phon_parts:
        return ""
    for h, p in zip(heb_parts, phon_parts):
        if h in _PLURAL_HEADS and p.startswith(_PLURAL_HEADS[h]):
            return f"head-plural {h}>{p}"
    h_tail, p_tail = heb_parts[-1], phon_parts[-1]
    if p_tail.endswith("ים") and not h_tail.endswith(("ים", "ם")):
        return "-im surplus"
    # A headword in bare ם is still singular (עולם -> אוילעמעס is the plural);
    # only the Hebrew plural ־ים blocks the test.
    if p_tail.endswith("עס") and not h_tail.endswith(("ות", "ים", "ת", "ס", "ץ")):
        return "-es surplus"
    return ""


def skeleton_key(key: str) -> str:
    return "".join(c for c in key if c not in _MATRES)


# =====================================================================
# OWNED KEYS (never re-stated, only agreed with or reported against)
# =====================================================================
def owning_tiers() -> list[tuple[str, dict]]:
    """(tier name, key -> primary IPA) for every table that outranks this one."""
    def prim(table, field):
        out = {}
        for k, v in table.items():
            ipa = v[0] if isinstance(v, tuple) else (v.get(field) or v.get("ipa"))
            if ipa:
                out[k] = ipa
        return out

    return [
        ("gold", prim(GOLD_LEXICON, "ipa_primary")),
        ("abbreviation", prim(_ABBREVIATIONS, "ipa_primary")),
        ("multiword", prim(_MULTIWORD, "ipa_primary")),
        ("multiword-legacy", {k: "" for k in _MULTIWORD_LEGACY}),
        ("audio-endorsed", prim(_AUDIO_ENDORSED, "ipa_primary")),
        ("audio-pe", prim(_AUDIO_PE, "ipa_primary")),
        ("audio-vowel", prim(_AUDIO_VOWEL, "ipa_primary")),
        ("sefaria-pointed", prim(_SEFARIA_POINTED, "ipa_primary")),
    ]


# =====================================================================
# PIPELINE
# =====================================================================
COLUMNS = ["word_key", "hebrew_as_printed", "phonetic_as_printed", "standard_ipa",
           "hasidic_ipa", "shift_rules_fired", "join_tier", "status", "reason"]
PERIPHRASTIC = ("זײַן", "ווערן", "זיך")


def build(records: list[list[str]]) -> dict:
    rows, dropped = assemble(records)
    stats: Counter[str] = Counter()
    stats["printed_lines"] = len(records)
    stats["rows_assembled"] = len(rows)
    stats["unplaceable_cells"] = len(dropped)

    staged: list[dict] = []
    plurals: list[dict] = []
    rejects: Counter[str] = Counter()

    for row in rows:
        heb = unicodedata.normalize("NFC", row.heb).strip()
        phon = unicodedata.normalize("NFC", row.phon).strip()
        if not heb or not phon or "…" in heb or "…" in phon:
            stats["dropped_stub"] += 1
            continue
        periphrastic = ""
        heb_words = heb.split()
        if len(heb_words) > 1 and heb_words[-1] in PERIPHRASTIC:
            periphrastic = heb_words[-1]
            heb = " ".join(heb_words[:-1])
            stats["periphrastic"] += 1
        if not set(phon) <= _PHON_ALLOWED:
            rejects["phonetic-cell-not-yivo"] += 1
            continue
        try:
            std_latin, has_latin, fired = read_phonetic(phon)
        except ReadError as exc:
            rejects[f"read-error:{exc}"] += 1
            continue
        std_latin, has_latin, rule_fired = apply_rule_classes(heb, std_latin, has_latin)
        fired |= rule_fired
        std_ipa, has_ipa = to_ipa(std_latin, has_latin)

        bad = ipa_phone_violations(has_ipa) + ipa_phone_violations(std_ipa)
        if bad:
            rejects["phone-inventory:" + "".join(sorted(set(bad)))] += 1
            continue
        if violates_vowel_ratio(has_ipa):
            rejects["vowel-ratio"] += 1
            continue

        notes = []
        if row.note:
            notes.append("note=" + row.note)
        if row.homograph:
            notes.append("homograph=" + row.homograph)
        if row.variant_of:
            notes.append("variant-of=" + row.variant_of)
        if periphrastic:
            notes.append("periphrastic=" + periphrastic)

        # An acronym line prints <acronym> :<expansion>; the acronym is the
        # form a corpus token will actually be, so it -- not the expansion --
        # is the key. A non-acronym note is a source citation and keys normally.
        key_form = row.note if '"' in row.note else heb
        rec = {
            "word_key": lexicon_key(key_form.replace("–", "-")),
            "hebrew_as_printed": heb,
            "phonetic_as_printed": phon,
            "standard_ipa": std_ipa,
            "hasidic_ipa": has_ipa,
            "shift_rules_fired": ",".join(sorted(fired)),
            "notes": notes,
            "is_multiword": ("–" in heb or " " in heb or "-" in row.phon or " " in phon),
            "is_abbrev": ('"' in heb or bool(row.note)),
        }
        reason = plural_reason(heb, phon)
        if reason:
            rec["reason"] = reason
            plurals.append(rec)
            continue
        staged.append(rec)

    # --- join tiers -------------------------------------------------------
    by_key: dict[str, set[str]] = defaultdict(set)
    for rec in staged:
        by_key[rec["word_key"]].add(rec["hasidic_ipa"])
    by_skel: dict[str, set[str]] = defaultdict(set)
    for key, readings in by_key.items():
        by_skel[skeleton_key(key)] |= readings

    owners = owning_tiers()
    conflicts: list[dict] = []
    kept: list[dict] = []
    for rec in staged:
        key = rec["word_key"]
        owner = next(((name, table[key]) for name, table in owners if key in table), None)
        if owner is not None:
            name, theirs = owner
            if theirs and theirs != rec["hasidic_ipa"]:
                conflicts.append({**rec, "owner_tier": name, "owner_ipa": theirs})
                stats["owned_disagree"] += 1
                stats[f"owned_disagree:{name}"] += 1
            else:
                stats["owned_agree"] += 1
                stats[f"owned_agree:{name}"] += 1
            continue
        if len(by_key[key]) > 1:
            rec["join_tier"] = "ambiguous"
            rec["status"] = "needs_review"
            rec["reason"] = f"homograph: {len(by_key[key])} readings share the key"
        else:
            unambiguous = len(by_skel[skeleton_key(key)]) == 1
            rec["join_tier"] = "exact+skeleton" if unambiguous else "exact"
            rec["status"] = "clean"
            rec["reason"] = ""
        if DISPUTED_RULES & set(rec["shift_rules_fired"].split(",")):
            rec["status"] = "needs_review"
            rec["reason"] = "disputed rule fired; gold keeps the unshifted shape"
        if rec["notes"]:
            rec["status"] = "needs_review"
            rec["reason"] = (rec["reason"] + "; " if rec["reason"] else "") + \
                "; ".join(rec["notes"])
        kept.append(rec)

    return {"kept": kept, "plurals": plurals, "conflicts": conflicts,
            "stats": stats, "rejects": rejects, "dropped": dropped}


def write_tsv(path: Path, recs: list[dict], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for rec in sorted(recs, key=lambda r: (r["word_key"], r["hasidic_ipa"],
                                           r["phonetic_as_printed"])):
        lines.append("\t".join(str(rec.get(c, "")).replace("\t", " ") for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path,
                    help="cached JSON dump of extract_records() (skips pdftotext)")
    args = ap.parse_args()

    if args.records and args.records.exists():
        records = json.loads(args.records.read_text(encoding="utf-8"))
    else:
        records = extract_records()
        if args.records:
            args.records.write_text(json.dumps(records, ensure_ascii=False),
                                    encoding="utf-8")

    result = build(records)
    OUT.mkdir(parents=True, exist_ok=True)

    singles = [r for r in result["kept"] if not r["is_multiword"] and not r["is_abbrev"]]
    multiwords = [r for r in result["kept"] if r["is_multiword"] and not r["is_abbrev"]]
    abbreviations = [r for r in result["kept"] if r["is_abbrev"]]
    write_tsv(OUT / "singles.tsv", singles, COLUMNS)
    write_tsv(OUT / "multiwords.tsv", multiwords, COLUMNS)
    write_tsv(OUT / "abbreviations.tsv", abbreviations, COLUMNS)
    write_tsv(OUT / "quarantine_plurals.tsv", result["plurals"],
              COLUMNS[:6] + ["reason"])
    write_tsv(OUT / "conflicts.tsv", result["conflicts"],
              COLUMNS[:6] + ["owner_tier", "owner_ipa"])

    stats = result["stats"]
    print("=== data/kodesh_index ===")
    for name, recs in (("singles.tsv", singles), ("multiwords.tsv", multiwords),
                       ("abbreviations.tsv", abbreviations),
                       ("quarantine_plurals.tsv", result["plurals"]),
                       ("conflicts.tsv", result["conflicts"])):
        print(f"  {name:26s} {len(recs):6d}")
    print("--- pipeline ---")
    for key in sorted(stats):
        print(f"  {key:34s} {stats[key]:6d}")
    print("--- status ---")
    for status, n in Counter(r["status"] for r in result["kept"]).most_common():
        print(f"  {status:34s} {n:6d}")
    print("--- join tier ---")
    for tier, n in Counter(r["join_tier"] for r in result["kept"]).most_common():
        print(f"  {tier:34s} {n:6d}")
    print("--- shift rules fired ---")
    rules: Counter[str] = Counter()
    for rec in result["kept"] + result["plurals"] + result["conflicts"]:
        for rule in rec["shift_rules_fired"].split(","):
            if rule:
                rules[rule] += 1
    for rule, n in rules.most_common():
        print(f"  {rule:34s} {n:6d}")
    print("--- rejected (never patched) ---")
    for reason, n in result["rejects"].most_common(12):
        print(f"  {reason:34s} {n:6d}")


if __name__ == "__main__":
    main()
