#!/usr/bin/env python
"""Mine loshn-koydesh lexicon candidates from Gemini-annotated audio.

For every annotated chunk we align text_yi tokens against ipa tokens (only chunks
where the two token counts match are used — a cheap but high-precision filter).
For each aligned pair we run the rule engine (yiddish_g2p.hebrew_to_ipa) on the
word and compare it to the observed IPA. A word is a candidate when:

  1. rule output != observed output (after normalisation), AND
  2. the word looks like loshn-koydesh (Hebrew-origin) rather than Germanic:
     Hebrew-only letters (ת ח שׂ תּ כּ ...), low vowel-letter density, LK
     suffix/prefix shapes, or membership in an optional Hebrew wordlist.

Output: data/lk_candidates.tsv with columns word, rule_ipa, observed_ipa, count
sorted by count desc. This is the Yiddish analogue of Phonikud's "manually fix
the 1K most common words" step: review the top of this file and paste the
confirmed entries into _LOSHN_KOYDESH in yiddish_g2p.py.

Usage:
  .venv/bin/python scripts/mine_lk_lexicon.py
  .venv/bin/python scripts/mine_lk_lexicon.py --min-count 2 --include-known
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonikud_yi.gateway import iter_jsonl  # noqa: E402
from yiddish_g2p import (  # noqa: E402
    _LK_ALL,
    _WORD_LATIN,
    _strip_points,
    hebrew_to_ipa,
    normalize_ipa_affricates,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ANNOT_DIR = DATA / "annotations"
OUT_TSV = DATA / "lk_candidates.tsv"
HEBREW_WORDLIST = DATA / "hebrew_wordlist.txt"

HEB = re.compile(r"[֐-׿]")
# Keep ' and " inside words: gershayim marks acronyms (תשפ"ו) which must survive intact.
NON_WORD = re.compile(r"[^֐-׿'\"]+")

# Letters that essentially never occur in the Germanic component of Yiddish.
LK_ONLY_LETTERS = set("תחט")  # ט is weak; scored lower below
LK_STRONG_LETTERS = set("תח")
# Yiddish vowel letters (Germanic component spells every syllable nucleus).
VOWEL_LETTERS = set("אעוי")


# Notation folding only — never fold distinct phonemes, those are the signal.
_IPA_FOLD = [
    ("ʦ", "ts"), ("ʧ", "tʃ"), ("ʤ", "dʒ"), ("ʣ", "dz"),  # ligatures -> sequences
    ("ɡ", "g"),  # U+0261 (rule engine) -> ASCII g (Gemini)
    ("ː", ""), ("ˈ", ""), ("ˌ", ""), ("ʔ", ""), ("ˑ", ""), ("͡", ""),
]


def normalize_ipa(s: str) -> str:
    s = normalize_ipa_affricates(unicodedata.normalize("NFC", s))
    for a, b in _IPA_FOLD:
        s = s.replace(a, b)
    s = re.sub(r"[^\wɐ-ʯͰ-Ͽᴀ-ᵿ]", "", s, flags=re.UNICODE)
    return s.strip().lower()


def load_hebrew_wordlist() -> set[str]:
    if not HEBREW_WORDLIST.exists():
        return set()
    return {
        _strip_points(w.strip())
        for w in HEBREW_WORDLIST.read_text(encoding="utf-8").splitlines()
        if w.strip()
    }


def lk_score(word: str, hebrew_words: set[str]) -> float:
    """Heuristic 0..1+ that `word` belongs to the loshn-koydesh component."""
    bare = _strip_points(word)
    letters = [c for c in bare if HEB.match(c)]
    if len(letters) < 2:
        return 0.0
    if bare in hebrew_words:
        return 1.0

    score = 0.0
    if '"' in word:  # gershayim acronym (roshei-teyves) — always Hebrew-component
        score += 1.0
    if any(c in LK_STRONG_LETTERS for c in letters):
        score += 0.6
    if "ט" in letters and any(c in LK_STRONG_LETTERS for c in letters):
        score += 0.05

    # vowel-letter density: Germanic Yiddish is ~>=0.33; LK is often <0.25
    density = sum(1 for c in letters if c in VOWEL_LETTERS) / len(letters)
    if density == 0.0:
        score += 0.6
    elif density < 0.25:
        score += 0.4
    elif density < 0.34:
        score += 0.15

    # LK morphology: feminine ־ה, plural ־ות / ־ים, construct ־ת
    if bare.endswith("ה") and len(letters) >= 3 and bare[-2] not in VOWEL_LETTERS:
        score += 0.25
    if bare.endswith(("ות", "ים")):
        score += 0.2
    # pointed forms that only exist in LK spelling
    if any(c in word for c in ("כּ", "פּ", "תּ", "שׂ", "בֿ", "פֿ")) and "אַ" not in word:
        score += 0.2

    # Germanic giveaways knock it back down
    if re.search(r"אַ|אָ|ײַ|ױ|וו|יי|ער\b|ען\b|געזאָגט", word):
        score -= 0.35
    if bare.startswith("גע"):
        score -= 0.3
    return max(0.0, score)


def token_pairs(rec: dict) -> list[tuple[str, str, str]]:
    """Yield (word, pointed_word, ipa) triples for word-aligned chunks."""
    text, ipa = rec.get("text_yi") or "", rec.get("ipa") or ""
    if not text or not ipa:
        return []
    yi = [t for t in text.split() if HEB.search(t)]
    ip = ipa.split()
    if not yi or len(yi) != len(ip):
        return []
    pointed = [t for t in (rec.get("text_yi_pointed") or "").split() if HEB.search(t)]
    if len(pointed) != len(yi):
        pointed = [""] * len(yi)
    out = []
    # Gemini often emits precomposed ligatures; fold to standard digraphs so
    # word keys match yiddish_g2p lexicons and the plain-text spelling.
    def fold(s: str) -> str:
        return s.replace("ײ", "יי").replace("ױ", "וי").replace("װ", "וו").replace("״", '"')

    for w, pt, p in zip(yi, pointed, ip):
        w = fold(NON_WORD.sub("", w))
        pt = fold(NON_WORD.sub("", pt))
        p = normalize_ipa(p)
        if w and p:
            out.append((w, pt, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default=str(ANNOT_DIR))
    ap.add_argument("--out", default=str(OUT_TSV))
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--min-score", type=float, default=0.5, help="LK heuristic threshold")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument(
        "--include-known",
        action="store_true",
        help="also emit words already in yiddish_g2p's lexicons",
    )
    args = ap.parse_args()

    annot_dir = Path(args.annotations)
    files = sorted(annot_dir.glob("*.jsonl")) if annot_dir.is_dir() else [annot_dir]
    if not files:
        print(f"no annotation files under {annot_dir}", file=sys.stderr)
        return 2

    hebrew_words = load_hebrew_wordlist()
    known = {_strip_points(k) for k in _LK_ALL} | {_strip_points(k) for k in _WORD_LATIN}

    # word -> Counter of observed IPA variants; word -> Counter of pointed spellings
    observed: dict[str, Counter] = defaultdict(Counter)
    pointed_forms: dict[str, Counter] = defaultdict(Counter)
    total_pairs = usable_chunks = seen_chunks = 0

    for f in files:
        for rec in iter_jsonl(f):
            seen_chunks += 1
            conf = rec.get("confidence")
            if args.min_confidence and isinstance(conf, (int, float)) and conf < args.min_confidence:
                continue
            pairs = token_pairs(rec)
            if pairs:
                usable_chunks += 1
            for w, pt, p in pairs:
                total_pairs += 1
                observed[w][p] += 1
                if pt:
                    pointed_forms[w][pt] += 1

    rows = []
    for word, variants in observed.items():
        bare = _strip_points(word)
        if not args.include_known and bare in known:
            continue
        score = lk_score(word, hebrew_words)
        if score < args.min_score:
            continue
        obs_ipa, count = variants.most_common(1)[0]
        if count < args.min_count:
            continue
        rule_ipa = hebrew_to_ipa(word)
        if normalize_ipa(rule_ipa) == obs_ipa:
            continue
        pointed = ""
        if word in pointed_forms and pointed_forms[word]:
            pointed = pointed_forms[word].most_common(1)[0][0]
        rows.append((word, pointed, rule_ipa, obs_ipa, count, score))

    rows.sort(key=lambda r: (-r[4], -r[5], r[0]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("word\tpointed\trule_ipa\tobserved_ipa\tcount\n")
        for word, pointed, rule_ipa, obs_ipa, count, _ in rows:
            fh.write(f"{word}\t{pointed}\t{rule_ipa}\t{obs_ipa}\t{count}\n")

    print(
        f"chunks={seen_chunks} aligned={usable_chunks} pairs={total_pairs} "
        f"distinct_words={len(observed)} candidates={len(rows)}\n-> {out}"
    )
    for word, pointed, rule_ipa, obs_ipa, count, score in rows[:15]:
        print(f"  {word}\t{pointed}\trule={rule_ipa}\tobs={obs_ipa}\tn={count}\tlk={score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
