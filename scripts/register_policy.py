#!/usr/bin/env python
"""Which REGISTER should a rescued loshn-koydesh word be read in?

Both pointing-based rescue builders (build_sefaria_lexicon.py,
build_model_guess_lexicon.py) hand a pointed form to a reader. Until now both
called read_pointed_wh() unconditionally, which is the register of a QUOTED
posuk: shuruk [u], shva-na [ə], final komets-hey [u]. That is the wrong default.
The native informant's verdicts in the gold CSV (spec v2 §5/§7) say an LK word
EMBEDDED in a Yiddish sentence takes the merged shifts instead — shuruk -> [i]
'near-exceptionless' (חידוש xˈidiʃ, שידוך ʃˈidəx, תשובה tshive), final
komets-hey -> [ə] (תורה tɔjrə, ברכה brˈuxə). Reading תרומה as tərˈumu instead of
trˈimə is not a subtle difference; it is a different word to a listener.

So: MERGED is the default, WH is the exception, and this module decides which
one each type gets, from three sources in descending authority.

  1. AUDIO      — how the word is actually said in the episodes. Decisive when
                  it separates the two readings at all (audio_verdict()).
  2. QUOTED-SPAN SHARE — how the word is USED in this corpus. A type that lives
                  inside quotations gets the quotation register. We have no
                  quote markup, so a span is approximated by a run of >= 3
                  consecutive loshn-koydesh tokens (quoted_shares()): running
                  Yiddish interleaves LK words with Germanic ones, a quoted
                  posuk does not. WH primary at >= 70% quoted share.
  3. DEFAULT    — merged.

The register that LOSES is never discarded: it ships as a variant, so an
aligner or a reviewer can still vote for it. Both tables stay LOW-confidence
and stay in the verification queue either way — this changes which reading is
the queue's starting guess, not how sure we are of it.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from yiddish_g2p import (  # noqa: E402
    STRESS,
    g2p_tokens,
    lexicon_key,
    read_pointed_merged,
    read_pointed_wh,
)

DATASET = REPO / "data" / "yiddish_tts_dataset.tsv"
AUDIO_DIR = REPO / "data" / "audio_lexicon"
HEBREW_VERIFY = AUDIO_DIR / "hebrew_verify.jsonl"
SHARE_CACHE = REPO / "data" / "quoted_share.json"

# A run of this many consecutive LK tokens reads as a quotation.
SPAN_MIN = 3
# ... and a type whose tokens are this often inside one is read as quoted.
WH_SHARE_MIN = 0.70

# Audio arbitration: a clip only votes when it heard the word at all, and the
# two registers only get to disagree when the clip separates them.
AUDIO_MIN_CLIPS = 2
AUDIO_MARGIN = 0.10


# ---------------------------------------------------------------- cache key


def span_fingerprint() -> str:
    """Hash of everything quoted_shares()' measurement depends on.

    The share cache is an ENGINE OUTPUT — quoted_shares() runs g2p_tokens() over
    the corpus — so a cache with no key is a table that cannot be rebuilt from
    the tree that shipped it. Anything that changes which tokens come back with
    layer == 'L', or how tokens are grouped into records, changes the measured
    shares and must invalidate the file.

    That is: the engine's own drift stamp (rules, gold, multiword table,
    clitics), plus the KEY SETS of the four rescue tables and the stemmer's
    parameters. Key sets, not readings: quoted_shares() reads only ``layer`` and
    ``word``, and a rescue's layer is 'L' whichever reading it stores. That is
    what makes the rebuild terminate — re-deciding a register changes the
    tables' IPA but not their keys, so the cache the builders read stays valid
    for the tables they write.
    """
    import yiddish_g2p as _g

    parts = [
        _g.g2p_fingerprint(),
        *(",".join(sorted(t)) for t in (
            _g._AUDIO_ENDORSED, _g._HOMOGRAPH_LK,
            _g._SEFARIA_POINTED, _g._MODEL_POINTED, _g._LK_BARE)),
        repr(_g._GERMANIC_SUFFIX_IPA), repr(sorted(_g._STEM_NO_SPLIT)),
        str(_g._MIN_STEM_ROOT), f"span_min={SPAN_MIN}",
    ]
    import hashlib
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- quoted spans

def _is_lk_token(rec: dict) -> bool:
    """Is this token a loshn-koydesh word, however it was routed?

    ``layer == 'L'`` is the engine's own answer to exactly that question, and it
    is the right one here because it does NOT depend on which table won: a gold
    LK word, an audio-endorsed one, a Sefaria rescue and a bare §6.3 fallback
    are all equally part of a quotation. Keying on route/reason instead would
    have made a span invisible the moment one of its words got rescued, which is
    precisely the direction this measurement is used in.
    """
    return rec.get("layer") == "L" and bool(rec.get("word", "").strip())


def _collapse_repeats(recs: list[dict]) -> list[dict]:
    """Drop immediately-repeated identical tokens.

    The corpus is ASR output and it loops: one row reads ``אכילת כזית כזית כזית
    …`` fourteen times over. Left in, such a loop is BOTH a fake quoted span (a
    run of LK tokens by construction) and a huge fake frequency — כזית measured
    at 1,653 occurrences / 98% quoted before this guard, almost all of it from
    stutters. A word repeated verbatim is evidence of a decoder loop, not of how
    the word is used, so each run contributes exactly one token.
    """
    out: list[dict] = []
    for rec in recs:
        if out and rec.get("word") == out[-1].get("word"):
            continue
        out.append(rec)
    return out


def quoted_shares(limit: int = 0, cache: bool = True) -> dict[str, dict]:
    """Per-type {'total', 'quoted', 'share'} over the corpus, by lexicon_key.

    A token counts as 'quoted' when it sits inside a run of >= SPAN_MIN
    consecutive LK tokens. The run is computed over the token stream of one
    corpus row, so a span never crosses a line boundary.

    NOT a real quotation detector, and it does not pretend to be: three LK words
    in a row is also what a dense rabbinic Yiddish clause looks like. It is
    calibrated for the decision it feeds — a type is only pushed into the WH
    register when MOST of its occurrences (>= WH_SHARE_MIN) are in such runs,
    which the frequent Yiddish-integrated LK words (שבת, תורה, ברוך השם) never
    reach because they also appear alone in ordinary sentences.
    """
    if cache and not limit and SHARE_CACHE.exists():
        blob = json.loads(SHARE_CACHE.read_text(encoding="utf-8"))
        if blob.get("fingerprint") == span_fingerprint():
            return blob["shares"]

    total: dict[str, int] = defaultdict(int)
    quoted: dict[str, int] = defaultdict(int)
    with DATASET.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if limit and i >= limit:
                break
            recs = _collapse_repeats(g2p_tokens(row["text"]))
            flags = [_is_lk_token(r) for r in recs]
            run = 0
            in_span = [False] * len(flags)
            for j, f in enumerate(flags):
                run = run + 1 if f else 0
                if run >= SPAN_MIN:
                    for k in range(j - run + 1, j + 1):
                        in_span[k] = True
            for rec, lk, span in zip(recs, flags, in_span):
                if not lk:
                    continue
                key = lexicon_key(rec["word"])
                total[key] += 1
                if span:
                    quoted[key] += 1

    out = {k: {"total": n, "quoted": quoted.get(k, 0),
               "share": round(quoted.get(k, 0) / n, 4)}
           for k, n in total.items()}
    if cache and not limit:
        SHARE_CACHE.write_text(
            json.dumps({"fingerprint": span_fingerprint(), "shares": out},
                       ensure_ascii=False, sort_keys=True),
            encoding="utf-8")
    return out


# ------------------------------------------------------------------ audio vote

_SPACED = re.compile(r"\s+")


def _phones(spaced: str) -> list[str]:
    return [p for p in _SPACED.split(spaced.strip()) if p]


def _align_score(pred: list[str], heard: list[str]) -> float:
    """Fraction of the predicted phones an aligned transcript matched.

    The same measure xeus_vote.score() uses for variant votes, so an audio
    verdict here means what an audio verdict means everywhere else in the repo.
    """
    from xeus_tag import align  # local: pulls torch-free helpers only

    if not pred:
        return 0.0
    hits = sum(1 for pi, hj in align(pred, heard)
               if pi is not None and hj is not None and pred[pi] == heard[hj])
    return hits / len(pred)


def _load_heard() -> dict[str, list[list[str]]]:
    """word -> [heard phone sequence, ...] from every audio source we have.

    hebrew_verify.jsonl is the relevant one (it targets quarantined LK words
    specifically); the xeus_tags_*.jsonl clip tags are folded in for the types
    that also appear there.
    """
    heard: dict[str, list[list[str]]] = defaultdict(list)
    if HEBREW_VERIFY.exists():
        with HEBREW_VERIFY.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("heard"):
                    heard[lexicon_key(rec["word"])].append(_phones(rec["heard"]))
    for path in glob.glob(str(AUDIO_DIR / "xeus_tags_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("heard"):
                    heard[lexicon_key(rec["word"])].append(_phones(rec["heard"]))
    return heard


_HEARD_CACHE: dict[str, list[list[str]]] | None = None


def audio_verdict(word: str, merged: str, wh: str) -> dict | None:
    """Which register the audio fits better, or None when it cannot say.

    Returns {'winner': 'merged'|'wh'|'tie', 'merged': f, 'wh': f, 'clips': n}.
    None means no usable evidence: fewer than AUDIO_MIN_CLIPS clips, or the two
    readings tokenize identically so no clip could tell them apart.

    'tie' is reported, not silently resolved — the two scores being within
    AUDIO_MARGIN is a real observation (the vowel in question was not heard
    clearly enough to decide) and the caller falls through to the usage-based
    rule, which is a better guess than a coin flip on a noisy margin.
    """
    global _HEARD_CACHE
    from xeus_map import tokenize_g2p_ipa

    m_ph = tokenize_g2p_ipa(merged.replace(" ", ""))
    w_ph = tokenize_g2p_ipa(wh.replace(" ", ""))
    if m_ph == w_ph:
        return None
    if _HEARD_CACHE is None:
        _HEARD_CACHE = _load_heard()
    clips = _HEARD_CACHE.get(lexicon_key(word)) or []
    if len(clips) < AUDIO_MIN_CLIPS:
        return None
    m = sum(_align_score(m_ph, h) for h in clips) / len(clips)
    w = sum(_align_score(w_ph, h) for h in clips) / len(clips)
    winner = "tie" if abs(m - w) < AUDIO_MARGIN else ("merged" if m > w else "wh")
    return {"winner": winner, "merged": round(m, 3), "wh": round(w, 3),
            "clips": len(clips)}


# ------------------------------------------------------- merged sanity checks
#
# The merged reader is the engine's ordinary Hebrew-SCRIPT reader, which has to
# survive text where the pointing is absent or Germanic. Handed a full book
# pointing it is right almost everywhere, and where it is wrong it is wrong
# because an orthographic default beat an explicit point. Those cases are reader
# defects, not register differences: the fix is to take the intact WH reading,
# and to say so in the record rather than quietly ship a mangled primary.


def _nuclei(ipa: str) -> list[str]:
    from xeus_map import VOWELS, tokenize_g2p_ipa

    return [p for p in tokenize_g2p_ipa(ipa.replace(" ", "")) if p in VOWELS]


def merged_drops_a_vowel(merged: str, wh: str) -> bool:
    """Did the merged reader swallow a vowel the POINTING actually writes?

    The merged register is expected to lose syllables — that is what dropping a
    shva-na is (bərˈuxu -> brˈuxə) and it is correct. What it must not lose is a
    FULL point. It occasionally does: כַּזַּיִת, whose yud carries its own chirik,
    comes back kˈazis from the merged path where the WH reader gives kazˈajis —
    a whole syllable of the written word gone.

    The test compares nucleus counts, using the WH reading's non-schwa nuclei as
    the number of full points the word spells (every shva-na and every reduced
    vowel in either register surfaces as [ə], so excluding schwas on the WH side
    and counting them on the merged side is exactly the 'shevas may vanish, real
    vowels may not' rule). Cheaper and steadier than re-parsing the pointing,
    and it cannot disagree with the readers about what they read.
    """
    full = sum(1 for v in _nuclei(wh) if v != "ə")
    return len(_nuclei(merged)) < full


def merged_overrides_a_point(merged: str, wh: str) -> bool:
    """Did an orthographic DEFAULT beat a point the edition actually printed?

    One environment does this, and it is systematic: a word-initial א/ע before a
    cholam-vav (אוֹתוֹ, אוֹתִיּוֹת, עוֹשׂוֹת). The §4 ambiguous-alef default claims
    the sequence and yields [oʊ], the Germanic diaphoneme, where the §5 nikud
    table that DEFINES the merged register maps cholam to [ɔj] — so the reading
    contradicts its own register, not just the WH one. Ten types in the Sefaria
    table, all the same shape.

    Detected on the output ([oʊ] appearing only in the merged reading) rather
    than by re-parsing the pointing, for the same reason as the check above: the
    readers' own answers are the ground truth about what they read.
    """
    return "oʊ" in merged and "oʊ" not in wh


def merged_orphans_an_h(merged: str, wh: str) -> bool:
    """Did dropping a shva-na strand an /h/ with no vowel in front of it?

    Dropping the sheva is the merged register working as intended — בְּרָכָה is
    brˈuxə, not bərˈuxə. But /h/ cannot be the second member of an onset in any
    language, and it is not a possible coda either, so the drop produces a
    string no one can say: תְּהִלִּים -> thˈilim, בְּהֵמָה -> bhˈajmə, קְהָת ->
    khus (which is not even distinguishable from כוס). 239 types in the two
    tables, all the identical shape, all with an intact WH reading sitting right
    next to them.

    The engine knows about this environment already — _word_to_latin() inserts a
    separator when an h lands after s/z/t/k — but only for those four letters
    and only inside the Latin layer. Here the whole reading can simply be
    declined in favour of the WH one, which never drops the sheva.
    """
    from xeus_map import VOWELS, tokenize_g2p_ipa

    def stranded(ipa: str) -> bool:
        ph = tokenize_g2p_ipa(ipa.replace(" ", ""))
        return any(p == "h" and ph[i - 1] not in VOWELS
                   for i, p in enumerate(ph) if i)

    return stranded(merged) and not stranded(wh)


def _stressed_nucleus(ipa: str) -> tuple[int, int | None]:
    """(number of nuclei, index of the stressed one counted from the end).

    The index is negative — -2 is the penult — so two readings with the same
    syllable count can be compared without caring how long they are. ``None``
    when the reading carries no stress mark at all (a monosyllable).
    """
    from xeus_map import VOWELS, tokenize_g2p_ipa

    marked: int | None = None
    phones: list[str] = []
    for chunk in ipa.replace(" ", "").split(STRESS):
        if phones or not chunk:
            marked = len(phones)
        phones.extend(tokenize_g2p_ipa(chunk))
    nuclei = [i for i, p in enumerate(phones) if p in VOWELS]
    if marked is None:
        return len(nuclei), None
    after = [i for i in nuclei if i >= marked]
    return len(nuclei), (nuclei.index(after[0]) - len(nuclei)) if after else None


def merged_mis_stresses(merged: str, wh: str) -> bool:
    """Did the merged reader retract the stress off the penult by one syllable?

    Both readers implement the SAME rule — the §11.5 loshn-koydesh penultimate
    default. So when they read the same pointing into the same number of
    syllables and disagree about which one is stressed, one of them has
    miscounted, and the one that is not on the penult is the one that did.

    In practice this is a single shape, 215 types of it: a segol-segol noun
    under a prefix. הַמֶּלֶךְ comes back hˈamələx (HA-melekh) from the merged path
    against WH hamˈɛlɛx (ha-MEY-lekh) — כְּסֵדֶר kˈasajdər, עֲשֶׂרֶת ˈasərəs,
    יוֹלֶדֶת jˈɔjlədəs all the same way.

    Only the retraction is treated as a defect, not any disagreement. Merged
    stress one syllable LATER than WH is usually merged being right (בְּשָׁעַת is
    biʃˈas, b'SHAS, not WH's bˈiʃas), and those 41 cases are left alone.
    """
    n_m, p_m = _stressed_nucleus(merged)
    n_w, p_w = _stressed_nucleus(wh)
    return n_m == n_w and p_w == -2 and p_m == -3


def merged_is_defective(merged: str, wh: str) -> str:
    """'' if the merged reading is sound, else the name of the defect."""
    if merged_drops_a_vowel(merged, wh):
        return "merged-drops-vowel"
    if merged_overrides_a_point(merged, wh):
        return "merged-overrides-point"
    if merged_orphans_an_h(merged, wh):
        return "merged-orphans-h"
    if merged_mis_stresses(merged, wh):
        return "merged-mis-stresses"
    return ""


# --------------------------------------------------------------- the decision

def decide(word: str, pointed: str, shares: dict[str, dict],
           validate=None) -> dict | None:
    """Pick primary/variant register for one pointed type.

    ``validate(ipa) -> bool`` is the caller's gate (closed inventory + §1 vowel
    shape). A reading that fails it cannot be a primary; if the merged reading
    fails and the WH one passes, WH is used and the fact is logged — the no-drop
    policy outranks the register preference, and shipping the wrong vowel beats
    shipping nothing.

    Returns None only when NEITHER reading is usable (the caller then rejects
    the type exactly as it did before), else a record carrying the primary, the
    losing reading as a variant, and why.
    """
    merged = read_pointed_merged(pointed)
    wh = read_pointed_wh(pointed)
    ok = validate or (lambda _ipa: True)
    m_ok, w_ok = bool(merged) and ok(merged), bool(wh) and ok(wh)
    if not m_ok and not w_ok:
        return None

    audio = audio_verdict(word, merged, wh) if (m_ok and w_ok) else None
    share = shares.get(lexicon_key(word), {})
    q = share.get("share")
    defect = merged_is_defective(merged, wh) if (m_ok and w_ok) else ""

    if not m_ok:
        register, why = "wh", "merged-reading-rejected"
    elif not w_ok:
        register, why = "merged", "wh-reading-rejected"
    elif defect:
        # The merged reading misreads the pointing (a swallowed syllable: לחיים
        # -> lxjim against WH ləxˈajim; or a default overriding a point: אותו ->
        # ˈoʊsɔj against ˈɔjsɔj). This outranks the audio vote deliberately. A
        # reading with one vowel in it can score well against a noisy transcript
        # for the wrong reason — there is almost nothing to match — and lxjim is
        # not a pronunciation of לחיים that any vote should be allowed to
        # select. A defect is not a register difference for evidence to
        # adjudicate, so it is settled before the evidence is consulted.
        register, why = "wh", defect
    elif audio and audio["winner"] == "wh":
        register, why = "wh", "audio-prefers-wh"
    elif audio and audio["winner"] == "merged":
        register, why = "merged", "audio-prefers-merged"
    elif q is not None and q >= WH_SHARE_MIN:
        register, why = "wh", "quoted-span-share"
    else:
        register, why = "merged", "default-embedded"

    primary, other = (merged, wh) if register == "merged" else (wh, merged)
    # The losing REGISTER ships as a variant; a losing DEFECT does not. Both
    # readings of תקון are pronunciations of the word and an aligner should be
    # free to pick either, but thˈilim and kˈazis are not pronunciations of
    # anything, and a variant list is a ballot — putting them on it is how they
    # get chosen. Suppressed precisely when a defect check is what decided the
    # type, which is the only way a defective reading can be the loser.
    keep = bool(other) and other != primary and ok(other) and not defect
    variants = [other] if keep else []
    return {"ipa": primary, "variants": variants, "register": register,
            "why": why, "merged": merged, "wh": wh, "audio": audio,
            "quoted_share": q, "quoted_total": share.get("total")}


def main() -> int:
    """Measure and report the quoted-span shares; refreshes the cache."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    shares = quoted_shares(limit=args.limit, cache=not args.no_cache)
    toks = sum(v["total"] for v in shares.values())
    quo = sum(v["quoted"] for v in shares.values())
    hi = [k for k, v in shares.items() if v["share"] >= WH_SHARE_MIN]
    print(f"{len(shares)} LK types, {toks} LK tokens, {quo} inside a >= "
          f"{SPAN_MIN}-token span ({quo / max(toks, 1):.1%})")
    print(f"{len(hi)} types at >= {WH_SHARE_MIN:.0%} quoted share")
    ranked = sorted(shares.items(), key=lambda kv: -kv[1]["total"])
    print(f"\n{'type':16s} {'total':>6s} {'quoted':>6s} {'share':>6s}")
    for k, v in ranked[:args.top]:
        print(f"{k:16s} {v['total']:6d} {v['quoted']:6d} {v['share']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
