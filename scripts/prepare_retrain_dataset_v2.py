#!/usr/bin/env python3
"""Retrain dataset v2 = v1 + the Hebrew evidence stamped in.

This is a MERGER over the v1 build, not a second builder.  v1
(``scripts/prepare_retrain_dataset.py``) decides supervision from the FROZEN
engine's lexicon route and the canonical pointing convention; everything it
masked for a *loshn-koydesh* reason -- quarantined type, homograph, no lexicon
entry -- stayed masked, which is why the Hebrew end of the corpus contributes
almost no supervision.  Since then three sources of Hebrew evidence landed:

  data/lexicons/audio_endorsed_lk.py  107 types whose corpus ``text_pointed`` reading was
                             confirmed against episode audio.  The evidence is
                             the *reading*, and the form that carries it is the
                             row's OWN text_pointed token -- that is the string
                             the recognizer agreed with -- so this pass stamps
                             per occurrence, never the type's citation form.
  data/lexicons/homograph_lk.py       215 voted types.  A type-level winner is NOT
                             evidence about a particular sentence (that is what
                             makes it a homograph), so nothing is stamped here.
                             The 778 occurrences the audio decider settled
                             individually are already stamped in
                             train_unmasked.jsonl by scripts/unmask_homographs.py
                             and are simply carried through.
  data/lexicons/sefaria_pointed_lk.py 3,647 types with a book pointing accepted from the
                             verified Sefaria editions.  Type-level and
                             context-free, but a quoted posuk is spelled the way
                             the edition spells it, so the accepted pointed form
                             is stamped wherever the letters match.

Rank follows the engine's own rescue order: audio outranks book pointing, and a
voted homograph type blocks the Sefaria stamp rather than being overwritten by
it.  Only tokens v1 left UNSUPERVISED are touched; an existing stamp is never
re-decided.

Letter identity is the invariant that makes the dataset trainable: ``text`` must
be exactly ``pointed`` with the marks removed.  Evidence forms are allowed to
differ from the corpus token by final-letter form only (a quoted מ vs the
corpus's ם); when they do, the marks are transferred onto the corpus letters
rather than the corpus letters being replaced.  Anything else -- a different
letter, a different length -- is counted and skipped, never guessed.

Output: data/retrain2/{train,val,test}.jsonl + dataset_stats.md.
test.jsonl is a byte-for-byte copy of v1's: the test set is the measuring stick
and this pass must not move it.

Usage:  .venv/bin/python scripts/prepare_retrain_dataset_v2.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import shutil
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yiddish_g2p as G  # FROZEN - read only  # noqa: E402

csv.field_size_limit(10_000_000)

V1 = REPO / "data" / "retrain"
OUTDIR = REPO / "data" / "retrain2"
CORPUS = REPO / "data" / "corpus" / "yiddish_tts_dataset.tsv"
TEST_EPISODE = "100313"


def _load_v1_module():
    """Reuse v1's normalisation verbatim -- one convention, one definition."""
    path = REPO / "scripts" / "prepare_retrain_dataset.py"
    spec = importlib.util.spec_from_file_location("_prep_v1", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_prep_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


P1 = _load_v1_module()
norm = P1.norm
strip_marks = P1.strip_marks
HEB_RE = P1.HEB_RE
IN_CONVENTION = P1.IN_CONVENTION

from data.lexicons.audio_endorsed_lk import AUDIO_ENDORSED_LK  # noqa: E402
from data.lexicons.homograph_lk import HOMOGRAPH_LK  # noqa: E402
from data.lexicons.sefaria_pointed_lk import SEFARIA_POINTED_LK  # noqa: E402

try:  # audio-confirmed pe flips (scripts/build_audio_pe_lexicon.py)
    from data.lexicons.audio_pe_lk import AUDIO_PE_LK  # noqa: E402
except ImportError:
    AUDIO_PE_LK = {}

try:  # audio-confirmed vowel corrections (scripts/build_audio_vowel_lexicon.py)
    from data.lexicons.audio_vowel_lk import AUDIO_VOWEL_LK  # noqa: E402
except ImportError:
    AUDIO_VOWEL_LK = {}

# Rescued readings the xeus LK sweep heard CONTRADICTED in the audio
# (scripts/xeus_lk_sweep.py, verdict SUSPECT). A suspect reading is never
# stamped as a training target: teaching the model a reading the audio
# refutes is worse than leaving the word unsupervised.
_LK_VOTES = Path(__file__).resolve().parent.parent / "data" / "audio_lexicon" / "lk_sweep_votes.tsv"


def _load_suspect_keys() -> set[str]:
    if not _LK_VOTES.exists():
        return set()
    with _LK_VOTES.open(encoding="utf-8") as fh:
        return {row["key"] for row in csv.DictReader(fh, delimiter="\t")
                if row["verdict"] == "SUSPECT"}


SUSPECT_KEYS = _load_suspect_keys()


# ------------------------------------------------------------------- helpers

def key_of(word: str) -> str:
    """§2.1 lookup key: marks stripped, final forms and ligatures folded."""
    return G.lexicon_key(word)


def letters(word: str) -> str:
    """Letter skeleton with no folding -- what strip_marks must reproduce."""
    return strip_marks(unicodedata.normalize("NFC", word))


def transfer_marks(pointed: str, target_bare: str) -> str | None:
    """Re-hang ``pointed``'s marks on ``target_bare``'s letters.

    Used when the evidence form and the corpus token are the same word spelled
    with a different final-letter form: the reading is the evidence, the
    spelling is the corpus's, and the row invariant demands the corpus's.
    Returns None if the two do not line up letter-for-letter.
    """
    pointed = unicodedata.normalize("NFC", pointed)
    src_letters = [ch for ch in pointed if not unicodedata.combining(ch)]
    if len(src_letters) != len(target_bare):
        return None
    out: list[str] = []
    i = -1
    for ch in pointed:
        if unicodedata.combining(ch):
            out.append(ch)
        else:
            i += 1
            out.append(target_bare[i])
    return "".join(out)


def fit_to_token(pointed: str, bare: str) -> str | None:
    """The stamp to emit for ``bare``, or None if the letters are not the same.

    Exact skeleton match passes through untouched; a match that needs only
    final-letter folding is repaired onto the corpus letters.
    """
    pointed = unicodedata.normalize("NFC", pointed)
    if letters(pointed) == bare:
        return pointed
    if key_of(pointed) != key_of(bare):
        return None
    return transfer_marks(pointed, bare)


def read_merged(pointed: str) -> str | None:
    """The reading of a pointed form on the same path the endorsement used.

    data/lexicons/audio_endorsed_lk.py's ``ipa`` came from
    ``hebrew_to_ipa(text_pointed, stress=True, quarantine=False)``
    (scripts/xeus_verify_hebrew.py); reading a candidate any other way would
    compare two different things.
    """
    try:
        return G.hebrew_to_ipa(pointed, stress=True, quarantine=False)
    except Exception:
        return None


def load_corpus_pointed() -> dict[str, str]:
    out: dict[str, str] = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["id"]] = row.get("text_pointed") or ""
    return out


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------- the stamper

class Stamper:
    def __init__(self, corpus_pointed: dict[str, str]) -> None:
        self.corpus_pointed = corpus_pointed
        self.counts: collections.Counter[str] = collections.Counter()
        self.by_word: collections.Counter[str] = collections.Counter()
        self.evidence_keys = (set(AUDIO_ENDORSED_LK) | set(HOMOGRAPH_LK)
                              | set(SEFARIA_POINTED_LK) | set(AUDIO_PE_LK)
                              | set(AUDIO_VOWEL_LK))
        # all audio tables carry {"ipa": ...} and stamp through try_audio
        self.audio_all = {**AUDIO_VOWEL_LK, **AUDIO_PE_LK, **AUDIO_ENDORSED_LK}
        # key -> {pointed form seen in some row's text_pointed: occurrences}.
        # Filled by the collect pass, collapsed to ONE form per key by
        # choose_canonical(); empty while collecting.
        self.audio_seen: dict[str, collections.Counter[str]] = \
            collections.defaultdict(collections.Counter)
        self.audio_canonical: dict[str, str] = {}
        self.collecting = True

    # -- audio ------------------------------------------------------------
    def aligned_corpus_tokens(self, row: dict) -> list[str] | None:
        """text_pointed split to the row's own token grid, or None.

        text_pointed is the corpus's UNVERIFIED tier and is not letter-faithful
        to ``text`` (it silently re-spells matres lectionis), so alignment is
        only claimed when the token counts agree; per-token letter identity is
        then checked individually at the token that is being stamped.
        """
        raw = self.corpus_pointed.get(row["id"])
        if not raw:
            return None
        ctoks = norm(raw).split()
        if len(ctoks) != len(row["pointed"].split()):
            return None
        return ctoks

    def corpus_cores_by_key(self, row: dict) -> dict[str, set[str]]:
        """key -> the distinct pointed cores text_pointed uses in this row.

        The fallback for rows whose text_pointed tokenizes to a different
        length than ``pointed`` (it re-spells matres lectionis and occasionally
        splits differently).  It is only ever consulted when the row spells the
        type exactly one way, so "the row's own reading of this word" is still
        unambiguous -- and the reading check below still has to pass.
        """
        raw = self.corpus_pointed.get(row["id"]) or ""
        out: dict[str, set[str]] = collections.defaultdict(set)
        for tok in norm(raw).split():
            _, ccore, _ = G.split_affixes(tok)
            if ccore:
                out[key_of(ccore)].add(ccore)
        return out

    def try_audio(self, core: str, key: str, ctok: str | None,
                  fallback: dict[str, set[str]]) -> str | None:
        """The stamp for one audio-endorsed occurrence, or None.

        The *evidence* is per occurrence: this row's own text_pointed form has
        to exist and to read back as the endorsed IPA.  The *target* is per
        type: a diacritic-restoration model may not be shown two different
        pointings of the same letters, so the form actually written is the
        canonical one chosen in choose_canonical().  That is a free swap --
        every candidate reads to the same endorsed IPA, which is the whole
        content of the audio endorsement -- and it is what keeps one type from
        carrying five conflicting labels (תפיסה used to carry five).
        """
        entry = self.audio_all[key]
        ccore = None
        if ctok is not None:
            _, cand, _ = G.split_affixes(ctok)
            if cand and key_of(cand) == key:
                ccore = cand
        if ccore is None:
            cands = fallback.get(key) or set()
            if len(cands) == 1:
                ccore = next(iter(cands))
                if not self.collecting:
                    self.counts["audio_via_row_unique_form"] += 1
        if ccore is None:
            if not self.collecting:
                self.counts["audio_skip_no_row_form"] += 1
            return None
        stamp = fit_to_token(ccore, core)
        if stamp is None:
            if not self.collecting:
                self.counts["audio_skip_letters_differ"] += 1
            return None
        ipa = read_merged(ccore)
        if ipa is None or ipa != entry["ipa"]:
            if not self.collecting:
                self.counts["audio_skip_reading_disagrees"] += 1
            return None
        if self.collecting:
            self.audio_seen[key][unicodedata.normalize("NFC", ccore)] += 1
            return None
        canon = self.audio_canonical.get(key, ccore)
        stamp = fit_to_token(canon, core)
        if stamp is None:
            # the canonical form cannot be re-hung on these letters (a spelling
            # this key folds together but that has a different length). Skipped
            # rather than stamped with a second target for the same type.
            self.counts["audio_skip_canonical_unfit"] += 1
            return None
        self.counts["stamp_audio_endorsed"] += 1
        return stamp

    def choose_canonical(self) -> None:
        """One pointed target per audio-endorsed type.

        Among the forms this corpus actually uses for the type -- all of which
        already read back as the endorsed IPA -- take the most explicitly
        pointed one, because a bare form teaches the model to restore nothing.
        Ties break on corpus frequency, then on codepoint order, so the choice
        is a function of the inputs and not of iteration order.
        """
        def rank(item: tuple[str, int]) -> tuple[int, int, str]:
            form, n = item
            marks = sum(1 for ch in form if unicodedata.combining(ch))
            return (marks, n, form)

        for key, forms in self.audio_seen.items():
            self.audio_canonical[key] = max(forms.items(), key=rank)[0]
            if len(forms) > 1:
                self.counts["audio_types_with_rival_forms"] += 1
        self.collecting = False

    # -- sefaria ----------------------------------------------------------
    def try_sefaria(self, core: str, key: str) -> str | None:
        stamp = fit_to_token(SEFARIA_POINTED_LK[key]["pointed"], core)
        if stamp is None:
            self.counts["sefaria_skip_letters_differ"] += 1
            return None
        self.counts["stamp_sefaria"] += 1
        return stamp

    # -- driver -----------------------------------------------------------
    def row(self, row: dict) -> dict:
        # the ``pointed`` grid is the grid ``supervised`` indexes.  It is NOT
        # always ``text``'s: one v1 row carries stray bare-pasekh "tokens" that
        # strip to nothing and vanish from ``text``.  Bare tokens are therefore
        # derived from ``pointed``, which keeps mask, target and source aligned.
        ptoks = row["pointed"].split()
        mask = list(row["supervised"])
        if len(ptoks) != len(mask):
            raise AssertionError(f"row {row['id']}: token/mask grid mismatch")
        ctoks: list[str] | None = None
        cmap: dict[str, set[str]] = {}
        ctoks_done = False
        changed = False

        for pos, ptok in enumerate(ptoks):
            lead, core, trail = G.split_affixes(strip_marks(ptok))
            if not core or not HEB_RE.search(core):
                continue
            key = key_of(core)
            if key not in self.evidence_keys:
                continue
            if key in SUSPECT_KEYS:
                if not self.collecting:
                    self.counts["suspect_not_stamped"] += 1
                continue  # audio refutes this reading; never a training target
            if self.collecting and key not in self.audio_all:
                continue  # the collect pass only surveys the audio forms
            if not self.collecting:
                self.counts["evidence_class_tokens"] += 1
            if mask[pos]:
                if not self.collecting:
                    self.counts["already_supervised"] += 1
                continue

            if key in self.audio_all:
                if not ctoks_done:
                    ctoks = self.aligned_corpus_tokens(row)
                    cmap = self.corpus_cores_by_key(row)
                    ctoks_done = True
                stamp = self.try_audio(core, key,
                                       ctoks[pos] if ctoks else None, cmap)
                source = ("audio-pe" if key in AUDIO_PE_LK
                          else "audio-vowel" if key in AUDIO_VOWEL_LK
                          else "audio")
            elif key in HOMOGRAPH_LK:
                # type-level winner only: this occurrence was never decided
                self.counts["homograph_type_not_stamped"] += 1
                continue  # unreachable while collecting
            else:
                stamp = self.try_sefaria(core, key)
                source = "sefaria"
            if stamp is None:
                continue
            if letters(stamp) != core:
                raise AssertionError(
                    f"row {row['id']} pos {pos}: stamp {stamp!r} does not "
                    f"strip to {core!r}")
            ptoks[pos] = lead + stamp + trail
            mask[pos] = True
            self.by_word[f"{source}:{key}"] += 1
            changed = True

        if changed:
            row = dict(row)
            row["pointed"] = " ".join(ptoks)
            row["supervised"] = mask
            row["n_supervised"] = sum(mask)
            if strip_marks(row["pointed"]) != row["text"]:
                raise AssertionError(f"row {row['id']}: letter identity broken")
            self.counts["rows_changed"] += 1
        return row


# ------------------------------------------------------------------ counting

def token_stats(rows: list[dict], evidence_keys: set[str]) -> dict[str, int]:
    """Token counts, overall and for the loshn-koydesh evidence classes.

    HEB_RE matches Yiddish too, so ``heb`` is effectively every token -- the
    population v1's headline share was computed over.  ``ev`` is the
    loshn-koydesh population this pass exists to supervise.
    """
    st = collections.Counter()
    for row in rows:
        st["all"] += row["n_tokens"]
        st["all_sup"] += sum(row["supervised"])
        for ptok, sup in zip(row["pointed"].split(), row["supervised"]):
            lead, core, trail = G.split_affixes(strip_marks(ptok))
            if not core or not HEB_RE.search(core):
                continue
            st["heb"] += 1
            st["heb_sup"] += bool(sup)
            if key_of(core) in evidence_keys:
                st["ev"] += 1
                st["ev_sup"] += bool(sup)
    return st


def pct(a: int, b: int) -> str:
    return f"{(a / b * 100):.2f}%" if b else "n/a"


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_in = read_jsonl(V1 / "train_unmasked.jsonl")
    val_in = read_jsonl(V1 / "val.jsonl")

    stamper = Stamper(load_corpus_pointed())
    before = token_stats(train_in + val_in, stamper.evidence_keys)

    # pass 1: survey the per-occurrence audio forms and fix one target per type
    for r in train_in + val_in:
        stamper.row(r)
    stamper.choose_canonical()

    # pass 2: stamp
    train = [stamper.row(r) for r in train_in]
    val = [stamper.row(r) for r in val_in]
    after = token_stats(train + val, stamper.evidence_keys)

    # ------------------------------------------------------------- gates
    fail: list[str] = []
    for split, rows in (("train", train), ("val", val)):
        for rec in rows:
            if strip_marks(rec["pointed"]) != rec["text"]:
                fail.append(f"{split}/{rec['id']}: pointed does not strip to text")
                break
            ptoks = rec["pointed"].split()
            if not (len(ptoks) == len(rec["supervised"]) == rec["n_tokens"]):
                fail.append(f"{split}/{rec['id']}: mask length != token count")
                break
            if rec["n_supervised"] != sum(rec["supervised"]):
                fail.append(f"{split}/{rec['id']}: n_supervised stale")
                break
            if rec["episode"] == TEST_EPISODE:
                fail.append(f"{split}/{rec['id']}: episode {TEST_EPISODE} leaked")
                break

    stray: collections.Counter[str] = collections.Counter()
    for rows in (train, val):
        for rec in rows:
            for ch in rec["pointed"]:
                if unicodedata.combining(ch) and ch not in IN_CONVENTION:
                    stray[ch] += 1
    if stray:
        fail.append("out-of-convention marks: "
                    + ", ".join(f"U+{ord(c):04X}x{n}" for c, n in stray.items()))

    # one pointed target per letter-identical type among this pass's stamps:
    # identical letters carrying different labels is noise for a
    # diacritic-restoration model, however consistent the phonetics.
    targets: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    for old, new in zip(train_in + val_in, train + val):
        n_p = new["pointed"].split()
        for i, (os_, ns) in enumerate(zip(old["supervised"], new["supervised"])):
            if ns and not os_:
                _, ncore, _ = G.split_affixes(n_p[i])
                targets[letters(ncore)][unicodedata.normalize("NFC", ncore)] += 1
    rival = {b: c for b, c in targets.items() if len(c) > 1}
    if rival:
        worst = sorted(rival.items(), key=lambda kv: -sum(kv[1].values()))[:5]
        fail.append(
            f"{len(rival)} type(s) stamped with >1 distinct pointed target: "
            + "; ".join(f"{b} {dict(c)}" for b, c in worst))

    # nothing may be UN-supervised by this pass, and no v1 stamp may move
    for old, new in zip(train_in + val_in, train + val):
        if old["id"] != new["id"]:
            fail.append("row order changed")
            break
        o_p, n_p = old["pointed"].split(), new["pointed"].split()
        for i, (os_, ns) in enumerate(zip(old["supervised"], new["supervised"])):
            if os_ and (not ns or o_p[i] != n_p[i]):
                fail.append(f"{new['id']}: v1 stamp at {i} was modified")
                break
            if not ns and o_p[i] != n_p[i]:
                fail.append(f"{new['id']}: unsupervised token {i} rewritten")
                break

    if fail:
        print("SANITY GATES FAILED:", file=sys.stderr)
        for f in fail[:20]:
            print("  -", f, file=sys.stderr)
        return 1

    # ------------------------------------------------------------- write
    for name, rows in (("train", train), ("val", val)):
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{path}  {len(rows)} rows")
    shutil.copyfile(V1 / "test.jsonl", args.out / "test.jsonl")
    for extra in ("train_episodes.txt", "val_episodes.txt"):
        if (V1 / extra).exists():
            shutil.copyfile(V1 / extra, args.out / extra)
    print(f"{args.out / 'test.jsonl'}  (byte copy of v1)")

    write_stats(args.out / "dataset_stats.md", stamper, before, after,
                train, val)
    n_new = (stamper.counts["stamp_audio_endorsed"]
             + stamper.counts["stamp_sefaria"])
    print(f"new Hebrew stamps: {n_new}")
    print(f"supervised (all tokens): "
          f"{after['all_sup']}/{after['all']} = {pct(after['all_sup'], after['all'])}"
          f"  (v1: {pct(before['all_sup'], before['all'])})")
    print(f"supervised (all Hebrew-script tokens): "
          f"{after['heb_sup']}/{after['heb']} = {pct(after['heb_sup'], after['heb'])}"
          f"  (v1: {pct(before['heb_sup'], before['heb'])})")
    print(f"supervised (evidence-class tokens): "
          f"{after['ev_sup']}/{after['ev']} = {pct(after['ev_sup'], after['ev'])}"
          f"  (v1: {pct(before['ev_sup'], before['ev'])})")
    return 0


def write_stats(path: Path, stamper: Stamper, before, after, train, val) -> None:
    c = stamper.counts
    n_new = c["stamp_audio_endorsed"] + c["stamp_sefaria"]
    lines = [
        "# Retrain dataset v2 — Hebrew evidence stamped in",
        "",
        "Merger over v1 (`data/retrain/`): starts from `train_unmasked.jsonl`",
        "(778 audio-decided homograph occurrences already stamped) + v1 `val.jsonl`,",
        "and stamps the Hebrew evidence onto tokens v1 left unsupervised.",
        "`test.jsonl` is a byte-for-byte copy of v1's.",
        "",
        "## Stamps by source",
        "",
        "| source | new stamps |",
        "| --- | ---: |",
        f"| audio-endorsed (`data/lexicons/audio_endorsed_lk.py`, per-occurrence text_pointed) "
        f"| {c['stamp_audio_endorsed']} |",
        f"| Sefaria book pointing (`data/lexicons/sefaria_pointed_lk.py`) | {c['stamp_sefaria']} |",
        f"| homograph type-level (`data/lexicons/homograph_lk.py`) | 0 (by policy) |",
        f"| **total new** | **{n_new}** |",
        "",
        "Carried through from v1: the 778 individually decided homograph",
        "occurrences stamped by `scripts/unmask_homographs.py`.",
        "",
        "## Skips (counted, never guessed)",
        "",
        "| reason | tokens |",
        "| --- | ---: |",
    ]
    for k in ("audio_via_row_unique_form", "audio_skip_no_row_form",
              "audio_skip_letters_differ", "audio_skip_reading_disagrees",
              "sefaria_skip_letters_differ", "audio_skip_canonical_unfit",
              "homograph_type_not_stamped", "already_supervised"):
        lines.append(f"| `{k}` | {c[k]} |")
    rivals = {k: v for k, v in stamper.audio_seen.items() if len(v) > 1}
    lines += [
        "",
        "## One target per type (audio path)",
        "",
        "The audio endorsement is about the *reading*, and a type is read the",
        "same way in every row that endorses it -- but the row's own",
        "`text_pointed` spells that reading with whatever pointing the",
        f"transcriber used. {len(rivals)} of {len(stamper.audio_seen)} audio types are",
        "spelled more than one way in the corpus; stamping each occurrence with",
        "its own spelling would hand a diacritic-restoration model conflicting",
        "labels on identical letters. One target is chosen per type: the most",
        "explicitly pointed rival (ties by frequency, then codepoint order).",
        "Every rival reads back as the same endorsed IPA, so nothing phonetic",
        "is decided here.",
        "",
        "| type | chosen | rivals dropped |",
        "| --- | --- | --- |",
    ]
    for key, forms in sorted(rivals.items(),
                             key=lambda kv: -sum(kv[1].values()))[:20]:
        chosen = stamper.audio_canonical[key]
        others = ", ".join(f"{f} ({n})" for f, n in forms.most_common()
                           if f != chosen)
        lines.append(f"| {key} | {chosen} ({forms[chosen]}) | {others} |")
    lines += [
        "",
        "## Supervision",
        "",
        "| population | v1 | v2 |",
        "| --- | ---: | ---: |",
        f"| all tokens (v1's headline convention) | "
        f"{pct(before['all_sup'], before['all'])} "
        f"({before['all_sup']}/{before['all']}) | "
        f"{pct(after['all_sup'], after['all'])} "
        f"({after['all_sup']}/{after['all']}) |",
        f"| all Hebrew-script tokens (train+val) | "
        f"{pct(before['heb_sup'], before['heb'])} "
        f"({before['heb_sup']}/{before['heb']}) | "
        f"{pct(after['heb_sup'], after['heb'])} "
        f"({after['heb_sup']}/{after['heb']}) |",
        f"| loshn-koydesh evidence-class tokens (audio/homograph/Sefaria types) | "
        f"{pct(before['ev_sup'], before['ev'])} "
        f"({before['ev_sup']}/{before['ev']}) | "
        f"{pct(after['ev_sup'], after['ev'])} "
        f"({after['ev_sup']}/{after['ev']}) |",
        "",
        f"Rows changed by this pass: {c['rows_changed']} "
        f"(train {len(train)} + val {len(val)} rows).",
        "",
        "## Most-stamped types",
        "",
        "| source:key | occurrences |",
        "| --- | ---: |",
    ]
    for w, n in stamper.by_word.most_common(30):
        lines.append(f"| `{w}` | {n} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
