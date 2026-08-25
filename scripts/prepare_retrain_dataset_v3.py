#!/usr/bin/env python3
"""Retrain dataset v3 = v2 + the chain-attested tier stamped in.

Merger over v2 (``data/retrain2/``), not a rebuild: v2's rows are the input,
nothing v2 supervised is touched, and ``test.jsonl`` stays a byte-for-byte
copy of v2's (asserted).

The new tier -- "chain-attested" -- supervises tokens whose READING the frozen
engine already vouches for, using pointings the corpus itself already writes:

  1. A type qualifies when the engine reads it with ``route='lexicon'`` and
     confidence HIGH or MED (``yiddish_g2p.g2p_token`` via ``yiddish_labels``).
     Homograph-lexicon types are excluded by v2's standing policy (a type-level
     winner is not evidence about a particular sentence), and so are the
     audio-refuted SUSPECT keys.
  2. The candidate pointings are the ATTESTED forms of that type across the
     retrain2 train+val ``pointed`` columns plus the corpus
     ``yiddish_tts_dataset_v2.tsv`` ``nikud`` column (test episode excluded).
     Nothing is ever synthesized.
  3. A pointing survives only if it reads back as one of the type's allowed
     readings (``ipa_primary`` + listed variants + gold-lexicon variants):
     the spec's ``text_to_ipa(P)`` membership check, PLUS the non-vacuous
     convention read-back -- v1's ``reconcile`` must return the pointing
     unrepaired against one allowed reading.  (The lexicon route strips marks
     before lookup, so ``text_to_ipa`` alone cannot see a wrong mark; the
     reconcile pass is what makes the read-back real.)
  4. ONE target per type: the most explicitly pointed survivor, ties by
     attested frequency, then codepoint order (v2's one-target policy).
  5. The target is stamped onto each unsupervised occurrence via letter-identity
     fitting (v2's ``fit_to_token``): exact skeleton or final-letter transfer
     only; misfits are counted and skipped, never guessed.

Output: data/retrain3/{train,val,test}.jsonl + dataset_stats.md.

Usage:  .venv/bin/python scripts/prepare_retrain_dataset_v3.py
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import yiddish_labels  # noqa: E402  (verify() runs at import -- deployment gate)
import yiddish_g2p as G  # FROZEN - read only  # noqa: E402

csv.field_size_limit(10_000_000)

V2DIR = REPO / "data" / "retrain2"
OUTDIR = REPO / "data" / "retrain3"
CORPUS_V2 = REPO / "data" / "corpus" / "yiddish_tts_dataset_v2.tsv"
TEST_EPISODE = "100313"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the standing helpers verbatim -- one convention, one definition.
P2 = _load_module("_prep_v2", "prepare_retrain_dataset_v2.py")
P1 = sys.modules["_prep_v1"]  # loaded by P2

norm = P1.norm
strip_marks = P1.strip_marks
HEB_RE = P1.HEB_RE
IN_CONVENTION = P1.IN_CONVENTION
reconcile = P1.reconcile  # pointing <-> reading convention read-back

key_of = P2.key_of
letters = P2.letters
fit_to_token = P2.fit_to_token
read_jsonl = P2.read_jsonl
SUSPECT_KEYS = P2.SUSPECT_KEYS
pct = P2.pct

from data.lexicons.gold_lexicon import GOLD_LEXICON  # noqa: E402
from data.lexicons.homograph_lk import HOMOGRAPH_LK  # noqa: E402


# ------------------------------------------------------------------- census

def iter_unsupervised(rows: list[dict]):
    """(row, pos, lead, core, trail, key) for every unsupervised Hebrew token."""
    for row in rows:
        ptoks = row["pointed"].split()
        mask = row["supervised"]
        if len(ptoks) != len(mask):
            raise AssertionError(f"row {row['id']}: token/mask grid mismatch")
        for pos, (ptok, sup) in enumerate(zip(ptoks, mask)):
            if sup:
                continue
            lead, core, trail = G.split_affixes(strip_marks(ptok))
            if not core or not HEB_RE.search(core):
                continue
            yield row, pos, lead, core, trail, key_of(core)


def collect_attested(rows: list[dict]) -> dict[str, collections.Counter]:
    """key -> Counter of pointed core forms attested across train+val+corpus."""
    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    def add(tok: str) -> None:
        _, ccore, _ = G.split_affixes(tok)
        if not ccore or not HEB_RE.search(strip_marks(ccore)):
            return
        seen[key_of(ccore)][unicodedata.normalize("NFC", ccore)] += 1

    for row in rows:  # retrain2 pointed column (already in convention)
        for tok in row["pointed"].split():
            add(tok)
    if CORPUS_V2.exists():  # corpus nikud column, test episode excluded
        with CORPUS_V2.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r.get("episode") == TEST_EPISODE:
                    continue
                for tok in norm(r.get("nikud") or "").split():
                    add(tok)
    return seen


def has_marks(form: str) -> bool:
    return any(unicodedata.combining(ch) for ch in form)


def in_convention(form: str) -> bool:
    return all(ch in IN_CONVENTION for ch in form if unicodedata.combining(ch))


# --------------------------------------------------------------- the builder

class ChainStamper:
    def __init__(self) -> None:
        self.counts: collections.Counter[str] = collections.Counter()
        self.by_key: collections.Counter[str] = collections.Counter()
        self.targets: dict[str, str] = {}       # key -> chosen pointed target
        self.allowed: dict[str, set[str]] = {}  # key -> allowed IPA readings
        self.skip_types: collections.Counter[str] = collections.Counter()
        self._readback_cache: dict[tuple[str, str], bool] = {}
        self._stamp_checked: set[str] = set()

    # -- type-level work (done once per distinct type) --------------------
    def route_types(self, rep_core: dict[str, str]) -> None:
        """Decide, per unsupervised type, whether the chain vouches for it."""
        for key, core in rep_core.items():
            if key in SUSPECT_KEYS:
                self.skip_types["suspect-key"] += 1
                continue
            if key in HOMOGRAPH_LK:
                self.skip_types["homograph-type"] += 1
                continue
            rec = yiddish_labels.token_detail(core)
            if rec["route"] != "lexicon" or rec["confidence"] not in ("HIGH", "MED"):
                self.skip_types["route-not-lexicon-high-med"] += 1
                continue
            allowed = {rec["ipa_primary"]} | set(rec.get("variants") or [])
            gold = GOLD_LEXICON.get(key)
            if gold:
                allowed |= set(gold.get("variants") or [])
                allowed.add(gold["ipa_primary"])
            allowed.discard("")
            if not allowed:
                self.skip_types["no-allowed-reading"] += 1
                continue
            self.allowed[key] = allowed

    def survives(self, form: str, key: str) -> bool:
        """The read-back check: the pointing must express an allowed reading."""
        if not in_convention(form):
            return False
        # spec check: the full pipeline's reading of the pointed form must be
        # in the allowed set (vacuously true on the lexicon route, but kept --
        # it is the stated contract and it does gate normalize_surface drift).
        try:
            if yiddish_labels.text_to_ipa(form) not in self.allowed[key]:
                return False
        except Exception:
            return False
        # the real read-back: the pointing's vowel slots must reconcile,
        # unrepaired, with one allowed reading.
        for ipa in self.allowed[key]:
            hit = self._readback_cache.get((form, ipa))
            if hit is None:
                try:
                    accepted, status = reconcile(form, ipa)
                except Exception:
                    accepted, status = None, "error"
                hit = accepted == form and status == "ok"
                self._readback_cache[(form, ipa)] = hit
            if hit:
                return True
        return False

    def choose_targets(self, attested: dict[str, collections.Counter]) -> None:
        """One target per type: most explicitly pointed survivor, ties by
        frequency then codepoint order (v2's one-target policy)."""
        def rank(item: tuple[str, int]) -> tuple[int, int, str]:
            form, n = item
            marks = sum(1 for ch in form if unicodedata.combining(ch))
            return (marks, n, form)

        for key in self.allowed:
            forms = attested.get(key)
            if not forms:
                self.skip_types["no-attested-pointing-survives"] += 1
                continue
            survivors = {f: n for f, n in forms.items() if self.survives(f, key)}
            if not survivors:
                self.skip_types["no-attested-pointing-survives"] += 1
                continue
            self.targets[key] = max(survivors.items(), key=rank)[0]

    # -- token-level stamping ---------------------------------------------
    def stamp_rows(self, rows: list[dict]) -> list[dict]:
        out = []
        misfit_keys: set[str] = set()
        for row in rows:
            ptoks = row["pointed"].split()
            mask = list(row["supervised"])
            changed = False
            for pos, (ptok, sup) in enumerate(zip(ptoks, mask)):
                if sup:
                    continue
                lead, core, trail = G.split_affixes(strip_marks(ptok))
                if not core or not HEB_RE.search(core):
                    continue
                key = key_of(core)
                target = self.targets.get(key)
                if target is None:
                    continue
                stamp = fit_to_token(target, core)
                if stamp is None:
                    self.counts["skip_letters_misfit"] += 1
                    misfit_keys.add(key)
                    continue
                if letters(stamp) != core:
                    raise AssertionError(
                        f"row {row['id']} pos {pos}: stamp {stamp!r} does not "
                        f"strip to {core!r}")
                self.check_stamp_reading(stamp, key, row["id"], pos)
                ptoks[pos] = lead + stamp + trail
                mask[pos] = True
                self.counts["stamp_chain_attested"] += 1
                self.by_key[key] += 1
                changed = True
            if changed:
                row = dict(row)
                row["pointed"] = " ".join(ptoks)
                row["supervised"] = mask
                row["n_supervised"] = sum(mask)
                if strip_marks(row["pointed"]) != row["text"]:
                    raise AssertionError(f"row {row['id']}: letter identity broken")
                self.counts["rows_changed"] += 1
            out.append(row)
        self.counts["types_letters_misfit_somewhere"] = len(misfit_keys)
        return out

    def check_stamp_reading(self, stamp: str, key: str, rid: str, pos: int) -> None:
        """Every distinct emitted form is read back once (covers 100% >= 5%)."""
        if stamp in self._stamp_checked:
            return
        self._stamp_checked.add(stamp)
        ipa = yiddish_labels.text_to_ipa(stamp)
        if ipa not in self.allowed[key]:
            raise AssertionError(
                f"row {rid} pos {pos}: stamped {stamp!r} reads {ipa!r}, "
                f"not in allowed set {sorted(self.allowed[key])!r}")


# ------------------------------------------------------------------ counting

def coverage(rows: list[dict]) -> tuple[int, int]:
    tot = sum(r["n_tokens"] for r in rows)
    sup = sum(sum(r["supervised"]) for r in rows)
    return sup, tot


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_in = read_jsonl(V2DIR / "train.jsonl")
    val_in = read_jsonl(V2DIR / "val.jsonl")
    all_in = train_in + val_in

    # census: distinct unsupervised types + a representative bare core each
    unsup_tok: collections.Counter[str] = collections.Counter()
    core_freq: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _, _, _, core, _, key in iter_unsupervised(all_in):
        unsup_tok[key] += 1
        core_freq[key][core] += 1
    rep_core = {k: c.most_common(1)[0][0] for k, c in core_freq.items()}
    print(f"unsupervised Hebrew-script tokens: {sum(unsup_tok.values())} "
          f"({len(unsup_tok)} types)")

    st = ChainStamper()
    st.route_types(rep_core)
    print(f"chain-vouched types (lexicon HIGH/MED): {len(st.allowed)}")

    attested = collect_attested(all_in)
    st.choose_targets(attested)
    print(f"types with a surviving attested target: {len(st.targets)}")

    before_sup, before_tot = coverage(all_in)
    train = st.stamp_rows(train_in)
    val = st.stamp_rows(val_in)
    after_sup, after_tot = coverage(train + val)
    assert before_tot == after_tot

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

    # one pointed target per letter-identical type among this pass's stamps
    targets: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
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

    # nothing may be UN-supervised by this pass, and no v2 stamp may move
    for old, new in zip(train_in + val_in, train + val):
        if old["id"] != new["id"]:
            fail.append("row order changed")
            break
        o_p, n_p = old["pointed"].split(), new["pointed"].split()
        for i, (os_, ns) in enumerate(zip(old["supervised"], new["supervised"])):
            if os_ and (not ns or o_p[i] != n_p[i]):
                fail.append(f"{new['id']}: v2 stamp at {i} was modified")
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
    shutil.copyfile(V2DIR / "test.jsonl", args.out / "test.jsonl")
    if (args.out / "test.jsonl").read_bytes() != (V2DIR / "test.jsonl").read_bytes():
        raise AssertionError("test.jsonl is not a byte copy of retrain2's")
    for extra in ("train_episodes.txt", "val_episodes.txt"):
        if (V2DIR / extra).exists():
            shutil.copyfile(V2DIR / extra, args.out / extra)
    print(f"{args.out / 'test.jsonl'}  (byte copy of retrain2, asserted)")

    write_stats(args.out / "dataset_stats.md", st, unsup_tok,
                (before_sup, before_tot), (after_sup, after_tot))
    c = st.counts
    print(f"new chain-attested stamps: {c['stamp_chain_attested']}")
    print(f"supervised (all tokens): {after_sup}/{after_tot} = "
          f"{pct(after_sup, after_tot)}  (v2: {pct(before_sup, before_tot)})")
    return 0


def write_stats(path: Path, st: ChainStamper, unsup_tok,
                before: tuple[int, int], after: tuple[int, int]) -> None:
    c = st.counts
    b_sup, b_tot = before
    a_sup, a_tot = after
    lines = [
        "# Retrain dataset v3 — chain-attested tier stamped in",
        "",
        "Merger over v2 (`data/retrain2/`): tokens v2 left unsupervised are",
        "stamped with a corpus-attested pointing of their type whenever the",
        "frozen engine vouches for the type's reading (route `lexicon`,",
        "confidence HIGH/MED) and the pointing reads back as an allowed",
        "reading (ipa_primary + listed/gold variants) under the convention",
        "read-back (`reconcile`, unrepaired). One target per type: the most",
        "explicitly pointed survivor, ties by attested frequency, then",
        "codepoint order. `test.jsonl` is a byte-for-byte copy of v2's",
        "(asserted).",
        "",
        "## Headline",
        "",
        "| metric | count |",
        "| --- | ---: |",
        f"| tokens newly supervised (chain-attested) | {c['stamp_chain_attested']} |",
        f"| rows changed | {c['rows_changed']} |",
        f"| types stamped | {len(st.by_key)} |",
        "",
        "## Coverage (train+val, all tokens)",
        "",
        "| | supervised | total | share |",
        "| --- | ---: | ---: | ---: |",
        f"| v2 (before) | {b_sup} | {b_tot} | {pct(b_sup, b_tot)} |",
        f"| v3 (after) | {a_sup} | {a_tot} | {pct(a_sup, a_tot)} |",
        "",
        "## Types skipped (counted, never guessed)",
        "",
        "| reason | types |",
        "| --- | ---: |",
    ]
    for k in ("route-not-lexicon-high-med", "homograph-type", "suspect-key",
              "no-attested-pointing-survives", "no-allowed-reading"):
        lines.append(f"| `{k}` | {st.skip_types[k]} |")
    lines += [
        f"| letters-misfit (types with >=1 misfit occurrence) "
        f"| {c['types_letters_misfit_somewhere']} |",
        "",
        f"Occurrence-level letter misfits skipped: {c['skip_letters_misfit']} tokens.",
        "",
        "## Top 20 stamped types by token count",
        "",
        "| type (lexicon key) | target | tokens stamped | unsupervised before |",
        "| --- | --- | ---: | ---: |",
    ]
    for key, n in st.by_key.most_common(20):
        lines.append(f"| {key} | {st.targets[key]} | {n} | {unsup_tok[key]} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
