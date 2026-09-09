#!/usr/bin/env python3
"""Retrain dataset v8 = retrain3 + the audio-attested tier.

Merger over v3 (``data/retrain3/``), not a rebuild: v3's rows are the input,
nothing v3 already supervised is touched, and ``test.jsonl`` stays a
byte-for-byte copy of v3's (asserted).

v3 supervises a token only when the frozen engine vouches for its reading
(lexicon HIGH/MED). The 35% of tokens the engine reads by RULE at LOW/MED
have never had a label. v8 gives them one where the AUDIO decides:

  1. ``scripts/xeus_attest.py`` scored, for every such occurrence, the
     spelling's legal readings (the open slots: א a/ɔ/u/aː, פ f/p,
     יי aj/aː/ej, וי ɔj/oʊ, ו i/u, ɛ/ə, final devoicing) against the clip of
     that very token with the fine-tuned Yiddish ear, and recorded the best
     reading and its margin over the runner-up (nats).
  2. An OCCURRENCE is decided when its margin >= --margin-occ (2.0). A TYPE is
     decided when >= --type-min occurrences with margin >= --margin-type (1.0)
     agree on one reading at >= --type-share (0.85); the type reading covers
     occurrences without a decision of their own. Occurrence beats type.
  3. Stress is not in the ear's output; the engine's stress position is put
     back onto the decided phones (same phone count: the graph only
     substitutes), giving the reading the convention read-back expects.
  4. The candidate pointings are v3's: attested forms of the type across
     retrain2 train+val and the corpus nikud column (test episode excluded).
     A pointing survives only if the engine reads it back to the decided
     phones and v1's ``reconcile`` returns it unrepaired against the
     decided (restressed) reading. Nothing is synthesized.
  5. One target per (type, reading): v3's rank — most explicit, then most
     attested, then codepoint order. Stamped by letter identity
     (``fit_to_token``); misfits counted and skipped.

Authority stays where it was: certain (Chezky) words and lexicon words are
v3's; this tier only ever touches tokens nobody had vouched for, with the
rabbi's own voice as the witness.

Output: data/retrain8/{train,val,test}.jsonl + dataset_stats.md.
Usage:  .venv/bin/python scripts/prepare_retrain_dataset_v8.py [--attest data/xeus_ft/attest.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import prepare_retrain_dataset_v3 as V3  # noqa: E402  (loads v2/v1 helpers, verify() gate)
import yiddish_labels  # noqa: E402
import yiddish_g2p as G  # noqa: E402
from xeus_ft_common import tokenize_ipa, lexicon_key  # noqa: E402

V3DIR = REPO / "data" / "retrain3"
OUTDIR = REPO / "data" / "retrain8"
_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")


# ------------------------------------------------------------------ readings

def restress(engine_ipa: str, phones: list[str]) -> str:
    """Put the engine's stress mark back at the same phone index in ``phones``."""
    from xeus_ft_common import YI2ID
    idx = None
    seen = 0
    s = engine_ipa.replace(" ", "")
    i = 0
    while i < len(s):
        if s[i] == "ˈ":
            idx = seen
            i += 1
            continue
        for m in ("aː", "ej", "aj", "ɔj", "oʊ"):
            if s.startswith(m, i):
                seen += 1
                i += len(m)
                break
        else:
            if s[i] in YI2ID:
                seen += 1
            i += 1
    out = []
    for k, p in enumerate(phones):
        if idx is not None and k == idx:
            out.append("ˈ")
        out.append(p)
    return "".join(out)


class AudioStamper:
    def __init__(self, margin_occ: float, margin_type: float, type_min: int, type_share: float):
        self.margin_occ = margin_occ
        self.margin_type = margin_type
        self.type_min = type_min
        self.type_share = type_share
        self.counts: collections.Counter[str] = collections.Counter()
        self.type_readings: dict[str, list[str]] = {}
        self.occ: dict[tuple[str, int], dict] = {}
        self.targets: dict[tuple[str, str], str] = {}        # (key, reading) -> pointed form
        self.no_target: set[tuple[str, str]] = set()
        self.attested: dict[str, collections.Counter] = {}
        self._engine_ipa: dict[str, str] = {}
        self._readback_cache: dict[tuple[str, str], bool] = {}

    # -- decisions --------------------------------------------------------
    def load(self, path: Path) -> None:
        per_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        n = 0
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            n += 1
            rid = f"{r['episode']}-{int(r['chunk_idx']):05d}"
            self.occ[(rid, r["wi"])] = r
            if r["episode"] == V3.TEST_EPISODE:
                continue      # the test episode's clips must not shape any type reading
            if r["margin"] >= self.margin_type:
                per_type[r["key"]][" ".join(r["chosen"])] += 1
        for key, c in per_type.items():
            total = sum(c.values())
            reading, top = c.most_common(1)[0]
            if total >= self.type_min and top / total >= self.type_share:
                self.type_readings[key] = reading.split()
        self.counts["attest_records"] = n
        self.counts["types_with_type_reading"] = len(self.type_readings)

    def decide(self, rid: str, heb_idx: int, key: str) -> tuple[list[str] | None, str]:
        r = self.occ.get((rid, heb_idx))
        if r is not None and r["key"] == key and r["margin"] >= self.margin_occ:
            return r["chosen"], "occurrence"
        t = self.type_readings.get(key)
        if t is not None:
            return t, "type"
        return None, "undecided"

    # -- read-back ----------------------------------------------------------
    def engine_ipa(self, core: str) -> str:
        if core not in self._engine_ipa:
            rec = yiddish_labels.token_detail(core)
            self._engine_ipa[core] = rec.get("ipa_primary") or ""
        return self._engine_ipa[core]

    def survives(self, form: str, phones: list[str], stressed: str) -> bool:
        if not V3.in_convention(form):
            return False
        hit = self._readback_cache.get((form, stressed))
        if hit is not None:
            return hit
        ok = False
        try:
            if tokenize_ipa(yiddish_labels.text_to_ipa(form)) == phones:
                accepted, status = V3.reconcile(form, stressed)
                ok = accepted == form and status == "ok"
        except Exception:  # noqa: BLE001
            ok = False
        self._readback_cache[(form, stressed)] = ok
        return ok

    def target_for(self, key: str, core: str, phones: list[str]) -> str | None:
        reading = " ".join(phones)
        tk = (key, reading)
        if tk in self.targets:
            return self.targets[tk]
        if tk in self.no_target:
            return None
        forms = self.attested.get(key)
        if not forms:
            self.no_target.add(tk)
            self.counts["type_reading_no_attested_form"] += 1
            return None
        stressed = restress(self.engine_ipa(core), phones)
        survivors = {f: n for f, n in forms.items() if self.survives(f, phones, stressed)}
        if not survivors:
            self.no_target.add(tk)
            self.counts["type_reading_no_surviving_form"] += 1
            return None

        def rank(item):
            f, n = item
            return (sum(1 for ch in f if unicodedata.combining(ch)), n, f)
        self.targets[tk] = max(survivors.items(), key=rank)[0]
        return self.targets[tk]

    # -- stamping -----------------------------------------------------------
    def stamp_rows(self, rows: list[dict]) -> list[dict]:
        out = []
        for row in rows:
            ptoks = row["pointed"].split()
            mask = list(row["supervised"])
            heb = 0
            changed = False
            for pos, (ptok, sup) in enumerate(zip(ptoks, mask)):
                bare = V3.strip_marks(ptok)
                k = len(_HEB.findall(bare))
                heb_idx = heb
                heb += k
                if sup or k != 1:
                    if not sup and k > 1:
                        self.counts["skip_multi_heb_token"] += 1
                    continue
                lead, core, trail = G.split_affixes(bare)
                if not core or not V3.HEB_RE.search(core):
                    continue
                key = lexicon_key(core)
                phones, how = self.decide(row["id"], heb_idx, key)
                if phones is None:
                    self.counts["undecided"] += 1
                    continue
                if len(phones) != len(tokenize_ipa(self.engine_ipa(core))):
                    self.counts["skip_reading_length_mismatch"] += 1
                    continue
                target = self.target_for(key, core, phones)
                if target is None:
                    self.counts[f"skip_no_target_{how}"] += 1
                    continue
                stamp = V3.fit_to_token(target, core)
                if stamp is None:
                    self.counts["skip_letters_misfit"] += 1
                    continue
                if V3.letters(stamp) != core:
                    raise AssertionError(f"row {row['id']} pos {pos}: stamp {stamp!r} != {core!r}")
                ptoks[pos] = lead + stamp + trail
                mask[pos] = True
                self.counts[f"stamp_{how}"] += 1
                changed = True
            if changed:
                row = dict(row)
                row["pointed"] = " ".join(ptoks)
                row["supervised"] = mask
                row["n_supervised"] = sum(mask)
                if V3.strip_marks(row["pointed"]) != row["text"]:
                    raise AssertionError(f"row {row['id']}: letter identity broken")
                self.counts["rows_changed"] += 1
            out.append(row)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attest", type=Path, default=REPO / "data/xeus_ft/attest.jsonl")
    ap.add_argument("--margin-occ", type=float, default=2.0)
    ap.add_argument("--margin-type", type=float, default=1.0)
    ap.add_argument("--type-min", type=int, default=5)
    ap.add_argument("--type-share", type=float, default=0.85)
    ap.add_argument("--out", type=Path, default=OUTDIR)
    args = ap.parse_args()

    train = V3.read_jsonl(V3DIR / "train.jsonl")
    val = V3.read_jsonl(V3DIR / "val.jsonl")
    st = AudioStamper(args.margin_occ, args.margin_type, args.type_min, args.type_share)
    st.load(args.attest)
    print(f"attest records {st.counts['attest_records']:,}  type readings {st.counts['types_with_type_reading']:,}", flush=True)
    st.attested = V3.collect_attested(train + val)
    before = V3.coverage(train + val)
    train2 = st.stamp_rows(train)
    val2 = st.stamp_rows(val)
    after = V3.coverage(train2 + val2)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train2), ("val", val2)):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copyfile(V3DIR / "test.jsonl", args.out / "test.jsonl")
    assert (args.out / "test.jsonl").read_bytes() == (V3DIR / "test.jsonl").read_bytes()
    for f in ("train_episodes.txt", "val_episodes.txt"):
        if (V3DIR / f).exists():
            shutil.copyfile(V3DIR / f, args.out / f)

    c = st.counts
    md = ["# Retrain dataset v8 — audio-attested tier stamped in", "",
          "Merger over v3 (`data/retrain3/`): tokens v3 left unsupervised (rule-path LOW/MED) are",
          "stamped with a corpus-attested pointing of their type whenever the fine-tuned Yiddish ear,",
          "scoring the spelling's legal readings against the clip of that token, decides the reading",
          f"(occurrence margin ≥ {args.margin_occ} nats, or a type reading agreed by ≥ {args.type_min} occurrences",
          f"at ≥ {args.type_share:.0%} with margin ≥ {args.margin_type}) and the pointing reads back to it under",
          "`reconcile`. `test.jsonl` is a byte-for-byte copy of v3's (asserted).", "",
          "## Headline", "", "| metric | count |", "| --- | ---: |",
          f"| attest records | {c['attest_records']:,} |",
          f"| tokens stamped, occurrence-level | {c['stamp_occurrence']:,} |",
          f"| tokens stamped, type-level | {c['stamp_type']:,} |",
          f"| types with a type-level reading | {c['types_with_type_reading']:,} |",
          f"| rows changed | {c['rows_changed']:,} |", "",
          "## Coverage (train+val, all tokens)", "", "| | supervised | total | share |", "| --- | ---: | ---: | ---: |",
          f"| v3 (before) | {before[0]:,} | {before[1]:,} | {before[0] / before[1]:.2%} |",
          f"| v8 (after) | {after[0]:,} | {after[1]:,} | {after[0] / after[1]:.2%} |", "",
          "## Skipped (counted, never guessed)", "", "| reason | tokens |", "| --- | ---: |"]
    for k, v in sorted(c.items()):
        if k.startswith("skip") or k in ("undecided", "type_reading_no_attested_form", "type_reading_no_surviving_form"):
            md.append(f"| `{k}` | {v:,} |")
    (args.out / "dataset_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
