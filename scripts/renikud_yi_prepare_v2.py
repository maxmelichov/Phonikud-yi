#!/usr/bin/env python3
"""ReNikud-yi v2 training data: per-letter (consonant, vowel, STRESS) labels
from the NEW ear (xeus-yi-ipa, stress-aware lattice decode).

Same frame as renikud_yi_prepare.py (v1), same output layout, so
renikud_yi_train.py runs unchanged; three things change:

  1. The attestation source is xeus-yi-ipa/data/attest_lattice.jsonl, one
     row for EVERY word of every chunk: {episode, chunk_idx, wi, w, key,
     route, engine, engine_ipa, chosen, chosen_ipa (with the ear's stress),
     stress_index, margin, posterior, ...}. The prepare needs no engine and
     no run3/segments.jsonl: the engine's reading of every word is already
     in attest_targets.jsonl and the ear's choice of dictionary variant is
     in the attest file.
  2. Stress is a label of its own. v1 wrote the ENGINE's stress index onto
     every reading; here the ear's stress is used where its margin clears
     --margin-stress, the engine's otherwise. train.py already has the
     stress head (ReNikudYi.stress_head, CE on the stress targets), so no
     trainer change is needed.
  3. Nothing from the six held-out episodes (--heldout-episodes) or the new
     ear's 14 val_eps episodes (build_stats.json) is labelled — not as
     occurrences, not in the type aggregation. The held-out six are written
     to test.jsonl with IGNORE everywhere (--heldout-labels none, default)
     or with dictionary/lexicon labels only (--heldout-labels table); the
     val_eps rows are not written at all. The trainer's val split, which
     selects the checkpoint, is --trainer-val-episodes episodes drawn with
     --seed from the labelled pool (v1 selected on the held-out episodes'
     own ear labels, which this removes).

Label policy per word (authority order):

  gold       key in dictionary.json (412 Chezky-verified words): the variant
             the ear chose when its margin >= --margin-gold (default 0: the
             ear is the only judge among the variants, as v1 took run3's
             choice), else the engine's variant / the primary. Stress: the
             ear's when its stress margin >= --margin-stress, else the
             engine's.
  lexicon    route lexicon at HIGH/MED: the engine's reading and stress
             (the lexicon is authority; v1 policy).
  audio-occ  rule-path word with an ear decision at margin >= --margin-occ:
             the ear's reading; stress the ear's if its stress margin >=
             --margin-stress, else the engine's ordinal placed on the ear's
             reading.
  audio-type rule-path word without an occurrence decision whose key has a
             type reading: >= --type-min decisions at margin >=
             --margin-type in the pool episodes, top reading >= --type-share
             of them. Stress: the majority ordinal among those decisions if
             it has >= --type-share, else the engine's.
  engine     only with --engine-all (the curriculum's noisy pretraining set).
  IGNORE     everything else (-100): no gradient, learned from context.

The stress margin is the row's ``stress_margin`` when the ear reports one
(gap between the chosen reading and the best reading with the stress
elsewhere), else ``margin``. A ``margin`` of null/inf is treated as 0 / +inf.

Format check: the first --check-rows rows of the attest file are validated
(required keys, types, chosen == tokenize(chosen_ipa), stress_index ==
ordinal of the mark in chosen_ipa); any violation aborts with the row.

Output: --out/{train,val,test}.jsonl, labels.json, dataset_stats.md (v1 layout,
rows carry an extra "src" list: the label source per character).

Usage:
  .venv/bin/python scripts/renikud_yi_prepare_v2.py \\
      --attest ../xeus-yi-ipa/data/attest_lattice.jsonl --out data/renikud_yi_v2
  .venv/bin/python scripts/renikud_yi_prepare_v2.py --check-only --attest FILE   # format check, no output
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import lexicon_key, tokenize_ipa  # noqa: E402
from yi_align import CONSONANTS, VOWELS, align_word, parse_chunk  # noqa: E402

IGNORE = -100
CONS_CLASSES = ("",) + CONSONANTS
VOWEL_CLASSES = ("",) + tuple(sorted(VOWELS, key=lambda v: (len(v), v)))
C2ID = {c: i for i, c in enumerate(CONS_CLASSES)}
V2ID = {v: i for i, v in enumerate(VOWEL_CLASSES)}
VOWEL_SET = set(VOWELS)
STRESS = "ˈ"
_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
HELDOUT = "100313,104192,104690,113370,58622,94226"
XEUS_YI = REPO.parent / "xeus-yi-ipa"
REQUIRED = {"episode": str, "chunk_idx": int, "wi": int, "w": str, "key": str, "route": str,
            "engine": list, "engine_ipa": str, "chosen": list, "chosen_ipa": str}


# ------------------------------------------------------------------ stress helpers

def stress_ordinal(ipa: str) -> int | None:
    """Ordinal (among the vowels) of the vowel that carries the mark, None if unmarked."""
    s = ipa.replace(" ", "")
    i = 0
    seen = 0
    pending = False
    while i < len(s):
        if s[i] in "ˈˌ":
            pending = True
            i += 1
            continue
        for m in ("aː", "ej", "aj", "ɔj", "oʊ"):
            if s.startswith(m, i):
                tok, i = m, i + len(m)
                break
        else:
            tok, i = s[i], i + 1
        if tok in VOWEL_SET:
            if pending:
                return seen
            seen += 1
        pending = False
    return None


def place_stress(phones: list[str], ordinal: int | None) -> str:
    """The reading with the mark before its ``ordinal``-th vowel (engine convention:
    no mark on a word with fewer than two vowels, never on ə); unmarked otherwise."""
    vpos = [i for i, p in enumerate(phones) if p in VOWEL_SET]
    if ordinal is None or len(vpos) < 2 or ordinal >= len(vpos) or phones[vpos[ordinal]] == "ə":
        return "".join(phones)
    at = vpos[ordinal]
    return "".join(STRESS + p if i == at else p for i, p in enumerate(phones))


def n_full_vowels(phones: list[str]) -> int:
    return sum(1 for p in phones if p in VOWEL_SET and p != "ə")


def fnum(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, str):
        return math.inf if x.lower() in ("inf", "infinity") else float(x)
    return float(x)


# ------------------------------------------------------------------ attest file

def check_row(r: dict, where: str) -> None:
    for k, t in REQUIRED.items():
        if k not in r:
            raise SystemExit(f"attest format: row {where} lacks {k!r}: {r}")
        if not isinstance(r[k], t):
            raise SystemExit(f"attest format: row {where} {k}={r[k]!r} is not {t.__name__}")
    if "margin" not in r:
        raise SystemExit(f"attest format: row {where} lacks 'margin'")
    if tokenize_ipa(r["chosen_ipa"]) != list(r["chosen"]):
        raise SystemExit(f"attest format: row {where}: chosen {r['chosen']} != tokenize(chosen_ipa {r['chosen_ipa']!r})")
    if tokenize_ipa(r["engine_ipa"]) != list(r["engine"]):
        raise SystemExit(f"attest format: row {where}: engine {r['engine']} != tokenize(engine_ipa {r['engine_ipa']!r})")
    if "stress_index" in r and r["stress_index"] != stress_ordinal(r["chosen_ipa"]):
        raise SystemExit(f"attest format: row {where}: stress_index {r['stress_index']} != mark in {r['chosen_ipa']!r}")


def load_attest(path: Path, check_rows: int) -> tuple[dict[tuple[str, int, int], dict], collections.Counter]:
    occ: dict[tuple[str, int, int], dict] = {}
    seen = collections.Counter()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            r = json.loads(line)
            if r.get("chosen") is None:        # scoreless: no candidates, or a lattice the ear declined (attest.py caps)
                seen["scoreless"] += 1
                continue
            if n < check_rows:
                check_row(r, f"{n} ({r.get('episode')}/{r.get('chunk_idx')}/{r.get('wi')})")
            occ[(r["episode"], int(r["chunk_idx"]), int(r["wi"]))] = r
            seen["rows"] += 1
            seen["synthetic"] += int(bool(r.get("synthetic")))
            seen["with_stress_margin"] += int("stress_margin" in r)
            seen["with_posterior"] += int(r.get("posterior") is not None)
    return occ, seen


def ear_stress(r: dict, margin_stress: float) -> tuple[int | None, bool]:
    """(the ear's stress ordinal, trusted?) for a row."""
    sm = fnum(r.get("stress_margin", r.get("margin")))
    k = r["stress_index"] if "stress_index" in r else stress_ordinal(r["chosen_ipa"])
    return k, sm >= margin_stress


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attest", default=str(XEUS_YI / "data/attest_lattice.jsonl"),
                    help="the new ear's decisions, one row per word (or the synthetic stand-in)")
    ap.add_argument("--targets", default=str(REPO / "data/xeus_ft/attest_targets.jsonl"),
                    help="engine reading/route/conf per word of every corpus chunk")
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--build-stats", default=str(XEUS_YI / "data/build_stats.json"),
                    help="the new ear's build_stats.json: its val_eps_episodes are never labelled")
    ap.add_argument("--heldout-episodes", default=HELDOUT, help="comma-separated; the yardstick episodes, never labelled")
    ap.add_argument("--heldout-labels", choices=("none", "table"), default="none",
                    help="labels on the held-out rows in test.jsonl: none (default) or dictionary/lexicon only")
    ap.add_argument("--margin-occ", type=float, default=2.0, help="T: ear reading on a rule-path word")
    ap.add_argument("--margin-stress", type=float, default=2.0, help="T_stress: ear stress (else the engine's)")
    ap.add_argument("--margin-gold", type=float, default=0.0, help="ear's dictionary variant when margin >= this")
    ap.add_argument("--margin-type", type=float, default=1.0)
    ap.add_argument("--type-min", type=int, default=5)
    ap.add_argument("--type-share", type=float, default=0.85)
    ap.add_argument("--trainer-val-episodes", type=int, default=8,
                    help="pool episodes held out of train.jsonl for the trainer's checkpoint selection")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--engine-all", action="store_true",
                    help="label every unvouched word with the engine's reading (curriculum pretraining set)")
    ap.add_argument("--check-rows", type=int, default=500, help="rows of the attest file to format-check")
    ap.add_argument("--check-only", action="store_true", help="validate the attest file and exit")
    ap.add_argument("--only-attested-chunks", action="store_true",
                    help="skip corpus chunks the attest file has no row for (developing on a partial / synthetic file)")
    ap.add_argument("--out", default=str(REPO / "data/renikud_yi_v2"))
    args = ap.parse_args()
    csv.field_size_limit(10_000_000)

    attp = Path(args.attest)
    if not attp.exists():
        raise SystemExit(f"no attest file at {attp}; build the synthetic one with scripts/renikud_v2_synth_attest.py")
    occ, seen = load_attest(attp, args.check_rows)
    print(f"attest {attp}: {seen['rows']:,} rows, first {min(args.check_rows, seen['rows'])} format-checked; "
          f"synthetic {seen['synthetic']:,}; with stress_margin {seen['with_stress_margin']:,}; with posterior {seen['with_posterior']:,}",
          flush=True)
    if args.check_only:
        return

    held = set(args.heldout_episodes.split(","))
    val_eps = set(json.loads(Path(args.build_stats).read_text())["val_eps_episodes"]) if Path(args.build_stats).exists() else set()
    if not val_eps:
        print(f"WARNING: no val_eps episodes read from {args.build_stats}", flush=True)
    excluded = held | val_eps
    dictionary = json.loads(Path(args.dictionary).read_text(encoding="utf-8"))
    gold_by_key = {v["key"]: v for v in dictionary.values()}

    # engine readings per chunk (route/conf/ipa), corpus text per chunk
    targets: dict[tuple[str, int], list[dict]] = {}
    for line in open(args.targets, encoding="utf-8"):
        t = json.loads(line)
        targets[(t["episode"], int(t["chunk_idx"]))] = t["words"]
    attested_chunks = {(ep, ci) for ep, ci, _ in occ}

    # type-level readings from the POOL episodes only (never a held-out / val_eps decision)
    per_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_type_stress: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for (ep, _, _), r in occ.items():
        if ep in excluded or r["route"] != "rule" or fnum(r.get("margin")) < args.margin_type:
            continue
        reading = " ".join(r["chosen"])
        per_type[r["key"]][reading] += 1
        k, trusted = ear_stress(r, args.margin_stress)
        if trusted:
            per_type_stress[(r["key"], reading)][k] += 1
    type_reading: dict[str, tuple[list[str], int | None]] = {}
    for key, c in per_type.items():
        total = sum(c.values())
        reading, top = c.most_common(1)[0]
        if total >= args.type_min and top / total >= args.type_share:
            sc = per_type_stress.get((key, reading))
            k = None
            if sc:
                kk, kn = sc.most_common(1)[0]
                if kn / sum(sc.values()) >= args.type_share:
                    k = kk
            type_reading[key] = (reading.split(), k)

    # trainer val episodes: a seeded draw from the pool
    pool_eps = sorted({ep for ep, _ in targets if ep not in excluded})
    rng = random.Random(args.seed)
    trainer_val = set(rng.sample(pool_eps, min(args.trainer_val_episodes, len(pool_eps))))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    stats: collections.Counter = collections.Counter()
    by_split: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    align_cache: dict[tuple[str, str], list | None] = {}
    align_fail: collections.Counter = collections.Counter()
    stress_src: collections.Counter = collections.Counter()

    for r in csv.DictReader(open(args.corpus, encoding="utf-8"), delimiter="\t"):
        ep, ci = r["episode"], int(r["chunk_idx"])
        if ep in val_eps:
            stats["rows_excluded_val_eps"] += 1
            continue
        if args.only_attested_chunks and (ep, ci) not in attested_chunks:
            continue
        split = "test" if ep in held else ("val" if ep in trainer_val else "train")
        text = r["text"]
        words = targets.get((ep, ci))
        if words is None:
            stats["rows_no_targets"] += 1
            continue
        cons = [IGNORE] * len(text)
        vow = [IGNORE] * len(text)
        strs = [IGNORE] * len(text)
        src = [""] * len(text)
        matches = list(_HEB.finditer(text))
        if len(matches) != len(words):
            stats["rows_token_mismatch"] += 1
            continue
        for hi, (m, e) in enumerate(zip(matches, words)):
            w = m.group(0)
            key = lexicon_key(w)
            stats["tokens"] += 1
            by_split[split]["tokens"] += 1
            rec = occ.get((ep, ci, hi))
            if rec is not None and rec["key"] != key:
                stats["attest_key_mismatch"] += 1
                rec = None
            eng_ph = list(e["ph"])
            eng_k = stress_ordinal(e["ipa"]) if e["ipa"] else None
            phones: list[str] | None = None
            k: int | None = None
            source = None
            ssrc = "engine"
            table_only = split == "test"
            if split == "test" and args.heldout_labels == "none":
                stats["ignore_heldout"] += 1
                continue
            if key in gold_by_key:
                variants = [list(v) for v in gold_by_key[key]["variants"]]
                if rec is not None and not table_only and fnum(rec.get("margin")) >= args.margin_gold and list(rec["chosen"]) in variants:
                    phones = list(rec["chosen"])
                    stats["gold_variant_from_ear"] += 1
                else:
                    phones = eng_ph if eng_ph in variants else variants[0]
                    stats["gold_variant_from_engine" if eng_ph in variants else "gold_variant_primary"] += 1
                k = eng_k
                if rec is not None and not table_only and list(rec["chosen"]) == phones:
                    ek, trusted = ear_stress(rec, args.margin_stress)
                    if trusted:
                        k, ssrc = ek, "ear"
                source = "gold"
            elif e["route"] == "lexicon" and e["conf"] in ("HIGH", "MED") and e["ipa"]:
                phones, k, source = eng_ph, eng_k, "lexicon"
            elif e["route"] == "rule" and e["ipa"] and not table_only:
                if rec is not None and fnum(rec.get("margin")) >= args.margin_occ:
                    phones, source = list(rec["chosen"]), "audio-occ"
                    ek, trusted = ear_stress(rec, args.margin_stress)
                    k, ssrc = (ek, "ear") if trusted else (eng_k, "engine")
                elif key in type_reading:
                    phones, tk = type_reading[key]
                    source = "audio-type"
                    k, ssrc = (tk, "type") if tk is not None else (eng_k, "engine")
            if phones is None and args.engine_all and e["ipa"] and not table_only:
                phones, k, source = eng_ph, eng_k, "engine"
            if phones is None:
                stats["ignore_" + (source or "unvouched")] += 1
                by_split[split]["ignore"] += 1
                continue
            ipa = place_stress(phones, k)
            ak = (w, ipa)
            if ak not in align_cache:
                align_cache[ak] = align_word(w, ipa)
            aligned = align_cache[ak]
            if aligned is None:
                stats["align_fail_" + source] += 1
                align_fail[w] += 1
                continue
            pos = m.start()
            letters_in_text = [(i, ch) for i, ch in enumerate(w, start=pos) if not ("֑" <= ch <= "ׇ")]
            if len(letters_in_text) != len(aligned):
                stats["align_len_mismatch"] += 1
                continue
            for (ti, _), (_, chunk) in zip(letters_in_text, aligned):
                c, v, st = parse_chunk(chunk) if chunk else ("", "", 0)
                cons[ti] = C2ID[c]
                vow[ti] = V2ID[v]
                strs[ti] = st
                src[ti] = source
            stats["labelled_" + source] += 1
            by_split[split]["labelled_" + source] += 1
            if n_full_vowels(phones) >= 2:
                stress_src[f"{source}:{ssrc}"] += 1
        files[split].write(json.dumps({"id": f"{ep}-{ci:05d}", "episode": ep, "text": text,
                                       "cons": cons, "vowel": vow, "stress": strs, "src": src}, ensure_ascii=False) + "\n")
        stats["rows_" + split] += 1
    for f in files.values():
        f.close()
    (out / "labels.json").write_text(json.dumps({"consonants": list(CONS_CLASSES), "vowels": list(VOWEL_CLASSES),
                                                  "ignore": IGNORE}, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg = vars(args) | {"heldout": sorted(held), "val_eps": sorted(val_eps), "trainer_val": sorted(trainer_val),
                        "type_readings": len(type_reading), "attest_rows": seen["rows"], "attest_synthetic": seen["synthetic"]}
    (out / "prepare_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    tot = stats["tokens"]
    lab = sum(v for k, v in stats.items() if k.startswith("labelled_"))
    md = ["# ReNikud-yi v2 dataset", "",
          f"attest: `{attp}` ({seen['rows']:,} rows{', SYNTHETIC' if seen['synthetic'] else ''})", "",
          f"thresholds: margin-occ {args.margin_occ}, margin-stress {args.margin_stress}, margin-gold {args.margin_gold}, "
          f"margin-type {args.margin_type}, type-min {args.type_min}, type-share {args.type_share}; "
          f"type readings {len(type_reading):,}", "",
          f"held-out (test.jsonl, labels={args.heldout_labels}): {', '.join(sorted(held))}; "
          f"val_eps excluded ({len(val_eps)}): {', '.join(sorted(val_eps))}; trainer val: {', '.join(sorted(trainer_val))}", "",
          f"tokens {tot:,}; labelled {lab:,} ({lab / max(1, tot):.1%})", "",
          "| source | tokens |", "| --- | ---: |"]
    for k, v in sorted(stats.items()):
        md.append(f"| `{k}` | {v:,} |")
    md += ["", "| split | tokens | labelled | ignore |", "| --- | ---: | ---: | ---: |"]
    for s in ("train", "val", "test"):
        c = by_split[s]
        md.append(f"| {s} | {c['tokens']:,} | {sum(v for k, v in c.items() if k.startswith('labelled_')):,} | {c['ignore']:,} |")
    md += ["", "Stress source on labelled polysyllables (>= 2 full vowels), `label source:stress source`:", ""]
    for k, v in sorted(stress_src.items()):
        md.append(f"- `{k}` {v:,}")
    md += ["", "Most frequent unalignable words:", ""]
    for w, n in align_fail.most_common(15):
        md.append(f"- {w} ({n})")
    (out / "dataset_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
