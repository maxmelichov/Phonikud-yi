#!/usr/bin/env python
"""Phonemize the corpus under spec v3 and enforce the §12 QA gates.

Reads the ``text`` column of data/corpus/yiddish_tts_dataset.tsv, routes every token
through yiddish_g2p.g2p_tokens (gold lexicon first, rules last), and writes:

    tokens.tsv      one row per token TYPE that is fit to emit:
                    word, ipa_primary, variants, layer, route, confidence, freq
    quarantine.tsv  types whose output is NOT fit for the training set
                    (route=fallback: vowel-less/ill-shaped rule output, unknown
                    abbreviations) — flagged approximations only, never emitted
    low_conf.tsv    LOW-confidence types sorted by token frequency — this is the
                    next native-verification batch (§12 iteration loop)
    oov_lk.tsv      types the LK detector claims that no lexicon knows, sorted
                    by frequency
    alef_defaults.tsv  every runtime application of the ambiguous-א default (§4)
    lines.tsv       id + the emitted IPA line, for rows with no quarantined token

QA gates (§12), all enforced, nonzero exit on any violation:
    (a) zero vowel-less / ill-shaped outputs in the EMITTED set
    (b) zero symbols outside the §1 closed inventory in the EMITTED set
    (c) both regression suites pass (scripts/test_g2p.py, test_g2p_spec.py)
    (d) the gold lexicon reproduces byte-identically (scripts/test_g2p_gold.py)

Run:  .venv/bin/python scripts/run_corpus_v3.py [--limit N] [--out DIR]
      --limit 0 processes the whole file.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yiddish_g2p import (  # noqa: E402
    A_DEFAULT_LOG,
    g2p_fingerprint,
    g2p_tokens,
    ipa_phone_violations,
    normalize_ipa_spacing,
    reset_default_logs,
    violates_vowel_ratio,
)

DATASET = ROOT / "data" / "corpus" / "yiddish_tts_dataset.tsv"
DEFAULT_OUT = ROOT / "data" / "phonemized" / "v3"
SUITES = ("test_g2p.py", "test_g2p_spec.py", "test_g2p_gold.py",
          "test_audio_evidence.py")


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def run_suites() -> dict[str, bool]:
    results = {}
    for name in SUITES:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name)],
            capture_output=True, text=True, cwd=ROOT,
        )
        results[name] = proc.returncode == 0
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.strip().splitlines()[-8:])
            print(f"--- {name} FAILED ---\n{tail}\n")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000,
                    help="corpus rows to process; 0 = all")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--skip-suites", action="store_true",
                    help="skip gates (c)/(d); for fast iteration only")
    ap.add_argument("--emit-partial", action="store_true",
                    help="keep a line with quarantined tokens elided, provided "
                         ">=4 word tokens survive and <20%% of the line is dropped; "
                         "lines.tsv then carries n_tokens/n_elided/elided_idx")
    ap.add_argument("--min-kept", type=int, default=4,
                    help="--emit-partial: minimum surviving word tokens")
    ap.add_argument("--max-elided-frac", type=float, default=0.20,
                    help="--emit-partial: max dropped share of a line's word tokens")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    reset_default_logs()

    types: dict[str, dict] = {}
    freq: Counter = Counter()
    tokens_seen = 0
    rows_read = 0
    lines: list[list[str]] = []
    emitted_line_tokens = 0          # word tokens actually present in lines.tsv
    elided_hist: Counter = Counter()  # n_elided -> number of partial lines
    partial_rescued = 0               # lines kept only because of --emit-partial
    partial_rejected = 0              # dirty lines the policy still refuses

    with args.dataset.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rows_read += 1
            records = g2p_tokens(text)
            word_idx = -1            # position among word tokens in this line
            n_words = 0
            elided: list[int] = []
            for rec in records:
                if rec["reason"] == "punctuation":
                    continue
                tokens_seen += 1
                n_words += 1
                word_idx += 1
                freq[rec["word"]] += 1
                types.setdefault(rec["word"], rec)
                if rec["route"] == "fallback":
                    elided.append(word_idx)
            clean = not elided

            def render(drop: set[int]) -> str:
                """Join the line, emitting only lead/trail for dropped tokens."""
                out, k = [], -1
                for r in records:
                    if r["reason"] == "punctuation":
                        out.append(r["lead"] + r["ipa_primary"] + r["trail"])
                        continue
                    k += 1
                    body = "" if k in drop else r["ipa_primary"]
                    out.append(r["lead"] + body + r["trail"])
                return normalize_ipa_spacing(" ".join(out))

            if clean:
                emitted_line_tokens += n_words
                if args.emit_partial:
                    lines.append([row.get("id", ""), render(set()),
                                  str(n_words), "0", ""])
                else:
                    lines.append([row.get("id", ""), render(set())])
            elif args.emit_partial:
                kept = n_words - len(elided)
                frac = len(elided) / n_words if n_words else 1.0
                if kept >= args.min_kept and frac < args.max_elided_frac:
                    emitted_line_tokens += kept
                    partial_rescued += 1
                    elided_hist[len(elided)] += 1
                    lines.append([
                        row.get("id", ""), render(set(elided)),
                        str(n_words), str(len(elided)),
                        ",".join(str(i) for i in elided),
                    ])
                else:
                    partial_rejected += 1
            if args.limit and rows_read >= args.limit:
                break

    emitted = {w: r for w, r in types.items() if r["route"] != "fallback"}
    quarantined = {w: r for w, r in types.items() if r["route"] == "fallback"}

    def sort_key(item):
        return (-freq[item[0]], item[0])

    write_tsv(
        args.out / "tokens.tsv",
        ["word", "ipa_primary", "variants", "layer", "route", "confidence", "freq"],
        [[w, r["ipa_primary"], "|".join(r["variants"]), r["layer"], r["route"],
          r["confidence"], freq[w]] for w, r in sorted(emitted.items(), key=sort_key)],
    )
    write_tsv(
        args.out / "quarantine.tsv",
        ["word", "flagged_approximation", "layer", "reason", "freq"],
        [[w, r["ipa_primary"], r["layer"], r["reason"], freq[w]]
         for w, r in sorted(quarantined.items(), key=sort_key)],
    )
    low = {w: r for w, r in types.items() if r["confidence"] == "LOW"}
    write_tsv(
        args.out / "low_conf.tsv",
        ["word", "ipa_primary", "layer", "route", "reason", "freq"],
        [[w, r["ipa_primary"], r["layer"], r["route"], r["reason"], freq[w]]
         for w, r in sorted(low.items(), key=sort_key)],
    )
    oov_lk = {w: r for w, r in types.items() if "lk-fallback" in r["reason"]}
    write_tsv(
        args.out / "oov_lk.tsv",
        ["word", "ipa_primary", "route", "reason", "freq"],
        [[w, r["ipa_primary"], r["route"], r["reason"], freq[w]]
         for w, r in sorted(oov_lk.items(), key=sort_key)],
    )
    write_tsv(
        args.out / "alef_defaults.tsv", ["word", "applications"],
        [[w, n] for w, n in A_DEFAULT_LOG.most_common()],
    )
    line_header = (["id", "ipa", "n_tokens", "n_elided", "elided_idx"]
                   if args.emit_partial else ["id", "ipa"])
    write_tsv(args.out / "lines.tsv", line_header, lines)

    # --- gates ---------------------------------------------------------------
    bad_shape = [w for w, r in emitted.items() if violates_vowel_ratio(r["ipa_primary"])]
    bad_phones = sorted({
        sym for r in emitted.values() for sym in ipa_phone_violations(r["ipa_primary"])
    })
    suites = {n: True for n in SUITES} if args.skip_suites else run_suites()

    low_tokens = sum(freq[w] for w in low)
    quar_tokens = sum(freq[w] for w in quarantined)
    conf_tokens = Counter()
    for w, r in types.items():
        conf_tokens[r["confidence"]] += freq[w]
    route_tokens = Counter()
    for w, r in types.items():
        route_tokens[r["route"]] += freq[w]

    print(f"fingerprint        {g2p_fingerprint()}")
    print(f"rows processed     {rows_read}")
    print(f"tokens             {tokens_seen} ({len(types)} types)")
    print(f"emitted types      {len(emitted)}   quarantined types {len(quarantined)}"
          f" ({quar_tokens} tokens, {quar_tokens / max(tokens_seen, 1):.2%})")
    for route in ("lexicon", "rule", "fallback"):
        print(f"  route {route:9s} {route_tokens[route]:7d} tokens "
              f"({route_tokens[route] / max(tokens_seen, 1):6.2%})")
    for conf in ("HIGH", "MED", "LOW"):
        print(f"  conf  {conf:9s} {conf_tokens[conf]:7d} tokens "
              f"({conf_tokens[conf] / max(tokens_seen, 1):6.2%})")
    print(f"LOW_CONF share     {low_tokens / max(tokens_seen, 1):.2%} of running tokens"
          f"  ({len(low)} types)")
    print(f"OOV-LK types       {len(oov_lk)}")
    print(f"א-defaults logged  {sum(A_DEFAULT_LOG.values())} applications, "
          f"{len(A_DEFAULT_LOG)} types")
    mode = "partial" if args.emit_partial else "strict"
    print(f"line policy        {mode}"
          + (f" (min_kept={args.min_kept}, max_elided_frac={args.max_elided_frac})"
             if args.emit_partial else ""))
    print(f"lines emitted      {len(lines)} of {rows_read} rows "
          f"({len(lines) / max(rows_read, 1):.2%})")
    print(f"line token yield   {emitted_line_tokens} of {tokens_seen} "
          f"({emitted_line_tokens / max(tokens_seen, 1):.2%})")
    if args.emit_partial:
        print(f"  partial lines    {partial_rescued} rescued, "
              f"{partial_rejected} dirty lines still dropped")
        print(f"  elided tokens    {sum(n * c for n, c in elided_hist.items())} "
              f"in rescued lines")
        for n in sorted(elided_hist):
            print(f"    n_elided={n:<3d} {elided_hist[n]:6d} lines")
    print(f"outputs            {args.out}")

    gates = [
        ("(a) zero vowel-less / ill-shaped emitted outputs",
         not bad_shape, f"{len(bad_shape)} offenders, e.g. {bad_shape[:5]}"),
        ("(b) zero symbols outside the §1 inventory",
         not bad_phones, f"stray symbols {bad_phones}"),
        ("(c) regression suites pass",
         all(suites[n] for n in ("test_g2p.py", "test_g2p_spec.py",
                                 "test_audio_evidence.py")),
         "see failures above"),
        ("(d) gold reproduces byte-identically",
         suites["test_g2p_gold.py"], "see failures above"),
    ]
    print("\nQA GATES")
    failed = 0
    for label, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f" -- {detail}"))
        failed += 0 if ok else 1
    print(f"\n{'ALL GATES PASS' if not failed else f'{failed} GATE(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
