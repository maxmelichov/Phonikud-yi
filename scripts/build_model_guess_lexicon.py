#!/usr/bin/env python
"""Final rescue: phonikud-yi v3 GUESSES the remaining quarantined Hebrew.

Policy (user directive 2026-08-08): no Hebrew word is ever dropped. For every
quarantined type no other evidence covers, the retrained pointing model (97%
held-out accuracy on evidence-backed Hebrew) points it IN SENTENCE CONTEXT;
the predicted pointing is read through the Whole-Hebrew register and emitted
at LOW confidence with reason 'model-pointed-guess'. Guesses stay in the
verification queue — any native verdict, audio vote, or book pointing
instantly outranks them (this table loads LAST in the rescue chain).

Method per type:
  - up to --contexts corpus sentences containing the type are pointed by the
    model (batched); the token's pointed form is extracted per sentence
  - each pointed form is read via read_pointed_wh; readings must pass the
    letter-identity, closed-inventory, and vowel-shape gates
  - if >=2 contexts agree on the reading -> agreement 'multi'; else the first
    valid reading is taken with agreement 'single'

Run: .venv/bin/python scripts/build_model_guess_lexicon.py
Output: data/model_pointed_lk.py + a printed summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from yiddish_g2p import (  # noqa: E402
    g2p_token,
    ipa_phone_violations,
    lexicon_key,
    read_pointed_wh,
    violates_vowel_ratio,
)

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״־]*")
_NIKUD = re.compile(r"[֑-ׇ]")
_PURE_HEB = re.compile(r"^[א-ת]+$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phonikud_yi_v3_gpu/best")
    ap.add_argument("--contexts", type=int, default=2)
    ap.add_argument("--batch-lines", type=int, default=64)
    ap.add_argument("--limit-types", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    # 1. remaining quarantined Hebrew types (skip abbreviations — letter-name
    # path owns them — and anything not purely Hebrew letters)
    targets: dict[str, int] = {}
    for row in csv.DictReader(open(REPO / "data/phonemized/v3/quarantine.tsv"), delimiter="\t"):
        w = row["word"]
        if '"' in w or not _PURE_HEB.match(w.replace("'", "")):
            continue
        rec = g2p_token(w)
        if rec["route"] == "fallback":
            targets[w] = int(row["freq"])
    if args.limit_types:
        targets = dict(sorted(targets.items(), key=lambda kv: -kv[1])[: args.limit_types])
    print(f"{len(targets)} quarantined Hebrew types to guess", file=sys.stderr)

    # 2. collect context sentences (first N rows containing each type)
    contexts: dict[str, list[str]] = defaultdict(list)
    need = set(targets)
    for row in csv.DictReader(open(REPO / "data/yiddish_tts_dataset.tsv"), delimiter="\t"):
        if not need:
            break
        toks = set(_HEB.findall(row["text"]))
        for w in list(need):
            if w in toks:
                contexts[w].append(row["text"])
                if len(contexts[w]) >= args.contexts:
                    need.discard(w)
    sentences = sorted({s for ss in contexts.values() for s in ss})
    print(f"{len(sentences)} unique sentences to point", file=sys.stderr)

    # 3. point all sentences with the model (batched over --stdin)
    pointed: dict[str, str] = {}
    for i in range(0, len(sentences), args.batch_lines):
        batch = sentences[i : i + args.batch_lines]
        r = subprocess.run(
            [str(REPO / ".venv/bin/python"), str(REPO / "scripts/point_text.py"),
             "-m", args.model, "--stdin"],
            input="\n".join(batch), capture_output=True, text=True)
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
        if len(lines) == len(batch):
            pointed.update(zip(batch, lines))
        else:
            print(f"  batch {i}: line mismatch ({len(lines)}/{len(batch)}), skipped",
                  file=sys.stderr)
        if (i // args.batch_lines) % 10 == 0:
            print(f"  pointed {min(i+args.batch_lines, len(sentences))}/{len(sentences)}",
                  file=sys.stderr)

    # 4. extract readings per type
    out: dict[str, dict] = {}
    stats = Counter()
    for w, sents in contexts.items():
        readings: list[tuple[str, str]] = []  # (ipa, pointed_form)
        for s in sents:
            ps = pointed.get(s)
            if not ps:
                continue
            toks = _HEB.findall(s)
            ptoks = _HEB.findall(ps)
            if len(toks) != len(ptoks):
                stats["align_mismatch"] += 1
                continue
            for t, pt in zip(toks, ptoks):
                if t != w:
                    continue
                if _NIKUD.sub("", pt) != t:
                    stats["letter_mismatch"] += 1
                    continue
                ipa = read_pointed_wh(pt)
                if not ipa or ipa_phone_violations(ipa) or violates_vowel_ratio(ipa):
                    stats["bad_reading"] += 1
                    continue
                readings.append((ipa, pt))
                break
        if not readings:
            stats["no_reading"] += 1
            continue
        tally = Counter(ipa for ipa, _ in readings)
        best_ipa, n = tally.most_common(1)[0]
        best_pointed = next(pt for ipa, pt in readings if ipa == best_ipa)
        out[w] = {"ipa": best_ipa, "pointed": best_pointed,
                  "agreement": "multi" if n >= 2 else "single",
                  "freq": targets[w]}
        stats["guessed"] += 1

    # 5. emit the generated module
    lines = [
        '"""GENERATED — model-guessed loshn-koydesh readings (final rescue).',
        '',
        'phonikud-yi v3 (held-out Hebrew accuracy 97% vs evidence readings)',
        'pointed each remaining quarantined type in sentence context; the',
        'predicted pointing is read via the Whole-Hebrew register. Emitted at',
        "LOW confidence, reason 'model-pointed-guess' — the LAST link in the",
        'rescue chain; every other evidence source outranks it, and these words',
        'remain in the verification queue.',
        'Regenerate: scripts/build_model_guess_lexicon.py',
        '"""',
        '',
        'MODEL_POINTED_LK = {',
    ]
    for w, v in sorted(out.items(), key=lambda kv: -kv[1]["freq"]):
        lines.append(f'    {w!r}: {{"ipa": {v["ipa"]!r}, "pointed": {v["pointed"]!r}, '
                     f'"agreement": {v["agreement"]!r}}},')
    lines.append("}")
    (REPO / "data/model_pointed_lk.py").write_text("\n".join(lines) + "\n")

    tok = sum(v["freq"] for v in out.values())
    print(f"\nguessed {stats['guessed']}/{len(targets)} types ({tok} tokens); "
          f"stats: {dict(stats)}")
    print(f"wrote data/model_pointed_lk.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
