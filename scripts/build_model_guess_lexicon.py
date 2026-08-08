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
  - each pointed form is read in BOTH registers and scripts/register_policy.py
    picks which one is primary (merged by default — the word is embedded in a
    Yiddish sentence; Whole-Hebrew only where audio or corpus usage says it is
    quoted). The loser is kept as a variant. Readings must pass the
    letter-identity, closed-inventory, and vowel-shape gates
  - if >=2 contexts agree on the reading -> agreement 'multi'; else the first
    valid reading is taken with agreement 'single'

Run: .venv/bin/python scripts/build_model_guess_lexicon.py
     .venv/bin/python scripts/build_model_guess_lexicon.py --reread
Output: data/model_pointed_lk.py + a printed summary.

--reread re-decides the REGISTER of the existing table from the pointings it
already stores, without touching the model. The model's job is to produce a
pointing; how that pointing is read is a separate decision, and re-running a
GPU pass over thousands of sentences to change the second one would risk
churning the first (a re-point is not deterministic across model revisions).
Use the full path only when the pointings themselves should change.
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
from register_policy import decide, quoted_shares  # noqa: E402
from yiddish_g2p import (  # noqa: E402
    g2p_token,
    lexicon_key,
    ipa_phone_violations,
    violates_vowel_ratio,
)


def readable(ipa: str) -> bool:
    """The §1 gate a reading must pass to be emitted at all."""
    return bool(ipa) and not ipa_phone_violations(ipa) and not violates_vowel_ratio(ipa)

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״־]*")
_NIKUD = re.compile(r"[֑-ׇ]")
_PURE_HEB = re.compile(r"^[א-ת]+$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phonikud_yi_v3_gpu/best")
    ap.add_argument("--contexts", type=int, default=2)
    ap.add_argument("--batch-lines", type=int, default=64)
    ap.add_argument("--limit-types", type=int, default=0, help="0 = all")
    ap.add_argument("--reread", action="store_true",
                    help="re-decide the register of the existing table from "
                         "its stored pointings; the model is not run")
    ap.add_argument("--flips", type=int, default=15)
    args = ap.parse_args()

    shares = quoted_shares()
    if args.reread:
        return reread(shares, args)

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
                verdict = decide(w, pt, shares, validate=readable)
                if verdict is None:
                    stats["bad_reading"] += 1
                    continue
                readings.append((verdict, pt))
                break
        if not readings:
            stats["no_reading"] += 1
            continue
        # Agreement is counted on the PRIMARY reading, which is what the table
        # ships and what a reviewer sees.
        tally = Counter(v["ipa"] for v, _ in readings)
        best_ipa, n = tally.most_common(1)[0]
        verdict, best_pointed = next((v, pt) for v, pt in readings
                                     if v["ipa"] == best_ipa)
        out[w] = {"ipa": best_ipa, "variants": verdict["variants"],
                  "pointed": best_pointed,
                  "register": verdict["register"], "why": verdict["why"],
                  "agreement": "multi" if n >= 2 else "single",
                  "freq": targets[w]}
        stats[f"register_{verdict['register']}"] += 1
        stats[f"why_{verdict['why']}"] += 1
        stats["guessed"] += 1

    # 5. emit the generated module
    write_table(out)

    tok = sum(v["freq"] for v in out.values())
    print(f"\nguessed {stats['guessed']}/{len(targets)} types ({tok} tokens); "
          f"stats: {dict(stats)}")
    print("wrote data/model_pointed_lk.py")
    return 0


HEADER = '''"""GENERATED — model-guessed loshn-koydesh readings (final rescue).

phonikud-yi v3 (held-out Hebrew accuracy 97% vs evidence readings) pointed each
remaining quarantined type in sentence context. The predicted pointing is read
in the register the type is actually used in (scripts/register_policy.py):
MERGED by default — read_pointed_merged(), the way the engine reads a
loshn-koydesh word embedded in a Yiddish sentence (shuruk/kubuts take the
Yiddish u->i shift, a final komets-hey is [ə]) — and Whole-Hebrew
(read_pointed_wh(), spec v2 §7.1) only where audio or corpus usage says the
word is being QUOTED. Each entry records which register won and why; the losing
register ships as a variant.

Emitted at LOW confidence, reason 'model-pointed-guess' — the LAST link in the
rescue chain; every other evidence source outranks it, and these words remain
in the verification queue.

Regenerate: scripts/build_model_guess_lexicon.py
Re-decide the register only (no model run): --reread
"""

MODEL_POINTED_LK = {'''


def write_table(out: dict[str, dict]) -> None:
    lines = [HEADER]
    for w, v in sorted(out.items(), key=lambda kv: -kv[1]["freq"]):
        lines.append(
            f'    {w!r}: {{"ipa": {v["ipa"]!r}, "variants": {v["variants"]!r}, '
            f'"pointed": {v["pointed"]!r}, "register": {v["register"]!r}, '
            f'"why": {v["why"]!r}, "agreement": {v["agreement"]!r}}},')
    lines.append("}")
    (REPO / "data/model_pointed_lk.py").write_text("\n".join(lines) + "\n")


def load_existing() -> dict[str, dict]:
    """The table as it stands, so --reread can re-decide from its pointings."""
    import importlib.util

    path = REPO / "data/model_pointed_lk.py"
    spec = importlib.util.spec_from_file_location("_model_lk_existing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.MODEL_POINTED_LK)


def reread(shares: dict[str, dict], args) -> int:
    """Re-decide the register of every existing entry; keep the pointings.

    Frequencies come from the corpus token counts that quoted_shares() already
    collected, keyed the same way. The quarantine snapshot cannot supply them:
    it is written AFTER the rescue chain runs, so every type this table covers
    has left it. Frequencies only order the output and the flip report.
    """
    existing = load_existing()
    freqs = {k: v["total"] for k, v in shares.items()}

    def freq_of(w: str) -> int:
        return freqs.get(lexicon_key(w), 0)

    out: dict[str, dict] = {}
    stats: Counter = Counter()
    flips: list[tuple[str, int, dict, str]] = []
    for w, old in existing.items():
        pointed = old.get("pointed")
        if not pointed:
            stats["no_pointing"] += 1
            continue
        verdict = decide(w, pointed, shares, validate=readable)
        if verdict is None:
            # Neither register is speakable — this entry was emitted before the
            # gates were applied to both readings. Drop it rather than ship it.
            stats["unreadable"] += 1
            continue
        out[w] = {"ipa": verdict["ipa"], "variants": verdict["variants"],
                  "pointed": pointed, "register": verdict["register"],
                  "why": verdict["why"],
                  "agreement": old.get("agreement", "single"),
                  "freq": freq_of(w)}
        stats[f"register_{verdict['register']}"] += 1
        stats[f"why_{verdict['why']}"] += 1
        if verdict["merged"] != verdict["wh"]:
            stats["distinguishable"] += 1
        if verdict["ipa"] != old["ipa"]:
            stats["changed"] += 1
            stats["changed_tokens"] += freq_of(w)
            flips.append((w, freq_of(w), verdict, old["ipa"]))

    write_table(out)
    print(f"re-read {len(existing)} entries -> {len(out)} kept")
    for k in sorted(stats):
        print(f"  {k:28s} {stats[k]}")
    flips.sort(key=lambda t: (-t[1], t[0]))
    print(f"\nREGISTER FLIPS: {len(flips)} types, "
          f"{stats['changed_tokens']} tokens")
    print(f"  {'word':14s} {'freq':>6s}  {'was (WH)':22s} {'now':22s} why")
    for w, f, v, was in flips[:args.flips]:
        print(f"  {w:14s} {f:6d}  {was:22s} {v['ipa']:22s} {v['why']}")
    print("wrote data/model_pointed_lk.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
