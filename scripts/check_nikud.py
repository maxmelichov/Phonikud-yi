#!/usr/bin/env python
"""Consistency and safety checks for a diacritized Yiddish corpus.

Run this on data/diacritics_r4 before training on it. Nothing here calls the API;
it is pure inspection, so it is cheap to run as often as you like.

Checks, in order of how badly a failure would hurt:

  1. LETTER SAFETY (fatal). Stripping the marks from the retagged corpus must
     reproduce the source corpus exactly. If it does not, the training targets no
     longer align with the inputs and the whole set is unusable.
  2. TYPE CONSISTENCY. How many bare words carry exactly one pointing. This is the
     number that was 0.26 in r3c and is what makes the voice pick a dialect at
     random when it is low.
  3. KOMETS/PASEKH SPOT CHECK. The words a native reviewer flagged -- האט, וואס,
     דאס, האבן -- must be pointed with a komets, and must phonemize to /u/.
  4. UNPOINTED WORDS. Hebrew-origin words left bare cannot be pronounced at all.
  5. WORST OFFENDERS. The remaining multi-variant types, so the next pass has a
     target list.

Usage:
  .venv/bin/python scripts/check_nikud.py                     # r4 vs r3c
  .venv/bin/python scripts/check_nikud.py --corpus data/diacritics_r3c
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yiddish_g2p as g  # noqa: E402
from scripts.apply_nikud_lexicon import split_affixes  # noqa: E402
from scripts.build_nikud_lexicon import bare_of  # noqa: E402
from scripts.nikud_yi import canon  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
KOMETS = "ָ"
_HEBREW = re.compile(r"[א-ת]")

# What a native Hasidic speaker said these must sound like. Each is a komets word
# whose realisation as /a/ or /o/ was the single loudest complaint about the voice.
SPOT_CHECKS = [
    ("האט", "hut"), ("האב", "hub"), ("האבן", "hˈubən"), ("וואס", "vus"),
    ("דאס", "dus"), ("דא", "du"), ("טאג", "tuɡ"), ("יאר", "jur"),
    ("נאך", "nux"), ("זאגן", "zˈuɡən"), ("אלעמאל", "ˈaləmul"), ("אבער", "ˈubər"),
]


def read(corpus: Path) -> dict[str, list[str]]:
    out = {}
    for name in ("train", "val", "test"):
        p = corpus / f"{name}.txt"
        if p.exists():
            out[name] = p.read_text(encoding="utf-8").splitlines()
    return out


def variants(lines: list[str]) -> dict[str, collections.Counter]:
    v: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in lines:
        for token in line.split():
            _, core, _ = split_affixes(token)
            b = bare_of(core)
            if b and _HEBREW.search(b):
                v[b][unicodedata.normalize("NFC", core)] += 1
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/diacritics_r4")
    ap.add_argument("--source", default="data/diacritics_r3c")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    corpus, source = REPO / args.corpus, REPO / args.source
    got = read(corpus)
    if not got:
        print(f"no corpus at {corpus}", file=sys.stderr)
        return 1
    src = read(source)
    failures = 0

    print(f"=== 1. LETTER SAFETY  ({args.corpus} vs {args.source})")
    for name, lines in got.items():
        if name not in src:
            print(f"  {name}: no source to compare against -- SKIPPED")
            continue
        bad = [i for i, (a, b) in enumerate(zip(lines, src[name])) if canon(a) != canon(b)]
        n_lines_differ = len(lines) != len(src[name])
        if bad or n_lines_differ:
            failures += 1
            print(f"  {name}: FAIL -- {len(bad)} lines changed letters"
                  + (", line count differs" if n_lines_differ else ""))
            for i in bad[:3]:
                print(f"      line {i}: {lines[i][:90]}")
        else:
            print(f"  {name}: OK -- {len(lines):,} lines reproduce the source exactly")

    print("\n=== 2. TYPE CONSISTENCY")
    all_lines = [ln for lines in got.values() for ln in lines]
    v = variants(all_lines)
    single = sum(1 for c in v.values() if len(c) == 1)
    inst_ok = sum(c.most_common(1)[0][1] for c in v.values())
    inst_all = sum(sum(c.values()) for c in v.values())
    print(f"  types                 : {len(v):,}")
    print(f"  single-pointing types : {single:,} = {single / len(v):.2%}")
    print(f"  instance consistency  : {inst_ok / inst_all:.2%}")
    print(f"  mean variants / type  : {sum(len(c) for c in v.values()) / len(v):.3f}")

    print("\n=== 3. KOMETS SPOT CHECK (native-speaker corrections)")
    for bare, want in SPOT_CHECKS:
        forms = v.get(bare)
        if not forms:
            print(f"  {bare:10} -- not in corpus")
            continue
        top, n = forms.most_common(1)[0]
        ipa = g.hebrew_to_ipa(top)
        ok = KOMETS in unicodedata.normalize("NFD", top) and ipa == want
        failures += not ok
        print(f"  {bare:10} {top:16} -> {ipa:12} want {want:12} "
              f"{'OK' if ok else 'FAIL'}  ({len(forms)} variant(s), n={n:,})")

    print("\n=== 4. UNPOINTED WORDS")
    unpointed: collections.Counter = collections.Counter()
    for b, forms in v.items():
        for form, n in forms.items():
            if len(b) >= 4 and '"' not in b and "'" not in b:
                if not any(unicodedata.category(c) == "Mn"
                           for c in unicodedata.normalize("NFD", form)):
                    unpointed[b] += n
    print(f"  {len(unpointed):,} types / {sum(unpointed.values()):,} instances "
          f"({sum(unpointed.values()) / inst_all:.2%}) carry no marks at all")
    for w, n in unpointed.most_common(8):
        print(f"      {w:16} {n:,}")

    print(f"\n=== 5. WORST REMAINING OFFENDERS (top {args.top})")
    worst = sorted(v.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))
    for b, forms in worst[: args.top]:
        if len(forms) == 1:
            break
        shown = ", ".join(f"{f}×{n}" for f, n in forms.most_common(4))
        print(f"  {b:14} {len(forms):>3} variants  {shown}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
