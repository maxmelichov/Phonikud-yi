#!/usr/bin/env python
"""Coverage test for the PhoneticXeus -> Yiddish phoneme conversion.

Asserts that every symbol in the model's 428-token vocabulary
(data/xeus_ipa_vocab.json) either folds into the closed v3 inventory or is a
documented deliberate drop — so a model update or mapper edit can never
silently produce out-of-inventory phones or silently lose votes.

Run: .venv/bin/python scripts/test_xeus_map.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from xeus_map import DELIBERATE_DROPS, INVENTORY, fold_phone_string, map_transcript  # noqa: E402


def main() -> int:
    toks = json.load(open(REPO / "data" / "xeus_ipa_vocab.json"))
    failures = []
    n_map = n_drop = 0
    for t in toks:
        if t.startswith("<") and t.endswith(">"):
            continue
        out = fold_phone_string(t)
        if not out:
            base = [c for c in t if not (0x0300 <= ord(c) <= 0x036F)]
            if all(c in DELIBERATE_DROPS or c in "ʰʲʷ" for c in base):
                n_drop += 1
            else:
                failures.append(f"undocumented drop: {t!r}")
        else:
            illegal = [p for p in out if p not in INVENTORY]
            if illegal:
                failures.append(f"illegal output: {t!r} -> {out}")
            else:
                n_map += 1

    # transcript-level sanity: special tokens vanish, phones fold
    sample = map_transcript("<sos>/ʃ/a/b/ə/s/<eos>")
    if sample != ["ʃ", "a", "b", "ə", "s"]:
        failures.append(f"map_transcript sanity: {sample}")

    print(f"{n_map} mapped, {n_drop} deliberate drops, {len(failures)} FAILED")
    for f in failures:
        print(" ", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
