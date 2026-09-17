#!/usr/bin/env python3
"""Round 3c data prep: drop the certain train clips whose וי variant contradicts Chezky's primary reading.

The dictionary lists exactly six words with both an oʊ and an ɔj variant (אויכ, אויס, ארויס,
לויט, הויז, טוישנ). The clip cutter picked the variant the run-1 aligner preferred, which is
the NON-primary one on most of those clips; the chunk stage of the attested curriculum
labels the same words with the primary variant, so the two stages contradict each other on
the very slot the oʊ probe measures. This writes a copy of segments.jsonl with every val row
and every train row kept, except train rows containing one of those words with variant != 0.
No label is edited; seg/ is not touched (symlink it next to the output).

  .venv/bin/python scripts/xeus_ft_filter_variants.py --data data/xeus_ft/run3 \
      --dictionary data/xeus_ft/dictionary.json --out data/xeus_ft/run3c/segments.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xeus_ft/run3")
    ap.add_argument("--dictionary", default="data/xeus_ft/dictionary.json")
    ap.add_argument("--out", default="data/xeus_ft/run3c/segments.jsonl")
    ap.add_argument("--pair", default="oʊ:ɔj", help="the two phones whose two-variant words are filtered")
    args = ap.parse_args()
    a, b = args.pair.split(":")
    dictionary = json.loads(Path(args.dictionary).read_text(encoding="utf-8"))
    two = sorted(v["key"] for v in dictionary.values()
                 if any(a in r for r in v["variants"]) and any(b in r for r in v["variants"]))
    print(f"two-variant {a}/{b} words: {two}")
    src = Path(args.data) / "segments.jsonl"
    out = Path(args.out)
    if out.resolve() == src.resolve():
        raise SystemExit("--out must not be the source segments.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    secs = 0.0
    tally: collections.Counter = collections.Counter()
    splits: collections.Counter = collections.Counter()
    ou_clips = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            if r["split"] == "train":
                bad = [(w["key"], w.get("variant", 0)) for w in r["words"]
                       if w["key"] in two and w.get("variant", 0) != 0]
                if bad:
                    dropped += 1
                    secs += r["dur_s"]
                    for k in bad:
                        tally[k] += 1
                    continue
                kept += 1
                ou_clips += a in r["target"]
            splits[r["split"]] += 1
            g.write(line)
    print(f"dropped {dropped:,} train clips ({secs / 3600:.2f} h), kept {kept:,}; "
          f"{' '.join(f'{s} {n}' for s, n in sorted(splits.items()))}; {a} clips {ou_clips}")
    for (k, v), n in tally.most_common():
        print(f"  {k} variant {v} ({''.join(dictionary_variant(dictionary, k, v))}): {n}")
    print(f"wrote {out}")


def dictionary_variant(dictionary: dict, key: str, v: int) -> list[str]:
    for e in dictionary.values():
        if e["key"] == key:
            return e["variants"][v]
    return []


if __name__ == "__main__":
    main()
