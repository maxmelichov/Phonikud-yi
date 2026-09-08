#!/usr/bin/env python3
"""Text side of the PhoneticXeus fine-tune: what is certain, and how to split it.

Runs locally (needs the G2P engine). Writes, under data/xeus_ft/:

  chunk_targets.jsonl  one row per corpus chunk: every word with its phones,
                       whether it is CERTAIN, and its gold variants if so
  split.json           held-out word types, held-out episodes, per-type counts
  dictionary.json      the certain words and their pronunciations — the
                       lexicon the decoder snaps to

"Certain" means a gold row whose source is chezky-verified or chezky-approved:
a native speaker said so. The 97 claude-annotated gold rows are NOT certain and
are treated like any other engine reading — they give a word its timing in the
alignment, never a training label.

For a certain word the training target is its gold pronunciation. For every
other word the engine's reading is kept only so the forced alignment knows
where the certain words sit; those words never contribute a label.

Usage:
  .venv/bin/python scripts/xeus_ft_text.py [--seed 20260907] [--val-types 60]
        [--val-episodes 14] [--quota 500]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import HARD_PHONES, tokenize_ipa, write_jsonl  # noqa: E402
from yiddish_g2p import hebrew_to_ipa, lexicon_key  # noqa: E402

CERTAIN_SOURCES = frozenset({"chezky-verified", "chezky-approved"})
_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")


def load_certain(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row["source"] not in CERTAIN_SOURCES:
            continue
        variants = []
        for v in row["gold_ipa"].split("|"):
            v = re.sub(r"\[.*?\]", "", v).strip()   # editorial notes ride along in a few rows
            if " " in v:
                continue                              # a spelled-out abbreviation is not a reading
            phones = tokenize_ipa(v)
            if phones and phones not in variants:
                variants.append(phones)
        if not variants:
            continue
        out[lexicon_key(row["word"])] = {
            "word": row["word"],
            "variants": variants,
            "layer": row["layer"],
            "freq": int(row["freq"] or 0),
            "source": row["source"],
        }
    return out


def choose_val_types(certain: dict[str, dict], counts: collections.Counter,
                     n: int, rng: random.Random) -> list[str]:
    """Held-out word types: mid-frequency, stratified by layer, with a floor of
    words carrying the phones the pretrained recognizer gets wrong.

    Frequent words are excluded (holding out די would remove half the corpus
    from training) and so are words seen fewer than 20 times (too few clips
    to measure anything).
    """
    cands = [k for k in certain if 20 <= counts.get(k, 0) <= 3000]
    rng.shuffle(cands)
    by_layer: dict[str, list[str]] = collections.defaultdict(list)
    for k in cands:
        by_layer[certain[k]["layer"]].append(k)

    want = {"L": max(12, n // 5), "E": max(4, n // 12)}
    chosen: list[str] = []
    for layer, k in want.items():
        chosen.extend(by_layer.get(layer, [])[:k])
    hard = set(HARD_PHONES)

    def has_hard(key: str) -> bool:
        return any(p in hard for v in certain[key]["variants"] for p in v)

    rest = [k for k in cands if k not in chosen]
    hard_rest = [k for k in rest if has_hard(k)]
    n_hard = sum(has_hard(k) for k in chosen)
    for k in hard_rest:
        if len(chosen) >= n or n_hard >= n // 2:
            break
        chosen.append(k)
        n_hard += 1
    for k in rest:
        if len(chosen) >= n:
            break
        if k not in chosen:
            chosen.append(k)
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--gold", default=str(REPO / "data/gold/g2p_gold_v3.csv"))
    ap.add_argument("--out", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--val-types", type=int, default=60)
    ap.add_argument("--val-episodes", type=int, default=14)
    ap.add_argument("--quota", type=int, default=500,
                    help="target number of training segments per certain type")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    certain = load_certain(Path(args.gold))
    print(f"certain types: {len(certain)}  "
          f"(with variants: {sum(len(v['variants']) > 1 for v in certain.values())})")

    corpus = list(csv.DictReader(open(args.corpus, encoding="utf-8"), delimiter="\t"))
    counts: collections.Counter = collections.Counter()
    cache: dict[str, list[str]] = {}
    rows = []
    n_tok = n_cert = n_empty = 0
    for r in corpus:
        words = []
        for w in _HEB.findall(r["text"]):
            key = lexicon_key(w)
            n_tok += 1
            if key in certain:
                g = certain[key]
                counts[key] += 1
                n_cert += 1
                words.append({"w": w, "key": key, "ph": g["variants"][0],
                              "certain": True, "variants": g["variants"]})
            else:
                if w not in cache:
                    cache[w] = tokenize_ipa(hebrew_to_ipa(w, stress=True))
                ph = cache[w]
                if not ph:
                    n_empty += 1
                words.append({"w": w, "key": key, "ph": ph, "certain": False})
        rows.append({
            "episode": r["episode"],
            "chunk_idx": int(r["chunk_idx"]),
            "file": f"data/chunks/{r['episode']}/chunk_{int(r['chunk_idx']):05d}.mp3",
            "start_s": float(r["start_s"]),
            "end_s": float(r["end_s"]),
            "words": words,
        })
    print(f"chunks {len(rows):,}  tokens {n_tok:,}  certain {n_cert:,} "
          f"({n_cert / n_tok:.1%})  quarantined/empty {n_empty:,}")

    rng = random.Random(args.seed)
    episodes = sorted({r["episode"] for r in rows})
    val_eps = sorted(rng.sample(episodes, args.val_episodes))
    val_types = choose_val_types(certain, counts, args.val_types, rng)
    print(f"held-out episodes: {len(val_eps)}  held-out types: {len(val_types)}  "
          f"(L={sum(certain[k]['layer'] == 'L' for k in val_types)}, "
          f"tokens covered={sum(counts[k] for k in val_types):,})")

    n = write_jsonl(out / "chunk_targets.jsonl", rows)
    split = {
        "seed": args.seed,
        "quota": args.quota,
        "val_episodes": val_eps,
        "val_types": val_types,
        "val_type_words": {k: certain[k]["word"] for k in val_types},
        "type_counts": dict(counts),
        "certain_types": len(certain),
        "chunks": n,
    }
    (out / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    dictionary = {
        v["word"]: {"key": k, "variants": v["variants"], "layer": v["layer"],
                    "freq": v["freq"], "source": v["source"]}
        for k, v in certain.items()
    }
    (out / "dictionary.json").write_text(json.dumps(dictionary, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {out / 'chunk_targets.jsonl'} ({n:,} rows), split.json, dictionary.json "
          f"({len(dictionary)} words)")


if __name__ == "__main__":
    main()
