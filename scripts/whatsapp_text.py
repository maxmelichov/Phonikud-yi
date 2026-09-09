#!/usr/bin/env python3
"""Text side for ivrit.ai's crowd-whatsapp-yi: the ear's first multi-speaker data.

20.5 h of scripted messages read by 581 volunteers into WhatsApp, in
Hasidic-American Yiddish (ivrit.ai licence: CC BY 4.0, AI training and
research only, no voice cloning of the speakers — the ear is a phone
recognizer, never a voice). Every message comes with its text.

This writes the same row schema xeus_ft_prepare.py consumes for the corpus
(chunk_targets.jsonl): per message, every Hebrew word with the engine's
reading, its lexicon key, and whether it is a certain (Chezky) word with its
gold variants. Messages containing Latin-script words (code-switched English:
``tickets``, ``phone``) are skipped: the forced alignment has no phones for
them and would hand their audio to the neighbouring Yiddish words.

"episode" is the recorder when the manifest names one, else the message id,
so held-out episodes are held-out SPEAKERS — the test the corpus, with its
single host, could never provide.

Output: data/xeus_ft/whatsapp_targets.jsonl, data/xeus_ft/whatsapp_split.json
Usage:  .venv/bin/python scripts/whatsapp_text.py
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

from xeus_ft_common import tokenize_ipa, write_jsonl  # noqa: E402
from yiddish_g2p import hebrew_to_ipa, lexicon_key  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
_LATIN = re.compile(r"[A-Za-z]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data/ivrit_whatsapp"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--out", default=str(REPO / "data/xeus_ft/whatsapp_targets.jsonl"))
    ap.add_argument("--val-share", type=float, default=0.15, help="share of speakers held out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root)
    certain = {v["key"]: v for v in json.loads(Path(args.dictionary).read_text(encoding="utf-8")).values()}
    recorder: dict[str, str] = {}
    mf = root / "manifest.csv"
    if mf.exists():
        with mf.open(encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames or []
            rcol = next((c for c in cols if "recorder" in c.lower() or "speaker" in c.lower() or "user" in c.lower()), None)
            idcol = next((c for c in cols if "entry" in c.lower() or c.lower() in ("id", "message_id", "session_id")), None)
            if rcol and idcol:
                for r in rd:
                    recorder[r[idcol]] = r[rcol]
        print(f"manifest columns: {cols}  recorder column: {rcol}", flush=True)

    cache: dict[str, list[str]] = {}
    rows = []
    stats = collections.Counter()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        tp, ap_ = d / "transcript.txt", d / "audio.light.opus"
        if not tp.exists() or not ap_.exists():
            stats["missing_files"] += 1
            continue
        text = tp.read_text(encoding="utf-8").strip().strip('"')
        if _LATIN.search(text):
            stats["skipped_latin"] += 1
            continue
        words = []
        for w in _HEB.findall(text):
            key = lexicon_key(w)
            if key in certain:
                g = certain[key]
                words.append({"w": w, "key": key, "ph": g["variants"][0], "certain": True, "variants": g["variants"]})
                stats["certain_tokens"] += 1
            else:
                if w not in cache:
                    cache[w] = tokenize_ipa(hebrew_to_ipa(w, stress=True))
                words.append({"w": w, "key": key, "ph": cache[w], "certain": False})
            stats["tokens"] += 1
        if not words:
            stats["empty"] += 1
            continue
        mid = d.name
        rows.append({"episode": recorder.get(mid, mid), "chunk_idx": 0, "message": mid,
                     "file": str(ap_.relative_to(REPO)), "start_s": 0.0, "words": words})
        stats["messages"] += 1

    speakers = sorted({r["episode"] for r in rows})
    rng = random.Random(args.seed)
    val = set(rng.sample(speakers, max(1, int(len(speakers) * args.val_share))))
    n = write_jsonl(args.out, rows)
    # the same held-out word types as the corpus split, so val_words stays comparable;
    # type counts over the messages drive prepare's rarest-first order and quota
    corpus_split = json.loads((REPO / "data/xeus_ft/split.json").read_text(encoding="utf-8"))
    counts = collections.Counter(w["key"] for r in rows for w in r["words"] if w.get("certain"))
    split = {"val_episodes": sorted(val), "val_types": corpus_split["val_types"],
             "val_type_words": corpus_split["val_type_words"], "type_counts": dict(counts),
             "quota": 300, "speakers": len(speakers), "messages": n, "stats": dict(stats)}
    Path(args.out).with_name("whatsapp_split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"messages {n:,} ({stats['skipped_latin']:,} skipped for Latin script)  speakers {len(speakers)}  "
          f"held-out speakers {len(val)}  tokens {stats['tokens']:,}  certain {stats['certain_tokens']:,} "
          f"({stats['certain_tokens'] / max(1, stats['tokens']):.1%})")


if __name__ == "__main__":
    main()
