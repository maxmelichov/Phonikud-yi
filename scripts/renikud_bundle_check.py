#!/usr/bin/env python3
"""Does the shipped path (yiddish_labels + ONNX ReNikud-yi) read what the measured one did?

renikud_yi_eval.py measured lexicon-first graph-constrained ReNikud-yi at
94.5% agreement with the audio on the rule-path words of six held-out
episodes, with torch and the training-side decode. This runs the same
words through the production path — hebrew_to_ipa with the context reader
installed, the ONNX export, the torch-free decode in yiddish_renikud.py —
and reports the same number, plus agreement with the torch decode word by
word, for the export named by PHONIKUD_YI_RENIKUD_MODEL (fp32 or int8).

  PHONIKUD_YI_RENIKUD_MODEL=models/renikud_yi_audio/onnx_int8 python scripts/renikud_bundle_check.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="100313,104192,104690,113370,58622,94226")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--torch", action="store_true", help="also run the torch decode (renikud_yi_g2p) for word-level agreement")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import yiddish_labels  # noqa: F401  (installs the context reader)
    import yiddish_g2p as g2p
    import yiddish_renikud
    from xeus_ft_common import lexicon_key, tokenize_ipa
    print("context reader:", yiddish_labels.CONTEXT_READER, flush=True)
    csv.field_size_limit(10_000_000)
    episodes = set(args.episode.split(","))
    rows = [r for r in csv.DictReader(open(REPO / "data/corpus/yiddish_tts_dataset.tsv", encoding="utf-8"), delimiter="\t") if r["episode"] in episodes]
    if args.limit:
        rows = rows[: args.limit]
    gold = {v["key"] for v in json.loads((REPO / "data/xeus_ft/dictionary.json").read_text(encoding="utf-8")).values()}
    audio = {}
    for line in open(REPO / "data/xeus_ft/attest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["episode"] in episodes and r["margin"] >= args.margin:
            audio[(r["episode"], int(r["chunk_idx"]), r["wi"])] = r["chosen"]
    torch_g2p = None
    if args.torch:
        from renikud_yi_g2p import YiG2P
        torch_g2p = YiG2P()
    hits = collections.Counter()
    n = 0
    agree = disagree = 0
    t_prod = 0.0
    examples = []
    for r in rows:
        text = r["text"]
        ci = int(r["chunk_idx"])
        t0 = time.time()
        recs = g2p.g2p_tokens(text)
        t_prod += time.time() - t0
        g2p.set_context_reader(None)
        plain = g2p.g2p_tokens(text)
        g2p.set_context_reader(yiddish_renikud.context_reader)
        # records to Hebrew-token index: multiword records span several tokens
        prod, eng = {}, {}
        hi = 0
        for pr, er in zip(recs, plain):
            parts = len(str(pr.get("word") or "").split())
            if not _HEB.search(str(pr.get("word") or "")):
                continue
            for k in range(parts):
                prod[hi + k] = (pr, k == 0)
                eng[hi + k] = er
            hi += parts
        heb = _HEB.findall(text)
        if hi != len(heb):
            hits["skipped_chunks_misaligned"] += 1
            continue
        tr = torch_g2p.read(text) if torch_g2p else None
        for i, w in enumerate(heb):
            key = lexicon_key(w)
            ref = audio.get((r["episode"], ci, i))
            if key in gold or ref is None:
                continue
            pr, first = prod[i]
            if not first:
                continue
            n += 1
            p_ipa = tokenize_ipa(pr["ipa_primary"] or "")
            e_ipa = tokenize_ipa(eng[i]["ipa_primary"] or "")
            hits["production"] += p_ipa == ref
            hits["engine"] += e_ipa == ref
            if tr is not None and i < len(tr):
                t_ipa = tokenize_ipa(tr[i]["ipa"] or "")
                if t_ipa == p_ipa:
                    agree += 1
                else:
                    disagree += 1
                    if len(examples) < 12:
                        examples.append((w, " ".join(p_ipa), " ".join(t_ipa), " ".join(ref)))
    print(f"rule-path words with an audio decision (margin >= {args.margin:g}): {n}")
    print(f"  engine alone      {hits['engine'] / max(1, n):.3%}")
    print(f"  production path   {hits['production'] / max(1, n):.3%}")
    if torch_g2p:
        print(f"  agreement with the torch decode: {agree}/{agree + disagree} ({agree / max(1, agree + disagree):.2%})")
        for w, p, t, ref in examples:
            print(f"    {w:14} prod={p:22} torch={t:22} audio={ref}")
    print(f"  production g2p_tokens: {t_prod / max(1, len(rows)) * 1000:.0f} ms / chunk over {len(rows)} chunks; "
          f"skipped {hits['skipped_chunks_misaligned']}")


if __name__ == "__main__":
    main()
