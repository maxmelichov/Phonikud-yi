#!/usr/bin/env python3
"""Score a ReNikud-yi model where it matters: on the words nobody labelled.

The trainer's word accuracy is agreement with the labels — gold and lexicon
readings the model has seen for those types — and it saturates near 100%.
The question this script answers is different: on the test episode's
RULE-PATH tokens, which no G2P label ever covered, does the model say what
the audio says?

For every test-episode token with an audio decision at margin >= --margin
(xeus_attest.py), three readings are compared with the ear's decision:

  engine      the frozen rule engine — production today
  model       the ReNikud-yi checkpoint(s) given
  (gold / lexicon tokens are reported too, as agreement with their label)

Word accuracy is exact match of the stress-stripped phone string. The
audio decision is the reference, so this is "agreement with the rabbi's
voice", the same yardstick the lattice used; where the ear is wrong the
number is wrong in the same direction for every system.

Usage:
  python scripts/renikud_yi_eval.py --models models/renikud_yi_noaudio models/renikud_yi_audio
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import lexicon_key, tokenize_ipa  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
TEST_EPISODE = "100313"


def load_model(run_dir: Path, device: str):
    import torch
    from transformers import AutoTokenizer, BertModel
    from renikud_yi_train import ReNikudYi
    heads = torch.load(run_dir / "heads.pt", map_location="cpu")
    labels = heads["labels"]
    enc = BertModel.from_pretrained(run_dir / "best_encoder", add_pooling_layer=False)
    tok = AutoTokenizer.from_pretrained(run_dir / "best_encoder")
    model = ReNikudYi(enc, len(labels["consonants"]), len(labels["vowels"]))
    model.load_state_dict({**{"encoder." + k: v for k, v in enc.state_dict().items()}, **heads["heads"]}, strict=False)
    return model.to(device).eval(), tok, labels


def predict_words(model, tok, labels, text: str, device: str) -> list[str]:
    """Phone string (stress-stripped) per Hebrew token of ``text``."""
    import torch
    enc = tok([text], return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        c, v, s = model(enc["input_ids"], enc["attention_mask"])
    pc, pv = c[0].argmax(-1).tolist(), v[0].argmax(-1).tolist()
    per_char: dict[int, tuple[str, str]] = {}
    for t, (a, z) in enumerate(offsets):
        if z - a == 1:
            per_char[a] = (labels["consonants"][pc[t]], labels["vowels"][pv[t]])
    out = []
    for m in _HEB.finditer(text):
        phones: list[str] = []
        for i in range(m.start(), m.end()):
            cc, vv = per_char.get(i, ("", ""))
            if cc:
                phones.append(cc)
            if vv:
                phones.append(vv)
        out.append(" ".join(phones))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="run dirs holding heads.pt + best_encoder/")
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--attest", default=str(REPO / "data/xeus_ft/attest.jsonl"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--episode", default=TEST_EPISODE)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO / "data/eval/renikud_yi_eval.json"))
    args = ap.parse_args()

    import csv
    import torch
    from yiddish_g2p import g2p_token
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    csv.field_size_limit(10_000_000)
    rows = [r for r in csv.DictReader(open(args.corpus, encoding="utf-8"), delimiter="\t") if r["episode"] == args.episode]
    gold = {v["key"]: v["variants"] for v in json.loads(Path(args.dictionary).read_text(encoding="utf-8")).values()}
    audio: dict[tuple[int, int], dict] = {}
    for line in open(args.attest, encoding="utf-8"):
        r = json.loads(line)
        if r["episode"] == args.episode and r["margin"] >= args.margin:
            audio[(int(r["chunk_idx"]), r["wi"])] = r

    models = {Path(m).name: load_model(Path(m), device) for m in args.models}
    engine_cache: dict[str, str] = {}
    # counters: bucket -> system -> [hits, n]
    acc: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    examples: list[dict] = []
    for r in rows:
        text = r["text"]
        ci = int(r["chunk_idx"])
        preds = {name: predict_words(m, t, l, text, device) for name, (m, t, l) in models.items()}
        for hi, m in enumerate(_HEB.finditer(text)):
            w = m.group(0)
            key = lexicon_key(w)
            if w not in engine_cache:
                tkn = g2p_token(w); tkn = tkn if isinstance(tkn, dict) else tkn.__dict__
                engine_cache[w] = " ".join(tokenize_ipa(tkn["ipa_primary"] or ""))
            eng = engine_cache[w]
            if key in gold:
                bucket, refs = "gold words (label agreement)", [" ".join(v) for v in gold[key]]
            elif (ci, hi) in audio:
                bucket, refs = f"rule-path words with audio decision (margin ≥ {args.margin:g})", [" ".join(audio[(ci, hi)]["chosen"])]
            else:
                continue
            systems = {"engine": eng, **{name: p[hi] for name, p in preds.items()}}
            for name, hyp in systems.items():
                acc[bucket][name][1] += 1
                acc[bucket][name][0] += int(hyp in refs)
            if bucket.startswith("rule") and len(examples) < 40 and any(systems[n] != eng for n in preds):
                examples.append({"word": w, "audio": refs[0], **systems})
    report = {b: {n: {"n": v[1], "acc": round(100 * v[0] / max(1, v[1]), 2)} for n, v in d.items()} for b, d in acc.items()}
    report["examples"] = examples
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for b, d in report.items():
        if b == "examples":
            continue
        print(f"\n{b}")
        for n, v in d.items():
            print(f"  {n:22} n={v['n']:5}  word acc {v['acc']:6.2f}%")
    print("\nexamples where a model differs from the engine (audio decision first):")
    for e in examples[:20]:
        print("  ", e)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
