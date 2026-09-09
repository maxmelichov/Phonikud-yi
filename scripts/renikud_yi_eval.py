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


def predict_words(model, tok, labels, text: str, device: str, want_logprobs: bool = False):
    """Phone string (stress-stripped) per Hebrew token of ``text``; optionally the
    per-character (consonant, vowel) log-probabilities for candidate scoring."""
    import torch
    enc = tok([text], return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        c, v, s = model(enc["input_ids"], enc["attention_mask"])
    lc, lv = torch.log_softmax(c[0].float(), -1).cpu(), torch.log_softmax(v[0].float(), -1).cpu()
    pc, pv = lc.argmax(-1).tolist(), lv.argmax(-1).tolist()
    per_char: dict[int, tuple[str, str]] = {}
    lp_char: dict[int, tuple] = {}
    for t, (a, z) in enumerate(offsets):
        if z - a == 1:
            per_char[a] = (labels["consonants"][pc[t]], labels["vowels"][pv[t]])
            lp_char[a] = (lc[t], lv[t])
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
    return (out, lp_char) if want_logprobs else out


def score_reading(word: str, phones: list[str], start: int, lp_char: dict, labels: dict) -> float:
    """Log-probability of a candidate reading under the model: align it to the
    letters, sum the per-letter (consonant, vowel) log-probs. -inf if it cannot
    be aligned to the spelling at all."""
    from yi_align import align_word, parse_chunk
    aligned = align_word(word, "".join(phones))
    if aligned is None:
        return float("-inf")
    letters = [(i, ch) for i, ch in enumerate(word, start=start) if not ("֑" <= ch <= "ׇ")]
    if len(letters) != len(aligned):
        return float("-inf")
    c2i = {c: i for i, c in enumerate(labels["consonants"])}
    v2i = {v: i for i, v in enumerate(labels["vowels"])}
    total = 0.0
    for (ti, _), (_, chunk) in zip(letters, aligned):
        if ti not in lp_char:
            continue
        cons, vowel, _ = parse_chunk(chunk) if chunk else ("", "", 0)
        lc, lv = lp_char[ti]
        total += float(lc[c2i.get(cons, 0)]) + float(lv[v2i.get(vowel, 0)])
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="ReNikud-yi run dirs holding heads.pt + best_encoder/")
    ap.add_argument("--phonikud", nargs="*", default=[],
                    help="phonikud-yi checkpoint dirs: the production path, text -> pointing model -> engine reads the pointed word")
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--attest", default=str(REPO / "data/xeus_ft/attest.jsonl"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--episode", default=TEST_EPISODE, help="comma-separated; e.g. the test episode plus the val episodes")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO / "data/eval/renikud_yi_eval.json"))
    args = ap.parse_args()

    import csv
    import torch
    from yiddish_g2p import g2p_token
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    csv.field_size_limit(10_000_000)
    episodes = set(args.episode.split(","))
    rows = [r for r in csv.DictReader(open(args.corpus, encoding="utf-8"), delimiter="\t") if r["episode"] in episodes]
    gold = {v["key"]: v["variants"] for v in json.loads(Path(args.dictionary).read_text(encoding="utf-8")).values()}
    audio: dict[tuple[int, int], dict] = {}
    for line in open(args.attest, encoding="utf-8"):
        r = json.loads(line)
        if r["episode"] in episodes and r["margin"] >= args.margin:
            audio[(r["episode"], int(r["chunk_idx"]), r["wi"])] = r

    models = {Path(m).name: load_model(Path(m), device) for m in args.models}
    pointers = {}
    if args.phonikud:
        from point_text import Pointer
        from yiddish_g2p import hebrew_to_ipa
        for d in args.phonikud:
            pointers[Path(d).parent.name if Path(d).name == "best" else Path(d).name] = Pointer(d, device="auto")

    def pointed_readings(text: str, pointer) -> list[str]:
        pointed = pointer.point([text])[0]
        out = []
        for m in _HEB.finditer(pointed):
            try:
                ipa = hebrew_to_ipa(m.group(0), stress=True, quarantine=False)
            except Exception:  # noqa: BLE001
                ipa = ""
            out.append(" ".join(tokenize_ipa(ipa)))
        return out
    engine_cache: dict[str, str] = {}
    # counters: bucket -> system -> [hits, n]
    acc: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    examples: list[dict] = []
    per_token: list[dict] = []
    disagreements: list[dict] = []
    for r in rows:
        text = r["text"]
        ci = int(r["chunk_idx"])
        preds = {}
        lps = {}
        for name, (m, t, l) in models.items():
            preds[name], lps[name] = predict_words(m, t, l, text, device, want_logprobs=True)
        for name, ptr in pointers.items():
            pr = pointed_readings(text, ptr)
            if len(pr) == len(_HEB.findall(text)):
                preds[name] = pr
        for hi, m in enumerate(_HEB.finditer(text)):
            w = m.group(0)
            key = lexicon_key(w)
            if w not in engine_cache:
                tkn = g2p_token(w); tkn = tkn if isinstance(tkn, dict) else tkn.__dict__
                engine_cache[w] = (" ".join(tokenize_ipa(tkn["ipa_primary"] or "")), tkn["route"] == "lexicon")
            eng, eng_lex = engine_cache[w]
            if key in gold:
                bucket, refs = "gold words (label agreement)", [" ".join(v) for v in gold[key]]
            elif (r["episode"], ci, hi) in audio:
                bucket, refs = f"rule-path words with audio decision (margin ≥ {args.margin:g})", [" ".join(audio[(r["episode"], ci, hi)]["chosen"])]
            else:
                continue
            systems = {"engine": eng, **{name: p[hi] for name, p in preds.items()}}
            # graph-constrained: the model only ranks the spelling's legal readings
            # (the engine's reading with its open slots branched), plus its own greedy guess
            for name, (mdl, tk, lab) in models.items():
                from xeus_lattice import graph_candidates
                cands = graph_candidates(eng.split(), max_candidates=64) if eng else []
                own = systems[name].split()
                if own and own not in cands:
                    cands.append(own)
                if cands:
                    best = max(cands, key=lambda c: score_reading(w, c, m.start(), lps[name], lab))
                    systems[name + "+graph"] = " ".join(best) if score_reading(w, best, m.start(), lps[name], lab) > float("-inf") else eng
                else:
                    systems[name + "+graph"] = systems[name]
            # lexicon-first: the table answers for gold and lexicon words, the model for the rest
            for name in list(models):
                systems[name + "+lexicon"] = eng if (key in gold or eng_lex) else systems[name]
                systems[name + "+graph+lexicon"] = eng if (key in gold or eng_lex) else systems[name + "+graph"]
            hits = {}
            for name, hyp in systems.items():
                acc[bucket][name][1] += 1
                acc[bucket][name][0] += int(hyp in refs)
                hits[name] = int(hyp in refs)
            per_token.append({"bucket": bucket[:4], "word": w, **hits})
            if bucket.startswith("rule"):
                disagreements.append({"word": w, "key": key, "audio": refs[0], **systems})
            if bucket.startswith("rule") and len(examples) < 40 and any(systems[n] != eng for n in preds):
                examples.append({"word": w, "audio": refs[0], **systems})
    report = {b: {n: {"n": v[1], "acc": round(100 * v[0] / max(1, v[1]), 2)} for n, v in d.items()} for b, d in acc.items()}
    report["examples"] = examples
    # paired sign tests on the rule-path bucket, every system against every other
    from math import comb
    names = [n for n in acc[next(b for b in acc if b.startswith("rule"))]]
    rule = [t for t in per_token if t["bucket"] == "rule"]
    paired = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fixed = sum(1 for t in rule if t[b] and not t[a]); broke = sum(1 for t in rule if t[a] and not t[b])
            k = fixed + broke
            pv = min(1.0, 2 * sum(comb(k, j) for j in range(0, min(fixed, broke) + 1)) / 2 ** k) if k else 1.0
            paired[f"{b} vs {a}"] = {"fixed": fixed, "broke": broke, "net": fixed - broke, "p": round(pv, 4)}
    report["paired_rule_path"] = paired
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    # Chezky's disagreement queue: rule-path words where the systems do not agree with the audio,
    # grouped by type, most frequent first — someone is wrong and it is cheap to find out who.
    q: dict[str, dict] = {}
    for d in disagreements:
        views = {k: v for k, v in d.items() if k not in ("word", "key", "audio")}
        if all(v == d["audio"] for v in views.values()):
            continue
        e = q.setdefault(d["key"], {"word": d["word"], "n": 0, "audio": collections.Counter(), **{k: collections.Counter() for k in views}})
        e["n"] += 1
        e["audio"][d["audio"]] += 1
        for k, v in views.items():
            e[k][v] += 1
    qp = Path(args.out).with_name(Path(args.out).stem + "_disagreements.tsv")
    with qp.open("w", encoding="utf-8") as fh:
        cols = ["word", "clips", "audio_says"] + [k for k in disagreements[0] if k not in ("word", "key", "audio")] if disagreements else ["word"]
        fh.write("\t".join(cols) + "\n")
        for key, e in sorted(q.items(), key=lambda kv: -kv[1]["n"]):
            fh.write("\t".join([e["word"], str(e["n"]), " | ".join(f"{r} ({n})" for r, n in e["audio"].most_common(2))]
                               + [" | ".join(f"{r} ({n})" for r, n in e[k].most_common(2)) for k in cols[3:]]) + "\n")
    print(f"disagreement queue: {len(q)} word types -> {qp}")
    for b, d in report.items():
        if b in ("examples", "paired_rule_path"):
            continue
        print(f"\n{b}")
        for n, v in d.items():
            print(f"  {n:22} n={v['n']:5}  word acc {v['acc']:6.2f}%")
    print("\npaired on rule-path words (b vs a: b right where a wrong / a right where b wrong):")
    for k, v in paired.items():
        print(f"  {k:40} fixed {v['fixed']:3} broke {v['broke']:3} net {v['net']:+4}  p={v['p']}")
    print("\nexamples where a model differs from the engine (audio decision first):")
    for e in examples[:20]:
        print("  ", e)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
