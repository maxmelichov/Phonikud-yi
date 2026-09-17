#!/usr/bin/env python3
"""ReNikud-yi v2 yardstick: rule-path words of the held-out episodes against the
ear's decision, segments AND stress, paired across models.

Engine-free version of renikud_yi_eval.py (the engine's reading of every
word is read from attest_targets.jsonl, so this runs on a pod without the
engine) that adds a stress metric and accepts either attestation format as
the reference:

  new   xeus-yi-ipa/data/attest_lattice.jsonl rows with chosen_ipa (stress)
  old   data/xeus_ft/attest.jsonl rows with chosen only (stressless; stress
        metrics are then skipped)

Buckets (per --episodes, decisions at margin >= --margin):
  rule   route == rule and not a dictionary word: what the systems are for
  gold   dictionary words: agreement with the ear's variant (new) / any
         dictionary variant (old)

Systems: engine (frozen rules), and per model M: M (free per-letter argmax),
M+graph (ranks the spelling graph's legal readings by the summed per-letter
log-probs, stress from the model's stress head along the winner's
alignment — the decode that ships), M+graph+engine-stress (production v1
convention: the engine's stress ordinal put on the winner).

Metrics on the rule bucket: seg_acc (stress-stripped exact match),
stress_acc (on polysyllables, >= 2 full vowels, with a marked reference:
stressed-vowel ordinal == the ear's, irrespective of the segments),
stress_given_seg (same, among words whose segments are right), full_acc
(segments and stress). Paired sign tests between every pair of systems on
seg (all rule words) and on stress (the polysyllables).

Model dirs: either heads.pt + best_encoder/ (models/renikud_yi_audio) or the
trainer's best/heads.pt + best/encoder/ (a run dir or its best/).

Usage:
  python scripts/renikud_yi_eval_v2.py --ref ../xeus-yi-ipa/data/attest_lattice.jsonl \\
      --models models/renikud_yi_audio models/renikud_yi_v2 --out data/eval/renikud_v2_vs_newear.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from renikud_yi_prepare_v2 import HELDOUT, VOWEL_SET, fnum, n_full_vowels, place_stress, stress_ordinal  # noqa: E402
from xeus_ft_common import lexicon_key  # noqa: E402
from xeus_lattice import graph_candidates  # noqa: E402
from yi_align import align_word, parse_chunk  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")


def resolve_model_dir(d: Path) -> tuple[Path, Path]:
    """(heads.pt, encoder dir) for either layout."""
    for heads, enc in ((d / "heads.pt", d / "best_encoder"), (d / "best" / "heads.pt", d / "best" / "encoder"),
                       (d / "heads.pt", d / "encoder")):
        if heads.exists() and enc.exists():
            return heads, enc
    raise SystemExit(f"no ReNikud-yi checkpoint under {d}")


def load_model(run_dir: Path, device: str):
    import torch
    from transformers import AutoTokenizer, BertModel
    from renikud_yi_train import ReNikudYi
    heads_p, enc_p = resolve_model_dir(run_dir)
    heads = torch.load(heads_p, map_location="cpu")
    labels = heads["labels"]
    enc = BertModel.from_pretrained(enc_p, add_pooling_layer=False)
    tok = AutoTokenizer.from_pretrained(enc_p)
    model = ReNikudYi(enc, len(labels["consonants"]), len(labels["vowels"]))
    model.load_state_dict({**{"encoder." + k: v for k, v in enc.state_dict().items()}, **heads["heads"]}, strict=False)
    return model.to(device).eval(), tok, labels


def logprobs(model, tok, text: str, device: str):
    """Per character: (log p cons, log p vowel, log p stress=1)."""
    import torch
    enc = tok([text], return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        c, v, s = model(enc["input_ids"], enc["attention_mask"])
    lc, lv = torch.log_softmax(c[0].float(), -1).cpu(), torch.log_softmax(v[0].float(), -1).cpu()
    ls = torch.log_softmax(s[0].float(), -1)[:, 1].cpu()
    out = {}
    for t, (a, z) in enumerate(offsets):
        if z - a == 1:
            out[a] = (lc[t], lv[t], float(ls[t]))
    return out


def letters_of(word: str, start: int) -> list[int]:
    return [i for i, ch in enumerate(word, start=start) if not ("֑" <= ch <= "ׇ")]


def free_reading(word: str, start: int, lp: dict, labels: dict) -> tuple[list[str], int | None]:
    """Greedy per-letter reading and the stress ordinal the stress head puts on it."""
    phones: list[str] = []
    vowel_ord: list[tuple[int, float]] = []   # (ordinal among vowels, log p stress) for full vowels
    n_v = 0
    for i in letters_of(word, start):
        if i not in lp:
            continue
        lc, lv, ls = lp[i]
        cons, vow = labels["consonants"][int(lc.argmax())], labels["vowels"][int(lv.argmax())]
        if cons:
            phones.append(cons)
        if vow:
            phones.append(vow)
            if vow != "ə":
                vowel_ord.append((n_v, ls))
            n_v += 1
    return phones, (max(vowel_ord, key=lambda x: x[1])[0] if len(vowel_ord) >= 2 else None)


def score_candidate(word: str, phones: list[str], start: int, lp: dict, labels: dict):
    """(log-prob of the stressless candidate, stress ordinal by the stress head) or None if unalignable."""
    aligned = align_word(word, "".join(phones))
    if aligned is None:
        return None
    letters = letters_of(word, start)
    if len(letters) != len(aligned):
        return None
    c2i = {c: i for i, c in enumerate(labels["consonants"])}
    v2i = {v: i for i, v in enumerate(labels["vowels"])}
    total = 0.0
    n_v = 0
    vowel_ord: list[tuple[int, float]] = []
    for ti, (_, chunk) in zip(letters, aligned):
        cons, vowel, _ = parse_chunk(chunk) if chunk else ("", "", 0)
        if ti in lp:
            lc, lv, ls = lp[ti]
            total += float(lc[c2i.get(cons, 0)]) + float(lv[v2i.get(vowel, 0)])
        else:
            ls = float("-inf")
        if vowel:
            if vowel != "ə":
                vowel_ord.append((n_v, ls))
            n_v += 1
    k = max(vowel_ord, key=lambda x: x[1])[0] if len(vowel_ord) >= 2 else None
    return total, k


def sign_test(fixed: int, broke: int) -> float:
    k = fixed + broke
    if not k:
        return 1.0
    return min(1.0, 2 * sum(comb(k, j) for j in range(0, min(fixed, broke) + 1)) / 2 ** k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="ReNikud-yi checkpoint dirs (either layout)")
    ap.add_argument("--names", nargs="*", default=None, help="display names, one per model (default: dir names)")
    ap.add_argument("--ref", required=True, help="attestation file: new attest_lattice.jsonl or old attest.jsonl")
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--targets", default=str(REPO / "data/xeus_ft/attest_targets.jsonl"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--episodes", default=HELDOUT, help="comma-separated")
    ap.add_argument("--max-candidates", type=int, default=64)
    ap.add_argument("--limit-chunks", type=int, default=None, help="first N chunks only (smoke)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO / "data/eval/renikud_yi_eval_v2.json"))
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    csv.field_size_limit(10_000_000)
    episodes = set(args.episodes.split(","))
    rows = [r for r in csv.DictReader(open(args.corpus, encoding="utf-8"), delimiter="\t") if r["episode"] in episodes]
    if args.limit_chunks:
        rows = rows[:args.limit_chunks]
    gold = {v["key"]: [list(x) for x in v["variants"]] for v in json.loads(Path(args.dictionary).read_text(encoding="utf-8")).values()}
    targets: dict[tuple[str, int], list[dict]] = {}
    for line in open(args.targets, encoding="utf-8"):
        t = json.loads(line)
        if t["episode"] in episodes:
            targets[(t["episode"], int(t["chunk_idx"]))] = t["words"]
    ref: dict[tuple[str, int, int], dict] = {}
    new_format = None
    for line in open(args.ref, encoding="utf-8"):
        r = json.loads(line)
        if r["episode"] not in episodes:
            continue
        if new_format is None:
            new_format = "chosen_ipa" in r
        if fnum(r.get("margin")) >= args.margin:
            ref[(r["episode"], int(r["chunk_idx"]), int(r["wi"]))] = r
    print(f"reference {args.ref}: {'new (stressed)' if new_format else 'old (stressless)'} format, "
          f"{len(ref):,} decisions at margin >= {args.margin:g} on {len(rows):,} chunks of {sorted(episodes)}", flush=True)

    names = args.names or [Path(m).name if Path(m).name != "best" else Path(m).parent.name for m in args.models]
    models = {n: load_model(Path(m), device) for n, m in zip(names, args.models)}

    seg_hits: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    per_token: list[dict] = []
    examples: list[dict] = []
    for r in rows:
        text, ep, ci = r["text"], r["episode"], int(r["chunk_idx"])
        words = targets.get((ep, ci))
        matches = list(_HEB.finditer(text))
        if words is None or len(words) != len(matches):
            continue
        lps = {n: logprobs(m, t, text, device) for n, (m, t, _) in models.items()}
        for hi, (m, e) in enumerate(zip(matches, words)):
            w = m.group(0)
            key = lexicon_key(w)
            rec = ref.get((ep, ci, hi))
            eng = list(e["ph"])
            eng_k = stress_ordinal(e["ipa"]) if e["ipa"] else None
            if key in gold:
                bucket = "gold"
                refs_seg = [list(rec["chosen"])] if (rec is not None and new_format) else gold[key]
                ref_k = None
            elif rec is not None and e["route"] == "rule" and rec["key"] == key:
                bucket = "rule"
                refs_seg = [list(rec["chosen"])]
                ref_k = stress_ordinal(rec["chosen_ipa"]) if new_format else None
            else:
                continue
            systems: dict[str, tuple[list[str], int | None]] = {"engine": (eng, eng_k)}
            for n, (_, _, lab) in models.items():
                lp = lps[n]
                fr, fk = free_reading(w, m.start(), lp, lab)
                systems[n] = (fr, fk)
                cands = graph_candidates(eng, max_candidates=args.max_candidates) if eng else []
                if fr and fr not in cands:
                    cands.append(fr)
                best = None
                for c in cands:
                    sc = score_candidate(w, c, m.start(), lp, lab)
                    if sc is not None and (best is None or sc[0] > best[0]):
                        best = (sc[0], c, sc[1])
                if best is None:
                    systems[n + "+graph"] = (eng, eng_k)
                    systems[n + "+graph+engine-stress"] = (eng, eng_k)
                else:
                    systems[n + "+graph"] = (best[1], best[2])
                    systems[n + "+graph+engine-stress"] = (best[1], eng_k)
            tok = {"bucket": bucket, "word": w, "key": key, "ref": " ".join(refs_seg[0]), "ref_k": ref_k,
                   "poly": bool(bucket == "rule" and ref_k is not None and n_full_vowels(refs_seg[0]) >= 2)}
            for n, (ph, k) in systems.items():
                seg_ok = int(ph in refs_seg)
                seg_hits[bucket][n][1] += 1
                seg_hits[bucket][n][0] += seg_ok
                tok[n] = {"seg": seg_ok, "stress": int(k == ref_k) if tok["poly"] else None, "hyp": place_stress(ph, k)}
            per_token.append(tok)
            if bucket == "rule" and len(examples) < 40 and any(systems[n][0] != eng for n in models):
                examples.append({"word": w, "ear": place_stress(refs_seg[0], ref_k), **{n: tok[n]["hyp"] for n in systems}})

    sys_names = list(next(iter(seg_hits.values())).keys()) if seg_hits else []
    rule = [t for t in per_token if t["bucket"] == "rule"]
    poly = [t for t in rule if t["poly"]]
    report: dict = {"ref": args.ref, "ref_format": "new" if new_format else "old", "margin": args.margin,
                    "episodes": sorted(episodes), "n_rule": len(rule), "n_poly": len(poly), "systems": {}}
    for n in sys_names:
        d = {"rule_n": len(rule), "seg_acc": round(100 * sum(t[n]["seg"] for t in rule) / max(1, len(rule)), 2)}
        if "gold" in seg_hits:
            g = seg_hits["gold"][n]
            d["gold_n"], d["gold_acc"] = g[1], round(100 * g[0] / max(1, g[1]), 2)
        if poly:
            seg_right = [t for t in poly if t[n]["seg"]]
            d["poly_n"] = len(poly)
            d["stress_acc"] = round(100 * sum(t[n]["stress"] for t in poly) / len(poly), 2)
            d["stress_given_seg"] = round(100 * sum(t[n]["stress"] for t in seg_right) / max(1, len(seg_right)), 2)
            d["full_acc"] = round(100 * sum(t[n]["stress"] and t[n]["seg"] for t in poly) / len(poly), 2)
        report["systems"][n] = d
    paired: dict[str, dict] = {}
    for i, a in enumerate(sys_names):
        for b in sys_names[i + 1:]:
            fixed = sum(1 for t in rule if t[b]["seg"] and not t[a]["seg"])
            broke = sum(1 for t in rule if t[a]["seg"] and not t[b]["seg"])
            entry = {"seg": {"fixed": fixed, "broke": broke, "net": fixed - broke, "p": round(sign_test(fixed, broke), 6)}}
            if poly:
                f2 = sum(1 for t in poly if t[b]["stress"] and not t[a]["stress"])
                b2 = sum(1 for t in poly if t[a]["stress"] and not t[b]["stress"])
                entry["stress"] = {"fixed": f2, "broke": b2, "net": f2 - b2, "p": round(sign_test(f2, b2), 6)}
            paired[f"{b} vs {a}"] = entry
    report["paired_rule_path"] = paired
    report["examples"] = examples
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    tp = Path(args.out).with_name(Path(args.out).stem + "_tokens.jsonl")
    with tp.open("w", encoding="utf-8") as fh:
        for t in per_token:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nrule-path words with a decision: n = {len(rule):,}; polysyllables with a marked reference: {len(poly):,}")
    hdr = f"  {'system':40} {'seg_acc':>8} {'gold':>7} {'stress':>7} {'st|seg':>7} {'full':>7}"
    print(hdr)
    for n, d in report["systems"].items():
        print(f"  {n:40} {d['seg_acc']:8.2f} {d.get('gold_acc', float('nan')):7.2f} {d.get('stress_acc', float('nan')):7.2f} "
              f"{d.get('stress_given_seg', float('nan')):7.2f} {d.get('full_acc', float('nan')):7.2f}")
    print("\npaired sign tests on the rule-path words (b vs a: b right where a wrong / a right where b wrong):")
    for k, v in paired.items():
        s = v["seg"]
        line = f"  {k:60} seg fixed {s['fixed']:4} broke {s['broke']:4} net {s['net']:+5} p={s['p']:.2g}"
        if "stress" in v:
            st = v["stress"]
            line += f" | stress fixed {st['fixed']:4} broke {st['broke']:4} net {st['net']:+5} p={st['p']:.2g}"
        print(line)
    print("\nexamples (ear first):")
    for e in examples[:15]:
        print("  ", e)
    print(f"\nwrote {args.out} and {tp}")


if __name__ == "__main__":
    main()
