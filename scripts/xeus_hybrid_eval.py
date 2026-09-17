#!/usr/bin/env python3
"""Decode-time hybrid of two ears: run 2 decides the וי slot, ckpt_att_ou the rest.

Round 3b (docs/xeus_finetune.md §27b) left two ears: run 2 (A), still the
better discriminator of oʊ vs ɔj, and ckpt_att_ou/best (B), better on every
other open slot. This measures the obvious combination without training
anything: score every graph candidate of every held-out clip under both
ears, cache the NLLs, and decode offline four ways.

  A       argmin NLL_A
  B       argmin NLL_B
  H1      B's argmin, then, among the candidates that differ from it only in
          וי slots (oʊ/ɔj), A's argmin            ("B first, A fixes וי")
  H2      within every וי-class A picks the member, then B picks among the
          class representatives                    ("A fixes וי, B ranks")
  S(λ)    argmin NLL_B(c) + λ·(NLL_A(c) − min_{c'∈class(c)} NLL_A(c'))
          — the soft version, λ ∈ {0.5, 1}

Candidates are the per-word graph readings (`xeus_lattice.graph_candidates`
around the clip's gold reading, ≤96 per word), each scored as the full clip
with the other words held at their gold reading — one coordinate step of
`xeus_lattice.choose` from the gold initialisation, so every ear and hybrid
sees the identical candidate list and the combination is a pure offline
re-ranking. Per-slot accuracy is counted only over *contested* slots, i.e.
positions where at least one scored candidate differs from gold.

  # stage 1, one ear resident at a time (MPS encoder, CPU ctc_loss)
  python scripts/xeus_hybrid_eval.py score --ear A --ckpt data/xeus_ft/ckpt/best
  python scripts/xeus_hybrid_eval.py score --ear B --ckpt data/xeus_ft/ear3/ckpt_att_ou/best
  # stage 2, offline
  python scripts/xeus_hybrid_eval.py combine
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import YI_BLANK, read_jsonl, yi_ids  # noqa: E402
from xeus_lattice import _OPEN_CLASSES, graph_candidates  # noqa: E402

VOY = frozenset({"ɔj", "oʊ"})
OPEN_PHONES = frozenset(p for cls in _OPEN_CLASSES for p in cls) | frozenset({"b", "d", "ɡ", "v", "z", "p", "t", "k", "f", "s"})
SLOT_GROUPS = {
    "oʊ": {"oʊ"}, "ɔj": {"ɔj"}, "ə": {"ə"}, "ɛ": {"ɛ"},
    "a/ɔ/u": {"a", "ɔ", "u", "aː"}, "i/u": {"i", "u"}, "aj/ej": {"aj", "ej", "aː"},
    "f/p": {"f", "p"},
}
EAR3 = REPO / "data/xeus_ft/ear3"


# ----------------------------------------------------------------------------
# clip selection and candidate lists (identical for every ear)
# ----------------------------------------------------------------------------

def select_clips(data: Path, per_split: int, seed: int, splits=("val_words", "val_eps")) -> list[dict]:
    rows = [r for r in read_jsonl(data / "segments.jsonl") if r["split"] in splits]
    out = []
    for s in splits:
        rs = [r for r in rows if r["split"] == s]
        rng = random.Random(seed)
        idx = list(range(len(rs)))
        rng.shuffle(idx)
        keep = set(idx[:per_split]) | {i for i, r in enumerate(rs) if "oʊ" in r["target"]}
        out += [rs[i] for i in sorted(keep)]
    return out


def word_targets(row: dict, by_key: dict) -> list[list[str]] | None:
    """The clip's per-word gold readings; None if a word is not in the dictionary
    or the concatenation is not the clip target."""
    ts = []
    for w in row["words"]:
        g = by_key.get(w["key"], {}).get("variants")
        if not g:
            return None
        ts.append(list(g[w["variant"]] if w["variant"] < len(g) else g[0]))
    if [p for t in ts for p in t] != row["target"]:
        return None
    return ts


def candidates(row: dict, by_key: dict, max_candidates: int) -> list[list[list[str]]] | None:
    ts = word_targets(row, by_key)
    if ts is None:
        return None
    return [graph_candidates(t, max_candidates) for t in ts]   # cand 0 of every word == gold


# ----------------------------------------------------------------------------
# stage 1: score
# ----------------------------------------------------------------------------

def nll_batch(lp, seqs: list[list[int]]) -> list[float]:
    """CTC NLL of every target sequence against one clip's log-probs (CPU)."""
    import torch
    import torch.nn.functional as F
    T = lp.shape[0]
    L = max(len(x) for x in seqs)
    tgt = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, x in enumerate(seqs):
        tgt[i, : len(x)] = torch.tensor(x)
    tl = torch.tensor([len(x) for x in seqs])
    lpe = lp.unsqueeze(1).expand(T, len(seqs), lp.shape[1])
    # zero_infinity=False: an impossible candidate must score +inf, not 0
    out = F.ctc_loss(lpe, tgt, torch.full((len(seqs),), T), tl, blank=YI_BLANK,
                     reduction="none", zero_infinity=False)
    return [float(x) if math.isfinite(float(x)) else float("inf") for x in out]


def score(args) -> None:
    import torch
    from xeus_ft_common import yi_logits
    from xeus_ft_train import Segments, collate
    from xeus_yi_decode import load_finetuned
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    data = Path(args.data)
    dictionary = json.loads((REPO / "data/xeus_ft/dictionary.json").read_text(encoding="utf-8"))
    by_key = {v["key"]: v for v in dictionary.values()}
    rows = select_clips(data, args.per_split, args.seed)
    cands = {}
    kept = []
    for r in rows:
        c = candidates(r, by_key, args.max_candidates)
        if c is not None:
            cands[r["id"]] = c
            kept.append(r)
    print(f"ear {args.ear}: {len(kept)} clips ({len(rows) - len(kept)} dropped: word not in dictionary / "
          f"target mismatch), device {device}", flush=True)
    ds = Segments(kept, data / "seg")
    inner, head = load_finetuned(Path(args.ckpt), device)
    out = Path(args.out or EAR3 / f"hybrid_nll_{args.ear}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh, torch.no_grad():
        for idx in ds.batches(args.seconds, shuffle=False, rng=random.Random(0)):
            speech, lens, _, _ = collate(ds, idx, device)
            logits, flens = yi_logits(inner, head, speech, lens)
            lp = torch.log_softmax(logits.float(), -1).cpu()   # ctc_loss has no MPS kernel
            for j, i in enumerate(idx):
                r = kept[i]
                wc = cands[r["id"]]
                gold = [c[0] for c in wc]
                seqs, keys = [], []
                for wi, cs in enumerate(wc):
                    for ci, c in enumerate(cs):
                        if ci == 0 and wi > 0:
                            continue                  # gold scored once
                        full = [p for k, t in enumerate(gold) for p in (c if k == wi else t)]
                        seqs.append(yi_ids(full))
                        keys.append((wi, ci))
                vals = nll_batch(lp[j, : int(flens[j])], seqs)
                nll = {k: v for k, v in zip(keys, vals)}
                g = nll[(0, 0)]
                rec = {"id": r["id"], "split": r["split"], "target": r["target"], "frames": int(flens[j]),
                       "words": [{"key": w["key"], "gold": c[0],
                                  "cands": [[c[ci], (g if ci == 0 else nll[(wi, ci)])] for ci in range(len(c))]}
                                 for wi, (w, c) in enumerate(zip(r["words"], wc))]}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            if n % 200 < len(idx):
                print(f"  {n}/{len(kept)} clips", flush=True)
    print(f"wrote {out}: {n} clips")


# ----------------------------------------------------------------------------
# stage 2: combine
# ----------------------------------------------------------------------------

def voy_class(phones: list[str]) -> tuple:
    return tuple("וי" if p in VOY else p for p in phones)


def decode_word(cands: list[list[str]], a: list[float], b: list[float], lam_soft=(0.5, 1.0)) -> dict[str, int]:
    """Chosen candidate index per decoder for one word."""
    n = len(cands)
    idx = range(n)
    res = {"A": min(idx, key=lambda k: a[k]), "B": min(idx, key=lambda k: b[k])}
    classes: dict[tuple, list[int]] = collections.defaultdict(list)
    for k in idx:
        classes[voy_class(cands[k])].append(k)
    # H1: B's argmin, then A within its class
    kb = res["B"]
    res["H1"] = min(classes[voy_class(cands[kb])], key=lambda k: a[k])
    # H2: A within every class, B among representatives
    reps = [min(m, key=lambda k: a[k]) for m in classes.values()]
    res["H2"] = min(reps, key=lambda k: b[k])
    # soft
    cmin = {c: min(a[k] for k in m) for c, m in classes.items()}
    for lam in lam_soft:
        def s(k):
            dev = a[k] - cmin[voy_class(cands[k])]
            if not math.isfinite(dev):
                dev = 0.0 if not math.isfinite(a[k]) and not math.isfinite(cmin[voy_class(cands[k])]) else dev
            return b[k] + lam * dev
        res[f"S{lam:g}"] = min(idx, key=s)
    return res


def sign_test(a_only: int, b_only: int) -> float:
    from math import comb
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def combine(args) -> None:
    A = {r["id"]: r for r in read_jsonl(args.nll_a)}
    B = {r["id"]: r for r in read_jsonl(args.nll_b)}
    ids = [i for i in A if i in B]
    print(f"A {len(A)} clips, B {len(B)} clips, common {len(ids)}")
    decoders = ["A", "B", "H1", "H2", "S0.5", "S1"]
    report = {"nll_a": str(args.nll_a), "nll_b": str(args.nll_b), "decoders": {
        "A": "run 2 argmin", "B": "ckpt_att_ou argmin",
        "H1": "B argmin, then A argmin within its וי class", "H2": "A argmin within each וי class, B among the reps",
        "S0.5": "NLL_B + 0.5·(NLL_A − class-min NLL_A)", "S1": "NLL_B + 1.0·(NLL_A − class-min NLL_A)"},
        "note": "candidates = per-word graph readings with the other words held at gold; "
                "slot accuracy counted over contested slots only (some candidate differs there)",
        "splits": {}}
    per_split_rows: dict[str, list] = collections.defaultdict(list)
    for i in ids:
        ra, rb = A[i], B[i]
        assert [w["cands"][0][0] for w in ra["words"]] == [w["cands"][0][0] for w in rb["words"]], i
        clip = {"id": i, "split": ra["split"], "words": []}
        for wa, wb in zip(ra["words"], rb["words"]):
            cands = [c[0] for c in wa["cands"]]
            assert cands == [c[0] for c in wb["cands"]], i
            a = [c[1] for c in wa["cands"]]
            b = [c[1] for c in wb["cands"]]
            ch = decode_word(cands, a, b)
            gold = cands[0]
            contested = [pos for pos in range(len(gold)) if any(c[pos] != gold[pos] for c in cands)]
            has_voy = any(gold[pos] in VOY or any(c[pos] in VOY for c in cands) for pos in contested)
            clip["words"].append({"gold": gold, "chosen": {d: cands[k] for d, k in ch.items()},
                                  "contested": contested, "has_voy": has_voy, "ncand": len(cands)})
        per_split_rows[ra["split"]].append(clip)

    for split, clips in per_split_rows.items():
        rec: dict = {"n_clips": len(clips), "n_words": sum(len(c["words"]) for c in clips),
                     "n_words_with_voy_slot": sum(w["has_voy"] for c in clips for w in c["words"]),
                     "n_clips_with_ou": sum(any("oʊ" in w["gold"] for w in c["words"]) for c in clips),
                     "mean_candidates_per_word": sum(w["ncand"] for c in clips for w in c["words"]) / max(1, sum(len(c["words"]) for c in clips)),
                     "clip_exact": {}, "word_exact": {}, "slots": {}, "sign_tests": {}}
        clip_ok = {d: [all(w["chosen"][d] == w["gold"] for w in c["words"]) for c in clips] for d in decoders}
        word_ok = {d: [w["chosen"][d] == w["gold"] for c in clips for w in c["words"]] for d in decoders}
        for d in decoders:
            rec["clip_exact"][d] = round(sum(clip_ok[d]) / len(clips), 4)
            rec["word_exact"][d] = round(sum(word_ok[d]) / len(word_ok[d]), 4)
        # slots
        slot_hits: dict[str, dict[str, list[bool]]] = {g: {d: [] for d in decoders} for g in SLOT_GROUPS}
        slot_hits_phone: dict[str, dict[str, list[bool]]] = collections.defaultdict(lambda: {d: [] for d in decoders})
        for c in clips:
            for w in c["words"]:
                for pos in w["contested"]:
                    gp = w["gold"][pos]
                    for d in decoders:
                        ok = w["chosen"][d][pos] == gp
                        slot_hits_phone[gp][d].append(ok)
                        for g, members in SLOT_GROUPS.items():
                            if gp in members:
                                slot_hits[g][d].append(ok)
        def slot_summary(h):
            n = len(h["A"])
            return {"n": n, **{d: (round(sum(h[d]) / n, 4) if n else None) for d in decoders}}
        rec["slots"] = {g: slot_summary(h) for g, h in slot_hits.items()}
        rec["slots_by_gold_phone"] = {p: slot_summary(h) for p, h in sorted(slot_hits_phone.items(), key=lambda kv: -len(kv[1]["A"]))}
        # paired sign tests
        def paired(x, y):
            xo = sum(p and not q for p, q in zip(x, y))
            yo = sum(q and not p for p, q in zip(x, y))
            return {"first_only": xo, "second_only": yo, "p": sign_test(xo, yo)}
        for h in ("H1", "H2", "S0.5", "S1"):
            for base in ("A", "B"):
                rec["sign_tests"][f"clip_exact {base} vs {h}"] = paired(clip_ok[base], clip_ok[h])
                rec["sign_tests"][f"word_exact {base} vs {h}"] = paired(word_ok[base], word_ok[h])
                for g in ("oʊ", "ɔj"):
                    rec["sign_tests"][f"slot {g} {base} vs {h}"] = paired(slot_hits[g][base], slot_hits[g][h])
        rec["sign_tests"]["clip_exact A vs B"] = paired(clip_ok["A"], clip_ok["B"])
        rec["sign_tests"]["word_exact A vs B"] = paired(word_ok["A"], word_ok["B"])
        for g in ("oʊ", "ɔj", "ə"):
            rec["sign_tests"][f"slot {g} A vs B"] = paired(slot_hits[g]["A"], slot_hits[g]["B"])
        report["splits"][split] = rec
        print(f"\n{split}: clips {rec['n_clips']}  words {rec['n_words']}  words with וי slot {rec['n_words_with_voy_slot']}  clips with oʊ {rec['n_clips_with_ou']}")
        print("  " + " ".join(f"{d:>7}" for d in decoders))
        print("  clip exact " + " ".join(f"{rec['clip_exact'][d]:7.3f}" for d in decoders))
        print("  word exact " + " ".join(f"{rec['word_exact'][d]:7.3f}" for d in decoders))
        for g, s in rec["slots"].items():
            if s["n"]:
                print(f"  slot {g:6} n={s['n']:5} " + " ".join(f"{s[d]:7.3f}" for d in decoders))
        for k, v in rec["sign_tests"].items():
            print(f"  {k:28} {v['first_only']:4} / {v['second_only']:4}  p={v['p']:.3g}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for split, clips in per_split_rows.items():
                for c in clips:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--data", default=str(REPO / "data/xeus_ft/run3"))
    s.add_argument("--ear", required=True, help="tag: A or B")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--per-split", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-candidates", type=int, default=96)
    s.add_argument("--seconds", type=float, default=40.0)
    s.add_argument("--device", default=None)
    s.add_argument("--out", default=None)
    c = sub.add_parser("combine")
    c.add_argument("--nll-a", default=str(EAR3 / "hybrid_nll_A.jsonl"))
    c.add_argument("--nll-b", default=str(EAR3 / "hybrid_nll_B.jsonl"))
    c.add_argument("--out", default=str(EAR3 / "hybrid_eval.json"))
    c.add_argument("--dump", default=None, help="optional per-clip decisions jsonl")
    args = ap.parse_args()
    (score if args.cmd == "score" else combine)(args)


if __name__ == "__main__":
    main()
