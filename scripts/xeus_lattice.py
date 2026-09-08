#!/usr/bin/env python3
"""The lattice: turn the ear's free decoding into multiple choice.

Free CTC decoding asks the recognizer to produce the right phone string out of
everything it could say. Scoring the same frame posteriors against a short
list of *legal* readings and taking the best one asks it only to tell them
apart — a much easier task for the same model. Three nested candidate sets,
each smaller and better informed than the last:

  graph   what the spelling can legally sound like. Built from the engine's
          reading by branching every slot the orthography leaves open
          (spec §4: א a/ɔ/u/aː, פ f/p, יי aj/aː/ej, וי ɔj/oʊ, ו i/u), plus
          the two surface processes the citation form excludes (ɛ/ə
          reduction, final devoicing). Nothing outside it can be chosen.
  menu    what people were actually recorded saying for that word: for
          every training clip, the graph candidate the ear scores best,
          counted per word. Corpus evidence filtered through the graph, so
          nothing illegal enters and nothing unattested stays.
  gold    the word's native-verified readings (dictionary.json).

For each word in a clip the candidate with the best CTC likelihood is chosen,
left to right, with the other words held at their current choice.

  python scripts/xeus_lattice.py mine  --data data/xeus_ft --ckpt ... [--limit N]
  python scripts/xeus_lattice.py eval  --data data/xeus_ft --ckpt ... [--menu data/xeus_ft/menu.json]

`eval` reports word accuracy at every level — free, graph, menu, gold — on the
held-out clips, so the remaining errors can be located: an error that the
gold level fixes is the ear's; one the gold level does not fix is the label's.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import YI_BLANK, greedy_decode, lexicon_key, read_jsonl, yi_ids, yi_logits  # noqa: E402

# ----------------------------------------------------------------------------
# Level 1: the reading graph
# ----------------------------------------------------------------------------

#: Slots the orthography leaves open (spec §4), as classes of mutually
#: substitutable phones. A phone in a class may be replaced by any other
#: member. ɛ/ə is the unstressed-reduction process; the stop/fricative pairs
#: are final devoicing (only applied word-finally).
_OPEN_CLASSES: tuple[frozenset[str], ...] = (
    frozenset({"a", "ɔ", "u", "aː"}),   # א
    frozenset({"f", "p"}),              # פ
    frozenset({"aj", "aː", "ej"}),      # יי
    frozenset({"ɔj", "oʊ"}),            # וי
    frozenset({"i", "u"}),              # ו (shuruk)
    frozenset({"ɛ", "ə"}),              # reduction
)
_FINAL_DEVOICE = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s"}
MAX_GRAPH = 96


def graph_candidates(reading: list[str], max_candidates: int = MAX_GRAPH) -> list[list[str]]:
    """Every legal reading reachable from ``reading`` by the open slots."""
    slots: list[list[str]] = []
    n = len(reading)
    for i, p in enumerate(reading):
        alts = {p}
        for cls in _OPEN_CLASSES:
            if p in cls:
                alts |= cls
        if i == n - 1 and p in _FINAL_DEVOICE:
            alts.add(_FINAL_DEVOICE[p])
        slots.append(sorted(alts, key=lambda x: (x != p, x)))
    out: list[list[str]] = []
    for combo in itertools.product(*slots):
        out.append(list(combo))
        if len(out) >= max_candidates:
            break
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def ctc_nll(lp, ids: list[int]) -> float:
    import torch
    import torch.nn.functional as F
    if not ids or lp.shape[0] < len(ids):
        return float("inf")
    tgt = torch.tensor([ids], device=lp.device)
    return float(F.ctc_loss(lp.unsqueeze(1), tgt, torch.tensor([lp.shape[0]], device=lp.device),
                            torch.tensor([len(ids)], device=lp.device), blank=YI_BLANK,
                            reduction="sum", zero_infinity=True))


def choose(lp, cands: list[list[list[str]]], init: list[int] | None = None) -> tuple[list[int], float]:
    """Per word, the candidate index with the best CTC score, left to right."""
    choice = list(init) if init else [0] * len(cands)

    def ids_for(ch):
        out: list[int] = []
        for c, k in zip(cands, ch):
            out.extend(yi_ids(c[k]))
        return out

    for i, c in enumerate(cands):
        if len(c) < 2:
            continue
        best, best_k = None, choice[i]
        for k in range(len(c)):
            trial = list(choice)
            trial[i] = k
            s = ctc_nll(lp, ids_for(trial))
            if best is None or s < best:
                best, best_k = s, k
        choice[i] = best_k
    return choice, ctc_nll(lp, ids_for(choice))


# ----------------------------------------------------------------------------
# Level 2: mining the menu
# ----------------------------------------------------------------------------

def _log_probs(inner, head, ds, idx, device, use_amp):
    import torch
    from xeus_ft_train import collate
    speech, lens, _, _ = collate(ds, idx, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits, flens = yi_logits(inner, head, speech, lens)
    lp = torch.log_softmax(logits.float(), -1)
    return [lp[k, : int(flens[k])] for k in range(len(idx))], logits.float(), flens


def mine(args) -> None:
    import torch
    from xeus_ft_train import Segments
    from xeus_yi_decode import load_finetuned
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    dictionary = json.loads((data / "dictionary.json").read_text(encoding="utf-8"))
    by_key = {v["key"]: v for v in dictionary.values()}
    rows = [r for r in read_jsonl(data / "segments.jsonl") if r["split"] == "train"]
    if args.limit:
        random.Random(0).shuffle(rows)
        rows = rows[: args.limit]
    ds = Segments(rows, data / "seg")
    inner, head = load_finetuned(Path(args.ckpt), device)
    use_amp = device == "cuda"
    menu: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n = 0
    for idx in ds.batches(args.batch_seconds, shuffle=False, rng=random.Random(0)):
        lps, _, _ = _log_probs(inner, head, ds, idx, device, use_amp)
        for i, lp in zip(idx, lps):
            r = ds.rows[i]
            cands = []
            for w in r["words"]:
                gold = by_key.get(w["key"], {}).get("variants") or []
                if not gold:
                    continue
                seen: list[list[str]] = []
                for g in gold:
                    for c in graph_candidates(g):
                        if c not in seen:
                            seen.append(c)
                cands.append(seen[: args.max_candidates])
            if len(cands) != len(r["words"]):
                continue
            ch, _ = choose(lp, cands)
            for w, c, k in zip(r["words"], cands, ch):
                menu[w["key"]][" ".join(c[k])] += 1
            n += 1
        if n % 5000 < len(idx):
            print(f"  mined {n:,} clips, {len(menu):,} words", flush=True)
    out = Path(args.out or data / "menu.json")
    payload = {k: dict(v.most_common()) for k, v in menu.items()}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {out}: {len(payload)} words from {n:,} clips")


# ----------------------------------------------------------------------------
# Eval: word accuracy per level
# ----------------------------------------------------------------------------

def menu_candidates(menu_row: dict | None, gold: list[list[str]], min_share: float, min_count: int) -> list[list[str]]:
    """Readings attested for the word, pruned by share; gold readings always stay."""
    out: list[list[str]] = [list(g) for g in gold]
    if menu_row:
        total = sum(menu_row.values())
        for reading, count in menu_row.items():
            if count >= min_count and count / total >= min_share:
                phones = reading.split()
                if phones not in out:
                    out.append(phones)
    return out


def evaluate(args) -> None:
    import torch
    from xeus_ft_train import Segments
    from xeus_yi_decode import load_finetuned
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    dictionary = json.loads((data / "dictionary.json").read_text(encoding="utf-8"))
    by_key = {v["key"]: v for v in dictionary.values()}
    menu = json.loads(Path(args.menu).read_text(encoding="utf-8")) if args.menu and Path(args.menu).exists() else {}
    inner, head = load_finetuned(Path(args.ckpt), device)
    use_amp = device == "cuda"
    levels = ("free", "graph", "menu", "gold")
    report: dict = {"ckpt": args.ckpt, "menu": args.menu if menu else None, "splits": {}}

    for split in ("val_words", "val_eps"):
        rows = [r for r in read_jsonl(data / "segments.jsonl") if r["split"] == split]
        if args.limit:
            rows = rows[: args.limit]
        ds = Segments(rows, data / "seg")
        # word-level counts: exact = chosen reading == the clip's target reading;
        # in_gold = chosen reading is one of the word's verified readings
        acc = {lv: {"exact": 0, "in_gold": 0, "n": 0} for lv in levels}
        single = {lv: {"exact": 0, "n": 0} for lv in levels}
        err_where = collections.Counter()
        for idx in ds.batches(args.batch_seconds, shuffle=False, rng=random.Random(0)):
            lps, logits, flens = _log_probs(inner, head, ds, idx, device, use_amp)
            frees = greedy_decode(logits, flens)
            for i, lp, free in zip(idx, lps, frees):
                r = ds.rows[i]
                words = r["words"]
                # the clip's per-word target readings
                targets: list[list[str]] = []
                pos = 0
                for w in words:
                    g = by_key.get(w["key"], {}).get("variants") or [[]]
                    v = g[w["variant"]] if w["variant"] < len(g) else g[0]
                    targets.append(v)
                golds = [by_key.get(w["key"], {}).get("variants") or [t] for w, t in zip(words, targets)]
                cand_sets = {
                    "gold": golds,
                    "graph": [list({tuple(c) for g in gs for c in graph_candidates(g)}) for gs in golds],
                    "menu": [menu_candidates(menu.get(w["key"]), gs, args.min_share, args.min_count)
                             for w, gs in zip(words, golds)],
                }
                cand_sets["graph"] = [[list(c) for c in cs][: args.max_candidates] for cs in cand_sets["graph"]]
                chosen = {}
                for lv in ("graph", "menu", "gold"):
                    ch, _ = choose(lp, cand_sets[lv])
                    chosen[lv] = [cand_sets[lv][k][c] for k, c in enumerate(ch)]
                # free level is only scorable per word on single-word clips
                for k, (w, t, gs) in enumerate(zip(words, targets, golds)):
                    for lv in ("graph", "menu", "gold"):
                        acc[lv]["n"] += 1
                        acc[lv]["exact"] += int(chosen[lv][k] == t)
                        acc[lv]["in_gold"] += int(chosen[lv][k] in gs)
                    if len(words) == 1:
                        single["free"]["n"] += 1
                        single["free"]["exact"] += int(free == t)
                        for lv in ("graph", "menu", "gold"):
                            single[lv]["n"] += 1
                            single[lv]["exact"] += int(chosen[lv][k] == t)
                        if chosen["gold"][k] != t:
                            err_where["gold wrong (ear cannot tell the gold readings apart, or label)"] += 1
                        elif chosen["menu"][k] != t:
                            err_where["menu wrong, gold right (menu has a distractor)"] += 1
                        elif chosen["graph"][k] != t:
                            err_where["graph wrong, menu right (menu removed the distractor)"] += 1
                        elif free != t:
                            err_where["free wrong, graph right (pure generation error)"] += 1
                        else:
                            err_where["all right"] += 1
        if not acc["gold"]["n"]:
            print(f"\n{split}: no clips", flush=True)
            continue
        report["splits"][split] = {
            "all_words": {lv: {"n": a["n"], "exact": a["exact"] / a["n"] if a["n"] else None,
                               "in_gold": a["in_gold"] / a["n"] if a["n"] else None}
                          for lv, a in acc.items() if a["n"]},
            "single_word_clips": {lv: {"n": s["n"], "exact": s["exact"] / s["n"] if s["n"] else None}
                                  for lv, s in single.items() if s["n"]},
            "where_errors_live": dict(err_where),
        }
        print(f"\n{split}: single-word clips (n={single['free']['n']})", flush=True)
        for lv in levels:
            s = single[lv]
            if s["n"]:
                print(f"  {lv:6} word accuracy {s['exact'] / s['n']:.3f}")
        aw = report["splits"][split]["all_words"]
        print(f"{split}: all words (n={acc['gold']['n']})")
        for lv in ("graph", "menu", "gold"):
            a = aw[lv]
            print(f"  {lv:6} exact {a['exact']:.3f}  in-gold {a['in_gold']:.3f}")
        for k, v in err_where.most_common():
            print(f"    {v:5}  {k}")

    out = Path(args.out or Path(args.ckpt) / "lattice_eval.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mine", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=str(REPO / "data/xeus_ft"))
        p.add_argument("--ckpt", default=str(REPO / "data/xeus_ft/ckpt/best"))
        p.add_argument("--device", default=None)
        p.add_argument("--batch-seconds", type=float, default=96.0)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--max-candidates", type=int, default=MAX_GRAPH)
        p.add_argument("--out", default=None)
    sub.choices["eval"].add_argument("--menu", default=str(REPO / "data/xeus_ft/menu.json"))
    sub.choices["eval"].add_argument("--min-share", type=float, default=0.05,
                                     help="menu reading must have this share of the word's clips")
    sub.choices["eval"].add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()
    (mine if args.cmd == "mine" else evaluate)(args)


if __name__ == "__main__":
    main()
