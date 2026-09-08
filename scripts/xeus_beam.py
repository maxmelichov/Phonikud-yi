#!/usr/bin/env python3
"""Decode-side improvements for the Yiddish ear: blank penalty and a
dictionary-guided prefix beam search. No retraining.

Two failure modes of greedy CTC decoding on this model, and what each does:

  blank penalty   the model's word-final ə is usually present in the
                  posteriors but loses the per-frame argmax to <blank>;
                  subtracting a constant from the blank log-probability
                  before decoding lets a phone the model half-believes in
                  come through. Applies to greedy and beam alike.

  prefix beam     greedy commits phone by phone and, on a long clip, one bad
  with lexicon    frame derails everything after it. A CTC prefix beam search
                  keeps the best few prefixes; a trie of dictionary
                  pronunciations adds a bonus whenever a prefix completes a
                  known word and a per-phone penalty whenever it walks off the
                  trie, so paths that spell real words are preferred without
                  being forced. Word boundaries are free, so the decoder can
                  return "word word phones word".

  python scripts/xeus_beam.py --data data/xeus_ft --ckpt data/xeus_ft/ckpt/best --limit 600
      compares greedy / greedy+penalty / beam / beam+penalty on the held-out clips
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

from xeus_ft_common import (  # noqa: E402
    PerAccumulator, YI_BLANK, YI_VOCAB, YI2ID, read_jsonl, yi_logits,
)

NEG_INF = float("-inf")


# ----------------------------------------------------------------------------
# Trie over dictionary pronunciations
# ----------------------------------------------------------------------------

class Trie:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[int, "Trie"] = {}
        self.terminal = False

    def add(self, ids: list[int]) -> None:
        node = self
        for i in ids:
            node = node.children.setdefault(i, Trie())
        node.terminal = True


def build_trie(dictionary: dict) -> Trie:
    root = Trie()
    for meta in dictionary.values():
        for v in meta["variants"]:
            if v:
                root.add([YI2ID[p] for p in v])
    return root


# ----------------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------------

def greedy(lp, blank_penalty: float = 0.0) -> list[str]:
    """Argmax per frame after penalising blank, then collapse."""
    lp = lp.clone()
    lp[:, YI_BLANK] -= blank_penalty
    ids = lp.argmax(-1).tolist()
    out: list[str] = []
    prev = None
    for t in ids:
        if t != YI_BLANK and t != prev:
            out.append(YI_VOCAB[t])
        prev = t
    return out


def _lse(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def prefix_beam(lp, trie: Trie | None, beam: int = 8, blank_penalty: float = 0.0,
                word_bonus: float = 1.0, oov_penalty: float = 0.5, top_k: int = 8) -> list[str]:
    """CTC prefix beam search with an optional lexicon trie.

    Each hypothesis is a phone prefix with the usual (p_blank, p_nonblank)
    split and a lexicon state: the trie node the current (unfinished) word
    is at, or None when the prefix has walked off the trie. Completing a
    word (terminal node) earns ``word_bonus`` and resets to the root;
    extending outside the trie costs ``oov_penalty`` per phone. Bonuses are
    added to the hypothesis score used for ranking, not to the CTC
    probabilities, so the acoustic evidence is never rewritten.
    """
    import torch
    lp = lp.clone()
    lp[:, YI_BLANK] -= blank_penalty
    lp = torch.log_softmax(lp, -1)  # renormalise after the penalty
    T, C = lp.shape
    root = trie
    # prefix -> (log p_blank, log p_nonblank, lexicon node, lexical score)
    beams: dict[tuple[int, ...], list] = {(): [0.0, NEG_INF, (root,) if root is not None else None, 0.0]}
    for t in range(T):
        frame = lp[t]
        # only the top-k phones of the frame are worth extending with
        vals, idxs = frame.topk(min(top_k, C))
        cand = [(int(i), float(v)) for v, i in zip(vals, idxs)]
        blank_lp = float(frame[YI_BLANK])
        nxt: dict[tuple[int, ...], list] = {}

        def merge(prefix, pb, pnb, node, lex):
            cur = nxt.get(prefix)
            if cur is None:
                nxt[prefix] = [pb, pnb, node, lex]
            else:
                cur[0] = _lse(cur[0], pb)
                cur[1] = _lse(cur[1], pnb)
                if lex > cur[3]:
                    cur[2], cur[3] = node, lex

        for prefix, (pb, pnb, node, lex) in beams.items():
            total = _lse(pb, pnb)
            # stay: blank, or repeat of the last phone
            merge(prefix, total + blank_lp, NEG_INF, node, lex)
            if prefix:
                merge(prefix, NEG_INF, pnb + float(frame[prefix[-1]]), node, lex)
            for c, clp in cand:
                if c == YI_BLANK:
                    continue
                if prefix and c == prefix[-1]:
                    # new phone equal to the last one needs a blank in between
                    new_pnb = pb + clp
                else:
                    new_pnb = total + clp
                if new_pnb == NEG_INF:
                    continue
                # lexicon transition: the state is the set of trie nodes the
                # unfinished word could be at (root included after a word ends)
                new_node, new_lex = node, lex
                if root is not None:
                    nxt_nodes = [n.children[c] for n in node if c in n.children] if node else []
                    if not nxt_nodes and c in root.children:
                        nxt_nodes = [root.children[c]]      # a new word starts here
                    if nxt_nodes:
                        if any(n.terminal for n in nxt_nodes):
                            new_lex = lex + word_bonus
                            nxt_nodes = nxt_nodes + [root]  # may end here, or go on
                        new_node = tuple(nxt_nodes)
                    else:
                        new_node = ()
                        new_lex = lex - oov_penalty
                merge(prefix + (c,), NEG_INF, new_pnb, new_node, new_lex)
        # prune by acoustic + lexical score
        ranked = sorted(nxt.items(), key=lambda kv: -(_lse(kv[1][0], kv[1][1]) + kv[1][3]))
        beams = dict(ranked[:beam])
    best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][3])[0]
    return [YI_VOCAB[i] for i in best]


# ----------------------------------------------------------------------------
# Comparison on the held-out clips
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--ckpt", default=str(REPO / "data/xeus_ft/ckpt/best"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=600, help="clips per split")
    ap.add_argument("--penalties", default="0,1,2,3")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--word-bonus", type=float, default=1.0)
    ap.add_argument("--oov-penalty", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from xeus_ft_train import Segments, collate
    from xeus_yi_decode import load_finetuned
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    dictionary = json.loads(Path(args.dictionary).read_text(encoding="utf-8"))
    trie = build_trie(dictionary)
    inner, head = load_finetuned(Path(args.ckpt), device)
    penalties = [float(x) for x in args.penalties.split(",")]
    configs = [("greedy", 0.0, False)] + [(f"greedy+blank{p:g}", p, False) for p in penalties if p] \
        + [("beam", 0.0, True)] + [(f"beam+blank{p:g}", p, True) for p in penalties if p]
    report: dict = {"ckpt": args.ckpt, "splits": {}}
    for split in ("val_words", "val_eps"):
        rows = [r for r in read_jsonl(data / "segments.jsonl") if r["split"] == split]
        random.Random(0).shuffle(rows)
        rows = rows[: args.limit]
        ds = Segments(rows, data / "seg")
        accs = {name: PerAccumulator() for name, _, _ in configs}
        singles = {name: [0, 0] for name, _, _ in configs}
        with torch.no_grad():
            for idx in ds.batches(64, shuffle=False, rng=random.Random(0)):
                speech, lens, _, _ = collate(ds, idx, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits, flens = yi_logits(inner, head, speech, lens)
                lp_all = torch.log_softmax(logits.float(), -1)
                for k, i in enumerate(idx):
                    lp = lp_all[k, : int(flens[k])].cpu()
                    r = ds.rows[i]
                    for name, pen, use_beam in configs:
                        hyp = prefix_beam(lp, trie, args.beam, pen, args.word_bonus, args.oov_penalty) if use_beam else greedy(lp, pen)
                        accs[name].add(r["target"], hyp)
                        if len(r["words"]) == 1:
                            singles[name][1] += 1
                            singles[name][0] += int(hyp == r["target"])
        print(f"\n{split} (n={len(rows)} clips, {singles['greedy'][1]} single-word)")
        print(f"  {'decoder':18} {'PER':>6} {'exact':>6} {'1-word':>7} {'ə recall':>9}")
        out = {}
        for name, _, _ in configs:
            a = accs[name]
            s = singles[name]
            rec, n = a.phone_recall("ə")
            out[name] = {"per": a.per, "exact": a.exact / a.n, "single_exact": s[0] / s[1] if s[1] else None, "schwa_recall": rec}
            print(f"  {name:18} {a.per:6.3f} {a.exact / a.n:6.3f} {(s[0] / s[1] if s[1] else float('nan')):7.3f} {rec:9.3f}")
        report["splits"][split] = out
    outp = Path(args.out or Path(args.ckpt) / "decode_eval.json")
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
