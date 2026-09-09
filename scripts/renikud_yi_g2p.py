#!/usr/bin/env python3
"""Yiddish text → IPA with lexicon-first, graph-constrained ReNikud-yi.

The best G2P measured in docs/xeus_finetune.md §19, packaged as one call:

  1. a word the engine holds in a table (gold, lexicon) is read from the
     table — a model cannot beat a lookup on words that are in it;
  2. any other word: the engine's rule reading defines the spelling's legal
     readings (its open slots branched, §12 graph); ReNikud-yi ranks those
     candidates, plus its own free guess, by the sum of its per-letter
     (consonant, vowel) log-probabilities along each candidate's alignment;
     the best legal reading wins. Stress is the engine's (the model predicts
     it, but the engine's placement is what the corpus labels carried).

On 3,748 unlabelled words across six held-out episodes this agrees with the
audio 94.5% of the time; the rule engine alone 88.0%.

  python scripts/renikud_yi_g2p.py "מיט א פאר יאר צוריק אין שפיטאל"
  python scripts/renikud_yi_g2p.py --explain "…"      # show the candidates and scores
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import tokenize_ipa  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
DEFAULT_MODEL = REPO / "models" / "renikud_yi_audio"


class YiG2P:
    def __init__(self, model_dir: Path = DEFAULT_MODEL, device: str | None = None):
        import torch
        from renikud_yi_eval import load_model
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model, self.tok, self.labels = load_model(Path(model_dir), self.device)

    def read(self, text: str, explain: bool = False) -> list[dict]:
        from renikud_yi_eval import predict_words, score_reading
        from xeus_lattice import graph_candidates
        from yiddish_g2p import g2p_token
        from prepare_retrain_dataset_v8 import restress
        free, lp = predict_words(self.model, self.tok, self.labels, text, self.device, want_logprobs=True)
        out = []
        for hi, m in enumerate(_HEB.finditer(text)):
            w = m.group(0)
            t = g2p_token(w)
            t = t if isinstance(t, dict) else t.__dict__
            eng_ipa = t["ipa_primary"] or ""
            eng = tokenize_ipa(eng_ipa)
            rec = {"word": w, "engine": eng_ipa, "route": t["route"]}
            if t["route"] == "lexicon" or not eng:
                rec.update({"ipa": eng_ipa, "source": "lexicon" if eng else "engine"})
                out.append(rec)
                continue
            cands = graph_candidates(eng, max_candidates=64)
            own = free[hi].split()
            if own and own not in cands:
                cands.append(own)
            scored = sorted(((score_reading(w, c, m.start(), lp, self.labels), c) for c in cands), key=lambda x: -x[0])
            best_score, best = scored[0]
            if best_score == float("-inf"):
                rec.update({"ipa": eng_ipa, "source": "engine (model could not align)"})
            else:
                rec.update({"ipa": restress(eng_ipa, best) if len(best) == len(eng) else "".join(best),
                            "source": "renikud+graph", "changed": best != eng})
            if explain:
                rec["candidates"] = [(" ".join(c), round(s, 2)) for s, c in scored[:6]]
            out.append(rec)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()
    g2p = YiG2P(Path(args.model))
    rows = g2p.read(args.text, explain=args.explain)
    print(" ".join(r["ipa"] for r in rows))
    for r in rows:
        flag = " ←" if r.get("changed") else ""
        print(f"  {r['word']:14} {r['ipa']:16} {r['source']:26} engine={r['engine']}{flag}")
        for c, s in r.get("candidates", []):
            print(f"      {s:8.2f}  {c}")


if __name__ == "__main__":
    main()
