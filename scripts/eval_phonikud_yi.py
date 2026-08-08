#!/usr/bin/env python3
"""
Evaluate a Yiddish diacritizer checkpoint on the retrain test split
(`data/retrain/test.jsonl` = the 409 canonically-pointed rows of episode 100313,
excluded from train and val).

Three families of numbers:

1. CHARACTER  -- per-head and joint accuracy on SUPERVISED positions only
   (a position is supervised when the frozen v3 engine verifies the word's
   reading), plus a per-diacritic breakdown so komets/pasekh confusion is
   visible directly.
2. WORD       -- exact-pointing rate: a word counts correct only when every
   Hebrew character in it carries exactly the gold marks.
3. DOWNSTREAM -- the predicted pointed text is run through the FROZEN engine's
   `hebrew_to_ipa` / `g2p_tokens` and compared, token by token, against the
   engine's own output for the same rows:
     * `vs_engine_verified`  -- pred-pointed vs engine-on-source, on tokens the
       engine routes to `lexicon`. The gold lexicon is keyed on the *unpointed*
       form, so a healthy model scores 100% here; anything less means the
       pointing corrupted letters or word boundaries. It is a letter-safety gate.
     * `vs_gold_pointing`    -- pred-pointed vs gold-pointed through the engine,
       broken down by route. On `rule`/`fallback` tokens the pointing actually
       drives the phonemization, so this is where a better model shows up.

Usage:
    python scripts/eval_phonikud_yi.py --model models/phonikud_yi/round4/stageB
    python scripts/eval_phonikud_yi.py -m models/phonikud_yi_v2/best --out data/retrain/v2_eval.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from phonikud_yi_data import (  # noqa: E402
    VOWEL_SET,
    canon,
    is_heb,
    parse_row,
    read_rows,
    strip_marks,
)
from point_text import DEFAULT_MODEL, Pointer  # noqa: E402

import yiddish_g2p  # noqa: E402  (FROZEN engine -- read only)

MARK_NAMES = {
    "ְ": "sheva", "ֱ": "hataf-segol", "ֲ": "hataf-pasekh", "ֳ": "hataf-komets",
    "ִ": "khirik", "ֵ": "tsere", "ֶ": "segol", "ַ": "pasekh", "ָ": "komets",
    "ֹ": "holam", "ֻ": "kubuts", "ּ": "dagesh", "ֿ": "rafe",
    "ׁ": "shin-dot", "ׂ": "sin-dot", "": "NO_MARK",
}


def char_marks(pointed: str):
    """(base chars, canonical mark string per char) for a pointed string."""
    ex = parse_row(pointed, None)
    return ex.text, ex.marks


def words_of(text: str):
    """Whitespace tokens with their (start, end) char span in `text`."""
    out, i = [], 0
    for tok in text.split(" "):
        if tok:
            out.append((tok, i, i + len(tok)))
        i += len(tok) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--test", type=Path, default=REPO / "data/retrain/test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None, help="markdown report")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = read_rows(args.test, args.limit)
    print(f"test rows: {len(rows)}  model: {args.model}", flush=True)

    pointer = Pointer(args.model, args.device)
    t0 = time.perf_counter()
    preds = []
    for i in range(0, len(rows), args.batch_size):
        chunk = [r["pointed"] for r in rows[i:i + args.batch_size]]
        preds.extend(pointer.point(chunk))
    infer_s = time.perf_counter() - t0
    print(f"inference: {infer_s:.1f}s", flush=True)

    # ---------------------------------------------------------- 1. character
    c_ok = c_n = 0
    marked_ok = marked_n = 0
    per_mark_ok, per_mark_n = Counter(), Counter()
    confusion = Counter()
    head_ok = Counter()
    head_n = Counter()
    # ---------------------------------------------------------- 2. word
    w_ok = w_n = 0
    # ---------------------------------------------------------- 3. downstream
    ds = defaultdict(lambda: [0, 0])  # key -> [agree, total]
    misalign = 0
    letter_unsafe = 0

    for row, pred in zip(rows, preds):
        gold_chars, gold_marks = char_marks(row["pointed"])
        pred_chars, pred_marks = char_marks(pred)
        sup = row["supervised"]

        if gold_chars != pred_chars:
            letter_unsafe += 1
            continue

        # per-word supervision spans, on the skeleton
        wsup = {}
        for wi, (tok, s, e) in enumerate(words_of(gold_chars)):
            wsup[(s, e)] = wi < len(sup) and bool(sup[wi])

        for (s, e), ok_word in wsup.items():
            if not ok_word:
                continue
            any_heb = False
            all_ok = True
            for i in range(s, e):
                if not is_heb(gold_chars[i]):
                    continue
                any_heb = True
                g = gold_marks[i] or ""
                p = pred_marks[i] or ""
                right = g == p
                c_n += 1
                c_ok += right
                all_ok &= right
                if g:
                    marked_n += 1
                    marked_ok += right
                gv = next((m for m in g if m in VOWEL_SET), "")
                pv = next((m for m in p if m in VOWEL_SET), "")
                per_mark_n[gv] += 1
                per_mark_ok[gv] += (gv == pv)
                if gv != pv:
                    confusion[(gv, pv)] += 1
                for name, mark in (("dagesh", "ּ"), ("rafe", "ֿ")):
                    head_n[name] += 1
                    head_ok[name] += ((mark in g) == (mark in p))
                if gold_chars[i] == "ש":
                    head_n["shin"] += 1
                    gd = "ׂ" if "ׂ" in g else "ׁ"
                    pd = "ׂ" if "ׂ" in p else "ׁ"
                    head_ok["shin"] += (gd == pd)
            if any_heb:
                w_n += 1
                w_ok += all_ok

        # ---- downstream through the FROZEN engine
        eng_src = yiddish_g2p.g2p_tokens(row.get("source_text_yi") or row["text"])
        eng_gold = yiddish_g2p.g2p_tokens(row["pointed"])
        eng_pred = yiddish_g2p.g2p_tokens(pred)
        if not (len(eng_src) == len(eng_gold) == len(eng_pred)):
            misalign += 1
        else:
            for rs, rg, rp in zip(eng_src, eng_gold, eng_pred):
                if rs["route"] == "lexicon":
                    ds["vs_engine_verified"][1] += 1
                    ds["vs_engine_verified"][0] += (rp["ipa_primary"] == rs["ipa_primary"])
                ds["vs_gold_all"][1] += 1
                ds["vs_gold_all"][0] += (rp["ipa_primary"] == rg["ipa_primary"])
                k = "vs_gold_" + rg["route"]
                ds[k][1] += 1
                ds[k][0] += (rp["ipa_primary"] == rg["ipa_primary"])

    pct = lambda a, b: round(100.0 * a / b, 2) if b else 0.0  # noqa: E731

    # ------------------------------------------------------------- 0. ceiling
    # The training targets and this gold are two pointings of the same
    # convention family, so the metric is capped below 100%. `ceiling_pointed`
    # (carried on every test row by scripts/prepare_retrain_dataset.py) is what
    # the pipeline's own stamper produces for these tokens, i.e. what a model
    # that fits the training objective perfectly emits. Score it the same way.
    ceiling = None
    if rows and all("ceiling_pointed" in r for r in rows):
        from prepare_retrain_dataset import score_ceiling  # noqa: E402
        ceiling = score_ceiling(rows)

    result = {
        "model": str(args.model),
        "test": str(args.test),
        "rows": len(rows),
        "rows_letter_unsafe": letter_unsafe,
        "rows_token_misaligned": misalign,
        "inference_seconds": round(infer_s, 1),
        "char": {
            "supervised_chars": c_n,
            "char_acc": pct(c_ok, c_n),
            "marked_char_acc": pct(marked_ok, marked_n),
            "dagesh_acc": pct(head_ok["dagesh"], head_n["dagesh"]),
            "rafe_acc": pct(head_ok["rafe"], head_n["rafe"]),
            "shin_acc": pct(head_ok["shin"], head_n["shin"]),
        },
        "per_vowel": {
            (MARK_NAMES.get(m, m) or "NO_MARK"): {
                "n": per_mark_n[m], "acc": pct(per_mark_ok[m], per_mark_n[m])}
            for m in sorted(per_mark_n, key=lambda x: -per_mark_n[x])
        },
        "top_vowel_confusions": [
            {"gold": MARK_NAMES.get(g, g) or "NO_MARK",
             "pred": MARK_NAMES.get(p, p) or "NO_MARK", "n": n}
            for (g, p), n in confusion.most_common(12)
        ],
        "word": {"supervised_words": w_n, "exact_pointing_rate": pct(w_ok, w_n)},
        "downstream": {k: {"n": v[1], "agree": pct(v[0], v[1])}
                       for k, v in sorted(ds.items())},
        "samples": [{"gold": rows[i]["pointed"], "pred": preds[i]}
                    for i in range(min(3, len(rows)))],
    }
    if ceiling:
        result["ceiling"] = ceiling
        result["vs_ceiling"] = {
            "char_acc_pct_of_ceiling":
                pct(result["char"]["char_acc"], ceiling["ceiling_char_acc"]),
            "word_exact_pct_of_ceiling":
                pct(result["word"]["exact_pointing_rate"],
                    ceiling["ceiling_word_exact"]),
            "char_headroom": round(
                ceiling["ceiling_char_acc"] - result["char"]["char_acc"], 2),
            "word_headroom": round(
                ceiling["ceiling_word_exact"] - result["word"]["exact_pointing_rate"], 2),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_markdown(result), encoding="utf-8")
        print(f"\nwrote {args.out}")


def to_markdown(r: dict) -> str:
    L = [f"# Eval: `{r['model']}`", "",
         f"Test set: `{r['test']}` -- {r['rows']} rows "
         f"(episode 100313, held out of train and val).",
         f"Inference {r['inference_seconds']}s. "
         f"Letter-unsafe rows: {r['rows_letter_unsafe']}. "
         f"Token-misaligned rows: {r['rows_token_misaligned']}.", "",
         "## 1. Character accuracy (supervised positions only)", "",
         "| metric | value |", "|---|---|"]
    for k, v in r["char"].items():
        L.append(f"| `{k}` | {v} |")
    c, vc = r.get("ceiling"), r.get("vs_ceiling")
    if c:
        L += [f"| `ceiling_char_acc` | {c['ceiling_char_acc']} |",
              f"| `char_headroom_to_ceiling` | {vc['char_headroom']} |",
              "",
              f"**Ceiling {c['ceiling_char_acc']}%** -- see the section below. "
              f"This model is at {vc['char_acc_pct_of_ceiling']}% of it."]
    L += ["", "## 2. Word-level exact pointing", "",
          f"{r['word']['exact_pointing_rate']}% of "
          f"{r['word']['supervised_words']:,} supervised words get *every* "
          f"Hebrew character exactly right."]
    if c:
        L += ["",
              f"Ceiling is {c['ceiling_word_exact']}%, so this is "
              f"{vc['word_exact_pct_of_ceiling']}% of what is reachable "
              f"({vc['word_headroom']} points of headroom)."]
    L += ["",
          "## 3. Downstream (frozen `yiddish_g2p`)", "",
          "| comparison | tokens | agreement |", "|---|---|---|"]
    for k, v in r["downstream"].items():
        L.append(f"| `{k}` | {v['n']:,} | {v['agree']}% |")
    L += ["", "`vs_engine_verified` = predicted pointing vs the engine's own "
          "reading on lexicon-route tokens; the gold lexicon is keyed unpointed, "
          "so <100% means the pointing broke letters or token boundaries.", "",
          "## Per-vowel accuracy", "", "| gold vowel | n | acc |", "|---|---|---|"]
    for name, v in r["per_vowel"].items():
        L.append(f"| {name} | {v['n']:,} | {v['acc']}% |")
    L += ["", "## Top vowel confusions", "", "| gold | predicted | n |", "|---|---|---|"]
    for conf in r["top_vowel_confusions"]:
        L.append(f"| {conf['gold']} | {conf['pred']} | {conf['n']:,} |")
    if c:
        L += ["", "## The ceiling on this test set", "",
              "The retrain targets and this gold are two different pointings of",
              "the same convention family, so the metric is capped below 100%.",
              "Feeding every gold-supervised test token through the pipeline's own",
              "stamper (`prepare_retrain_dataset.Builder.token`) -- i.e. exactly",
              "what a model that fits the training objective perfectly emits --",
              "and scoring it with the metric above gives:", "",
              "| quantity | value |", "|---|---|",
              f"| gold-supervised test tokens | {c['supervised_test_tokens']:,} |",
              f"| stamper agrees with gold | {c['stamp_identical']:,} |",
              f"| stamper disagrees with gold | {c['stamp_different']:,} |",
              f"| stamper does not supervise (gold credited) | {c['stamp_unsupervised']:,} |",
              f"| **ceiling char_acc** | **{c['ceiling_char_acc']}%** |",
              f"| **ceiling word-exact** | **{c['ceiling_word_exact']}%** |",
              "",
              f"{round(100 - c['ceiling_char_acc'], 2)}% of supervised characters "
              "are therefore unreachable by construction. Read every delta on this",
              "test set against the ceiling, not against 100%: the retrain can move",
              f"char accuracy by at most {vc['char_headroom']} points and word-exact",
              f"by at most {vc['word_headroom']} from here.", ""]
    L += ["", "## Samples", ""]
    for s in r["samples"]:
        L += [f"- gold: {s['gold']}", f"  pred: {s['pred']}"]
    L += ["", "---", "", "## Reading these numbers", "",
          "* **Never read char/word accuracy against 100%.** The ceiling above is",
          "  the score of the pipeline's own stamper on this gold; the difference",
          "  is a disagreement between two pointings, not something training can",
          "  learn away.",
          "* **`vs_engine_verified` is a letter-safety gate, not a quality metric.**",
          "  `yiddish_g2p.lexicon_key()` strips nikud, so on lexicon-route tokens the",
          "  engine's reading is pointing-independent by construction; anything below",
          "  100% means the model corrupted letters or word boundaries.",
          "* **The signal to move is `vs_gold_rule` and `vs_gold_fallback`.** Those",
          "  are the tokens where the pointing actually drives the phonemization,",
          "  i.e. the OOV words the runtime dictionary cannot cover. They are also",
          "  not capped by the convention gap the way the char metric is.",
          "",
          "## Model selection",
          "",
          "Val (`data/retrain/val.jsonl`) is near-saturated before a single gradient",
          "step -- the val episodes were in round 4's training data and their",
          "supervision comes from the same lexicon -- so val `char_acc` ties epoch to",
          "epoch and cannot rank checkpoints on its own. `scripts/train_phonikud_yi.py`",
          "therefore selects `best` on `--select-on` (default `val_char` = val",
          "char_acc with val loss as tie-break) and records `best_select_key`,",
          "`best_val` and `best_test_peek` in `run_summary.json`. `--select-on",
          "test_char` selects on the test-peek split instead: a stronger signal, but",
          "it spends the held-out set, so the number it reports stops being an",
          "honest estimate.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
