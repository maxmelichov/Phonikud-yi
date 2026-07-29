#!/usr/bin/env python3
"""
OOV / in-vocab and word-level evaluation of a trained Yiddish diacritizer.

No retraining -- pure inference on the canonical test split.

  (a) Character accuracy split by whether the word's *bare* form was ever seen in
      the training split. The OOV number is the real generalization measure.
  (b) Word-level accuracy: a word counts correct only if EVERY Hebrew character in
      it gets the exactly right diacritics.

Gold is projected through the model's own label space (`to_mirror_ids` ->
`mirror_to_canon`), i.e. the target the model was actually trained against, so
these numbers are on the same scale as the reported test accuracy.

Usage:
    python scripts/eval_oov_wordlevel.py \
        --model models/phonikud_yi/round4/stageB \
        --train data/diacritics_r3c/train.txt \
        --test  data/diacritics_r3c/test.txt \
        --out   models/phonikud_yi/round4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phonikud" / "model"))

from src.model.phonikud_model import PhonikudModel  # noqa: E402
from src.train.yi_data import (  # noqa: E402
    IGNORE,
    mirror_to_canon,
    parse_pointed,
    read_pointed_file,
    to_mirror_ids,
)

HEB = re.compile(r"[א-ת]")
MARKS = re.compile(r"[֑-ׇ]")


def bare(word: str) -> str:
    """Unpointed key for vocabulary matching: drop diacritics, then trim any
    leading/trailing characters that are not Hebrew letters (punctuation,
    quotes, digits) so `וואס,` and `וואס` count as the same word."""
    w = MARKS.sub("", word)
    return w.strip("".join(c for c in set(w) if not HEB.match(c))) if w else w


def build_vocab(path: Path) -> set[str]:
    vocab = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        for tok in line.split():
            b = bare(tok)
            if b and HEB.search(b):
                vocab.add(b)
    return vocab


@torch.no_grad()
def predict_marks(model, tokenizer, examples, batch_size=8, max_length=512, log_every=20):
    """Return, per example, a list of predicted canonical mark strings (one per
    character, None where the character is not a Hebrew letter)."""
    classes = model.menaked.yi_nikud_classes
    out_all = []
    t0 = time.time()
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        texts = [e.text for e in chunk]
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True,
        )
        offsets = enc.pop("offset_mapping")
        out = model(dict(enc))
        nk = out.yi_nikud_logits.argmax(-1)
        sh = out.yi_shin_logits.argmax(-1)
        rf = out.yi_rafe_logits.argmax(-1)

        for b, ex in enumerate(chunk):
            preds = [None] * len(ex.text)
            for t in range(offsets.size(1)):
                s, e = int(offsets[b, t, 0]), int(offsets[b, t, 1])
                if e - s != 1 or s >= len(ex.marks) or ex.marks[s] is None:
                    continue
                preds[s] = mirror_to_canon(
                    int(nk[b, t]), int(sh[b, t]), int(rf[b, t]), ex.text[s]
                )
            out_all.append(preds)

        if log_every and (i // batch_size) % log_every == 0:
            done = min(i + batch_size, len(examples))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(examples)} segments  ({rate:.1f}/s)", flush=True)
    return out_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"building train vocabulary from {args.train}", flush=True)
    vocab = build_vocab(args.train)
    print(f"  {len(vocab):,} distinct bare word forms", flush=True)

    examples = read_pointed_file(args.test, args.max_length - 32)
    print(f"test: {len(examples)} segments", flush=True)

    model = PhonikudModel.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    print(f"running inference on CPU ({torch.get_num_threads()} threads)", flush=True)

    preds = predict_marks(model, tokenizer, examples, args.batch_size, args.max_length)

    # ---- aggregate per word ------------------------------------------------
    groups = {"in_vocab": [], "oov": []}
    words = []  # (group, n_chars, n_correct, n_marked, n_marked_correct, all_ok, ...)

    for ex, pred in zip(examples, preds):
        # walk whitespace-delimited words, tracking absolute char offsets
        pos = 0
        for tok in ex.text.split(" "):
            start = ex.text.find(tok, pos)
            if start < 0:
                start = pos
            pos = start + len(tok)
            if not tok or not HEB.search(tok):
                continue

            key = bare(tok)
            if not key:
                continue
            grp = "in_vocab" if key in vocab else "oov"

            n_c = n_ok = n_m = n_m_ok = 0
            gold_word, pred_word = [], []
            for off in range(len(tok)):
                ci = start + off
                ch = ex.text[ci]
                g_marks = ex.marks[ci]
                if g_marks is None:
                    gold_word.append(ch)
                    pred_word.append(ch)
                    continue
                # project gold through the model's label space
                gi = to_mirror_ids(g_marks, ch)
                gold = mirror_to_canon(gi[0], gi[1] if gi[1] != IGNORE else 0, gi[2], ch)
                p = pred[ci] if pred[ci] is not None else ""
                gold_word.append(ch + gold)
                pred_word.append(ch + p)
                n_c += 1
                ok = p == gold
                n_ok += ok
                if gold != "":
                    n_m += 1
                    n_m_ok += ok

            if n_c == 0:
                continue
            rec = {
                "group": grp, "word": key, "n_chars": n_c, "n_correct": n_ok,
                "n_marked": n_m, "n_marked_correct": n_m_ok,
                "correct": n_ok == n_c, "len": len(key),
                "gold": "".join(gold_word), "pred": "".join(pred_word),
            }
            words.append(rec)
            groups[grp].append(rec)

    # ---- metrics -----------------------------------------------------------
    def stats(rs):
        c = sum(r["n_chars"] for r in rs)
        ok = sum(r["n_correct"] for r in rs)
        m = sum(r["n_marked"] for r in rs)
        mok = sum(r["n_marked_correct"] for r in rs)
        wc = sum(1 for r in rs if r["correct"])
        long = [r for r in rs if r["len"] >= 4]
        wl = sum(1 for r in long if r["correct"])
        pct = lambda a, b: (100.0 * a / b) if b else float("nan")  # noqa: E731
        return {
            "n_words": len(rs), "n_chars": c, "n_marked_chars": m,
            "char_acc": pct(ok, c), "marked_char_acc": pct(mok, m),
            "word_acc": pct(wc, len(rs)),
            "n_words_ge4": len(long), "word_acc_ge4": pct(wl, len(long)),
        }

    res = {
        "model": str(args.model), "test": str(args.test),
        "train_vocab_size": len(vocab),
        "overall": stats(words),
        "in_vocab": stats(groups["in_vocab"]),
        "oov": stats(groups["oov"]),
    }
    res["oov_word_rate"] = 100.0 * len(groups["oov"]) / max(len(words), 1)
    res["distinct_oov_types"] = len({r["word"] for r in groups["oov"]})

    rng = random.Random(args.seed)
    oov_ok = [r for r in groups["oov"] if r["correct"] and r["n_marked"] > 0]
    oov_bad = [r for r in groups["oov"] if not r["correct"]]
    res["examples_oov_correct"] = rng.sample(oov_ok, min(10, len(oov_ok)))
    res["examples_oov_wrong"] = rng.sample(oov_bad, min(10, len(oov_bad)))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oov_wordlevel_report.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- markdown ----------------------------------------------------------
    L = ["# Round 4 — OOV and word-level evaluation\n"]
    L.append(f"Model: `{args.model}` · test: `{args.test}` · no retraining, inference only.\n")
    L.append(f"\nA word is **in-vocab** if its bare (unpointed) form appears anywhere in "
             f"`{args.train.name}` — {len(vocab):,} distinct forms. "
             f"{res['oov_word_rate']:.1f}% of test word tokens are OOV "
             f"({res['distinct_oov_types']:,} distinct OOV types).\n")

    L.append("\n## Character-level\n")
    L.append("| group | words | chars | char acc | marked chars | marked-char acc |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for key, name in (("overall", "overall"), ("in_vocab", "in-vocab"), ("oov", "**OOV**")):
        s = res[key]
        L.append(f"| {name} | {s['n_words']:,} | {s['n_chars']:,} | {s['char_acc']:.2f}% | "
                 f"{s['n_marked_chars']:,} | {s['marked_char_acc']:.2f}% |")

    L.append("\n## Word-level (every character in the word must be exactly right)\n")
    L.append("| group | words | word acc | words ≥4 chars | word acc ≥4 chars |")
    L.append("|---|---:|---:|---:|---:|")
    for key, name in (("overall", "overall"), ("in_vocab", "in-vocab"), ("oov", "**OOV**")):
        s = res[key]
        L.append(f"| {name} | {s['n_words']:,} | {s['word_acc']:.2f}% | "
                 f"{s['n_words_ge4']:,} | {s['word_acc_ge4']:.2f}% |")

    L.append("\n## Example OOV words the model got right\n")
    L.append("| bare word | gold pointing | predicted |")
    L.append("|---|---|---|")
    for r in res["examples_oov_correct"]:
        L.append(f"| `{r['word']}` | `{r['gold']}` | `{r['pred']}` |")

    L.append("\n## Example OOV words the model got wrong\n")
    L.append("| bare word | gold pointing | predicted |")
    L.append("|---|---|---|")
    for r in res["examples_oov_wrong"]:
        L.append(f"| `{r['word']}` | `{r['gold']}` | `{r['pred']}` |")

    (args.out / "oov_wordlevel_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n" + json.dumps(
        {k: res[k] for k in ("overall", "in_vocab", "oov", "oov_word_rate")}, indent=2))


if __name__ == "__main__":
    main()
