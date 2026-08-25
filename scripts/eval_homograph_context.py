#!/usr/bin/env python3
"""Homograph-in-context + וי-class slice for a phonikud-yi checkpoint.

The pointing model is supposed to split real homographs by sentence rather
than collapsing every occurrence to the type-level winner. This script:

  1. Points constructed contrastive sentences for חלה, מקדש, מדבר (and a few
     other split-verdict types) and records which pointing / IPA the model
     emitted for the target token.
  2. Replays audio-decided vote contexts whose two winners disagree, labelled
     as train-memorization if the row id is in the train split.
  3. Runs a tiny וי class slice through the frozen G2P on the *model's*
     pointing: הויט vs טויט, מויל vs בוים, גרויס as ɔj not oʊ. These are
     lexicon/G2P facts, not pointing facts — the check is "did the model
     mangle the grapheme so G2P can no longer recover the class?"

Usage:
    .venv/bin/python scripts/eval_homograph_context.py \
        --model models/phonikud_yi_v6/best \
        --out data/retrain7/homograph_context_v6.json
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from point_text import Pointer  # noqa: E402
import yiddish_g2p as G  # noqa: E402

MARKS = set("ְֱֲֳִֵֶַָׇֹֺֻּֿׁׂ")


def strip_marks(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return "".join(ch for ch in s if ch not in MARKS and not unicodedata.combining(ch))


def token_at(pointed: str, surface: str) -> str | None:
    """First whitespace token whose skeleton equals ``surface``."""
    want = strip_marks(surface)
    for tok in pointed.split():
        if strip_marks(tok) == want:
            return tok
    return None


# Constructed sentences. Expected pointing is the linguistically distinct
# reading, taken from data/lexicons/homograph_lk.py / candidates.json — not
# invented. "split" means the two sentences of a pair must not collapse to
# the same pointing.
CONTRAST_PAIRS = [
    {
        "type": "מדבר",
        "readings": [
            {
                "tag": "speaks",
                "text": "דער רבי מדבר צו דעם עולם",
                "expect_pointed": "מְדַבֵּר",
                "expect_ipa_substr": "dabajr",
            },
            {
                "tag": "desert",
                "text": "משה רבינו איז געווען אין מדבר סיני",
                "expect_pointed": "מִדְבָּר",
                "expect_ipa_substr": "idbur",
            },
        ],
    },
    {
        "type": "חלה",
        "readings": [
            {
                "tag": "challah",
                "text": "מען עסט חלה שבת צו דער סעודה",
                "expect_pointed": "חַלָּה",
                "expect_ipa_substr": "alu",
            },
            {
                "tag": "fell-ill",
                "text": "ער איז חלה געווארן און ליגט אין בעט",
                "expect_pointed": "חָלָה",
                "expect_ipa_substr": "ulu",
            },
        ],
    },
    {
        "type": "מקדש",
        "readings": [
            {
                "tag": "sanctifies",
                "text": "מען מקדש דעם חודש בזמנו",
                "expect_pointed": "מְקַדֵּשׁ",
                "expect_ipa_substr": "kadajʃ",
            },
            {
                "tag": "temple",
                "text": "דער מקדש אין ירושלים איז חרוב געווארן",
                "expect_pointed": "מִקְדָּשׁ",
                "expect_ipa_substr": "ikduʃ",
            },
        ],
    },
    {
        "type": "טויב",
        # Same pointing (טוֹיב); the split is phonemic, not nikud. The pointing
        # model cannot disambiguate. G2P keeps both IPAs at type level; a
        # sentence-conditioned reading would need a context model we do not have.
        "readings": [
            {
                "tag": "deaf",
                "text": "דער מענטש איז טויב ער הערט נישט",
                "expect_pointed": "טוֹיב",
                "expect_ipa_substr": "ɔj",
            },
            {
                "tag": "dove",
                "text": "א טויב פליט איבערן דאך",
                "expect_pointed": "טוֹיב",
                "expect_ipa_substr": "oʊ",
            },
        ],
    },
]


# Lexicon/G2P class facts. oʊ only for attested û-class; ɔj otherwise.
OY_SLICE = [
    {"word": "הויט", "expect_dip": "oʊ", "note": "skin, Weinreich 54"},
    {"word": "טויט", "expect_dip": "ɔj", "note": "death, class 44"},
    {"word": "מויל", "expect_dip": "oʊ", "note": "mouth, Weinreich 54"},
    {"word": "בוים", "expect_dip": "ɔj", "note": "tree, class 42/44"},
    {"word": "גרויס", "expect_dip": "ɔj", "note": "big; gold ɡrɔjs"},
    {"word": "אויף", "expect_dip": "oʊ", "note": "on; keep class 54"},
    {"word": "ארויס", "expect_dip": "oʊ", "note": "out; keep class 54"},
    {"word": "אויס", "expect_dip": "oʊ", "note": "prefix; keep class 54"},
    {"word": "בלויז", "expect_dip": "ɔj", "note": "only; gold blɔjz"},
    {"word": "אויך", "expect_dip": "ɔj", "note": "also; class 44, gold ɔjx"},
]


def load_train_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def vote_contrast_pairs(votes: Path, min_margin: float) -> list[dict]:
    """Types with at least two distinct decided pointings, with one example each."""
    by_word: dict[str, dict[str, dict]] = defaultdict(dict)
    with votes.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            v = json.loads(line)
            if v.get("decision") != "decided" or not v.get("winner_pointed"):
                continue
            if float(v.get("margin") or 0.0) < min_margin:
                continue
            word = v["word"]
            pointed = v["winner_pointed"]
            if pointed not in by_word[word]:
                ctx = " ".join(
                    (v.get("context_left") or []) + [word] + (v.get("context_right") or [])
                )
                by_word[word][pointed] = {
                    "tag": pointed,
                    "text": ctx,
                    "expect_pointed": pointed,
                    "expect_ipa_substr": "",
                    "vote_id": v["id"],
                    "margin": v.get("margin"),
                    "winner_ipa": v.get("winner_ipa"),
                }
    pairs = []
    for word, readings in by_word.items():
        if len(readings) < 2:
            continue
        pairs.append({"type": word, "readings": list(readings.values())[:3]})
    pairs.sort(key=lambda p: p["type"])
    return pairs


def score_pair(pointer: Pointer, pair: dict, train_ids: set[str]) -> dict:
    recs = []
    pointings = []
    for r in pair["readings"]:
        pred = pointer.point([r["text"]])[0]
        tok = token_at(pred, pair["type"])
        ipa = G.hebrew_to_ipa(tok, stress=True) if tok else None
        in_train = r.get("vote_id") in train_ids if r.get("vote_id") else False
        hit_pointed = bool(tok and strip_marks(tok) == strip_marks(r["expect_pointed"])
                           and tok == r["expect_pointed"]) if r.get("expect_pointed") else None
        # looser: skeleton match already required; compare folded pointing
        folded_ok = tok == r["expect_pointed"] if r.get("expect_pointed") else None
        ipa_ok = None
        if r.get("expect_ipa_substr") and ipa:
            ipa_ok = r["expect_ipa_substr"] in ipa
        recs.append({
            "tag": r["tag"],
            "text": r["text"],
            "pred_sentence": pred,
            "pred_token": tok,
            "pred_ipa": ipa,
            "expect_pointed": r.get("expect_pointed"),
            "pointed_match": folded_ok,
            "ipa_substr_match": ipa_ok,
            "in_train_split": in_train,
            "vote_id": r.get("vote_id"),
        })
        pointings.append(tok)
    split = len({p for p in pointings if p is not None}) >= 2
    return {
        "type": pair["type"],
        "split_by_sentence": split,
        "n_distinct_pointings": len({p for p in pointings if p is not None}),
        "readings": recs,
    }


def score_oy(pointer: Pointer) -> list[dict]:
    out = []
    for c in OY_SLICE:
        pred = pointer.point([c["word"]])[0]
        tok = token_at(pred, c["word"]) or pred.split()[0]
        ipa = G.hebrew_to_ipa(tok, stress=True)
        has_ou = "oʊ" in ipa
        has_oj = "ɔj" in ipa
        ok = (c["expect_dip"] == "oʊ" and has_ou) or (c["expect_dip"] == "ɔj" and has_oj)
        out.append({
            **c,
            "pred_token": tok,
            "pred_ipa": ipa,
            "ok": ok,
            "has_oʊ": has_ou,
            "has_ɔj": has_oj,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", type=Path, required=True)
    ap.add_argument("--votes", type=Path, default=REPO / "data/homographs/votes.jsonl")
    ap.add_argument("--train", type=Path, default=REPO / "data/retrain7/train.jsonl")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pointer = Pointer(args.model, args.device)
    train_ids = load_train_ids(args.train)

    constructed = [score_pair(pointer, p, train_ids) for p in CONTRAST_PAIRS]
    vote_pairs = vote_contrast_pairs(args.votes, args.margin)
    # keep the named types plus a cap of other split-verdict types
    named = {"מדבר", "חלה", "מקדש", "טויב"}
    extra = [p for p in vote_pairs if p["type"] not in named][:12]
    named_vote = [p for p in vote_pairs if p["type"] in named]
    vote_scored = [score_pair(pointer, p, train_ids) for p in named_vote + extra]
    oy = score_oy(pointer)

    def split_rate(rows: list[dict]) -> dict:
        n = len(rows)
        s = sum(1 for r in rows if r["split_by_sentence"])
        return {"n_types": n, "n_split": s, "split_rate": round(100.0 * s / n, 1) if n else 0.0}

    result = {
        "model": str(args.model),
        "constructed": constructed,
        "constructed_summary": split_rate(constructed),
        "vote_contexts": vote_scored,
        "vote_context_summary": split_rate(vote_scored),
        "oy_slice": oy,
        "oy_slice_ok": sum(1 for r in oy if r["ok"]),
        "oy_slice_n": len(oy),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
