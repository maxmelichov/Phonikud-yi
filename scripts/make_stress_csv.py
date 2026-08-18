#!/usr/bin/env python3
"""Build a TTS-ready CSV: Phonikud pointing -> IPA -> LLM-placed stress.

Rule-based stress placement measured worse than none, so here the stress marks
come from the LLMs instead: our engine supplies the phonemes (stress=False) and
Gemini Flash and Gemini 3.1 Pro each insert ˈ before the stressed vowel. The
models may move stress marks ONLY -- any answer whose phoneme string differs
from ours once the marks are stripped is rejected and logged, so the phonetics
stay ours and only prosody comes from the LLM.

Usage:
  GATEWAY_REASONING_EFFORT=low .venv/bin/python scripts/make_stress_csv.py \
      --n 20 --out data/stress/stress_tts.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from phonikud_yi.gateway import (  # noqa: E402
    Gateway,
    MODEL_FLASH,
    MODEL_PRO,
    text_message,
)
from yiddish_g2p import hebrew_to_ipa  # noqa: E402

STRESS = "ˈ"
HEB = re.compile(r"[א-ת]")

PROMPT = """You are a native Hasidic (Central/Poylish) Yiddish speaker and phonetician.

Below is a Yiddish sentence and its phonemic IPA. The IPA has NO stress marks.
Insert the primary stress mark ˈ immediately BEFORE the stressed VOWEL of every
polysyllabic word (TTS convention: /sˈefer/, not /ˈsefer/). Leave one-syllable
words unmarked.

Yiddish reminders:
- Germanic words stress the first syllable of the ROOT; the prefixes ge-, ba-,
  far-, der-, tse-, ant-, ent- are UNSTRESSED (gekˈimen, farʃtˈayn).
- Separable prefixes ARE stressed (ˈuntergeyn, ˈibergebn).
- Words like azoy, amol, arayn, aroys, arop stress the SECOND syllable (azˈoy).
- Loshn-koydesh (Hebrew-origin) words are usually penultimate (ʃˈabes, brˈokhe)
  but not always -- use how the word is actually said in Hasidic speech.
- Loanwords keep their donor-language stress (interesˈant).

CRITICAL: change NOTHING except inserting ˈ characters. Do not alter, add or
remove any phoneme, space or punctuation. The output with all ˈ removed must be
byte-identical to the input IPA.

Return STRICT JSON: {"ipa": "<the IPA with stress marks>"}

Yiddish: %s
IPA: %s"""


def strip_stress(s: str) -> str:
    return s.replace(STRESS, "")


def ask(gw: Gateway, model: str, sent: str, ipa: str) -> tuple[str, str]:
    """Return (stressed_ipa, status)."""
    try:
        obj = gw.chat_json(
            [text_message("user", PROMPT % (sent, ipa))], model=model
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
        return "", f"error: {type(exc).__name__}"
    got = (obj.get("ipa") or "").strip() if isinstance(obj, dict) else ""
    if not got:
        return "", "empty"
    if strip_stress(got) != ipa:
        return got, "REJECTED (phonemes changed)"
    return got, "ok"


def pick_sentences(n: int, seed: int) -> list[str]:
    """Clean, self-contained sentences from the annotated corpus."""
    random.seed(seed)
    files = sorted(glob.glob(str(REPO / "data/annotations/*.jsonl")))
    random.shuffle(files)
    out: list[str] = []
    seen: set[str] = set()
    for f in files:
        for line in open(f, encoding="utf-8"):
            rec = json.loads(line)
            if (rec.get("confidence") or 0) < 0.9:
                continue
            for sent in re.split(r"(?<=[.!?])\s+", rec.get("text_yi") or ""):
                sent = sent.strip()
                words = sent.split()
                if not (6 <= len(words) <= 14):
                    continue
                if not all(HEB.search(w) for w in words):
                    continue
                if sent in seen or '"' in sent:
                    continue
                seen.add(sent)
                out.append(sent)
                if len(out) >= n * 4:
                    break
            if len(out) >= n * 4:
                break
        if len(out) >= n * 4:
            break
    random.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/stress/stress_tts.csv")
    ap.add_argument("--model-onnx", default="models/phonikud_yi_small/student.onnx")
    ap.add_argument("--dict", default="data/corpus/canonical_pointing.tsv")
    args = ap.parse_args()

    from infer_onnx import Diacritizer  # noqa: PLC0415 - heavy import

    dia = Diacritizer(REPO / args.model_onnx, dictionary=REPO / args.dict)
    gw = Gateway()

    rows = []
    for i, sent in enumerate(pick_sentences(args.n, args.seed), 1):
        pointed = dia.point(sent)
        ipa = hebrew_to_ipa(pointed, stress=False)
        flash, flash_status = ask(gw, MODEL_FLASH, sent, ipa)
        pro, pro_status = ask(gw, MODEL_PRO, sent, ipa)
        agree = bool(flash) and flash == pro
        rows.append(
            {
                "id": i,
                "sentence_yi": sent,
                "pointed_yi": pointed,
                "ipa_no_stress": ipa,
                "ipa_flash": flash,
                "ipa_pro": pro,
                "flash_status": flash_status,
                "pro_status": pro_status,
                "models_agree": "yes" if agree else "no",
            }
        )
        print(f"[{i}/{args.n}] flash={flash_status} pro={pro_status} agree={agree}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok_f = sum(r["flash_status"] == "ok" for r in rows)
    ok_p = sum(r["pro_status"] == "ok" for r in rows)
    ag = sum(r["models_agree"] == "yes" for r in rows)
    print(f"\nflash ok {ok_f}/{len(rows)}  pro ok {ok_p}/{len(rows)}  identical {ag}/{len(rows)}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
