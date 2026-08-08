#!/usr/bin/env python
"""Re-diacritize the Yiddish corpus with Gemini under a rule-pinned prompt.

WHY THIS EXISTS: the pointing in data/annotations was produced by the audio
annotation prompt, which gave no pointing conventions at all. The result
disagrees with itself -- האט appears pointed 18 different ways, האבן 31 -- and
the mean word type carried 2.97 distinct pointings. Since the komets/pasekh
choice is exactly what decides /u/ vs /a/ downstream, a diacritizer trained on
that learns three dialects at once, which is what the released voice does.

scripts/prompts/yiddish_nikud.txt pins the conventions -- komets vs pasekh word
lists, פּ/פֿ, שׁ/שׂ, כּ/כ, תּ/ת, וּ, ײַ/ייִ/יי, and full Ashkenazi nikud on
loshn-koydesh -- so the same word is pointed the same way every time. The
orthographic knowledge stays the model's; the prompt only fixes the conventions.

The audio is NOT re-transcribed: the transcripts are fine, only the pointing was
broken, and text-in/text-out costs ~100x less.

TWO THINGS MAKE THE OUTPUT SAFE TO TRAIN ON:

  1. Letter repair. Models rewrite letters they consider equivalent -- the ײ
     ligature for two yuds, non-final פ for ף when attaching a rafe. Those are
     the same letter, so once a row round-trips under normalisation we rebuild it
     from the SOURCE letters with the model's marks. Training data then aligns to
     the corpus character for character, by construction.

  2. Round-trip validation. Anything that still does not reproduce the source
     after repair -- dropped vav in חופש, dropped yud in בסייעתא -- is kept with
     ok=false and never enters training. Measured ~80% clean on flash, ~60% on
     flash-lite, which is why this defaults to flash.

Output: data/nikud/<episode_id>.jsonl, one record per sentence:
  {chunk_idx, sent_idx, text_yi, pointed, ok, problems}

Usage:
  .venv/bin/python scripts/nikud_yi.py --limit 1            # one episode
  .venv/bin/python scripts/nikud_yi.py                      # all episodes
  .venv/bin/python scripts/nikud_yi.py --model google/gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonikud_yi.gateway import Gateway, GatewayError, text_message  # noqa: E402
from scripts.phonemize_yi import split_sentences  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ANNOT_DIR = REPO / "data" / "annotations"
OUT_DIR = REPO / "data" / "nikud"
PROMPT_PATH = REPO / "scripts" / "prompts" / "yiddish_nikud.txt"

MODEL_FLASH = "google/gemini-3.5-flash"

SYSTEM = (
    "You are an expert in Yiddish orthography and the Hasidic pronunciation "
    "tradition. You add diacritics mechanically and identically every time. "
    "You never change, drop or transliterate a letter."
)

_HEBREW = re.compile(r"[֐-׿]")

# Codepoints that are the same letter written differently. Normalising these
# before comparison is lossless: ligatures decompose to the letters the corpus
# uses, and final forms are positional variants of their base letter.
_LIGATURES = {"װ": "וו", "ױ": "וי", "ײ": "יי"}
_FINALS = {"ך": "כ", "ם": "מ", "ן": "נ",
           "ף": "פ", "ץ": "צ"}
# Invisible joiners and bidi marks the model sometimes sprinkles in.
_INVISIBLE = re.compile("[​-‏‪-‮⁠﻿]")


def canon(text: str) -> str:
    """Letters only, ligatures split, final forms folded, marks removed."""
    text = _INVISIBLE.sub("", unicodedata.normalize("NFD", text))
    out = []
    for ch in text:
        if unicodedata.category(ch) == "Mn":
            continue
        ch = _LIGATURES.get(ch, ch)
        out.append("".join(_FINALS.get(c, c) for c in ch))
    return "".join(out)


def repair(src: str, pointed: str) -> str | None:
    """Rebuild ``pointed`` on ``src``'s letters, keeping the model's marks.

    Returns None when the two do not describe the same letter sequence, which is
    a real corruption rather than a spelling variant.
    """
    if canon(src) != canon(pointed):
        return None
    # Walk the pointed string collecting the marks that follow each base letter,
    # then re-emit using the source's own letters (and its final forms).
    groups: list[str] = []
    for ch in _INVISIBLE.sub("", unicodedata.normalize("NFD", pointed)):
        if unicodedata.category(ch) == "Mn":
            if groups:
                groups[-1] += ch
        else:
            # A ligature stands for two source letters; the second gets the marks.
            for c in _LIGATURES.get(ch, ch):
                groups.append(c)
    src_nfd = _INVISIBLE.sub("", unicodedata.normalize("NFD", src))
    out: list[str] = []
    gi = 0
    for ch in src_nfd:
        if unicodedata.category(ch) == "Mn":
            continue  # source marks are replaced wholesale by the model's
        if gi >= len(groups):
            return None
        out.append(ch)              # the source's letter, incl. its final form
        out.append(groups[gi][1:])  # the model's marks for that letter
        gi += 1
    return unicodedata.normalize("NFC", "".join(out))


def validate(src: str, pointed: str) -> list[str]:
    problems: list[str] = []
    if not pointed.strip():
        return ["empty"]
    if canon(pointed).split() != canon(src).split():
        problems.append("letters-differ-from-source")
    # A row where the model simply echoed the input teaches nothing.
    if not any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", pointed)):
        problems.append("no-diacritics-added")
    # Hebrew-origin words left bare are the failure the prompt targets; flag a
    # row where a long all-Hebrew word came back with no marks at all.
    for w in pointed.split():
        if len(w) >= 4 and _HEBREW.match(w) and '"' not in w and "'" not in w:
            if not any(unicodedata.category(c) == "Mn"
                       for c in unicodedata.normalize("NFD", w)):
                problems.append(f"unpointed-word({w})")
    return problems


def parse_tsv(raw: str, ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() in ids:
            out[parts[0].strip()] = parts[1].strip()
    return out


def load_sentences(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = (d.get("text_yi") or "").strip()
        if not text or not _HEBREW.search(text):
            continue
        for si, sent in enumerate(split_sentences(text)):
            if _HEBREW.search(sent):
                rows.append({
                    "chunk_idx": d.get("chunk_idx"),
                    "sent_idx": si,
                    "id": f"{d.get('chunk_idx')}.{si}",
                    "text_yi": sent,
                })
    return rows


def run_batch(gw: Gateway, prompt: str, batch: list[dict], model: str) -> dict[str, str]:
    body = "\n".join(f"{r['id']}\t{r['text_yi']}" for r in batch)
    msgs = [text_message("system", SYSTEM), text_message("user", prompt + body)]
    raw = gw.chat(msgs, model=model)
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    return parse_tsv(raw, {r["id"] for r in batch})


def process(gw, prompt, rows, model, batch_size, workers=8, retry=True):
    """Return {id: pointed}; retries the rows that failed once, in isolation.

    Batches go out concurrently -- serially this is ~12k requests for the full
    corpus, which is about a day of wall clock for work that is entirely
    network-bound.
    """
    got: dict[str, str] = {}
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_batch, gw, prompt, b, model): b for b in batches}
        for fut in as_completed(futures):
            try:
                got.update(fut.result())
            except (GatewayError, Exception) as exc:  # noqa: BLE001
                print(f"    batch failed: {exc}", file=sys.stderr)
    if not retry:
        return got
    bad = [r for r in rows
           if r["id"] not in got or validate(r["text_yi"], repair(r["text_yi"], got[r["id"]]) or got[r["id"]])]
    if bad:
        # Smaller batches: most failures are the model losing alignment on a long
        # request, and the same row often succeeds on its own.
        retried = process(gw, prompt, bad, model, max(1, batch_size // 4),
                          workers=workers, retry=False)
        for rid, val in retried.items():
            src = next(r["text_yi"] for r in bad if r["id"] == rid)
            if not validate(src, repair(src, val) or val):
                got[rid] = val
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--episode", action="append")
    ap.add_argument("--model", default=MODEL_FLASH)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    files = sorted(ANNOT_DIR.glob("*.jsonl"))
    if args.episode:
        files = [f for f in files if f.stem in set(args.episode)]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("no episodes matched", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gw = Gateway()
    grand = {"rows": 0, "ok": 0}

    for n, fp in enumerate(files, 1):
        out_path = OUT_DIR / fp.name
        if out_path.exists() and not args.overwrite:
            continue
        rows = load_sentences(fp)
        if not rows:
            continue
        got = process(gw, prompt, rows, args.model, args.batch, args.workers)

        results = []
        for r in rows:
            raw_pt = got.get(r["id"], "")
            fixed = repair(r["text_yi"], raw_pt) if raw_pt else None
            pointed = fixed or raw_pt
            problems = ["missing-from-reply"] if not raw_pt else validate(r["text_yi"], pointed)
            results.append({
                "chunk_idx": r["chunk_idx"], "sent_idx": r["sent_idx"],
                "text_yi": r["text_yi"], "pointed": pointed,
                "ok": not problems, "problems": problems,
            })
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in results:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        good = sum(r["ok"] for r in results)
        grand["rows"] += len(results)
        grand["ok"] += good
        print(f"[{n}/{len(files)}] {fp.stem}: {good}/{len(results)} clean "
              f"| running {grand['ok']}/{grand['rows']} = "
              f"{grand['ok'] / max(grand['rows'], 1):.1%}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
