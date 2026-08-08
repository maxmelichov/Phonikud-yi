#!/usr/bin/env python
"""Build a pronunciation lexicon from the AUDIO, not from introspection.

WHY: the hard distinctions in this dialect are not derivable from spelling or
from diacritics. סְפָרִים and מִשְׁפָּחָה carry the identical komets on the
identical pe yet take different vowels (sfoorim long, mishpuche short). And e
before ר splits three ways -- ee (shveer, veert), a (barg, vark), ay (zayer) --
"depending on the specific word and the speaker's family background", with no
rule behind it. Both were settled word-by-word by asking a native speaker, which
does not scale past a couple of hundred words.

But we already have 264 episodes of real Williamsburg speech with aligned
transcripts. So instead of asking anyone how a word *should* sound, this finds
the places the word is actually said and asks Gemini to report what it hears.
Several independent clips per word, from different episodes where possible, so
one speaker's idiolect cannot decide the entry on its own; the per-clip answers
are kept alongside the majority so disagreement stays visible rather than being
averaged away.

Output: data/audio_lexicon/lexicon.jsonl, one record per word:
  {word, n_clips, consensus, agreement, readings:[{episode, chunk, reading}]}

RE-RUNNING IS CHEAP. The file is an append-only cache keyed by word; a second
run only mines words not already in it. --refresh re-decides cached words.

Usage:
  .venv/bin/python scripts/mine_audio_lexicon.py --words שבת ספרים משפחה --plan
  .venv/bin/python scripts/mine_audio_lexicon.py --words שבת ספרים --clips 3
  .venv/bin/python scripts/mine_audio_lexicon.py --word-file data/oo_u_targets.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonikud_yi.gateway import Gateway, GatewayError, audio_message, text_message  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ANNOT_DIR = REPO / "data" / "annotations"
CHUNK_DIR = REPO / "data" / "chunks"
OUT_DIR = REPO / "data" / "audio_lexicon"
LEXICON = OUT_DIR / "lexicon.jsonl"

MODEL = "google/gemini-3.1-pro-preview"

SYSTEM = (
    "You are a phonetician and a native speaker of Hungarian/Satmar Hasidic "
    "Yiddish as spoken in Williamsburg and Kiryas Joel. You report what you "
    "actually hear on the recording, not what the dictionary or Standard/YIVO "
    "Yiddish says the word should be."
)

PROMPT = """\
This is a clip of Hasidic Yiddish speech from Williamsburg / Kiryas Joel
(Hungarian-Satmar dialect).

Transcript of the clip:
{transcript}

Listen for the word: {word}

Report EXACTLY how the speaker pronounces that one word, in this clip. Return
STRICT JSON:

{{
  "heard": true or false,
  "reading": "<simple English-phonetic spelling of the word as pronounced>",
  "stress": "<the syllable that is stressed, e.g. SHA-bes>",
  "vowel_note": "<see below>",
  "confidence": 0.0-1.0
}}

Rules for "reading":
- Write it the way an English speaker would have to spell it to say it correctly,
  e.g. "shabes", "mishpuche", "sfoorim", "gemure", "shveer", "barg".
- CRITICALLY, distinguish the two u-sounds and say which you hear:
    "oo" = LONG, as in zoo / food  (e.g. sfoorim, shoolem)
    "u"  = SHORT, as in should / put (e.g. mishpuche, bruche, gemure)
  Put which one in "vowel_note", e.g. "long oo as in zoo" or "short u as in should".
- Do NOT normalise toward Standard/YIVO Yiddish. If the speaker says "hut" where
  the book says "hot", write hut. If he says "barg" where the book says "berg",
  write barg.
- If the word is not audible in this clip, or you are not sure you found it, set
  "heard": false and leave "reading" empty. Do not guess.

Output ONLY the JSON object."""

_MARKS = re.compile(r"[֑-ׇ]")


def bare(text: str) -> str:
    return _MARKS.sub("", unicodedata.normalize("NFC", text))


def build_index() -> dict[str, list[tuple[str, int, str]]]:
    """bare word -> [(episode, chunk_idx, transcript), ...]"""
    idx: dict[str, list[tuple[str, int, str]]] = collections.defaultdict(list)
    for fp in sorted(ANNOT_DIR.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (d.get("text_yi") or "").strip()
            ci = d.get("chunk_idx")
            if not text or ci is None:
                continue
            for w in {bare(t).strip(".,!?;:\"'()") for t in text.split()}:
                if w:
                    idx[w].append((fp.stem, int(ci), text))
    return idx


def pick_clips(hits: list[tuple[str, int, str]], n: int) -> list[tuple[str, int, str]]:
    """Prefer one clip per episode, so different speakers get a vote."""
    by_ep: dict[str, list] = collections.defaultdict(list)
    for h in hits:
        by_ep[h[0]].append(h)
    picked: list = []
    # round-robin across episodes before taking a second clip from any one of them
    while len(picked) < n and by_ep:
        for ep in list(by_ep):
            if not by_ep[ep]:
                del by_ep[ep]
                continue
            picked.append(by_ep[ep].pop(0))
            if len(picked) >= n:
                break
    return picked


def ask(gw: Gateway, word: str, ep: str, ci: int, transcript: str, model: str) -> dict:
    path = CHUNK_DIR / ep / f"chunk_{ci:05d}.mp3"
    if not path.exists():
        return {"heard": False, "error": f"missing {path.name}"}
    msgs = [
        text_message("system", SYSTEM),
        audio_message(path, PROMPT.format(transcript=transcript[:1200], word=word)),
    ]
    obj = gw.chat_json(msgs, model=model)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    return obj if isinstance(obj, dict) else {"heard": False}


def norm_reading(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if LEXICON.exists():
        for line in LEXICON.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    out[r["word"]] = r
                except json.JSONDecodeError:
                    pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="*", default=[])
    ap.add_argument("--word-file", help="one word per line")
    ap.add_argument("--clips", type=int, default=3, help="clips to sample per word")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--plan", action="store_true", help="report cost, call nothing")
    args = ap.parse_args()

    words = [bare(w) for w in args.words]
    if args.word_file:
        words += [bare(l.strip()) for l in Path(args.word_file).read_text(
            encoding="utf-8").splitlines() if l.strip()]
    words = list(dict.fromkeys(w for w in words if w))
    if not words:
        print("no words given (--words or --word-file)", file=sys.stderr)
        return 1

    print("indexing transcripts ...", flush=True)
    idx = build_index()
    cache = load_cache()
    todo = words if args.refresh else [w for w in words if w not in cache]

    jobs: list[tuple[str, str, int, str]] = []
    missing: list[str] = []
    for w in todo:
        hits = idx.get(w, [])
        if not hits:
            missing.append(w)
            continue
        for ep, ci, tr in pick_clips(hits, args.clips):
            jobs.append((w, ep, ci, tr))

    print(f"words requested : {len(words)}  (cached {len(words) - len(todo)})")
    print(f"not in any transcript: {len(missing)}"
          + (f" -> {' '.join(missing[:8])}" if missing else ""))
    print(f"audio clips to send  : {len(jobs)}")
    if args.plan or not jobs:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gw = Gateway()
    results: dict[str, list[dict]] = collections.defaultdict(list)
    lock = threading.Lock()
    done = [0]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(ask, gw, w, ep, ci, tr, args.model): (w, ep, ci)
                for w, ep, ci, tr in jobs}
        for fut in as_completed(futs):
            w, ep, ci = futs[fut]
            try:
                obj = fut.result()
            except (GatewayError, Exception) as exc:  # noqa: BLE001
                obj = {"heard": False, "error": str(exc)[:120]}
            with lock:
                results[w].append({"episode": ep, "chunk": ci, **obj})
                done[0] += 1
                print(f"  [{done[0]}/{len(jobs)}] {w} @ {ep}#{ci}: "
                      f"{obj.get('reading') or obj.get('error') or 'not heard'}",
                      flush=True)

    with LEXICON.open("a", encoding="utf-8") as fh:
        for w, reads in results.items():
            heard = [r for r in reads if r.get("heard") and r.get("reading")]
            counts = collections.Counter(norm_reading(r["reading"]) for r in heard)
            top, n = counts.most_common(1)[0] if counts else ("", 0)
            # keep the best-spelled surface form for the winning normalisation
            surface = next((r["reading"] for r in heard
                            if norm_reading(r["reading"]) == top), "")
            rec = {
                "word": w,
                "n_clips": len(reads),
                "n_heard": len(heard),
                "consensus": surface,
                "agreement": round(n / len(heard), 2) if heard else 0.0,
                "readings": reads,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{'word':12} {'consensus':16} {'agree':>6} clips")
    for w in todo:
        if w in results:
            heard = [r for r in results[w] if r.get("heard") and r.get("reading")]
            c = collections.Counter(norm_reading(r["reading"]) for r in heard)
            top, n = c.most_common(1)[0] if c else ("-", 0)
            surface = next((r["reading"] for r in heard
                            if norm_reading(r["reading"]) == top), "-")
            print(f"{w:12} {surface:16} {n}/{len(heard) or 0:<4} {len(results[w])}")
    print(f"\nwrote {LEXICON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
