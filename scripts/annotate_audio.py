#!/usr/bin/env python
"""Annotate Yiddish audio episodes with Gemini: verbatim text + phonemic IPA.

Input manifest: data/audio_manifest.jsonl (falls back to data/episodes.jsonl),
one JSON object per line with at least an id and an audio path. Recognised keys:
  id | episode_id | slug        -> episode id
  audio | audio_path | path | file | mp3 -> path to the mp3 (abs or repo-relative)

Output: data/annotations/<episode_id>.jsonl with one record per ~30s chunk:
  {chunk_idx, start_s, end_s, text_yi, ipa}

Usage:
  .venv/bin/python scripts/annotate_audio.py --limit 3
  .venv/bin/python scripts/annotate_audio.py --episode ep0042 --model google/gemini-3.6-flash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonikud_yi.gateway import (  # noqa: E402
    MODEL_FLASH,
    Gateway,
    GatewayError,
    audio_message,
    iter_jsonl,
    text_message,
)
from phonikud_yi.segment import chunk_mp3, require_ffmpeg  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ANNOT_DIR = DATA / "annotations"
CHUNK_DIR = DATA / "chunks"

SYSTEM = (
    "You are a phonetician and native speaker of Hasidic (Central/Poylish) Yiddish. "
    "You transcribe audio exactly as spoken, with no normalisation, no translation, "
    "and no correction of grammar or dialect."
)

PROMPT = """\
Listen to this Yiddish audio clip and return STRICT JSON with exactly these keys:

{
  "text_yi": "<verbatim transcript in Hebrew script, standard Yiddish orthography, WITHOUT diacritics>",
  "text_yi_pointed": "<the SAME transcript, word for word, WITH full correct diacritics>",
  "ipa": "<phonemic IPA of what is ACTUALLY pronounced>",
  "confidence": <0.0-1.0>,
  "notes": "<short note, or empty string>"
}

Rules:
- text_yi: verbatim Yiddish in Hebrew script. Keep loshn-koydesh (Hebrew-origin) words
  in their historical Hebrew spelling, e.g. שבת, מסתמא, בעל-הבית — do NOT respell them
  phonetically. Include filler words. Use normal punctuation. No Latin transliteration.
- Acronyms / roshei-teyves (תשפ"ו, חז"ל, ב"ה, ר"ת...): keep them EXACTLY as written,
  as ONE token with the gershayim (") in place, in BOTH text fields. NEVER spell out
  the letter names. Their ipa token is what the speaker actually says; if an acronym
  is pronounced as several words, join its IPA with hyphens (e.g. ב"ה -> borəx-haʃem)
  so token counts still match text word-for-word.
- IMPORTANT: inside the JSON strings, write the acronym mark as the Hebrew gershayim
  character ״ (U+05F4), NEVER an ASCII double quote, so the JSON stays valid
  (e.g. תשפ״ו not תשפ"ו).
- text_yi_pointed: identical words in identical order, but with the right diacritics:
  YIVO pointing on Germanic-component words (אַ אָ ייִ וּ פּ בֿ פֿ כּ שׂ תּ) and full Hebrew
  nikud on loshn-koydesh words reflecting the ACTUAL Hasidic pronunciation heard.
- ipa: broad phonemic IPA for Central/Hasidic Yiddish, matching text_yi word for word,
  words separated by single spaces, no slashes, no brackets, no stress marks needed.
  Transcribe the ACTUAL Hasidic pronunciation (e.g. שבת -> ʃabəs, אויף -> af,
  Central Yiddish /ej/->aɪ, /aj/->aː, /u/->i where the speaker does so).
- If the clip has no intelligible Yiddish speech (music, silence, another language),
  return {"text_yi": "", "ipa": "", "confidence": 0.0, "notes": "<reason>"}.
- Output ONLY the JSON object.
"""


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for r in iter_jsonl(path):
        eid = r.get("id") or r.get("episode_id") or r.get("slug")
        audio = (
            r.get("audio")
            or r.get("audio_path")
            or r.get("path")
            or r.get("file")
            or r.get("mp3")
        )
        if not eid or not audio:
            continue
        p = Path(audio)
        if not p.is_absolute():
            p = REPO / p
        rows.append({"id": str(eid), "audio": p, "meta": r})
    return rows


def annotate_chunk(gw: Gateway, audio_path: Path, model: str) -> dict:
    msgs = [
        text_message("system", SYSTEM),
        audio_message(audio_path, PROMPT),
    ]
    obj = gw.chat_json(msgs, model=model)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    def _txt(key: str) -> str:
        # Model writes gershayim as ״ to keep JSON valid; store as ASCII ".
        return (obj.get(key) or "").strip().replace("״", '"')

    return {
        "text_yi": _txt("text_yi"),
        "text_yi_pointed": _txt("text_yi_pointed"),
        "ipa": _txt("ipa"),
        "confidence": obj.get("confidence"),
        "notes": (obj.get("notes") or "").strip(),
    }


def annotate_episode(
    gw: Gateway,
    eid: str,
    audio: Path,
    model: str,
    chunk_s: float,
    max_chunks: int | None,
    out_dir: Path = ANNOT_DIR,
) -> Path:
    out_path = out_dir / f"{eid}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        done = {r["chunk_idx"] for r in iter_jsonl(out_path)}

    chunks = chunk_mp3(audio, CHUNK_DIR / eid, chunk_s=chunk_s)
    if max_chunks:
        chunks = chunks[:max_chunks]

    with open(out_path, "a", encoding="utf-8") as fh:
        for ch in chunks:
            if ch.idx in done:
                continue
            try:
                res = annotate_chunk(gw, ch.path, model)
            except (GatewayError, ValueError, json.JSONDecodeError) as exc:
                print(f"  [{eid} #{ch.idx}] FAILED: {exc}", file=sys.stderr)
                continue
            rec = {
                "chunk_idx": ch.idx,
                "start_s": round(ch.start_s, 2),
                "end_s": round(ch.end_s, 2),
                "text_yi": res["text_yi"],
                "text_yi_pointed": res["text_yi_pointed"],
                "ipa": res["ipa"],
                "confidence": res["confidence"],
                "notes": res["notes"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{eid} #{ch.idx}] {rec['text_yi'][:60]}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=None, help="jsonl manifest of episodes")
    ap.add_argument("--episode", action="append", help="only these episode ids")
    ap.add_argument("--model", default=MODEL_FLASH)
    ap.add_argument("--chunk-s", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None, help="max episodes")
    ap.add_argument("--max-chunks", type=int, default=None, help="max chunks per episode")
    ap.add_argument("--out-dir", default=str(ANNOT_DIR), help="annotation output directory")
    args = ap.parse_args()

    manifest = Path(args.manifest) if args.manifest else None
    if manifest is None:
        for cand in (DATA / "audio_manifest.jsonl", DATA / "episodes.jsonl"):
            if cand.exists():
                manifest = cand
                break
    if manifest is None or not manifest.exists():
        print(
            f"No manifest found. Expected {DATA/'audio_manifest.jsonl'} or "
            f"{DATA/'episodes.jsonl'} (the scraper writes these).",
            file=sys.stderr,
        )
        return 2

    require_ffmpeg()
    rows = load_manifest(manifest)
    if args.episode:
        want = set(args.episode)
        rows = [r for r in rows if r["id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("nothing to do", file=sys.stderr)
        return 1

    gw = Gateway()
    print(f"{len(rows)} episode(s), model={args.model}, manifest={manifest}")
    for r in rows:
        if not r["audio"].exists():
            print(f"  [{r['id']}] missing audio {r['audio']}", file=sys.stderr)
            continue
        print(f"episode {r['id']}")
        out = annotate_episode(
            gw, r["id"], r["audio"], args.model, args.chunk_s, args.max_chunks,
            out_dir=Path(args.out_dir),
        )
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
