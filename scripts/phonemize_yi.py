#!/usr/bin/env python
"""Re-phonemize the Yiddish corpus with Gemini under a rule-pinned prompt.

WHY THIS EXISTS: the `ipa` column in data/annotations was written by the audio
annotation prompt, which told the model to transcribe "the ACTUAL pronunciation
... where the speaker does so". That instruction produced labels that disagree
with themselves: האָט appears as hut, hot AND hat; וואָס as vus, vos AND vas. Of
44,685 words whose only vowel is a komets, the stored IPA realises it as /o/
49.7% of the time, /a/ 5.7% and /u/ only 2.6%. The inventory drifted to 200+
symbols including Cyrillic, Thai, katakana and bare Hebrew letters. A TTS model
trained on that learns to pick a dialect at random -- which is exactly what the
released voice does.

This script replaces those labels using scripts/prompts/yiddish_phonemize.txt,
which pins the vowel correspondences (komets->u, shurek->i, ...), a closed
phoneme inventory and the stress rules, so the same word always phonemizes the
same way. The phonology still comes from the model's own knowledge of Hasidic
Yiddish; the prompt only fixes the conventions it must apply consistently.

Text in, text out -- the audio is NOT re-transcribed. The transcripts were fine;
only the phonemization was broken, and re-running audio would cost ~100x more
and would re-open transcription that is already aligned to the chunk timings.

Output: data/phonemized/<episode_id>.jsonl, one record per source chunk:
  {chunk_idx, text_yi, pointed, ipa, ok, problems}
Rows failing validation are kept with ok=false so they can be inspected or
re-run rather than silently poisoning the training set.

Usage:
  .venv/bin/python scripts/phonemize_yi.py --limit 2 --dry-run
  .venv/bin/python scripts/phonemize_yi.py --limit 20
  .venv/bin/python scripts/phonemize_yi.py --model google/gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonikud_yi.gateway import Gateway, GatewayError, text_message  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ANNOT_DIR = REPO / "data" / "annotations"
OUT_DIR = REPO / "data" / "phonemized"
PROMPT_PATH = REPO / "scripts" / "prompts" / "yiddish_phonemize.txt"

# "gemini lite" on the gateway. Verified present via GET /models.
MODEL_LITE = "google/gemini-3.5-flash-lite"

SYSTEM = (
    "You are an expert phonologist of Central (Poylish/Hungarian) Hasidic Yiddish. "
    "You apply a fixed set of transcription conventions mechanically and identically "
    "every time. You never vary a transcription for naturalness."
)

# Must match the inventory block in the prompt. Anything outside this set in the
# phoneme column is a hallucinated symbol and the row is rejected. ɪ is here as
# the second element of the diphthongs aɪ and ɔɪ; it never stands alone.
PHONEMES = set("a ɛ i u ɔ ə ɪ b d f ɡ h j k l m n p r s t v x z ʃ ʒ ʦ ʧ ʣ ʤ".split())
ALLOWED_EXTRA = set(" ˈ-.,!?;:'״\"()")
_HEBREW = re.compile(r"[֐-׿]")

# Pure glyph slips: same phoneme, wrong codepoint. Normalising these is safe and
# keeps a good row from being thrown away. Vowel confusions (e/o for ɛ/ə/u/ɔ) are
# deliberately NOT normalised -- those are the dialect errors we are hunting.
GLYPH_FIXES = {"g": "ɡ", "ɹ": "r", "ʀ": "r", "ʁ": "r", "ŋ": "n", "ʣ": "ʣ"}


def normalize_glyphs(ipa: str) -> str:
    return "".join(GLYPH_FIXES.get(c, c) for c in ipa)


def strip_points(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def validate(src: str, pointed: str, ipa: str) -> list[str]:
    """Return a list of problems; empty means the row is usable."""
    problems: list[str] = []

    if not pointed.strip() or not ipa.strip():
        problems.append("empty")
        return problems

    # The pointed text must be the source text plus marks -- nothing added,
    # dropped or reordered. This catches the model paraphrasing or translating.
    # Strip BOTH sides: the source transcripts are not bare, they already carry
    # partial pointing (האַלט, דאָ, אָן), so comparing against raw src flags every row.
    if strip_points(pointed).split() != strip_points(src).split():
        problems.append("pointed-text-differs-from-source")

    if len(pointed.split()) != len(ipa.split()):
        problems.append(
            f"token-mismatch pointed={len(pointed.split())} ipa={len(ipa.split())}"
        )

    # No Hebrew letters may survive into the phoneme column. The old labels
    # leaked untranscribed words (זיין appeared 204x in the IPA field).
    if _HEBREW.search(ipa):
        problems.append("hebrew-letters-in-ipa")

    bad = {c for c in ipa if c not in PHONEMES and c not in ALLOWED_EXTRA}
    if bad:
        problems.append("out-of-inventory:" + "".join(sorted(bad)))

    # Every multi-syllable token carries exactly one stress mark. Hyphen-joined
    # loshn-koydesh phrases (bˈurəx-haʃˈɛm) are one token but several words, so
    # each hyphen-separated part is checked on its own.
    vowels = set("aɛiuɔə")
    for tok in ipa.split():
        for core in tok.strip("".join(ALLOWED_EXTRA - {"ˈ"})).split("-"):
            if not core:
                continue
            nvow = sum(1 for c in core if c in vowels)
            nstress = core.count("ˈ")
            if (nvow >= 2 and nstress != 1) or (nvow <= 1 and nstress > 1):
                problems.append(f"stress({core})={nstress}")

    return problems


def parse_tsv(raw: str, ids: list[str]) -> dict[str, tuple[str, str]]:
    """Parse the model's TSV reply into {id: (pointed, ipa)}."""
    want = set(ids)
    out: dict[str, tuple[str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rid = parts[0].strip()
        if rid in want:
            out[rid] = (parts[1].strip(), normalize_glyphs(parts[2].strip()))
    return out


# Sentence-ish split. A 30s chunk runs ~100 words, and at that length the model
# drops and merges tokens: batching whole chunks scored 4/71 clean, where the
# same prompt over sentences scores far higher. Alignment is per row, so shorter
# rows also mean a bad row costs less.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
MAX_WORDS = 25


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        # Long run-on stretches with no final punctuation still have to be cut,
        # so fall back to splitting on commas, then on a hard word count.
        if len(words) <= MAX_WORDS:
            out.append(sent)
            continue
        piece: list[str] = []
        for w in words:
            piece.append(w)
            if (w.endswith(",") and len(piece) >= 8) or len(piece) >= MAX_WORDS:
                out.append(" ".join(piece))
                piece = []
        if piece:
            out.append(" ".join(piece))
    return out


def load_chunks(path: Path) -> list[dict]:
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
                rows.append(
                    {
                        "chunk_idx": d.get("chunk_idx"),
                        "sent_idx": si,
                        "id": f"{d.get('chunk_idx')}.{si}",
                        "text_yi": sent,
                    }
                )
    return rows


def phonemize_batch(
    gw: Gateway, prompt: str, rows: list[dict], model: str
) -> dict[str, tuple[str, str]]:
    ids = [r["id"] for r in rows]
    body = "\n".join(f"{i}\t{r['text_yi']}" for i, r in zip(ids, rows))
    msgs = [text_message("system", SYSTEM), text_message("user", prompt + body)]
    raw = gw.chat(msgs, model=model)
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    return parse_tsv(raw, ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max episodes")
    ap.add_argument("--episode", action="append", help="episode id (repeatable)")
    ap.add_argument("--model", default=MODEL_LITE)
    ap.add_argument("--batch", type=int, default=10, help="sentences per request")
    ap.add_argument("--dry-run", action="store_true", help="print one batch, send nothing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    files = sorted(ANNOT_DIR.glob("*.jsonl"))
    if args.episode:
        keep = set(args.episode)
        files = [f for f in files if f.stem in keep]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("no episodes matched", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gw = None if args.dry_run else Gateway()

    totals = {"rows": 0, "ok": 0, "missing": 0}
    problem_counts: dict[str, int] = {}

    for fp in files:
        out_path = OUT_DIR / fp.name
        if out_path.exists() and not args.overwrite:
            print(f"skip {fp.stem} (exists)")
            continue
        rows = load_chunks(fp)
        if not rows:
            continue

        results = []
        for i in range(0, len(rows), args.batch):
            batch = rows[i : i + args.batch]
            if args.dry_run:
                ids = [r["id"] for r in batch]
                body = "\n".join(f"{j}\t{r['text_yi']}" for j, r in zip(ids, batch))
                print(prompt + body)
                return 0
            try:
                got = phonemize_batch(gw, prompt, batch, args.model)
            except GatewayError as exc:
                print(f"  {fp.stem} batch {i}: {exc}", file=sys.stderr)
                got = {}
            for r in batch:
                rid = r["id"]
                pointed, ipa = got.get(rid, ("", ""))
                problems = (
                    ["missing-from-reply"]
                    if rid not in got
                    else validate(r["text_yi"], pointed, ipa)
                )
                for p in problems:
                    key = p.split(":")[0].split("(")[0]
                    problem_counts[key] = problem_counts.get(key, 0) + 1
                totals["rows"] += 1
                totals["ok"] += not problems
                totals["missing"] += rid not in got
                results.append(
                    {
                        "chunk_idx": r["chunk_idx"],
                        "sent_idx": r["sent_idx"],
                        "text_yi": r["text_yi"],
                        "pointed": pointed,
                        "ipa": ipa,
                        "ok": not problems,
                        "problems": problems,
                    }
                )

        with out_path.open("w", encoding="utf-8") as fh:
            for rec in results:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        good = sum(r["ok"] for r in results)
        print(f"{fp.stem}: {good}/{len(results)} clean -> {out_path.name}")

    if totals["rows"]:
        print(
            f"\ntotal {totals['ok']}/{totals['rows']} clean "
            f"({totals['ok'] / totals['rows']:.1%}), {totals['missing']} missing"
        )
        for k, v in sorted(problem_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
