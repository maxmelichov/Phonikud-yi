#!/usr/bin/env python
"""Build a bare-word -> pointed-word lexicon for the Yiddish diacritics corpus.

WHY A LEXICON AND NOT PER-LINE POINTING: the corpus is 805,704 word instances but
only 45,073 distinct types, and 13,404 of those cover 95% of instances. Pointing
each TYPE once makes the dataset consistent by construction -- האט cannot come
back 18 different ways, the way it does in the current data, because the choice
is made exactly once. It is also ~18x less work than pointing every instance.

The source is data/diacritics_r3c/*.txt (plain text). The audio, the chunks and
data/annotations are NOT touched.

RE-RUNNING IS FREE. The lexicon is an append-only cache keyed by bare word and is
flushed after every batch. A second run sends only types that are not in it yet,
so an interrupted run resumes where it stopped, and a completed run costs nothing
to repeat. Use --refresh to deliberately re-decide words already in the cache.

Homographs (פאר = far vs pur) cannot be settled from a word in isolation. The
model flags them; they are stored with ambiguous=true and an `alt` pointing, and
are left for a context pass rather than silently forced to one reading.

Output: data/nikud_lexicon/
  lexicon.jsonl   one record per type: bare, pointed, ambiguous, alt, ok, problems
  stats.json      coverage and rejection counts

Usage:
  .venv/bin/python scripts/build_nikud_lexicon.py --plan          # no API calls
  .venv/bin/python scripts/build_nikud_lexicon.py --min-count 3
  .venv/bin/python scripts/build_nikud_lexicon.py                 # every type
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

from phonikud_yi.gateway import Gateway, GatewayError, text_message  # noqa: E402
from scripts.nikud_yi import canon  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "data" / "diacritics_r3c"
OUT_DIR = REPO / "data" / "nikud_lexicon"
LEXICON = OUT_DIR / "lexicon.jsonl"
STATS = OUT_DIR / "stats.json"
PROMPT_PATH = REPO / "scripts" / "prompts" / "yiddish_nikud_types.txt"

MODEL = "google/gemini-3.5-flash"
SYSTEM = (
    "You are an expert in Yiddish orthography and the Hasidic pronunciation "
    "tradition. You point each word the same way every time and never change, "
    "drop or add a letter."
)

_HEBREW = re.compile(r"[א-ת]")
_STRIP_PUNCT = ".,!?;:\"'()[]{}—–-«»„“”…"


def bare_of(word: str) -> str:
    """The word with all diacritics and edge punctuation removed."""
    w = "".join(
        c for c in unicodedata.normalize("NFD", word)
        if unicodedata.category(c) != "Mn"
    )
    return w.strip(_STRIP_PUNCT)


def corpus_types() -> collections.Counter:
    types: collections.Counter = collections.Counter()
    for name in ("train", "val", "test"):
        path = SRC_DIR / f"{name}.txt"
        if not path.exists():
            continue
        for word in path.read_text(encoding="utf-8").split():
            b = bare_of(word)
            if b and _HEBREW.search(b):
                types[b] += 1
    return types


def load_lexicon() -> dict[str, dict]:
    """Read the append-only cache. Later records win, so --refresh can rewrite."""
    lex: dict[str, dict] = {}
    if LEXICON.exists():
        for line in LEXICON.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            lex[rec["bare"]] = rec
    return lex


def parse_reply(raw: str, batch: list[str]) -> list[dict]:
    """Map the model's TSV rows back onto the words we sent, by index."""
    rows: dict[int, tuple[str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            idx = int(parts[0].strip())
            flag = parts[2].strip() if len(parts) > 2 else "-"
            rows[idx] = (parts[1].strip(), flag)

    out = []
    for i, word in enumerate(batch):
        if i not in rows:
            out.append({"bare": word, "pointed": "", "ambiguous": False,
                        "alt": "", "ok": False, "problems": ["missing-from-reply"]})
            continue
        pointed, flag = rows[i]
        problems = []
        # The whole point of the lexicon is that the letters never move; a word
        # whose letters changed is rejected rather than repaired, because at type
        # level there is no context to repair it against.
        if canon(pointed) != canon(word):
            problems.append("letters-differ")
        if not any(unicodedata.category(c) == "Mn"
                   for c in unicodedata.normalize("NFD", pointed)):
            # Acronyms legitimately come back unpointed; everything else should not.
            if '"' not in word and "'" not in word:
                problems.append("no-diacritics")
        ambiguous = flag.upper().startswith("AMBIG")
        alt = flag.split(":", 1)[1].strip() if ambiguous and ":" in flag else ""
        out.append({"bare": word, "pointed": pointed, "ambiguous": ambiguous,
                    "alt": alt, "ok": not problems, "problems": problems})
    return out


def run_batch(gw: Gateway, prompt: str, batch: list[str]) -> list[dict]:
    body = "\n".join(f"{i}\t{w}" for i, w in enumerate(batch))
    raw = gw.chat(
        [text_message("system", SYSTEM), text_message("user", prompt + body)],
        model=MODEL,
    )
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    return parse_reply(raw, batch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=1,
                    help="only point types occurring at least this often")
    ap.add_argument("--batch", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many new types to send (for a costed trial)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-decide types already cached")
    ap.add_argument("--plan", action="store_true",
                    help="report what would be sent and exit without calling the API")
    args = ap.parse_args()

    types = corpus_types()
    if not types:
        print(f"no text found under {SRC_DIR}", file=sys.stderr)
        return 1
    total_inst = sum(types.values())
    wanted = [w for w, c in types.most_common() if c >= args.min_count]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lex = load_lexicon()
    todo = wanted if args.refresh else [w for w in wanted if w not in lex]
    if args.limit:
        todo = todo[: args.limit]

    covered = sum(types[w] for w in wanted)
    print(f"corpus     : {total_inst:,} instances / {len(types):,} types")
    print(f"selected   : {len(wanted):,} types (min-count {args.min_count}) "
          f"= {covered / total_inst:.2%} of instances")
    print(f"cached     : {len(lex):,} already in {LEXICON.name}")
    print(f"to request : {len(todo):,} types "
          f"-> {(len(todo) + args.batch - 1) // args.batch:,} requests")
    if args.plan or not todo:
        if not todo and not args.plan:
            print("nothing to do; lexicon is already complete for this selection")
        return 0

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    gw = Gateway()
    lock = threading.Lock()
    done = {"batches": 0, "ok": 0, "bad": 0}

    # Append + flush per batch so an interrupted run keeps everything it earned.
    with LEXICON.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_batch, gw, prompt, b): b for b in batches}
            for fut in as_completed(futs):
                try:
                    recs = fut.result()
                except (GatewayError, Exception) as exc:  # noqa: BLE001
                    print(f"  batch failed: {exc}", file=sys.stderr)
                    continue
                with lock:
                    for rec in recs:
                        rec["count"] = types.get(rec["bare"], 0)
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done["ok" if rec["ok"] else "bad"] += 1
                    fh.flush()
                    done["batches"] += 1
                    print(f"  [{done['batches']}/{len(batches)}] "
                          f"ok={done['ok']:,} rejected={done['bad']:,}", flush=True)

    lex = load_lexicon()
    good = {w: r for w, r in lex.items() if r["ok"]}
    amb = [w for w, r in good.items() if r["ambiguous"]]
    stats = {
        "corpus_instances": total_inst,
        "corpus_types": len(types),
        "lexicon_types": len(lex),
        "lexicon_ok": len(good),
        "lexicon_rejected": len(lex) - len(good),
        "ambiguous_types": len(amb),
        "instance_coverage": sum(types.get(w, 0) for w in good) / total_inst,
        "ambiguous_instance_share": sum(types.get(w, 0) for w in amb) / total_inst,
        # One pointing per bare word is guaranteed by the dict, so type
        # consistency is 1.0 by construction rather than by measurement.
        "type_consistency": 1.0,
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nlexicon {len(good):,} usable / {len(lex):,} decided")
    print(f"instance coverage {stats['instance_coverage']:.2%}, "
          f"{len(amb):,} ambiguous types "
          f"({stats['ambiguous_instance_share']:.2%} of instances)")
    print(f"wrote {LEXICON} and {STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
