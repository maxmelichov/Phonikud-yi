#!/usr/bin/env python3
"""
Canonicalize the nikud pointing of the Hasidic-Yiddish diacritics corpus.

Yiddish pronunciation is lexical: one word type -> one pronunciation -> one
correct pointed form (rare homographs aside).  The Gemini annotations, however,
point the same word inconsistently -- mark placement inside digraphs
(וַואיל / וואַיל), full vs partial pointing (אוֹפֶן / אופֿן), doubled marks, etc.

This script derives ONE canonical pointed form per word type from the *train*
split and writes it to data/corpus/canonical_pointing.tsv.  scripts/apply_canonical.py
then rewrites train/val/test with that map.

Stages
  1. COLLECT   word type (bare, unpointed) -> Counter of pointed variants, over
               data/diacritics_r2/train.txt only.
  2. MECHANICAL deterministic repairs that need no model: dedup repeated marks,
               drop colliding same-class vowels, and re-attach vowels to the
               canonical slot inside vowel-letter digraphs.  Variant counters
               are re-merged afterwards.
  3. GEMINI    word types still holding >=2 variants with total count >=3 are
               batched (~40/call) to google/gemini-3.6-flash for adjudication.
               Results are cached to data/corpus/canonical_cache.jsonl (resumable).
  4. OUTPUT    data/corpus/canonical_pointing.tsv + data/candidates/homographs.tsv.

Usage
    python scripts/canonicalize_pointing.py --stage mechanical   # no LLM calls
    python scripts/canonicalize_pointing.py                      # full run
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ------------------------------------------------------------------ inventory

NIQQUD = "ְֱֲֳִֵֶַָׇֹֺֻ"  # vowel points (incl. qamats qatan U+05C7)
DAGESH = "ּ"  # U+05BC
RAFE = "ֿ"  # U+05BF
SHIN_DOT, SIN_DOT = "ׁ", "ׂ"  # U+05C1 / U+05C2
MARKS = set(NIQQUD) | {DAGESH, RAFE, SHIN_DOT, SIN_DOT}
MARK_RE = re.compile("[" + re.escape("".join(sorted(MARKS))) + "]")

# mark -> equivalence class; two marks of the same class cannot coexist on one letter
MARK_CLASS = {c: "vowel" for c in NIQQUD}
MARK_CLASS[DAGESH] = "dagesh"
MARK_CLASS[RAFE] = "rafe"
MARK_CLASS[SHIN_DOT] = "sindot"
MARK_CLASS[SIN_DOT] = "sindot"

HEB_LETTER = re.compile(r"[א-ת]")
VOWEL_LETTERS = set("אוי")  # digraph carriers: shtumer alef, vov, yud
PATAH, QAMATS = "ַ", "ָ"

# folding identical to scripts/prepare_diacritics_dataset.py
FOLD = {
    "װ": "וו", "ױ": "וי", "ײ": "יי",
    "׳": "'", "״": '"', "־": "-",
    "‎": "", "‏": "", "‍": "", "‌": "", "﻿": "",
}

# stripped from both ends of a whitespace token; ' and " survive *inside* a word
EDGE_PUNCT = " \t.,!?;:()[]{}<>\"'«»„“”‚‘’…-–—/\\|*_~`+=%$#@&^־ "


def fold(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for src, dst in FOLD.items():
        text = text.replace(src, dst)
    return text


def strip_marks(text: str) -> str:
    return MARK_RE.sub("", text)


def split_token(tok: str) -> tuple[str, str, str]:
    """token -> (leading punct, core, trailing punct)."""
    i, j = 0, len(tok)
    while i < j and tok[i] in EDGE_PUNCT:
        i += 1
    while j > i and tok[j - 1] in EDGE_PUNCT:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


def has_hebrew(s: str) -> bool:
    return bool(HEB_LETTER.search(s))


# ------------------------------------------------------------ letter clusters


def clusters(word: str) -> list[tuple[str, str]]:
    """Split a word into (base char, marks) pairs, preserving order."""
    out: list[list[str]] = []
    for ch in word:
        if ch in MARKS and out:
            out[-1][1] += ch
        else:
            out.append([ch, ""])
    return [(b, m) for b, m in out]


def join(cl: Iterable[tuple[str, str]]) -> str:
    return "".join(b + m for b, m in cl)


# ------------------------------------------------------------ stage 2: rules


def dedup_marks(cl: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop repeated identical marks, and the 2nd of two same-class marks."""
    out = []
    for base, marks in cl:
        seen_chars: set[str] = set()
        seen_class: set[str] = set()
        keep = []
        for m in marks:
            if m in seen_chars:
                continue  # exact duplicate, e.g. פֿֿ
            klass = MARK_CLASS.get(m, m)
            if klass in seen_class:
                continue  # same-class collision, e.g. two vowels on one letter
            seen_chars.add(m)
            seen_class.add(klass)
            keep.append(m)
        # Unicode canonical order (== NFC): vowel < dagesh < rafe < shin/sin dot
        keep.sort(key=unicodedata.combining)
        out.append((base, "".join(keep)))
    return out


def n_vowels(word: str) -> int:
    """Number of vocalic marks.

    A niqqud point counts, and so does a dagesh sitting on a vov -- that is the
    melupm vov וּ (/u/), the *only* vowel many Yiddish words carry (אוּן, צוּ, דוּ).
    Missing this case makes an otherwise-pointed word look unpointed.
    """
    n = 0
    for base, marks in clusters(word):
        if any(MARK_CLASS.get(m) == "vowel" for m in marks):
            n += 1
        elif base == "ו" and DAGESH in marks:
            n += 1
    return n


def _vowel_of(marks: str) -> str:
    for m in marks:
        if MARK_CLASS.get(m) == "vowel":
            return m
    return ""


def reattach_digraphs(cl: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Enforce one placement convention inside runs of vowel letters (א ו י).

    * a non-run-initial א owns the patah/qamats of its run  (וַואיל -> וואַיל,
      וָואס -> וואָס);  a *run-initial* alef is the shtumer alef of אויף/אויב and
      is left alone.
    * a pure וו / יי run carries its vowel on the SECOND letter
      (matches the ligature folding ײַ -> ייַ, װאַ -> וואַ):  וֶוען -> ווֶען, יִיד -> ייִד
    * the same vowel smeared over several letters of one run (וָואָס, וֶוֶען) is
      collapsed onto that one slot.
    Runs carrying two *different* vowels, or none, are left alone -- those are
    real disagreements for the adjudicator, not placement noise.
    """
    cl = list(cl)
    n = len(cl)

    def put(k: int, v: str) -> None:
        b, m = cl[k]
        cl[k] = (b, "".join(sorted(m + v, key=unicodedata.combining)))

    def drop(k: int, v: str) -> None:
        b, m = cl[k]
        cl[k] = (b, m.replace(v, "", 1))

    # -- pass A: inside a run of vowel letters, an internal alef owns patah/qamats
    i = 0
    while i < n:
        if cl[i][0] not in VOWEL_LETTERS:
            i += 1
            continue
        j = i
        while j < n and cl[j][0] in VOWEL_LETTERS:
            j += 1
        run = list(range(i, j))
        alefs = [k for k in run[1:] if cl[k][0] == "א"]
        if len(run) >= 2 and alefs:
            voweled = [(k, _vowel_of(cl[k][1])) for k in run if _vowel_of(cl[k][1])]
            distinct = {v for _, v in voweled}
            if len(distinct) == 1:
                v = distinct.pop()
                tgt = alefs[0]
                if v in (PATAH, QAMATS) and [k for k, _ in voweled] != [tgt]:
                    for k, _ in voweled:
                        drop(k, v)
                    put(tgt, v)
        i = j

    # -- pass B: in an identical pair (וו, יי) the vowel sits on the SECOND letter
    for k in range(n - 1):
        a, b = cl[k][0], cl[k + 1][0]
        if a != b or a not in VOWEL_LETTERS:
            continue
        v1, v2 = _vowel_of(cl[k][1]), _vowel_of(cl[k + 1][1])
        if v1 and (not v2 or v1 == v2):
            drop(k, v1)
            if not v2:
                put(k + 1, v1)
    return cl


def mechanical(word: str) -> str:
    cl = clusters(word)
    cl = dedup_marks(cl)
    cl = reattach_digraphs(cl)
    return unicodedata.normalize("NFC", join(cl))


# -------------------------------------------------------------- stage 1: read


def tokenize_line(line: str) -> list[str]:
    out = []
    for tok in fold(line).split():
        _, core, _ = split_token(tok)
        if core and has_hebrew(core):
            out.append(core)
    return out


def collect(path: Path) -> dict[str, collections.Counter]:
    types: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            for core in tokenize_line(line):
                types[strip_marks(core)][core] += 1
    return types


def collect_contexts(path: Path, wanted: set[str], per_word: int = 2) -> dict[str, list[str]]:
    """A couple of short usage contexts per wanted bare word type."""
    ctx: dict[str, list[str]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            toks = tokenize_line(line)
            bares = [strip_marks(t) for t in toks]
            for idx, bare in enumerate(bares):
                if bare in wanted and len(ctx[bare]) < per_word:
                    lo, hi = max(0, idx - 3), min(len(toks), idx + 4)
                    snippet = " ".join(toks[lo:hi])
                    if snippet not in ctx[bare]:
                        ctx[bare].append(snippet)
    return ctx


# ---------------------------------------------------------------- consistency


def consistency(types: dict[str, collections.Counter], min_count: int = 3) -> dict:
    """Self-consistency of the pointing over word types seen >= min_count."""
    elig = {b: c for b, c in types.items() if sum(c.values()) >= min_count}
    if not elig:
        return {}
    n_types = len(elig)
    single = sum(1 for c in elig.values() if len(c) == 1)
    inst_tot = sum(sum(c.values()) for c in elig.values())
    inst_major = sum(c.most_common(1)[0][1] for c in elig.values())
    return {
        "types_ge_min": n_types,
        "types_single_variant": single,
        "type_consistency": single / n_types,
        "instance_consistency": inst_major / inst_tot,
        "mean_variants_per_type": sum(len(c) for c in elig.values()) / n_types,
        "mean_variants_all_types": sum(len(c) for c in types.values()) / len(types),
    }


# ------------------------------------------------------------- stage 3: gemini

SYSTEM = """You are an expert in Hasidic Yiddish orthography and nikud (pointing).

Task: for each word type you are given several pointed spellings that a noisy
annotator produced for the SAME word, plus their frequencies and one or two
usage contexts. Choose the ONE correct canonical pointed form.

Rules:
- Hasidic Yiddish, FULL nikud pointing style: EVERY syllable carries a vowel
  point. Never return a bare or partially pointed form -- if the most frequent
  variant is unpointed or half-pointed, return the fully pointed form even when
  it appears in none of the listed variants. This holds even when the vowel is
  already carried by a vowel letter: אוּן, פֿוּן, צוּ, אוֹיף, ווִי, שוֹין, מִיט -- never
  און, פון, צו, אויף, ווי, שוין, מיט.
- Words of loshn-koydesh (Hebrew/Aramaic) origin keep their traditional Hebrew
  nikud exactly (e.g. שִׂמְחָה, הֲלָכוֹת, בְּרָכָה), including dagesh and shin/sin dots.
- Germanic/Slavic-origin Yiddish words: pasekh alef אַ, komets alef אָ, melupm vov
  וּ, khirik yud יִ; rafe on פֿ בֿ כֿ where the pronunciation is fricative.
- Digraph placement conventions (already enforced elsewhere -- follow them):
  * tsvey vovn/tsvey yudn: the vowel point goes on the SECOND letter of the pair
    (ווִיל, ווֶען, ייִד -- never וִויל, וֶוען, יִיד).
  * when a vowel alef follows the pair, the pasekh/komets goes on the ALEF
    (וואָס, וואַיל -- never וָואס, וַואיל).
  * the diphthong /ay/ spelled ייַ takes its pasekh on the second yud
    (זייַן, מייַן -- never זַיין, מַיין).
- The consonant skeleton MUST stay exactly as given (same letters, same order);
  only the diacritics may change. Never add or remove letters.
- Yiddish pronunciation is lexical: one word, one pronunciation, one pointing.
  Only if the bare spelling really covers TWO different words with different
  pronunciations (a true homograph) return the homograph form instead.

Output STRICT JSON, no prose, no markdown fence:
{"results": {"<bare word>": {"canonical": "<pointed form>"}, ...}}
For a genuine homograph use {"homograph": ["<form1>", "<form2>"]} instead.
Include every word you were given, keyed by its bare (unpointed) spelling."""


def build_user_msg(batch: list[tuple[str, collections.Counter]], ctx: dict[str, list[str]]) -> str:
    lines = []
    for bare, cnt in batch:
        vs = ", ".join(f"{v} ({n})" for v, n in cnt.most_common(6))
        lines.append(f"- {bare}\n  variants: {vs}")
        for c in ctx.get(bare, [])[:2]:
            lines.append(f"  context: …{c}…")
    return "Words to adjudicate:\n" + "\n".join(lines)


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "word" in rec:
                    cache[rec["word"]] = rec
    return cache


# google/gemini-3.6-flash list price, $ per 1M tokens (for the spend estimate).
PRICE_IN, PRICE_OUT = 0.30, 2.50


def adjudicate(
    todo: list[tuple[str, collections.Counter]],
    ctx: dict[str, list[str]],
    cache_path: Path,
    batch_size: int = 40,
    limit_batches: int | None = None,
    workers: int = 6,
) -> tuple[dict[str, dict], dict]:
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from phonikud_yi.gateway import Gateway, MODEL_FLASH, parse_json_loose, text_message

    cache = load_cache(cache_path)
    todo = [(b, c) for b, c in todo if b not in cache]
    gw = Gateway()
    usage = collections.Counter()
    lock = threading.Lock()
    n_bad = 0
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    if limit_batches:
        batches = batches[:limit_batches]
    fh = cache_path.open("a", encoding="utf-8")
    t0 = time.time()
    done = 0

    def run_batch(item: tuple[int, list]) -> None:
        nonlocal n_bad, done
        bi, batch = item
        msgs = [text_message("system", SYSTEM), text_message("user", build_user_msg(batch, ctx))]
        payload = {
            "model": MODEL_FLASH,
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        # NB: the gateway rejects response_format for google/gemini-3.6-flash
        # ("400 Invalid input, param: response_format"), so we rely on the
        # prompt + parse_json_loose instead.
        effort = __import__("os").environ.get("GATEWAY_REASONING_EFFORT")
        if effort:
            payload["reasoning_effort"] = effort
        try:
            raw = gw._post("/chat/completions", payload)
            u = raw.get("usage") or {}
            with lock:
                usage["prompt"] += int(u.get("prompt_tokens") or 0)
                usage["completion"] += int(u.get("completion_tokens") or 0)
            data = parse_json_loose(raw["choices"][0]["message"]["content"] or "")
        except Exception as exc:  # noqa: BLE001
            print(f"  batch {bi}: FAILED {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
            with lock:
                n_bad += len(batch)
                done += 1
            return
        res = data.get("results") if isinstance(data, dict) else None
        if not isinstance(res, dict):
            res = data if isinstance(data, dict) else {}
        # tolerate keys returned pointed instead of bare
        res = {strip_marks(fold(str(k))): v for k, v in res.items()}
        recs, bad = [], 0
        for bare, cnt in batch:
            r = res.get(bare)
            if not isinstance(r, dict):
                bad += 1
                continue
            rec = {"word": bare}
            if isinstance(r.get("homograph"), list) and len(r["homograph"]) >= 2:
                forms = [fold(str(x)) for x in r["homograph"]]
                if not all(strip_marks(f) == bare for f in forms):
                    bad += 1
                    continue
                rec["homograph"] = [mechanical(f) for f in forms]
            else:
                canon = fold(str(r.get("canonical") or ""))
                if not canon or strip_marks(canon) != bare:
                    bad += 1  # hallucinated a different skeleton
                    continue
                # reject a bare answer when the corpus knows a pointed form:
                # the model fell back on an unpointed majority variant.
                if not n_vowels(canon) and any(n_vowels(v) for v in cnt):
                    rec["reject"] = "unpointed"
                    bad += 1
                else:
                    rec["canonical"] = mechanical(canon)
            recs.append(rec)
        with lock:
            n_bad += bad
            done += 1
            for rec in recs:
                cache[rec["word"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            usage["batches"] += 1
            if done % 20 == 0 or done == len(batches):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(batches) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(batches)} batches  {el:.0f}s  eta {eta:.0f}s  "
                      f"bad={n_bad}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_batch, enumerate(batches, 1)))
    fh.close()
    cost = usage["prompt"] / 1e6 * PRICE_IN + usage["completion"] / 1e6 * PRICE_OUT
    return cache, {
        "batches": usage["batches"],
        "unresolved": n_bad,
        "prompt_tokens": usage["prompt"],
        "completion_tokens": usage["completion"],
        "cost_usd": round(cost, 3),
    }


# --------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=REPO / "data/diacritics_r2/train.txt")
    ap.add_argument("--out-tsv", type=Path, default=REPO / "data/corpus/canonical_pointing.tsv")
    ap.add_argument("--homographs", type=Path, default=REPO / "data/candidates/homographs.tsv")
    ap.add_argument("--cache", type=Path, default=REPO / "data/corpus/canonical_cache.jsonl")
    ap.add_argument("--stage", choices=["collect", "mechanical", "all"], default="all")
    ap.add_argument("--min-count", type=int, default=3, help="min total count for LLM adjudication")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print(f"[1] COLLECT  {args.train}")
    raw = collect(args.train)
    n_inst = sum(sum(c.values()) for c in raw.values())
    before = consistency(raw)
    print(f"    word types: {len(raw):,}   instances: {n_inst:,}")
    print(f"    multi-variant types: {sum(1 for c in raw.values() if len(c) > 1):,}")
    print(f"    mean variants/type: {before['mean_variants_all_types']:.3f}")
    print(f"    types seen >=3x: {before['types_ge_min']:,}")
    print(f"    self-consistency (types, >=3x): {100 * before['type_consistency']:.1f}%")
    print(f"    self-consistency (instances, >=3x): {100 * before['instance_consistency']:.1f}%")
    if args.stage == "collect":
        return 0

    print("\n[2] MECHANICAL")
    folded: dict[str, collections.Counter] = {}
    n_changed_mech = 0
    for bare, cnt in raw.items():
        m: collections.Counter = collections.Counter()
        for v, n in cnt.items():
            mv = mechanical(v)
            if mv != v:
                n_changed_mech += n
            m[mv] += n
        folded[bare] = m
    resolved_mech = sum(1 for b in raw if len(raw[b]) > 1 and len(folded[b]) == 1)
    still_multi = {b: c for b, c in folded.items() if len(c) > 1}
    after_mech = consistency(folded)
    print(f"    instances rewritten by rules: {n_changed_mech:,}")
    print(f"    multi-variant types resolved mechanically: {resolved_mech:,}")
    print(f"    types still multi-variant: {len(still_multi):,}")
    print(f"    self-consistency (types, >=3x): {100 * after_mech['type_consistency']:.1f}%")

    todo = sorted(
        ((b, c) for b, c in still_multi.items() if sum(c.values()) >= args.min_count),
        key=lambda x: -sum(x[1].values()),
    )
    print(f"    -> gemini queue: {len(todo):,} types "
          f"({sum(sum(c.values()) for _, c in todo):,} instances)")

    cache: dict[str, dict] = load_cache(args.cache)
    stats: dict = {"batches": 0, "unresolved": 0, "cost_usd": 0.0}
    if args.stage == "all" and todo:
        print("\n[3] GEMINI ADJUDICATION")
        need_ctx = {b for b, _ in todo if b not in cache}
        print(f"    cached: {len(todo) - len(need_ctx):,}  to query: {len(need_ctx):,}")
        ctx = collect_contexts(args.train, need_ctx) if need_ctx else {}
        cache, stats = adjudicate(
            todo, ctx, args.cache, batch_size=args.batch_size,
            limit_batches=args.limit_batches, workers=args.workers,
        )

    print("\n[4] OUTPUT")
    rows = []
    homographs = []
    src_count: collections.Counter = collections.Counter()
    for bare, cnt in sorted(folded.items(), key=lambda x: -sum(x[1].values())):
        total = sum(cnt.values())
        nvar = len(raw[bare])
        rec = cache.get(bare)
        if rec and "homograph" in rec and len(folded[bare]) > 1:
            homographs.append((bare, " | ".join(rec["homograph"]), nvar, total))
            src_count["homograph"] += 1
            continue
        if len(cnt) == 1:
            canon, source = next(iter(cnt)), "mechanical"
        elif rec and rec.get("canonical"):
            canon, source = rec["canonical"], "gemini"
        else:
            # majority fallback, but never demote to a bare/unpointed variant
            pointed = [(v, n) for v, n in cnt.most_common() if n_vowels(v)]
            canon = (pointed or cnt.most_common())[0][0]
            source = "majority"
        src_count[source] += 1
        rows.append((bare, canon, nvar, total, source))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8") as fh:
        fh.write("word_bare\tcanonical_pointed\tn_variants_before\ttotal_count\tsource\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    with args.homographs.open("w", encoding="utf-8") as fh:
        fh.write("word_bare\tforms\tn_variants_before\ttotal_count\n")
        for r in homographs:
            fh.write("\t".join(str(x) for x in r) + "\n")

    single_before = sum(1 for b, _, nv, _, s in rows if s == "mechanical" and nv == 1)
    print(f"    {args.out_tsv}  ({len(rows):,} entries)")
    print(f"    {args.homographs}  ({len(homographs):,} entries)")
    print(f"    sources: {dict(src_count)}")
    print(f"      (of the {src_count['mechanical']:,} 'mechanical', {single_before:,} were "
          f"already single-variant, {src_count['mechanical'] - single_before:,} were collapsed "
          f"by the rules)")
    print(f"    gemini batches: {stats['batches']}  unresolved/malformed: {stats['unresolved']}")
    if stats.get("cost_usd"):
        print(f"    tokens in/out: {stats['prompt_tokens']:,} / {stats['completion_tokens']:,}"
              f"   est. spend: ${stats['cost_usd']:.2f}")

    # projected: every mapped type collapses to one variant, homographs keep theirs
    mapped = {r[0] for r in rows}
    proj = {b: (collections.Counter({dict((r[0], r[1]) for r in rows)[b]: sum(c.values())})
                if b in mapped else c)
            for b, c in folded.items()}
    after = consistency(proj)
    print("\n[5] PROJECTED (map applied to train)")
    print(f"    mean variants/type: {before['mean_variants_all_types']:.3f}"
          f"  ->  {after['mean_variants_all_types']:.3f}")
    print(f"    self-consistency (types, >=3x): {100 * before['type_consistency']:.1f}%"
          f"  ->  {100 * after['type_consistency']:.1f}%")
    print(f"    self-consistency (instances, >=3x): {100 * before['instance_consistency']:.1f}%"
          f"  ->  {100 * after['instance_consistency']:.1f}%")

    summary = {
        "types": len(raw), "instances": n_inst,
        "before": before, "after_mechanical": after_mech, "projected": after,
        "sources": dict(src_count), "gemini": stats,
        "homographs": len(homographs),
    }
    (args.out_tsv.parent / "canonical_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
