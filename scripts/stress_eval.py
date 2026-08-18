#!/usr/bin/env python
"""Audio-grounded evaluation of the rule-based Yiddish stress stage.

Samples annotated 30s chunks, runs yiddish_g2p's stress rules over text_yi,
and asks Gemini Flash to listen to the audio and judge, per word, whether our
stressed syllable matches the speaker. Results cache to
data/stress/stress_eval_cache.jsonl so runs are resumable.

Usage:
    .venv/bin/python scripts/stress_eval.py sample     # build the sample split
    .venv/bin/python scripts/stress_eval.py judge      # call Gemini (resumable)
    .venv/bin/python scripts/stress_eval.py score      # accuracy + breakdowns
    .venv/bin/python scripts/stress_eval.py harvest    # overrides + needs-review
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yiddish_g2p as G  # noqa: E402
from phonikud_yi.gateway import Gateway, audio_message, iter_jsonl  # noqa: E402

DATA = REPO / "data"
ANN = DATA / "annotations"
CHUNKS = DATA / "chunks"
SAMPLE_PATH = DATA / "stress" / "stress_eval_sample.json"
CACHE_PATH = DATA / "stress" / "stress_eval_cache.jsonl"

N_CHUNKS = 60
SEED = 20260803


# ---------------------------------------------------------------- sampling
def build_sample() -> dict:
    rng = random.Random(SEED)
    pool: list[dict] = []
    for f in sorted(ANN.glob("*.jsonl")):
        ep = f.stem
        cdir = CHUNKS / ep
        if not cdir.is_dir():
            continue
        rows = []
        for r in iter_jsonl(f):
            if (r.get("confidence") or 0) < 0.9:
                continue
            txt = (r.get("text_yi") or "").strip()
            if len(txt) < 120:
                continue
            mp3 = cdir / f"chunk_{r['chunk_idx']:05d}.mp3"
            if not mp3.exists():
                continue
            rows.append({"episode": ep, "chunk_idx": r["chunk_idx"],
                         "text_yi": txt, "mp3": str(mp3.relative_to(REPO))})
        if rows:
            pool.append(rng.choice(rows))  # one chunk per episode
    rng.shuffle(pool)
    sample = pool[:N_CHUNKS]
    cut = int(round(len(sample) * 0.7))
    out = {"harvest": sample[:cut], "test": sample[cut:]}
    SAMPLE_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"episodes with usable chunks: {len(pool)}")
    print(f"sample: {len(sample)}  harvest={len(out['harvest'])} test={len(out['test'])}")
    return out


def load_sample() -> dict:
    return json.loads(SAMPLE_PATH.read_text())


# ------------------------------------------------------- word-level analysis
_HEB = re.compile(r"[֐-׿]")


# Syllabification is display-only (it makes the indices we send Gemini
# unambiguous); the engine itself marks stress directly before the vowel and no
# longer needs onset logic, so these tables live here.
_C_UNITS = ("dzh", "tsh", "kh", "sh", "zh", "ts", "dz")
_LEGAL_ONSETS = {
    "sht", "shp", "shtr", "shpr", "shm", "shn", "shl", "shv", "shr", "shk",
    "st", "sp", "str", "spr", "sk", "sl", "sm", "sn", "sv",
    "tr", "dr", "kr", "gr", "pr", "br", "fr", "vr",
    "kl", "gl", "pl", "bl", "fl", "kn", "gn", "tsv", "kv", "shtsh",
}


def _split_consonants(cluster: str) -> list[str]:
    units, i = [], 0
    while i < len(cluster):
        for u in _C_UNITS:
            if cluster.startswith(u, i):
                units.append(u)
                i += len(u)
                break
        else:
            units.append(cluster[i])
            i += 1
    return units


def syllabify(latin: str) -> list[str]:
    """Split a Latin word into syllables (maximal legal onset)."""
    nuc = G._nuclei(latin)
    if not nuc:
        return [latin]
    starts = [0]
    for (s, _e) in nuc[1:]:
        j = s
        while j > 0 and latin[j - 1].isalpha() and latin[j - 1] not in G._LATIN_VOWELS:
            j -= 1
        units = _split_consonants(latin[j:s])
        while len(units) > 1 and "".join(units) not in _LEGAL_ONSETS:
            units.pop(0)
        starts.append(s - len("".join(units)))
    starts.append(len(latin))
    return [latin[starts[i]:starts[i + 1]] for i in range(len(nuc))]


def analyze_word(token: str) -> dict | None:
    """Our stress decision for one whitespace token, or None if not analyzable."""
    core = token.strip()
    if not _HEB.search(core):
        return None
    pre = G._preprocess_hebrew(core)
    toks = pre.split()
    if len(toks) != 1:
        return None  # contraction expanded into two words: skip
    part = toks[0].split("-")[0]
    m = G._PUNCT_SPLIT.match(part)
    lead, cw, _trail = m.group(1), m.group(2), m.group(3)
    is_lk_sent = G._LK_SENTINEL in lead or G._LK_SENTINEL in cw
    cw = cw.replace(G._LK_SENTINEL, "")
    if not cw:
        return None
    bare = G._strip_points(cw)
    work = cw
    if bare in G._WORD_LATIN and not G._vowel_point(cw):
        latin = G._WORD_LATIN[bare]
    else:
        for stem, repl in G._STEM_SUBS:
            if stem in work:
                work = work.replace(stem, repl)
        latin = G._word_to_latin(work)
    if not latin:
        return None
    nuc = G._nuclei(latin)
    if len(nuc) < 2:
        return None  # monosyllables carry no mark, nothing to evaluate
    is_lk = is_lk_sent or G._looks_lk(bare)
    idx = G._stress_nucleus(bare, latin, is_lk, len(nuc))
    syls = syllabify(latin)
    # unstressed-prefix detection (mirrors _stress_nucleus's loop)
    rest, pref = latin, []
    while True:
        for p in G._UNSTRESSED_PREFIXES:
            if rest.startswith(p) and G._nuclei(rest[len(p):]):
                pref.append(p)
                rest = rest[len(p):]
                break
        else:
            break
    return {
        "word": bare,
        "bare": bare,
        "latin": latin,
        "ipa": G.normalize_ipa_affricates(G.latin_to_ipa(G._apply_stress(latin, bare, is_lk))),
        "syllables": [G.normalize_ipa_affricates(G.latin_to_ipa(s)) for s in syls],
        "n_syl": len(nuc),
        "ours": idx,
        "is_lk": is_lk,
        "in_lexicon": is_lk_sent,
        "prefix": "+".join(pref),
        "overridden": bare in G._STRESS_OVERRIDE,
    }


def chunk_words(text: str) -> list[dict]:
    seen: set[str] = set()
    out = []
    for tok in G.strip_tags(text).split():
        w = analyze_word(tok)
        if w and w["bare"] not in seen:
            seen.add(w["bare"])
            out.append(w)
    return out


# ------------------------------------------------------------------ judging
PROMPT = """You hear a 30-second clip of spoken Hasidic/Central Yiddish.

For each word below, the syllables are listed 0-indexed and the one WE think is
stressed is marked with *. Listen for the word in the audio and decide whether
our stressed syllable is the one the speaker actually stresses (loudest /
highest-pitch / longest vowel of the word).

Words:
{words}

Reply with STRICT JSON only:
{{"words":[{{"word":"<Hebrew word exactly as given>","ours_ok":true|false,
"correct_index":<0-based syllable index the speaker stresses, or null if ours_ok>,
"confidence":<0.0-1.0>}}]}}

Rules: one entry per word, same order. If you cannot find the word in the audio
or are unsure, still answer but set confidence below 0.5. Do not explain."""


def fmt_words(words: list[dict]) -> str:
    lines = []
    for w in words:
        syl = " ".join(
            f"{i}:{'*' if i == w['ours'] else ''}{s}" for i, s in enumerate(w["syllables"])
        )
        lines.append(f"- {w['word']}  [{syl}]")
    return "\n".join(lines)


def cache_key(c: dict) -> str:
    return f"{c['episode']}:{c['chunk_idx']}"


def load_cache(tag: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if CACHE_PATH.exists():
        for r in iter_jsonl(CACHE_PATH):
            if r.get("tag") == tag:
                out[r["key"]] = r
    return out


def judge(tag: str = "base", limit: int | None = None) -> None:
    os.environ.setdefault("GATEWAY_REASONING_EFFORT", "low")
    sample = load_sample()
    chunks = sample["harvest"] + sample["test"] if tag == "base" else sample["test"]
    done = load_cache(tag)
    todo = [c for c in chunks if cache_key(c) not in done]
    if limit:
        todo = todo[:limit]
    print(f"tag={tag} cached={len(done)} todo={len(todo)}")
    if not todo:
        return
    gw = Gateway()

    def run(c: dict) -> dict | None:
        words = chunk_words(c["text_yi"])
        if not words:
            return None
        msg = audio_message(REPO / c["mp3"], PROMPT.format(words=fmt_words(words)))
        try:
            res = gw.chat_json([msg], model="google/gemini-3.6-flash",
                               reasoning_effort="low", max_tokens=8000)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {cache_key(c)}: {exc}")
            return None
        items = res.get("words") if isinstance(res, dict) else res
        if not isinstance(items, list):
            print(f"  BAD  {cache_key(c)}: {str(res)[:120]}")
            return None
        return {"tag": tag, "key": cache_key(c), "episode": c["episode"],
                "chunk_idx": c["chunk_idx"], "split": ("test" if c in sample["test"] else "harvest"),
                "words": words, "judgments": items}

    with CACHE_PATH.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, rec in enumerate(ex.map(run, todo), 1):
                if rec:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  [{i}/{len(todo)}] ok" if rec else f"  [{i}/{len(todo)}] skipped")


# ------------------------------------------------------------------ scoring
MIN_CONF = 0.5


def pairs(tag: str, split: str | None = None):
    """Yield (word_analysis, judgment) for confident judgments."""
    for rec in load_cache(tag).values():
        if split and rec.get("split") != split:
            continue
        by_word = {w["word"]: w for w in rec["words"]}
        # Gemini sometimes echoes the word re-spelled phonetically, so pair
        # positionally when the list lengths line up and fall back to the string.
        positional = len(rec["judgments"]) == len(rec["words"])
        for pos, j in enumerate(rec["judgments"]):
            if not isinstance(j, dict):
                continue
            w = by_word.get(str(j.get("word", "")).strip())
            if w is None and positional:
                w = rec["words"][pos]
            if w is None:
                continue
            conf = j.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else 0.0
            if conf < MIN_CONF:
                continue
            ok = bool(j.get("ours_ok"))
            ci = j.get("correct_index")
            ci = int(ci) if isinstance(ci, (int, float)) else None
            if not ok and (ci is None or not (0 <= ci < w["n_syl"])):
                continue  # unusable disagreement
            if not ok and ci == w["ours"]:
                continue  # self-contradictory: "wrong" but names our own syllable
            if ok:
                ci = w["ours"]
            yield rec, w, j, ok, ci, conf


def acc(rows) -> tuple[float, int]:
    rows = list(rows)
    if not rows:
        return float("nan"), 0
    return sum(rows) / len(rows), len(rows)


def score(tag: str, split: str | None = None, quiet: bool = False) -> dict:
    buckets: dict[str, list[int]] = defaultdict(list)
    fails: Counter = Counter()
    examples: dict[str, tuple] = {}
    for _rec, w, _j, ok, ci, _conf in pairs(tag, split):
        hit = int(ok)
        buckets["ALL"].append(hit)
        buckets["LK" if w["is_lk"] else "Germanic"].append(hit)
        if w["in_lexicon"]:
            buckets["LK(lexicon)"].append(hit)
        elif w["is_lk"]:
            buckets["LK(heuristic ת/ח)"].append(hit)
        buckets[f"{min(w['n_syl'], 5)}-syl"].append(hit)
        buckets["prefixed" if w["prefix"] else "no-prefix"].append(hit)
        if not hit:
            key = f"{'LK' if w['is_lk'] else 'Ger'}|{w['n_syl']}syl|ours={w['ours']}->{ci}"
            fails[key] += 1
            examples.setdefault(key, (w["word"], w["ipa"], w["syllables"], ci))
    res = {k: acc(v) for k, v in buckets.items()}
    if not quiet:
        label = f"{tag}/{split or 'all'}"
        print(f"\n=== {label} ===")
        order = ["ALL", "LK", "LK(lexicon)", "LK(heuristic ת/ח)", "Germanic",
                 "2-syl", "3-syl", "4-syl", "5-syl", "prefixed", "no-prefix"]
        for k in order:
            if k in res:
                a, n = res[k]
                print(f"  {k:22s} {a*100:5.1f}%  (n={n})")
        print("  top failure patterns:")
        for key, cnt in fails.most_common(8):
            word, ipa, syls, ci = examples[key]
            print(f"    {cnt:3d}  {key}  e.g. {word} ours={ipa} correct={syls[ci] if ci < len(syls) else '?'} ({'-'.join(syls)})")
    return {"res": res, "fails": fails, "examples": examples}


# ----------------------------------------------------------------- harvest
HARVEST_CONF = 0.8


def harvest() -> None:
    votes: dict[str, Counter] = defaultdict(Counter)
    meta: dict[str, dict] = {}
    for _rec, w, _j, ok, ci, conf in pairs("base", "harvest"):
        if ok or conf < HARVEST_CONF:
            continue
        votes[w["bare"]][ci] += 1
        meta[w["bare"]] = w
    overrides: dict[str, int] = {}
    review: list[tuple] = []
    for bare, c in sorted(votes.items()):
        idx, n = c.most_common(1)[0]
        w = meta[bare]
        if n >= 2:
            overrides[bare] = idx
        else:
            review.append((bare, w["ipa"], w["ours"], idx, w["n_syl"],
                           "LK" if w["is_lk"] else "Ger", "-".join(w["syllables"])))
    lines = ["# Harvested from audio (Gemini Flash judgments, conf>=0.8, >=2 agreeing",
             "# occurrences in the 70% harvest split). bare Hebrew word -> nucleus index.",
             "_STRESS_OVERRIDE: dict[str, int] = {"]
    for bare, idx in sorted(overrides.items()):
        w = meta[bare]
        lines.append(f'    "{bare}": {idx},  # {w["ipa"]} -> stress {"-".join(w["syllables"])!r}[{idx}]')
    lines.append("}")
    (DATA / "lexicons" / "stress_overrides.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (DATA / "stress" / "stress_needs_review.tsv").open("w", encoding="utf-8") as fh:
        fh.write("word\tour_ipa\tour_index\tsuggested_index\tn_syl\ttype\tsyllables\n")
        for r in review:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"overrides harvested: {len(overrides)}  needs-review: {len(review)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "sample":
        build_sample()
    elif cmd == "judge":
        judge(arg or "base")
    elif cmd == "score":
        score(arg or "base")
        score(arg or "base", "test")
    elif cmd == "harvest":
        harvest()
    elif cmd == "words":
        s = load_sample()
        tot = sum(len(chunk_words(c["text_yi"])) for c in s["harvest"] + s["test"])
        print("polysyllabic word instances:", tot)
