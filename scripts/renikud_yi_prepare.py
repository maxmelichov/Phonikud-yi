#!/usr/bin/env python3
"""ReNikud-yi training data: per-letter (consonant, vowel, stress) labels.

The Yiddish version of ReNikud's pseudo-labelled corpus, with one change:
where ReNikud took whatever a phoneme ASR wrote, every label here is a
reading something vouched for, in this order of authority:

  gold      a certain (Chezky) word: its gold reading; when it has several,
            the one the ear chose for that clip (run-3 segments) if the clip
            was cut, else the primary
  lexicon   a word the frozen engine reads from its lexicon at HIGH/MED
  audio     a rule-path word whose reading the ear decided against the
            audio (xeus_attest.py; occurrence margin >= --margin-occ, or a
            type reading agreed by >= --type-min clips at >= --type-share)
  —         everything else is IGNORE (-100): the model gets no gradient
            there and is free to learn it from context

Each labelled word is aligned letter by letter with scripts/yi_align.py;
a word the aligner cannot place (spelled-out abbreviations, a few irregular
LK words) is IGNORE too. Non-Hebrew characters are IGNORE.

Splits follow retrain3's episodes: test = episode 100313 (never touched),
val = retrain3 val episodes, train = the rest.

Output: data/renikud_yi/{train,val,test}.jsonl, one row per corpus chunk:
  {id, episode, text, cons: [..], vowel: [..], stress: [..], src: [..]}
with one label id per character of ``text`` (-100 = ignore), and
labels.json (the class lists) + dataset_stats.md.

Usage:  .venv/bin/python scripts/renikud_yi_prepare.py [--attest data/xeus_ft/attest.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import lexicon_key, tokenize_ipa  # noqa: E402
from yi_align import CONSONANTS, VOWELS, align_word, parse_chunk  # noqa: E402
from yiddish_g2p import g2p_token  # noqa: E402

IGNORE = -100
CONS_CLASSES = ("",) + CONSONANTS
VOWEL_CLASSES = ("",) + tuple(sorted(VOWELS, key=lambda v: (len(v), v)))
C2ID = {c: i for i, c in enumerate(CONS_CLASSES)}
V2ID = {v: i for i, v in enumerate(VOWEL_CLASSES)}
_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
TEST_EPISODE = "100313"


def restress(engine_ipa: str, phones: list[str]) -> str:
    from prepare_retrain_dataset_v8 import restress as _r  # noqa: E402
    return _r(engine_ipa, phones)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--segments", default=str(REPO / "data/xeus_ft/run3/segments.jsonl"),
                    help="ear-chosen gold variant per certain-word clip (optional)")
    ap.add_argument("--attest", default=str(REPO / "data/xeus_ft/attest.jsonl"), help="optional")
    ap.add_argument("--margin-occ", type=float, default=2.0)
    ap.add_argument("--margin-type", type=float, default=1.0)
    ap.add_argument("--type-min", type=int, default=5)
    ap.add_argument("--type-share", type=float, default=0.85)
    ap.add_argument("--val-episodes", default=str(REPO / "data/retrain3/val_episodes.txt"))
    ap.add_argument("--out", default=str(REPO / "data/renikud_yi"))
    args = ap.parse_args()
    csv.field_size_limit(10_000_000)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dictionary = json.loads(Path(args.dictionary).read_text(encoding="utf-8"))
    gold_by_key = {v["key"]: v for v in dictionary.values()}

    # ear-chosen variant per (chunk, heb index) for certain words that were cut
    chosen: dict[tuple[str, int, int], int] = {}
    segp = Path(args.segments)
    if segp.exists():
        # segments carry word surface + variant but not the heb index; recover by
        # matching the words in order within the chunk
        by_chunk: dict[tuple[str, int], list[tuple[str, int]]] = collections.defaultdict(list)
        for line in segp.open(encoding="utf-8"):
            r = json.loads(line)
            for w in r["words"]:
                by_chunk[(r["episode"], r["chunk_idx"])].append((w["key"], w["variant"]))
    else:
        by_chunk = {}

    # audio decisions
    occ: dict[tuple[str, int, int], dict] = {}
    type_reading: dict[str, list[str]] = {}
    attp = Path(args.attest)
    if attp.exists():
        per_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for line in attp.open(encoding="utf-8"):
            r = json.loads(line)
            occ[(r["episode"], int(r["chunk_idx"]), r["wi"])] = r
            if r["margin"] >= args.margin_type:
                per_type[r["key"]][" ".join(r["chosen"])] += 1
        for key, c in per_type.items():
            total = sum(c.values())
            reading, top = c.most_common(1)[0]
            if total >= args.type_min and top / total >= args.type_share:
                type_reading[key] = reading.split()

    val_eps = set(Path(args.val_episodes).read_text().split()) if Path(args.val_episodes).exists() else set()
    engine_cache: dict[str, dict] = {}
    align_cache: dict[tuple[str, str], list | None] = {}
    stats = collections.Counter()
    align_fail: collections.Counter = collections.Counter()
    files = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}

    for r in csv.DictReader(open(args.corpus, encoding="utf-8"), delimiter="\t"):
        text = r["text"]
        ep, ci = r["episode"], int(r["chunk_idx"])
        cons = [IGNORE] * len(text)
        vow = [IGNORE] * len(text)
        strs = [IGNORE] * len(text)
        src = [""] * len(text)
        seg_words = list(by_chunk.get((ep, ci), []))
        seg_ptr = 0
        for hi, m in enumerate(_HEB.finditer(text)):
            w = m.group(0)
            key = lexicon_key(w)
            stats["tokens"] += 1
            if w not in engine_cache:
                t = g2p_token(w)
                t = t if isinstance(t, dict) else t.__dict__
                engine_cache[w] = {"ipa": t["ipa_primary"] or "", "route": t["route"], "conf": t["confidence"]}
            e = engine_cache[w]
            ipa = None
            source = None
            if key in gold_by_key:
                variants = gold_by_key[key]["variants"]
                vi = 0
                # ear-chosen variant if this clip was cut (words appear in order)
                while seg_ptr < len(seg_words) and seg_words[seg_ptr][0] != key:
                    seg_ptr += 1
                if seg_ptr < len(seg_words):
                    vi = min(seg_words[seg_ptr][1], len(variants) - 1)
                    seg_ptr += 1
                phones = variants[vi]
                ipa = restress(e["ipa"], phones) if e["ipa"] and len(tokenize_ipa(e["ipa"])) == len(phones) else "".join(phones)
                source = "gold"
            elif e["route"] == "lexicon" and e["conf"] in ("HIGH", "MED") and e["ipa"]:
                ipa, source = e["ipa"], "lexicon"
            elif e["route"] == "rule" and e["ipa"]:
                rec = occ.get((ep, ci, hi))
                phones = None
                if rec is not None and rec["key"] == key and rec["margin"] >= args.margin_occ:
                    phones, source = rec["chosen"], "audio-occ"
                elif key in type_reading:
                    phones, source = type_reading[key], "audio-type"
                if phones is not None and len(phones) == len(tokenize_ipa(e["ipa"])):
                    ipa = restress(e["ipa"], phones)
            if ipa is None:
                stats["ignore_" + (source or "unvouched")] += 1
                continue
            ak = (w, ipa)
            if ak not in align_cache:
                align_cache[ak] = align_word(w, ipa)
            aligned = align_cache[ak]
            if aligned is None:
                stats["align_fail_" + source] += 1
                align_fail[w] += 1
                continue
            # map aligned letters back onto text positions (aligner strips nikud/finals/ligatures)
            pos = m.start()
            letters_in_text = [(i, ch) for i, ch in enumerate(w, start=pos) if not ("֑" <= ch <= "ׇ")]
            if len(letters_in_text) != len(aligned):
                stats["align_len_mismatch"] += 1
                continue
            for (ti, _), (_, chunk) in zip(letters_in_text, aligned):
                c, v, st = parse_chunk(chunk) if chunk else ("", "", 0)
                cons[ti] = C2ID[c]
                vow[ti] = V2ID[v]
                strs[ti] = st
                src[ti] = source
            stats["labelled_" + source] += 1
        split = "test" if ep == TEST_EPISODE else ("val" if ep in val_eps else "train")
        files[split].write(json.dumps({"id": f"{ep}-{ci:05d}", "episode": ep, "text": text,
                                       "cons": cons, "vowel": vow, "stress": strs}, ensure_ascii=False) + "\n")
        stats["rows_" + split] += 1
    for f in files.values():
        f.close()
    (out / "labels.json").write_text(json.dumps({"consonants": list(CONS_CLASSES), "vowels": list(VOWEL_CLASSES),
                                                  "ignore": IGNORE}, ensure_ascii=False, indent=1), encoding="utf-8")
    tot = stats["tokens"]
    lab = sum(v for k, v in stats.items() if k.startswith("labelled_"))
    md = ["# ReNikud-yi dataset", "", f"tokens {tot:,}; labelled {lab:,} ({lab / tot:.1%})", "",
          "| source | tokens |", "| --- | ---: |"]
    for k, v in sorted(stats.items()):
        md.append(f"| `{k}` | {v:,} |")
    md += ["", "Most frequent unalignable words:", ""]
    for w, n in align_fail.most_common(15):
        md.append(f"- {w} ({n})")
    (out / "dataset_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
