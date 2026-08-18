#!/usr/bin/env python
"""Per-occurrence audio scoring for the homograph-conflict bucket.

The Sefaria rescue skips 1,223 loshn-koydesh types because their pointed
sources disagree with no >=80% winner (`homograph-conflict`).  For 1,149 of them
scripts/build_homograph_candidates.py showed the disagreement is REAL — two or
more candidate readings that genuinely sound different — so no lexicon entry
can be right for every sentence.  The decision has to be made per occurrence,
and the only witness is the audio.

This script does exactly that:

  1. select corpus chunks containing true-homograph tokens (greedy set cover,
     up to --clips-per-type occurrences of each type, --limit chunks total);
  2. PhoneticXeus-transcribe each chunk once and Needleman-Wunsch align the
     heard phones against the chunk's expected sequence (engine output for
     ordinary words; the homograph's FIRST candidate as a placeholder, so the
     alignment has something plausible to anchor on);
  3. for every homograph occurrence, cut the heard window the alignment
     assigned to that word (padded 2 phones each side) and re-score EVERY
     candidate against that window with the same NW scorer.  The verdict is
     relative — which candidate fits this clip better — never an absolute
     threshold, because the recognizer has known systematic biases (z->s,
     ej->ɛ, aː->a, ɔ~a~u looseness) that would sink all candidates alike.
  4. a verdict counts only if the winner's FIT-WEIGHTED LEAD reaches --margin
     (default 0.05); otherwise the occurrence is 'undecided' and simply
     contributes no vote.

     confidence = (best_fit - runner_fit) * clamp(best_fit, 0, 1)

     Both fits are already length-normalised to <= 1.0, so their difference is
     the lead on one scale for every word — and multiplying by the winner's own
     fit makes the confidence rise with how well the winner actually matches
     what was heard. The earlier form divided the lead BY best_fit, which
     inverted exactly that: a window the word is not in scores near zero for
     every candidate, and dividing a crumb of a lead by a crumb of a fit
     reported near-total confidence (r(best_fit, margin) = -0.90 over a real
     run; a כדת window with no k in it decided at margin 1.00). Since
     build_homograph_lexicon.py promotes on counted decisions, that handed the
     noisiest windows the most weight.

     Still no absolute threshold on a single clip's score: the bar is on the
     LEAD between candidates heard through the same recognizer biases, the fit
     only weights it.

Output data/homographs/votes.jsonl, one record per occurrence, carrying the
+-5-word context so the retrain-unmasking consumer (prepare_retrain_dataset.py,
counter 'homograph') can find the same token in the same sentence.

Usage:
  .venv/bin/python scripts/xeus_score_homographs.py --limit 12 --print   # smoke
  .venv/bin/python scripts/xeus_score_homographs.py --limit 4000 --clips-per-type 8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_map import VOWELS, tokenize_g2p_ipa  # noqa: E402
from xeus_tag import align, load_model, slice_audio, transcribe  # noqa: E402
from yiddish_g2p import hebrew_to_ipa  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״־]*")

CANDIDATES = REPO / "data" / "homographs" / "candidates.json"
VOTES = REPO / "data" / "homographs" / "votes.jsonl"

GAP = -1.0
CONTEXT = 5   # words of context recorded each side
PAD = 2       # heard phones of slack each side of the aligned window


def sim(a: str, b: str) -> float:
    """Same substitution scores xeus_tag.align uses, so windows and re-scores
    live on one scale."""
    if a == b:
        return 2.0
    if (a in VOWELS) == (b in VOWELS):
        return 0.5 if (a in VOWELS) else 0.0
    return -1.5


def nw_score(pred: list[str], heard: list[str]) -> float:
    """Needleman-Wunsch best global-alignment score (no traceback needed)."""
    n, m = len(pred), len(heard)
    prev = [j * GAP for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [i * GAP] + [0.0] * m
        pi = pred[i - 1]
        for j in range(1, m + 1):
            cur[j] = max(prev[j - 1] + sim(pi, heard[j - 1]),
                         prev[j] + GAP,
                         cur[j - 1] + GAP)
        prev = cur
    return prev[m]


def fit(cand: list[str], window: list[str]) -> float:
    """Length-normalised fit in (-inf, 1]: 1.0 == every candidate phone heard."""
    if not cand:
        return -1.0
    return nw_score(cand, window) / (2.0 * len(cand))


def load_candidates() -> dict[str, list[dict]]:
    """word -> candidate readings, deduped on the phone string (two pointings
    that sound identical are one candidate; the audio cannot separate them)."""
    raw = json.load(open(CANDIDATES, encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for word, rec in raw.items():
        seen: set[str] = set()
        cands = []
        for c in rec["candidates"]:
            phones = c["phones"].split()
            key = " ".join(phones)
            if not phones or key in seen:
                continue
            seen.add(key)
            cands.append({"pointed": c["pointed"], "register": c["register"],
                          "ipa": c["ipa"], "phones": phones,
                          "share": c.get("share", 0.0)})
        if len(cands) >= 2:
            out[word] = cands
    return out


def select_jobs(cands: dict[str, list[dict]], limit: int,
                clips_per_type: int) -> list[tuple[dict, list[tuple[int, str]]]]:
    """Greedy set cover over chunks: keep taking the chunk that adds the most
    still-needed occurrences (types under `clips_per_type`)."""
    rows = list(csv.DictReader(open(REPO / "data" / "corpus" / "yiddish_tts_dataset.tsv"),
                               delimiter="\t"))
    have_audio: dict[str, bool] = {}
    pool: list[tuple[dict, list[tuple[int, str]]]] = []
    for r in rows:
        ep = r["episode"]
        if ep not in have_audio:
            have_audio[ep] = (REPO / "data" / "audio" / f"{ep}.mp3").exists()
        if not have_audio[ep]:
            continue
        toks = _HEB.findall(r["text"])
        hits = [(i, t) for i, t in enumerate(toks) if t in cands]
        if hits:
            pool.append((r, hits))
    print(f"pool: {len(pool)} chunks contain true homographs", file=sys.stderr)

    coverage: Counter[str] = Counter()
    jobs: list[tuple[dict, list[tuple[int, str]]]] = []
    remaining = pool
    while remaining and len(jobs) < limit:
        def gain(item: tuple[dict, list[tuple[int, str]]]) -> int:
            return sum(1 for _, t in item[1] if coverage[t] < clips_per_type)
        remaining.sort(key=gain, reverse=True)
        best = remaining.pop(0)
        if gain(best) == 0:
            break
        jobs.append(best)
        for _, t in best[1]:
            coverage[t] += 1
    done = sum(1 for v in coverage.values() if v >= clips_per_type)
    print(f"selected {len(jobs)} chunks; {len(coverage)} types touched, "
          f"{done} at >={clips_per_type} clips", file=sys.stderr)
    return jobs


def score_chunk(model, device, row: dict, hits: list[tuple[int, str]],
                cands: dict[str, list[dict]], margin_min: float) -> list[dict]:
    toks = _HEB.findall(row["text"])
    target_at = dict(hits)

    words: list[tuple[int, str, list[str], bool]] = []  # tok idx, word, phones, is_target
    for i, t in enumerate(toks):
        if i in target_at:
            words.append((i, t, list(cands[t][0]["phones"]), True))
            continue
        ph = tokenize_g2p_ipa(hebrew_to_ipa(t, stress=True).replace(" ", ""))
        if ph:
            words.append((i, t, ph, False))
    if not any(is_t for *_, is_t in words):
        return []

    heard = transcribe(model, device, slice_audio(
        REPO / "data" / "audio" / f"{row['episode']}.mp3",
        float(row["start_s"]), float(row["end_s"])))
    if not heard:
        return []

    flat: list[str] = []
    owner: list[int] = []
    for wi, (_, _, ph, _) in enumerate(words):
        flat.extend(ph)
        owner.extend([wi] * len(ph))

    span: dict[int, list[int]] = defaultdict(list)
    for pi, hj in align(flat, heard):
        if pi is not None and hj is not None:
            span[owner[pi]].append(hj)

    recs = []
    for wi, (ti, word, _, is_target) in enumerate(words):
        if not is_target or not span[wi]:
            continue
        lo = max(0, min(span[wi]) - PAD)
        hi = min(len(heard), max(span[wi]) + 1 + PAD)
        window = heard[lo:hi]
        scored = [(fit(c["phones"], window), c) for c in cands[word]]
        scored.sort(key=lambda x: -x[0])
        best_s, best_c = scored[0]
        runner = scored[1][0] if len(scored) > 1 else -1.0
        # Lead between candidates, weighted by how well the winner fits: a
        # window the word is not in leads nowhere, whatever the runner-up did.
        margin = max(0.0, best_s - runner) * max(0.0, min(best_s, 1.0))
        decided = margin >= margin_min
        recs.append({
            "id": row["id"],
            "episode": row["episode"],
            "chunk_idx": int(row["chunk_idx"]),
            "word": word,
            "position": ti,
            "winner_ipa": best_c["ipa"] if decided else None,
            "winner_pointed": best_c["pointed"] if decided else None,
            "winner_register": best_c["register"] if decided else None,
            "winner_phones": " ".join(best_c["phones"]) if decided else None,
            "margin": round(margin, 4),
            "best_fit": round(best_s, 4),
            "lead": round(best_s - runner, 4),
            "decision": "decided" if decided else "undecided",
            "heard_window": " ".join(window),
            "scores": [{"pointed": c["pointed"], "register": c["register"],
                        "ipa": c["ipa"], "phones": " ".join(c["phones"]),
                        "score": round(s, 4)} for s, c in scored],
            "context_left": toks[max(0, ti - CONTEXT):ti],
            "context_right": toks[ti + 1:ti + 1 + CONTEXT],
        })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="max chunks")
    ap.add_argument("--clips-per-type", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="required fit-weighted lead: (best-runner)*clamp(best,0,1)")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    cands = load_candidates()
    print(f"{len(cands)} true-homograph types", file=sys.stderr)
    jobs = select_jobs(cands, args.limit, args.clips_per_type)
    if not jobs:
        print("nothing to do", file=sys.stderr)
        return 0

    model, device = load_model()
    print(f"device {device}", file=sys.stderr)

    VOTES.parent.mkdir(parents=True, exist_ok=True)
    n_dec = n_und = 0
    winners: Counter[tuple[str, str]] = Counter()
    with open(VOTES, "a" if args.append else "w") as out:
        for k, (row, hits) in enumerate(jobs, 1):
            try:
                recs = score_chunk(model, device, row, hits, cands, args.margin)
            except Exception as e:  # noqa: BLE001 — keep the sweep alive
                print(f"  {row['id']}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["decision"] == "decided":
                    n_dec += 1
                    winners[(rec["word"], rec["winner_ipa"])] += 1
                else:
                    n_und += 1
                if args.do_print:
                    alt = " | ".join(f"{s['ipa']}={s['score']:+.2f}"
                                     for s in rec["scores"])
                    print(f"{rec['id']} #{rec['position']:>3} {rec['word']:12s} "
                          f"-> {rec['winner_ipa'] or 'UNDECIDED':12s} "
                          f"m={rec['margin']:+.2f}  heard[{rec['heard_window']}]  {alt}")
            print(f"  [{k}/{len(jobs)}] {row['id']}: {len(recs)} occurrence(s)",
                  file=sys.stderr)

    total = n_dec + n_und
    print(f"\n{total} occurrences scored: {n_dec} decided "
          f"({n_dec/max(total,1):.0%}), {n_und} undecided")
    if winners:
        print("top verdicts:")
        for (w, ipa), c in winners.most_common(10):
            print(f"  {w:12s} {ipa:14s} x{c}")
    print(f"votes -> {VOTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
