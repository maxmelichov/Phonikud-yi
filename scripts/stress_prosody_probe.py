#!/usr/bin/env python3
"""Can word stress be read off the audio from the prosody of the ear's forced alignment?

The ear (run 2, data/xeus_ft/ckpt/best) has no stress in its vocabulary; stress
is the G2P's rule stage. This probe asks whether the acoustic correlates of
stress — vowel duration, energy, F0 — measured over the frame spans the ear's
forced alignment assigns to each vowel, predict which vowel the engine stresses.

  1. Clips: training clips of data/xeus_ft/run3 (all certain gold words). Every
     word in a clip with >= 2 vowels is a case, provided the engine's stressed
     reading (yiddish_labels.text_to_ipa) has the same vowel count as the gold
     target, so the engine's stressed-vowel index transfers to the gold phones.
  2. Forced alignment with run 2 (encoder on MPS, forced_align on CPU). A
     token's span runs from its first emitted frame to the next token's first
     frame (CTC is peaky: the blanks after a token belong to it).
  3. Per vowel: duration (frames), RMS energy over the span, F0 (torchaudio
     detect_pitch_frequency, frame_time 0.02), word-relative versions of each,
     position, vowel identity. Conditional-logit (within-word softmax) fitted by
     numpy; word-type-disjoint split by lexicon key.
  4. The slice where Gemini disagreed with the engine (data/stress/stress_eval_cache.jsonl):
     whose side is the prosody classifier on? Those types are excluded from training.

Usage:
  python scripts/stress_prosody_probe.py --stage all [--n-clips 4000]
  python scripts/stress_prosody_probe.py --stage fit      # reuse cached features
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from xeus_ft_common import YI_BLANK, YI_VOWELS, lexicon_key, read_jsonl, tokenize_ipa, yi_ids, yi_logits  # noqa: E402

SR = 16000
VOWELS = set(YI_VOWELS)
FEATS = REPO / "data/eval/stress_prosody_feats.jsonl"


# ----------------------------------------------------------------------------
# Engine stress
# ----------------------------------------------------------------------------

def engine_stress_index(word: str) -> tuple[int | None, list[str]]:
    """Index (among vowels) of the vowel carrying ˈ in the engine's reading, and the vowel list."""
    import yiddish_labels
    ipa = yiddish_labels.text_to_ipa(word)
    # tokenize with stress kept: walk the string, count vowels before ˈ
    idx = None
    vowels: list[str] = []
    pending = False
    s = ipa.replace(" ", "")
    i = 0
    multi = ("aː", "ej", "aj", "ɔj", "oʊ")
    while i < len(s):
        ch = s[i]
        if ch == "ˈ":
            pending = True
            i += 1
            continue
        if ch == "ˌ":
            i += 1
            continue
        tok = None
        for m in multi:
            if s.startswith(m, i):
                tok = m
                break
        if tok is None:
            tok = ch
        i += len(tok)
        if tok in VOWELS:
            if pending and idx is None:
                idx = len(vowels)
            pending = False
            vowels.append(tok)
    return idx, vowels


# ----------------------------------------------------------------------------
# Gemini disagreements
# ----------------------------------------------------------------------------

def gemini_disagreements(path: Path) -> dict[str, dict]:
    """lexicon key -> {word, gemini_index, ours_cache, n_syl, votes}. Only judgments where
    Gemini said ours_ok=False with a correct_index != ours and confidence >= 0.5."""
    out: dict[str, dict] = {}
    for r in read_jsonl(path):
        meta = {w["word"]: w for w in r["words"]}
        for j in r["judgments"]:
            if j.get("ours_ok") or j.get("correct_index") is None or (j.get("confidence") or 0) < 0.5:
                continue
            m = meta.get(j["word"])
            if m is None or j["correct_index"] == m["ours"]:
                continue
            key = lexicon_key(m["bare"])
            d = out.setdefault(key, {"word": m["bare"], "n_syl": m["n_syl"], "ours_cache": m["ours"], "votes": collections.Counter()})
            d["votes"][j["correct_index"]] += 1
    for d in out.values():
        d["gemini_index"] = d["votes"].most_common(1)[0][0]
        d["votes"] = dict(d["votes"])
    return out


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------

def select_clips(rows: list[dict], bykey: dict, n_clips: int, per_type: int, dis_keys: set[str], rng: random.Random, args_all_splits: bool = False):
    """Training clips with >= 1 case word; capped per word type. Plus every clip (any split)
    holding a Gemini-disagreement type, capped at 40 per type."""
    cases_cache: dict[str, tuple] = {}

    def word_cases(r):
        out = []
        pos = 0
        target = r["target"]
        for w in r["words"]:
            ph = bykey[w["key"]]["variants"][w["variant"]]
            span = (pos, pos + len(ph))
            pos += len(ph)
            nv = sum(p in VOWELS for p in ph)
            if nv < 2:
                continue
            if w["w"] not in cases_cache:
                cases_cache[w["w"]] = engine_stress_index(w["w"])
            eidx, evow = cases_cache[w["w"]]
            if eidx is None or len(evow) != nv:
                continue
            out.append({"w": w["w"], "key": w["key"], "span": span, "engine": eidx, "n_vowels": nv})
        return out

    main, dis = [], []
    count = collections.Counter()
    dcount = collections.Counter()
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in order:
        r = rows[i]
        if len(r["words"]) > 4:
            continue
        try:
            t = [p for w in r["words"] for p in bykey[w["key"]]["variants"][w["variant"]]]
        except (KeyError, IndexError):
            continue
        if t != r["target"]:
            continue
        cases = word_cases(r)
        if not cases:
            continue
        keys = {c["key"] for c in cases}
        hit = keys & dis_keys
        if hit:
            if any(dcount[k] < 40 for k in hit):
                for k in hit:
                    dcount[k] += 1
                dis.append((r, cases))
            continue
        if (not args_all_splits and r["split"] != "train") or len(main) >= n_clips:
            continue
        if all(count[k] >= per_type for k in keys):
            continue
        for k in keys:
            count[k] += 1
        main.append((r, cases))
    return main, dis


def extract(args) -> None:
    import torch
    import torchaudio.functional as taf
    from xeus_ft_train import Segments, collate
    from xeus_yi_decode import load_finetuned

    rows = list(read_jsonl(Path(args.data) / "segments.jsonl"))
    d = json.load(open(REPO / "data/xeus_ft/dictionary.json", encoding="utf-8"))
    bykey = {}
    for v in d.values():
        bykey.setdefault(v["key"], v)
    dis = gemini_disagreements(REPO / "data/stress/stress_eval_cache.jsonl")
    print(f"Gemini-disagreement word types: {len(dis)}")
    rng = random.Random(args.seed)
    main, dclips = select_clips(rows, bykey, args.n_clips, args.per_type, set(dis), rng, args.all_splits)
    print(f"main clips {len(main)} (cases {sum(len(c) for _, c in main)}), disagreement clips {len(dclips)} "
          f"(cases {sum(len(c) for _, c in dclips)})")
    allclips = [(r, c, "main") for r, c in main] + [(r, c, "dis") for r, c in dclips]
    seg_rows = [r for r, _, _ in allclips]
    ds = Segments(seg_rows, Path(args.data) / "seg")

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    inner, head = load_finetuned(Path(args.ckpt), device)
    inner.eval(); head.eval()
    out_rows = []
    n_fail = 0
    with torch.no_grad(), open(FEATS, "w", encoding="utf-8") as fh:
        batches = ds.batches(args.seconds, shuffle=False, rng=random.Random(0))
        for bi, idx in enumerate(batches):
            speech, lens, _, _ = collate(ds, idx, device)
            logits, flens = yi_logits(inner, head, speech, lens)
            lp = torch.log_softmax(logits.float(), -1).cpu()
            speech = speech.cpu()
            lens = lens.cpu()
            for j, i in enumerate(idx):
                r, cases, group = allclips[i]
                T = int(flens[j])
                nsamp = int(lens[j])
                wav = speech[j, :nsamp]
                ids = torch.tensor(yi_ids(r["target"]), dtype=torch.long)
                try:
                    labels, scores = taf.forced_align(lp[j:j + 1, :T], ids.unsqueeze(0),
                                                      torch.tensor([T]), torch.tensor([len(ids)]), blank=YI_BLANK)
                except Exception as e:  # target longer than frames etc.
                    n_fail += 1
                    continue
                lab = labels[0].tolist()
                # first frame of each target token
                starts = []
                prev = YI_BLANK
                for t, x in enumerate(lab):
                    if x != YI_BLANK and x != prev:
                        starts.append(t)
                    prev = x
                if len(starts) != len(ids):  # CTC puts a blank between repeats, so this should not happen
                    n_fail += 1
                    continue
                ends = starts[1:] + [T]
                spf = nsamp / T  # samples per frame
                # win_length = median-smoothing window in frames; must not exceed the clip's frame count
                f0 = taf.detect_pitch_frequency(wav.unsqueeze(0), SR, frame_time=0.02,
                                                win_length=max(1, min(5, nsamp // int(SR * 0.02) - 2)),
                                                freq_low=60, freq_high=400)[0].numpy()
                clip_f0_med = float(np.median(np.log2(np.maximum(f0, 1.0))))
                sq = (wav.numpy().astype(np.float64)) ** 2
                for c in cases:
                    a, b = c["span"]
                    vow = []
                    for k in range(a, b):
                        p = r["target"][k]
                        if p not in VOWELS:
                            continue
                        s, e = starts[k], ends[k]
                        s_s, e_s = int(s * spf), max(int(e * spf), int(s * spf) + 1)
                        rms = float(np.sqrt(sq[s_s:e_s].mean() + 1e-12))
                        f_s, f_e = min(s, len(f0) - 1), max(min(e, len(f0)), min(s, len(f0) - 1) + 1)
                        fseg = f0[f_s:f_e]
                        vow.append({"v": p, "start": s, "end": e, "dur": e - s, "rms": rms,
                                    "f0": float(np.median(fseg)), "f0_rel": float(np.median(np.log2(np.maximum(fseg, 1.0))) - clip_f0_med)})
                    out_rows.append(1)
                    fh.write(json.dumps({"clip": r["id"], "split": r["split"], "group": group, "w": c["w"], "key": c["key"],
                                         "engine": c["engine"], "n_vowels": c["n_vowels"], "n_words": len(r["words"]),
                                         "gemini": dis[c["key"]]["gemini_index"] if c["key"] in dis else None,
                                         "ours_cache": dis[c["key"]]["ours_cache"] if c["key"] in dis else None,
                                         "n_syl_cache": dis[c["key"]]["n_syl"] if c["key"] in dis else None,
                                         "vowels": vow}, ensure_ascii=False) + "\n")
            if bi == 0:
                print(f"  samples per frame {spf:.1f}")
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(batches)}  cases {len(out_rows)}  align failures {n_fail}", flush=True)
    print(f"wrote {len(out_rows)} cases to {FEATS}; alignment failures {n_fail}")


# ----------------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------------

VOWEL_LIST = list(YI_VOWELS)
FEATURE_GROUPS = {
    "duration": ["log_dur", "dur_rel", "dur_rank", "dur_max"],
    "energy": ["log_rms", "rms_rel", "rms_rank", "rms_max"],
    "f0": ["f0_rel_clip", "f0_rel_word", "f0_rank", "f0_max"],
    "position": ["pos_frac", "is_first", "is_last", "is_penult", "n_vowels"],
    "vowel": [f"v_{v}" for v in VOWEL_LIST] + ["is_schwa"],
}


def featurize(case: dict) -> tuple[np.ndarray, list[str]]:
    vs = case["vowels"]
    n = len(vs)
    ld = np.log(np.array([v["dur"] for v in vs], float))
    lr = np.log(np.array([v["rms"] for v in vs], float) + 1e-9)
    f0 = np.log2(np.maximum(np.array([v["f0"] for v in vs], float), 1.0))
    f0c = np.array([v["f0_rel"] for v in vs], float)

    def rank(x):  # 1.0 for the max, 0 for the min
        order = x.argsort().argsort()
        return order / max(n - 1, 1)

    names, cols = [], []
    def add(name, col):
        names.append(name); cols.append(np.asarray(col, float))
    add("log_dur", ld); add("dur_rel", ld - ld.mean()); add("dur_rank", rank(ld)); add("dur_max", (ld == ld.max()).astype(float))
    add("log_rms", lr); add("rms_rel", lr - lr.mean()); add("rms_rank", rank(lr)); add("rms_max", (lr == lr.max()).astype(float))
    add("f0_rel_clip", f0c); add("f0_rel_word", f0 - f0.mean()); add("f0_rank", rank(f0)); add("f0_max", (f0 == f0.max()).astype(float))
    pos = np.arange(n)
    add("pos_frac", pos / (n - 1)); add("is_first", pos == 0); add("is_last", pos == n - 1); add("is_penult", pos == n - 2); add("n_vowels", np.full(n, n))
    for v in VOWEL_LIST:
        add(f"v_{v}", [x["v"] == v for x in vs])
    add("is_schwa", [x["v"] == "ə" for x in vs])
    return np.stack(cols, 1), names


class CondLogit:
    """Within-word softmax over vowel scores w·x; L2-regularised; Adam in numpy."""

    def __init__(self, l2: float = 1e-3, iters: int = 1500, lr: float = 0.05):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X: np.ndarray, groups: np.ndarray, y: np.ndarray):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        d = Z.shape[1]
        w = np.zeros(d); m = np.zeros(d); v = np.zeros(d)
        starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
        ends = np.r_[starts[1:], len(groups)]
        for it in range(1, self.iters + 1):
            s = Z @ w
            p = np.empty_like(s)
            for a, b in zip(starts, ends):
                e = np.exp(s[a:b] - s[a:b].max()); p[a:b] = e / e.sum()
            g = Z.T @ (p - y) / len(starts) + self.l2 * w
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            w -= self.lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        self.w = w
        return self

    def scores(self, X):
        return ((X - self.mu) / self.sd) @ self.w


def build(cases: list[dict], cols: list[int] | None):
    Xs, gs, ys = [], [], []
    for gi, c in enumerate(cases):
        X, _ = featurize(c)
        if cols is not None:
            X = X[:, cols]
        Xs.append(X); gs.append(np.full(len(X), gi)); ys.append(np.eye(len(X))[c["engine"]])
    return np.concatenate(Xs), np.concatenate(gs), np.concatenate(ys)


def predict(model, cases, cols):
    out = []
    for c in cases:
        X, _ = featurize(c)
        if cols is not None:
            X = X[:, cols]
        out.append(int(np.argmax(model.scores(X))))
    return out


def split_key(key: str, test_frac: float = 0.25) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000
    return "test" if h < test_frac * 1000 else "train"


def pct(a, n):
    return f"{100 * a / n:.1f} %" if n else "—"


def fit(args) -> None:
    global _ARGS
    _ARGS = args
    cases = list(read_jsonl(FEATS))
    main = [c for c in cases if c["group"] == "main"]
    dis = [c for c in cases if c["group"] == "dis" and c["gemini"] is not None and c["n_syl_cache"] == c["n_vowels"]]
    _, names = featurize(main[0])
    col = {n: i for i, n in enumerate(names)}
    train = [c for c in main if split_key(c["key"]) == "train"]
    test = [c for c in main if split_key(c["key"]) == "test"]
    types = lambda cs: len({c["key"] for c in cs})
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P("# Stress from prosody: can the ear's forced alignment recover the engine's stress?")
    P()
    P(f"Script: `scripts/stress_prosody_probe.py`. Ear: run 2 (`data/xeus_ft/ckpt/best`). Clips: `data/xeus_ft/run3` "
      f"{'all splits (train + val_words + val_eps)' if args.all_splits else 'training split'}, 1–4 words per clip, every word with >= 2 vowels whose engine reading (`yiddish_labels.text_to_ipa`) "
      f"has the same vowel count as the gold target is a case. Label = the vowel the engine stresses. "
      f"Split is word-type-disjoint by lexicon key (md5 hash, 25 % of types held out).")
    P()
    P(f"| | cases | clips | word types |\n|---|---|---|---|")
    P(f"| train | {len(train)} | {len({c['clip'] for c in train})} | {types(train)} |")
    P(f"| test  | {len(test)} | {len({c['clip'] for c in test})} | {types(test)} |")
    P(f"| Gemini-disagreement slice (excluded from train) | {len(dis)} | {len({c['clip'] for c in dis})} | {types(dis)} |")
    P()
    # schwa check
    n_schwa_stressed = sum(1 for c in main if c["vowels"][c["engine"]]["v"] == "ə")
    schwa_words = collections.Counter(c["w"] for c in main if c["vowels"][c["engine"]]["v"] == "ə")
    P(f"Engine stresses ə in {n_schwa_stressed} / {len(main)} cases (should be 0): {dict(schwa_words) or '—'} "
      f"— the engine reads these with a full vowel where the gold has ə, so the transferred index lands on a schwa.")
    P()
    # vowel-level correlates
    P("## Raw correlates (train cases): stressed vs unstressed vowels, non-schwa only")
    P()
    stats = collections.defaultdict(list)
    for c in train:
        for i, v in enumerate(c["vowels"]):
            if v["v"] == "ə":
                continue
            stats["stressed" if i == c["engine"] else "unstressed"].append((v["dur"], v["rms"], v["f0_rel"]))
    P("| | n | mean duration (frames) | median duration | mean RMS | median F0 rel. clip (semitones) |\n|---|---|---|---|---|---|")
    for k in ("stressed", "unstressed"):
        a = np.array(stats[k])
        P(f"| {k} | {len(a)} | {a[:,0].mean():.2f} | {np.median(a[:,0]):.0f} | {a[:,1].mean():.4f} | {12*np.median(a[:,2]):.2f} |")
    P()
    # baselines
    P("## Word-level accuracy on held-out word types (label = engine stress)")
    P()
    def acc(preds, cs):
        return sum(p == c["engine"] for p, c in zip(preds, cs)), len(cs)
    results = {}
    results["first syllable"] = acc([0] * len(test), test)
    results["last syllable"] = acc([c["n_vowels"] - 1 for c in test], test)
    results["penultimate"] = acc([c["n_vowels"] - 2 for c in test], test)
    results["first non-schwa"] = acc([next(i for i, v in enumerate(c["vowels"]) if v["v"] != "ə") if any(v["v"] != "ə" for v in c["vowels"]) else 0 for c in test], test)
    results["longest vowel"] = acc([int(np.argmax([v["dur"] for v in c["vowels"]])) for c in test], test)
    results["loudest vowel"] = acc([int(np.argmax([v["rms"] for v in c["vowels"]])) for c in test], test)
    results["highest F0"] = acc([int(np.argmax([v["f0"] for v in c["vowels"]])) for c in test], test)
    ablations = {
        "duration only": FEATURE_GROUPS["duration"],
        "energy only": FEATURE_GROUPS["energy"],
        "F0 only": FEATURE_GROUPS["f0"],
        "dur + energy + F0 (acoustic only)": FEATURE_GROUPS["duration"] + FEATURE_GROUPS["energy"] + FEATURE_GROUPS["f0"],
        "position only": FEATURE_GROUPS["position"],
        "position + vowel identity (no audio)": FEATURE_GROUPS["position"] + FEATURE_GROUPS["vowel"],
        "acoustic + vowel identity": FEATURE_GROUPS["duration"] + FEATURE_GROUPS["energy"] + FEATURE_GROUPS["f0"] + FEATURE_GROUPS["vowel"],
        "all": names,
    }
    models = {}
    for name, feats in ablations.items():
        cols = [col[f] for f in feats]
        X, g, y = build(train, cols)
        m = CondLogit().fit(X, g, y)
        models[name] = (m, cols)
        results[name] = acc(predict(m, test, cols), test)
    P("| model | correct | n | accuracy |\n|---|---|---|---|")
    for k, (a, n) in results.items():
        P(f"| {k} | {a} | {n} | {pct(a, n)} |")
    P()
    # hard subsets
    P("## Hard subsets of the held-out cases")
    P()
    P("Vowel identity solves most words because the unstressed syllable holds ə. The audio question only bites where it does not.")
    P()
    subsets = {
        "all held-out": test,
        "no ə in the word": [c for c in test if all(v["v"] != "ə" for v in c["vowels"])],
        "engine stress not on first vowel": [c for c in test if c["engine"] != 0],
        "no ə AND stress not first": [c for c in test if c["engine"] != 0 and all(v["v"] != "ə" for v in c["vowels"])],
        ">= 2 non-ə vowels": [c for c in test if sum(v["v"] != "ə" for v in c["vowels"]) >= 2],
    }
    show = ["first syllable", "penultimate", "longest vowel", "loudest vowel", "highest F0", "duration only", "energy only", "F0 only",
            "dur + energy + F0 (acoustic only)", "position + vowel identity (no audio)", "acoustic + vowel identity", "all"]
    P("| subset | n | types | " + " | ".join(show) + " |")
    P("|---|---|---|" + "---|" * len(show))
    for sname, cs in subsets.items():
        if not cs:
            continue
        vals = []
        for mname in show:
            if mname in models:
                m, cols = models[mname]
                preds = predict(m, cs, cols)
            elif mname == "first syllable":
                preds = [0] * len(cs)
            elif mname == "penultimate":
                preds = [c["n_vowels"] - 2 for c in cs]
            elif mname == "longest vowel":
                preds = [int(np.argmax([v["dur"] for v in c["vowels"]])) for c in cs]
            elif mname == "loudest vowel":
                preds = [int(np.argmax([v["rms"] for v in c["vowels"]])) for c in cs]
            else:
                preds = [int(np.argmax([v["f0"] for v in c["vowels"]])) for c in cs]
            vals.append(pct(*acc(preds, cs)))
        P(f"| {sname} | {len(cs)} | {types(cs)} | " + " | ".join(vals) + " |")
    P()
    P("## Learned weights (acoustic only, standardised)")
    P()
    P("| feature | weight |\n|---|---|")
    m0, cols0 = models["dur + energy + F0 (acoustic only)"]
    for f, wv in sorted(zip([names[i] for i in cols0], m0.w), key=lambda t: -abs(t[1])):
        P(f"| {f} | {wv:+.3f} |")
    P()
    # by syllables
    P("## By number of vowels (held-out types)")
    P()
    bysyl = collections.defaultdict(lambda: collections.Counter())
    preds_all = predict(models["all"][0], test, models["all"][1])
    preds_ac = predict(models["acoustic + vowel identity"][0], test, models["acoustic + vowel identity"][1])
    preds_noaudio = predict(models["position + vowel identity (no audio)"][0], test, models["position + vowel identity (no audio)"][1])
    for c, pa, pc, pn in zip(test, preds_all, preds_ac, preds_noaudio):
        k = min(c["n_vowels"], 4)
        bysyl[k]["n"] += 1
        bysyl[k]["first"] += c["engine"] == 0
        bysyl[k]["all"] += pa == c["engine"]
        bysyl[k]["ac"] += pc == c["engine"]
        bysyl[k]["noaudio"] += pn == c["engine"]
    P("| vowels | n | first syllable | no audio (pos + vowel) | acoustic + vowel | all |\n|---|---|---|---|---|---|")
    for k in sorted(bysyl):
        b = bysyl[k]
        P(f"| {k}{'+' if k == 4 else ''} | {b['n']} | {pct(b['first'], b['n'])} | {pct(b['noaudio'], b['n'])} | {pct(b['ac'], b['n'])} | {pct(b['all'], b['n'])} |")
    P()
    # the acoustic-only classifier's confusion: when wrong, what does it pick?
    P("## Learned weights (all features, standardised)")
    P()
    m, cols = models["all"]
    P("| feature | weight |\n|---|---|")
    for f, wv in sorted(zip(names, m.w), key=lambda t: -abs(t[1]))[:14]:
        P(f"| {f} | {wv:+.3f} |")
    P()
    # type-level accuracy (majority vote over clips of a type)
    P("## Type-level (majority vote over a type's clips, held-out types)")
    P()
    def type_vote(preds, cs):
        votes = collections.defaultdict(collections.Counter)
        eng = {}
        for p, c in zip(preds, cs):
            votes[c["key"]][p] += 1; eng[c["key"]] = c["engine"]
        multi = [k for k in votes if sum(votes[k].values()) >= 3]
        return sum(votes[k].most_common(1)[0][0] == eng[k] for k in multi), len(multi)
    P("| model | types (>= 3 clips) correct | n | accuracy |\n|---|---|---|---|")
    for name in ("acoustic + vowel identity", "all"):
        a, n = type_vote(predict(models[name][0], test, models[name][1]), test)
        P(f"| {name} | {a} | {n} | {pct(a, n)} |")
    P()
    # 5-fold type-disjoint CV: every type is held out once, so the hard subsets have all the types there are
    P("## 5-fold type-disjoint cross-validation (pooled over all types)")
    P()
    P("The single split holds out only 30 types. Here every type is held out exactly once (fold = md5(key) mod 5), "
      "so the hard subsets carry every polysyllabic gold type the dictionary has.")
    P()
    fold_of = {c["key"]: int(hashlib.md5(c["key"].encode()).hexdigest(), 16) % 5 for c in main}
    cv_models = ["duration only", "energy only", "F0 only", "dur + energy + F0 (acoustic only)", "position only",
                 "position + vowel identity (no audio)", "acoustic + vowel identity", "all"]
    cv_pred = {m: [None] * len(main) for m in cv_models}
    for f in range(5):
        tr = [c for c in main if fold_of[c["key"]] != f]
        te_idx = [i for i, c in enumerate(main) if fold_of[c["key"]] == f]
        te = [main[i] for i in te_idx]
        for mname in cv_models:
            cols = [col[x] for x in ablations[mname]]
            X, g, y = build(tr, cols)
            mdl = CondLogit().fit(X, g, y)
            for i, pr in zip(te_idx, predict(mdl, te, cols)):
                cv_pred[mname][i] = pr
    cv_base = {
        "first syllable": [0] * len(main),
        "penultimate": [c["n_vowels"] - 2 for c in main],
        "first non-schwa": [next((i for i, v in enumerate(c["vowels"]) if v["v"] != "ə"), 0) for c in main],
        "longest vowel": [int(np.argmax([v["dur"] for v in c["vowels"]])) for c in main],
        "loudest vowel": [int(np.argmax([v["rms"] for v in c["vowels"]])) for c in main],
        "highest F0": [int(np.argmax([v["f0"] for v in c["vowels"]])) for c in main],
    }
    cv_all = {**cv_base, **cv_pred}
    summary = {}
    cv_subsets = {
        "all": lambda c: True,
        "2 vowels": lambda c: c["n_vowels"] == 2,
        "3 vowels": lambda c: c["n_vowels"] == 3,
        "4+ vowels": lambda c: c["n_vowels"] >= 4,
        "no ə in the word": lambda c: all(v["v"] != "ə" for v in c["vowels"]),
        ">= 2 non-ə vowels": lambda c: sum(v["v"] != "ə" for v in c["vowels"]) >= 2,
        "engine stress not on first vowel": lambda c: c["engine"] != 0,
        "no ə AND stress not first": lambda c: c["engine"] != 0 and all(v["v"] != "ə" for v in c["vowels"]),
        "stress not first, >= 2 non-ə vowels": lambda c: c["engine"] != 0 and sum(v["v"] != "ə" for v in c["vowels"]) >= 2,
    }
    P("| subset | n | types | random | " + " | ".join(cv_all) + " |")
    P("|---|---|---|---|" + "---|" * len(cv_all))
    for sname, fn in cv_subsets.items():
        idx = [i for i, c in enumerate(main) if fn(c)]
        if not idx:
            continue
        cs = [main[i] for i in idx]
        vals = [pct(sum(cv_all[m][i] == main[i]["engine"] for i in idx), len(idx)) for m in cv_all]
        chance = sum(1.0 / c["n_vowels"] for c in cs) / len(cs)
        summary[sname] = {"n": len(idx), "types": types(cs), "chance": round(100 * chance, 1),
                          **{m: round(100 * sum(cv_all[m][i] == main[i]["engine"] for i in idx) / len(idx), 1) for m in cv_all}}
        P(f"| {sname} | {len(idx)} | {types(cs)} | {100*chance:.1f} % | " + " | ".join(vals) + " |")
    P()
    # type-level, CV
    P("Type-level (majority vote over each type's clips, types with >= 3 clips), CV predictions:")
    P()
    P("| model | types correct | n | accuracy | of which stress-not-first types correct / n |\n|---|---|---|---|---|")
    for mname in ["first syllable", "first non-schwa", "longest vowel", "dur + energy + F0 (acoustic only)", "acoustic + vowel identity", "position + vowel identity (no audio)", "all"]:
        votes = collections.defaultdict(collections.Counter); eng = {}
        for pr, c in zip(cv_all[mname], main):
            votes[c["key"]][pr] += 1; eng[c["key"]] = c["engine"]
        ks = [k for k in votes if sum(votes[k].values()) >= 3]
        ok = [votes[k].most_common(1)[0][0] == eng[k] for k in ks]
        nf = [k for k in ks if eng[k] != 0]
        okf = sum(votes[k].most_common(1)[0][0] == eng[k] for k in nf)
        P(f"| {mname} | {sum(ok)} | {len(ks)} | {pct(sum(ok), len(ks))} | {okf} / {len(nf)} |")
    P()
    # Gemini slice
    P("## The Gemini-disagreement slice")
    P()
    P("Word types where the Gemini judge (`data/stress/stress_eval_cache.jsonl`, `ours_ok=false`, "
      "`correct_index` != engine's index at judging time, confidence >= 0.5) put the stress elsewhere than the engine. "
      "Classifiers were trained without these types. `engine` = the engine's index NOW (overrides applied since the judging); "
      "types whose current engine index already equals Gemini's are listed separately.")
    P()
    P("Two sub-slices. (a) **still open**: the engine's current index != Gemini's, so the classifier chooses between them. "
      "(b) **override applied**: an override since moved the engine onto Gemini's index; the classifier chooses between the "
      "OLD rule index (`ours_cache`) and Gemini's = the current engine's. Only types whose cached syllable count equals the gold vowel count.")
    P()
    if dis:
        table_rows = []
        for name in ("all", "acoustic + vowel identity", "dur + energy + F0 (acoustic only)", "duration only", "energy only", "F0 only", "position + vowel identity (no audio)"):
            m, cols = models[name]
            preds = predict(m, dis, cols)
            cnt = {"open": collections.Counter(), "applied": collections.Counter()}
            tv = {"open": collections.defaultdict(collections.Counter), "applied": collections.defaultdict(collections.Counter)}
            info = {}
            for p, c in zip(preds, dis):
                g, e, o = c["gemini"], c["engine"], c["ours_cache"]
                sub = "applied" if e == g else "open"
                rule = o if sub == "applied" else e  # the rule-stage index Gemini disagreed with
                cnt[sub]["clips"] += 1
                cnt[sub]["gemini" if p == g else ("rule" if p == rule else "neither")] += 1
                tv[sub][c["key"]][p] += 1
                info[c["key"]] = c
            P(f"### {name}")
            P()
            P("| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |\n|---|---|---|---|---|---|---|---|---|")
            for sub in ("open", "applied"):
                tcnt = collections.Counter()
                for k, v in tv[sub].items():
                    c = info[k]
                    g = c["gemini"]; rule = c["ours_cache"] if sub == "applied" else c["engine"]
                    p = v.most_common(1)[0][0]
                    side = "Gemini" if p == g else ("rule" if p == rule else "neither")
                    tcnt[side] += 1
                    if name == "all":
                        table_rows.append((sub, c["w"], "".join(x["v"] + " " for x in c["vowels"]).strip(), rule, g, p, sum(v.values()), side))
                P(f"| {sub} | {cnt[sub]['clips']} | {cnt[sub]['gemini']} | {cnt[sub]['rule']} | {cnt[sub]['neither']} | "
                  f"{sum(tcnt.values())} | {tcnt['Gemini']} | {tcnt['rule']} | {tcnt['neither']} |")
            P()
        P("### Per type (model: all)")
        P()
        P("| slice | word | vowels | rule idx | Gemini idx | prosody idx | clips | side |\n|---|---|---|---|---|---|---|---|")
        for r in sorted(table_rows, key=lambda t: (t[0], -t[6])):
            P("| " + " | ".join(str(x) for x in r) + " |")
        P()
    P("## Findings")
    P()
    a, h, nf = summary["all"], summary["no ə in the word"], summary["no ə AND stress not first"]
    P(f"1. **Prosody alone reproduces the engine's stress weakly.** 5-fold type-disjoint CV over {a['n']} cases / {a['types']} types: "
      f"duration only {a['duration only']} %, energy only {a['energy only']} %, F0 only {a['F0 only']} %, all three {a['dur + energy + F0 (acoustic only)']} % "
      f"— against first-syllable {a['first syllable']} %, first-non-ə {a['first non-schwa']} % and random {a['chance']} %. "
      f"The ear's vowel spans are 3–5 frames (60–100 ms) and CTC-peaky, so duration is coarse; the raw correlates go the right way "
      f"(stressed vowels are longer, louder, higher) but the separation is small.")
    P(f"2. **Vowel identity does the work, not audio.** Position + vowel identity with no audio at all: {a['position + vowel identity (no audio)']} %; "
      f"adding the acoustics: {a['acoustic + vowel identity']} % (all features {a['all']} %). In the gold lexicon the unstressed syllable is nearly always ə "
      f"(and ə is never stressed by the engine except the transferred-index anomaly נאכדעם, engine nuxdˈejm vs gold nuxdəm).")
    P(f"3. **Where it bites (no ə in the word, n={h['n']}, {h['types']} types, random {h['chance']} %)**: acoustic only {h['dur + energy + F0 (acoustic only)']} %, "
      f"vs first syllable {h['first syllable']} %, no-audio model {h['position + vowel identity (no audio)']} %, acoustic + vowel {h['acoustic + vowel identity']} %. "
      f"Non-initial stress with no ə (n={nf['n']}, {nf['types']} types): acoustic only {nf['dur + energy + F0 (acoustic only)']} % vs random {nf['chance']} % "
      f"and 0 % for every position rule. So the audio carries a real but weak stress signal (~+7–10 points over chance per clip), "
      f"and it is the only signal available on exactly the words where the rule stage fails.")
    P("4. **On the Gemini-disagreement types** (the a-initial adverbs etc., 16 types with the override already applied): the acoustic-only classifier sides with "
      "Gemini on the majority of clips and 10 / 16 types; with vowel identity 13 / 16. Two types are still open (אנגעהויבן, וויפיל) and prosody sides with the rule on both. "
      "Per-clip prosody is noisy; the per-type majority over ~40 clips is what moves.")
    P("5. **Caveat on n**: the certain gold lexicon has only 163 polysyllabic types with an engine reading of matching vowel count, 49 of them with non-initial stress. "
      "Clip counts are large (4,795) but the type count is what the split is over, and the within-type clips are mostly the same few speakers.")
    P()
    out = REPO / "data/eval/stress_prosody.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("extract", "fit", "all"), default="all")
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft/run3"))
    ap.add_argument("--ckpt", default=str(REPO / "data/xeus_ft/ckpt/best"))
    ap.add_argument("--n-clips", type=int, default=4000)
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--all-splits", action="store_true", help="draw main clips from val splits too (the ear is not being evaluated)")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.stage in ("extract", "all"):
        extract(args)
    if args.stage in ("fit", "all"):
        fit(args)


if __name__ == "__main__":
    main()
