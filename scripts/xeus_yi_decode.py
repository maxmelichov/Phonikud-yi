#!/usr/bin/env python3
"""Speech → Yiddish phones with the fine-tuned recognizer, plus a dictionary.

Three ways to read a clip:

  free        greedy CTC over the 34-phone inventory: what was said, phone by
              phone, whether or not it is a known word.

  --words     the words are known (a transcript) and the question is HOW each
              one was pronounced. Every certain-gold variant of every word is
              scored by CTC likelihood over the clip and the best one is
              reported per word — speech-to-pronunciation through the
              dictionary. This is the tool for homographs and for words with
              several accepted readings (də/di, hut/hɔt).

  --snap      no transcript: decode freely, then segment the phone string into
              dictionary words by dynamic programming over edit distance. Runs
              of phones that match no word are kept as-is and flagged. This is
              how a recognizer output becomes a checkable claim about words.

The dictionary is data/xeus_ft/dictionary.json — the certain gold words only.
Anything the engine merely guessed is deliberately absent: the decoder must
not snap audio onto a pronunciation nobody has verified.

Usage:
  python scripts/xeus_yi_decode.py clip.wav
  python scripts/xeus_yi_decode.py clip.wav --words "מיט א פאר יאר צוריק"
  python scripts/xeus_yi_decode.py clip.wav --snap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import (  # noqa: E402
    YI_BLANK, YI_VOCAB, build_yi_head, edit_distance, greedy_decode, lexicon_key,
    load_pretrained, yi_ids, yi_logits,
)

SR = 16000


def load_finetuned(ckpt: Path, device: str = "cpu"):
    """Pretrained architecture + fine-tuned weights + Yiddish head."""
    import torch
    from safetensors.torch import load_file
    _, inner = load_pretrained(device)
    head = build_yi_head(inner)
    if (ckpt / "inner.safetensors").exists():
        state = load_file(str(ckpt / "inner.safetensors"), device=device)
        missing, unexpected = inner.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  note: {len(missing)} missing / {len(unexpected)} unexpected keys", file=sys.stderr)
        head.load_state_dict(torch.load(ckpt / "yi_head.pt", map_location=device))
    else:
        print("  note: no checkpoint found, using the warm-started head on the pretrained encoder", file=sys.stderr)
    inner.eval()
    head.eval()
    return inner, head.to(device)


def read_wav(path: Path):
    import numpy as np
    import soundfile as sf
    import torch
    if path.suffix == ".npy":
        wav = np.load(path).astype("float32") / 32767.0
        return torch.from_numpy(wav)
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return torch.from_numpy(wav)


def log_probs(inner, head, wav, device):
    import torch
    speech = wav.unsqueeze(0).to(device)
    lens = torch.tensor([speech.shape[1]], device=device)
    with torch.no_grad():
        logits, flens = yi_logits(inner, head, speech, lens)
    return torch.log_softmax(logits.float(), -1)[0, : int(flens[0])]


def ctc_nll(lp, phones: list[str]) -> float:
    import torch
    import torch.nn.functional as F
    ids = yi_ids(phones)
    if not ids or lp.shape[0] < len(ids):
        return float("inf")
    tgt = torch.tensor([ids], device=lp.device)
    return float(F.ctc_loss(lp.unsqueeze(1), tgt, torch.tensor([lp.shape[0]], device=lp.device),
                            torch.tensor([len(ids)], device=lp.device), blank=YI_BLANK,
                            reduction="sum", zero_infinity=True))


def load_dictionary(path: Path) -> dict[str, dict]:
    """dictionary.json re-indexed by lexicon key, so pointed, final-letter and
    ligature spellings of a word all reach its entry."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for word, meta in raw.items():
        out[meta.get("key") or lexicon_key(word)] = {**meta, "word": word}
    return out


def choose_pronunciations(lp, words: list[str], dictionary: dict[str, dict]) -> list[dict]:
    """Per word, the dictionary variant the clip supports best (left to right).

    Words not in the dictionary keep no pronunciation — the decoder does not
    invent one — but still occupy their place so the neighbours are scored in
    the right context.
    """
    variants = [dictionary.get(lexicon_key(w), {}).get("variants", []) for w in words]
    choice = [0] * len(words)

    def seq(ch):
        out = []
        for v, c in zip(variants, ch):
            if v:
                out.extend(v[c])
        return out

    base = ctc_nll(lp, seq(choice))
    result = []
    for i, v in enumerate(variants):
        if len(v) > 1:
            scores = []
            for c in range(len(v)):
                trial = list(choice)
                trial[i] = c
                scores.append(ctc_nll(lp, seq(trial)))
            choice[i] = min(range(len(v)), key=lambda c: scores[c])
            margin = sorted(scores)[1] - sorted(scores)[0] if len(scores) > 1 else 0.0
        else:
            margin = 0.0
        result.append({
            "word": words[i],
            "in_dictionary": bool(v),
            "pronunciation": " ".join(v[choice[i]]) if v else None,
            "alternatives": [" ".join(x) for k, x in enumerate(v) if k != choice[i]],
            "margin_nats": round(margin, 2),
        })
    return result, base, ctc_nll(lp, seq(choice))


def snap_to_dictionary(phones: list[str], dictionary: dict[str, dict], max_cost: float = 0.34) -> list[dict]:
    """Segment a phone string into dictionary words by edit-distance DP.

    Cost of a word = edit distance between its pronunciation and the phone
    span, normalised by pronunciation length; only matches at or under
    ``max_cost`` are allowed. Spans no word can claim cost 1 per phone and come
    back as ``{"word": None, "phones": ...}``.
    """
    prons: list[tuple[str, list[str]]] = []
    for meta in dictionary.values():
        for v in meta["variants"]:
            prons.append((meta["word"], v))
    n = len(phones)
    INF = float("inf")
    best = [INF] * (n + 1)
    back: list[tuple | None] = [None] * (n + 1)
    best[0] = 0.0
    for i in range(n):
        if best[i] == INF:
            continue
        # unknown phone
        if best[i] + 1.0 < best[i + 1]:
            best[i + 1] = best[i] + 1.0
            back[i + 1] = (i, None, None, 1.0)
        for w, v in prons:
            L = len(v)
            for span in range(max(1, L - 2), min(n - i, L + 2) + 1):
                d = edit_distance(v, phones[i:i + span]) / L
                if d > max_cost:
                    continue
                cost = best[i] + d * L
                if cost < best[i + span]:
                    best[i + span] = cost
                    back[i + span] = (i, w, v, d)
    out = []
    j = n
    while j > 0:
        i, w, v, d = back[j]
        out.append({"word": w, "phones": " ".join(phones[i:j]),
                    "pronunciation": " ".join(v) if v else None, "cost": round(d, 3)})
        j = i
    return out[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="wav/mp3 (any rate) or a segment .npy")
    ap.add_argument("--ckpt", default=str(REPO / "data/xeus_ft/ckpt/best"))
    ap.add_argument("--dictionary", default=str(REPO / "data/xeus_ft/dictionary.json"))
    ap.add_argument("--words", default=None, help="space-separated transcript to score pronunciations for")
    ap.add_argument("--snap", action="store_true", help="segment the free decode into dictionary words")
    ap.add_argument("--greedy", action="store_true",
                    help="plain greedy CTC for the free decode (default: dictionary-guided prefix beam, "
                         "which roughly doubles single-word accuracy on unseen words — see xeus_beam.py)")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--blank-penalty", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inner, head = load_finetuned(Path(args.ckpt), device)
    wav = read_wav(Path(args.audio))
    lp = log_probs(inner, head, wav, device)
    dictionary = load_dictionary(Path(args.dictionary))
    if args.greedy:
        free = greedy_decode(lp.unsqueeze(0), torch.tensor([lp.shape[0]]))[0]
        decoder = "greedy"
    else:
        from xeus_beam import build_trie, prefix_beam
        raw = json.loads(Path(args.dictionary).read_text(encoding="utf-8"))
        free = prefix_beam(lp.cpu(), build_trie(raw), beam=args.beam, blank_penalty=args.blank_penalty)
        decoder = f"beam{args.beam}+dictionary"
    result = {"audio": args.audio, "seconds": round(wav.shape[0] / SR, 2), "decoder": decoder, "free": " ".join(free)}

    if args.words or args.snap:
        if args.words:
            per_word, nll_first, nll_best = choose_pronunciations(lp, args.words.split(), dictionary)
            result["words"] = per_word
            result["nll_first_variants"] = round(nll_first, 2)
            result["nll_chosen_variants"] = round(nll_best, 2)
        if args.snap:
            result["snap"] = snap_to_dictionary(free, dictionary)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
