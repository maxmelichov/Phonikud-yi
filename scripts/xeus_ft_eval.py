#!/usr/bin/env python3
"""Score a fine-tuned Yiddish recognizer against the pretrained baseline.

Both are measured on the same segments, so the comparison is paired:

  val_words   segments containing a held-out WORD TYPE — never seen in training
              in any clip. Measures whether the model learned Yiddish phones or
              memorised word shapes.
  val_eps     segments from held-out EPISODES (any word). Measures whether it
              transfers to recordings it has not heard. (The corpus is one
              host, so this is recording- not speaker-generalisation.)

Reports PER, exact-match rate, recall on the phones the pretrained recognizer
mishears (ʦ, aj, ej, z, ʃ …), and per-held-out-type exact match on the
single-word segments. Writes eval.json and eval.md next to the checkpoint.

Usage:
  python scripts/xeus_ft_eval.py --data data/xeus_ft --ckpt data/xeus_ft/ckpt/best
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import HARD_PHONES, PerAccumulator, read_jsonl  # noqa: E402
from xeus_ft_train import Segments, baseline, evaluate  # noqa: E402
from xeus_yi_decode import load_finetuned  # noqa: E402


def fmt(x: float) -> str:
    return "—" if x != x else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/xeus_ft"))
    ap.add_argument("--ckpt", default=None, help="defaults to <data>/ckpt/best")
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-seconds", type=float, default=96.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--noise-snr", type=float, default=15.0,
                    help="also score every split with white noise added at this SNR (dB); "
                         "the baseline column for that row is the pretrained model on the same "
                         "noisy audio. 0 disables.")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    ckpt = Path(args.ckpt or data / "ckpt/best")
    # A directory with no weights scores the pretrained encoder with the
    # warm-started head — the "before" report — and gets its eval files too.
    ckpt.mkdir(parents=True, exist_ok=True)
    split = json.loads((data / "split.json").read_text(encoding="utf-8"))
    val_types = set(split["val_types"])
    rows = list(read_jsonl(data / "segments.jsonl"))
    by = {s: [r for r in rows if r["split"] == s] for s in ("val_words", "val_eps")}
    if args.limit:
        by = {s: v[: args.limit] for s, v in by.items()}

    inner, head = load_finetuned(ckpt, device)
    use_amp = device == "cuda"
    report: dict = {"ckpt": str(ckpt), "splits": {}}

    def add_noise(speech, snr_db: float):
        g = torch.Generator(device="cpu").manual_seed(7)
        rms = speech.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-5)
        noise = torch.randn(speech.shape, generator=g).to(speech.device) * rms / (10.0 ** (snr_db / 20.0))
        return (speech + noise).clamp(-1.0, 1.0)

    base_model = None
    if args.noise_snr:
        # The pretrained model has to be re-run on the noisy audio: the stored
        # baseline column was computed on clean clips.
        from xeus_ft_common import load_pretrained, xeus_vocab
        from xeus_map import fold_phone_string
        from xeus_ft_prepare import greedy_fold
        _, base_model = load_pretrained(device)
        base_vocab, _ = xeus_vocab(base_model)
    per_type: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: {"base": [], "ft": []})

    for s, seg_rows in by.items():
        ds = Segments(seg_rows, data / "seg")
        base = baseline(ds)
        ft = PerAccumulator()
        inner.eval()
        head.eval()
        import random
        from xeus_ft_common import greedy_decode, yi_logits
        from xeus_ft_train import collate
        with torch.no_grad():
            for idx in ds.batches(args.eval_seconds, shuffle=False, rng=random.Random(0)):
                speech, lens, _, _ = collate(ds, idx, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits, flens = yi_logits(inner, head, speech, lens)
                for i, hyp in zip(idx, greedy_decode(logits.float(), flens)):
                    r = ds.rows[i]
                    ft.add(r["target"], hyp)
                    if len(r["words"]) == 1 and r["words"][0]["key"] in val_types:
                        k = r["words"][0]["key"]
                        per_type[k]["base"].append(int(r["baseline"] == r["target"]))
                        per_type[k]["ft"].append(int(hyp == r["target"]))
        report["splits"][s] = {"baseline": base.summary(), "finetuned": ft.summary()}

        if base_model is not None:
            nb, nf = PerAccumulator(), PerAccumulator()
            with torch.no_grad():
                for idx in ds.batches(args.eval_seconds, shuffle=False, rng=random.Random(0)):
                    speech, lens, _, _ = collate(ds, idx, device)
                    speech = add_noise(speech, args.noise_snr)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        logits, flens = yi_logits(inner, head, speech, lens)
                        blg, bfl = base_model.ctc_logits(speech, lens)
                    blp = torch.log_softmax(blg.float(), -1)
                    for k, (i, hyp) in enumerate(zip(idx, greedy_decode(logits.float(), flens))):
                        r = ds.rows[i]
                        nf.add(r["target"], hyp)
                        nb.add(r["target"], greedy_fold(blp[k, : int(bfl[k])], base_vocab, base_model.blank_id, fold_phone_string))
            report["splits"][f"{s} @ {args.noise_snr:g} dB SNR"] = {"baseline": nb.summary(), "finetuned": nf.summary()}

    types = []
    for k, d in per_type.items():
        if d["ft"]:
            types.append({"key": k, "word": split["val_type_words"].get(k, k), "n": len(d["ft"]),
                          "base_exact": sum(d["base"]) / len(d["base"]), "ft_exact": sum(d["ft"]) / len(d["ft"])})
    types.sort(key=lambda t: -t["n"])
    report["held_out_types"] = types
    if types:
        report["held_out_types_summary"] = {
            "types": len(types),
            "base_exact_macro": sum(t["base_exact"] for t in types) / len(types),
            "ft_exact_macro": sum(t["ft_exact"] for t in types) / len(types),
            "improved": sum(t["ft_exact"] > t["base_exact"] for t in types),
            "regressed": sum(t["ft_exact"] < t["base_exact"] for t in types),
        }

    (ckpt / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# PhoneticXeus → Yiddish fine-tune: evaluation", "", f"Checkpoint: `{ckpt}`", ""]
    md += ["| split | segments | PER baseline | PER fine-tuned | exact baseline | exact fine-tuned |", "|---|---|---|---|---|---|"]
    for s, v in report["splits"].items():
        b, f = v["baseline"], v["finetuned"]
        md.append(f"| {s} | {f['segments']:,} | {fmt(b['per'])} | **{fmt(f['per'])}** | {fmt(b['exact_match'])} | **{fmt(f['exact_match'])}** |")
    md += ["", "## Phones the pretrained recognizer mishears (val_words recall)", "",
           "| phone | n | baseline recall | fine-tuned recall | baseline heard as | fine-tuned heard as |", "|---|---|---|---|---|---|"]
    vw = report["splits"].get("val_words", {})
    for p in HARD_PHONES:
        b = vw.get("baseline", {}).get("hard_phones", {}).get(p)
        f = vw.get("finetuned", {}).get("hard_phones", {}).get(p)
        if not b or not b["n"]:
            continue
        ha = lambda h: ", ".join(f"{k} {v}" for k, v in h["heard_as"])  # noqa: E731
        md.append(f"| {p} | {b['n']} | {fmt(b['recall'])} | **{fmt(f['recall'])}** | {ha(b)} | {ha(f)} |")
    if types:
        s = report["held_out_types_summary"]
        md += ["", f"## Held-out word types ({s['types']} with single-word clips)", "",
               f"Macro exact match: baseline {fmt(s['base_exact_macro'])} → fine-tuned **{fmt(s['ft_exact_macro'])}**; "
               f"improved {s['improved']}, regressed {s['regressed']}.", "",
               "| word | clips | baseline exact | fine-tuned exact |", "|---|---|---|---|"]
        for t in types[:40]:
            md.append(f"| {t['word']} | {t['n']} | {fmt(t['base_exact'])} | **{fmt(t['ft_exact'])}** |")
    (ckpt / "eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
