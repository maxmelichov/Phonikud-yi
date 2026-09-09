#!/usr/bin/env python3
"""Fine-tune ivrit.ai's Yiddish Whisper on the corpus: a Hasidic transcriber.

Zero-shot, ``ivrit-ai/yi-whisper-large-v3`` scores ~50% WER on this corpus,
mostly because it writes another community's orthography. The corpus holds
196 h of the host's speech with Yiddish Labs transcripts in the conventions
every other tool here expects — the training pairs for a transcriber that
writes them, which can then transcribe the rest of yiddish24 for nothing.

LoRA on the attention and feed-forward projections of encoder and decoder
(peft), bf16, one epoch over the 30 s chunks of every episode except the six
held-out ones (the test episode + retrain3's val episodes). WER is measured
on the held-out episodes after the same normalisation as
whisper_yi_probe.py: nikud stripped, punctuation dropped, finals folded.

  python scripts/whisper_yi_finetune.py --corpus data/corpus/yiddish_tts_dataset.tsv --root . --out models/yi_whisper_hasidic
  python scripts/whisper_yi_finetune.py --eval-only --adapter models/yi_whisper_hasidic/adapter --n-eval 200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from xeus_ft_common import edit_distance, lexicon_key  # noqa: E402

_HEB = re.compile(r"[֐-׿][֐-׿'\"׳״-]*")
MODEL = "ivrit-ai/yi-whisper-large-v3"
HELD_OUT = ("100313", "104192", "104690", "113370", "58622", "94226")
SR = 16000


def norm_words(text: str) -> list[str]:
    return [lexicon_key(w).strip("'\"-") for w in _HEB.findall(text) if lexicon_key(w).strip("'\"-")]


def load_rows(corpus: Path):
    csv.field_size_limit(10_000_000)
    return list(csv.DictReader(open(corpus, encoding="utf-8"), delimiter="\t"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "data/corpus/yiddish_tts_dataset.tsv"))
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--out", default=str(REPO / "models/yi_whisper_hasidic"))
    ap.add_argument("--adapter", default=None, help="eval-only: LoRA adapter dir")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=240, help="held-out chunks to transcribe for WER")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from xeus_ft_prepare import load_audio
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(Path(args.corpus))
    train_rows = [r for r in rows if r["episode"] not in HELD_OUT]
    test_rows = [r for r in rows if r["episode"] in HELD_OUT]
    random.Random(0).shuffle(train_rows)
    random.Random(1).shuffle(test_rows)
    if args.limit:
        train_rows = train_rows[: args.limit]
    test_rows = test_rows[: args.n_eval]
    print(f"train chunks {len(train_rows):,}  eval chunks {len(test_rows)} (held-out episodes)", flush=True)

    processor = WhisperProcessor.from_pretrained(MODEL)
    # The labels must carry the same prefix the decoder is given at generation
    # time — <|startoftranscript|><|yi|><|transcribe|><|notimestamps|>. Without
    # set_prefix_tokens the tokenizer emits only <|startoftranscript|><|notimestamps|>,
    # the model trains on that, and generation with the language token forced is
    # off-distribution: the first run did exactly this and got worse (63% → 66% WER).
    processor.tokenizer.set_prefix_tokens(language="yi", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.generation_config.language = "yi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        from peft import LoraConfig, get_peft_model
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        cfg = LoraConfig(r=args.lora_r, lora_alpha=64, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        model.to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
        steps_total = max(1, math.ceil(len(train_rows) / args.batch / args.grad_accum * args.epochs))
        warm = max(1, int(0.05 * steps_total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps_total - warm)))))

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=args.workers)

        def prep(r):
            wav = load_audio(Path(args.root) / f"data/chunks/{r['episode']}/chunk_{int(r['chunk_idx']):05d}.mp3")
            return wav[: SR * 30], r["text"]

        model.train()
        t0 = time.time()
        step = 0
        log = open(out / "train_log.jsonl", "a", encoding="utf-8")
        n_batches = math.ceil(len(train_rows) / args.batch)
        for ep in range(math.ceil(args.epochs)):
            futures = [pool.submit(prep, r) for r in train_rows[: args.batch * 24]]
            queue_idx = args.batch * 24
            run, k = 0.0, 0
            for bi in range(n_batches):
                items = []
                for _ in range(args.batch):
                    if not futures:
                        break
                    items.append(futures.pop(0).result())
                    if queue_idx < len(train_rows):
                        futures.append(pool.submit(prep, train_rows[queue_idx])); queue_idx += 1
                if not items:
                    break
                feats = processor.feature_extractor([w for w, _ in items], sampling_rate=SR, return_tensors="pt").input_features
                labels = processor.tokenizer([t for _, t in items], padding=True, return_tensors="pt", truncation=True, max_length=448).input_ids
                labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
                feats = feats.to(device, dtype=model.dtype if hasattr(model, "dtype") else torch.bfloat16)
                loss = model(input_features=feats, labels=labels.to(device)).loss / args.grad_accum
                loss.backward()
                run += loss.item() * args.grad_accum; k += 1
                if (bi + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                    if step % 25 == 0:
                        el = time.time() - t0
                        print(f"  ep{ep + 1} step {step}/{steps_total} loss {run / k:.4f} lr {sched.get_last_lr()[0]:.2e} {el / 60:.1f} min", flush=True)
                        log.write(json.dumps({"step": step, "loss": run / k}) + "\n"); run, k = 0.0, 0
                if args.epochs < 1 and bi + 1 >= n_batches * args.epochs:
                    break
        model.save_pretrained(out / "adapter")
        print(f"saved adapter to {out / 'adapter'}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        model.eval()
    else:
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model.to(device).eval()

    # ---- WER on held-out chunks, before (zero-shot) is a separate run without --adapter ----
    tot_err = tot_ref = 0
    samples = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(test_rows), 4):
            batch = test_rows[i:i + 4]
            wavs = [load_audio(Path(args.root) / f"data/chunks/{r['episode']}/chunk_{int(r['chunk_idx']):05d}.mp3")[: SR * 30] for r in batch]
            feats = processor.feature_extractor(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(device, dtype=next(model.parameters()).dtype)
            ids = model.generate(input_features=feats, language="yi", task="transcribe", max_new_tokens=440, num_beams=1)
            hyps = processor.batch_decode(ids, skip_special_tokens=True)
            for r, hyp in zip(batch, hyps):
                ref_w, hyp_w = norm_words(r["text"]), norm_words(hyp)
                err = edit_distance(ref_w, hyp_w)
                tot_err += err; tot_ref += len(ref_w)
                if len(samples) < 12:
                    samples.append({"ref": " ".join(ref_w[:16]), "hyp": " ".join(hyp_w[:16]), "wer": round(err / max(1, len(ref_w)), 3)})
    wer = tot_err / max(1, tot_ref)
    tag = "adapter" if (args.adapter or not args.eval_only) else "zero-shot"
    res = {"model": MODEL, "which": tag, "eval_chunks": len(test_rows), "wer": round(wer, 4),
           "seconds_per_chunk": round((time.time() - t0) / max(1, len(test_rows)), 2), "samples": samples}
    (out / f"wer_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"WER ({tag}) over {len(test_rows)} held-out chunks: {wer:.3f}   ({res['seconds_per_chunk']} s/chunk)", flush=True)
    for s in samples[:6]:
        print("  ref:", s["ref"]); print("  hyp:", s["hyp"], f"(wer {s['wer']})")


if __name__ == "__main__":
    main()
