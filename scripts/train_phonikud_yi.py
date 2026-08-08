#!/usr/bin/env python3
"""
Fine-tune phonikud-yi on the masked-supervision retrain dataset.

Warm start from the shipping round-4 checkpoint (`models/phonikud_yi/round4/stageB`)
-- this is a *continued* fine-tune, never a from-scratch run: the encoder and all
three Yiddish heads keep their weights and only the supervision changes.

What is new relative to the upstream trainer (`phonikud/model/src/train/yi_train.py`,
untracked): the dataset carries a per-word supervision mask. Only words whose
reading the FROZEN v3 G2P engine verifies from its gold lexicon contribute
gradient; every other character is labelled IGNORE (-100) and its pointing is
left alone. Loss and every accuracy number below are therefore computed on
supervised positions only.

Label scheme, tokenizer and head layout are taken from the checkpoint itself
(round-4 "Hebrew-mirror": yi_nikud 22-way, yi_shin 2-way on ש only, yi_rafe
2-way on every Hebrew char), so the output dir stays a drop-in for
`scripts/export_onnx.py`, `scripts/eval_oov_wordlevel.py` and `scripts/point_text.py`.

Usage:
    # smoke
    python scripts/train_phonikud_yi.py --limit 300 --epochs 1 --batch-size 4 \
        --out models/phonikud_yi_v2_smoke
    # full
    python scripts/train_phonikud_yi.py --epochs 3 --batch-size 8 \
        --out models/phonikud_yi_v2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "phonikud" / "model"))

from src.model.phonikud_model import PhonikudModel  # noqa: E402

from phonikud_yi_data import (  # noqa: E402
    IGNORE,
    Collator,
    MirrorLabels,
    make_loader,
    read_jsonl,
)

DEFAULT_INIT = REPO / "models/phonikud_yi/round4/stageB"
DEFAULT_DATA = REPO / "data/retrain"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def move(batch, device):
    batch.input = {k: v.to(device) for k, v in batch.input.items()}
    batch.nikud = batch.nikud.to(device)
    batch.shin = batch.shin.to(device)
    batch.rafe = batch.rafe.to(device)
    return batch


def head_loss(out, batch, loss_fct) -> torch.Tensor:
    """Sum of the three head cross-entropies over supervised positions.

    A head whose targets are ALL `IGNORE` in this batch must be skipped, not
    summed: `CrossEntropyLoss(ignore_index=...)` divides by a zero weight and
    returns NaN, which under masked supervision is routine (`yi_shin` is
    supervised only on ש inside a supervised word, so most small batches have
    none) and would otherwise NaN out every gradient in the model.
    """
    loss = None
    for logits, target in ((out.yi_nikud_logits, batch.nikud),
                           (out.yi_shin_logits, batch.shin),
                           (out.yi_rafe_logits, batch.rafe)):
        if not bool((target != IGNORE).any()):
            continue
        l = loss_fct(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss = l if loss is None else loss + l
    if loss is None:
        return out.yi_nikud_logits.sum() * 0.0
    return loss


@torch.no_grad()
def evaluate(model, loader, device, loss_fct) -> dict:
    """Accuracy over SUPERVISED positions only (everything else is IGNORE)."""
    model.eval()
    tot = dict(nikud_ok=0, nikud_n=0, shin_ok=0, shin_n=0, rafe_ok=0, rafe_n=0,
               char_ok=0, char_n=0, marked_ok=0, marked_n=0, loss=0.0, batches=0)
    for batch in loader:
        batch = move(batch, device)
        out = model(batch.input)
        tot["loss"] += float(head_loss(out, batch, loss_fct))
        tot["batches"] += 1

        pn = out.yi_nikud_logits.argmax(-1)
        ps = out.yi_shin_logits.argmax(-1)
        pr = out.yi_rafe_logits.argmax(-1)

        m_n = batch.nikud != IGNORE
        m_s = batch.shin != IGNORE
        m_r = batch.rafe != IGNORE

        tot["nikud_ok"] += int(((pn == batch.nikud) & m_n).sum())
        tot["nikud_n"] += int(m_n.sum())
        tot["shin_ok"] += int(((ps == batch.shin) & m_s).sum())
        tot["shin_n"] += int(m_s.sum())
        tot["rafe_ok"] += int(((pr == batch.rafe) & m_r).sum())
        tot["rafe_n"] += int(m_r.sum())

        # a character is right only if every head that supervises it is right
        ok = (pn == batch.nikud) & ((ps == batch.shin) | ~m_s) & ((pr == batch.rafe) | ~m_r)
        tot["char_ok"] += int((ok & m_n).sum())
        tot["char_n"] += int(m_n.sum())
        marked = m_n & ((batch.nikud != 0) | (batch.rafe == 1) | m_s)
        tot["marked_ok"] += int((ok & marked).sum())
        tot["marked_n"] += int(marked.sum())

    pct = lambda a, b: round(100.0 * a / b, 3) if b else 0.0  # noqa: E731
    model.train()
    # A frozen encoder must go back to eval mode: its dropout is not just
    # useless (no gradients flow), it crashes MPS -- scaled_dot_product_attention
    # does not support dropout there. This is what killed the first full run at
    # step 1001, right after the first eval.
    if getattr(model, "_frozen_encoder", False):
        model.bert.eval()
    return {
        "loss": round(tot["loss"] / max(tot["batches"], 1), 4),
        "char_acc": pct(tot["char_ok"], tot["char_n"]),
        "marked_char_acc": pct(tot["marked_ok"], tot["marked_n"]),
        "nikud_acc": pct(tot["nikud_ok"], tot["nikud_n"]),
        "shin_acc": pct(tot["shin_ok"], tot["shin_n"]),
        "rafe_acc": pct(tot["rafe_ok"], tot["rafe_n"]),
        "n_supervised_chars": tot["char_n"],
    }


def save(model, tokenizer, init_dir: Path, out: Path, meta: dict):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    # the custom modelling code is shipped in-checkpoint (auto_map), so the saved
    # dir stays loadable standalone -- same contract as upstream `yi_train.save()`
    for name in ("phonikud_model.py", "dicta_model.py", "yi_labels.json"):
        src = Path(init_dir) / name
        # resuming with --init <out>/best makes src and dst the same file
        if src.exists() and src.resolve() != (out / name).resolve():
            shutil.copy2(src, out / name)
    (out / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--init", type=Path, default=DEFAULT_INIT,
                    help="warm-start checkpoint (NOT from scratch)")
    ap.add_argument("--out", type=Path, default=REPO / "models/phonikud_yi_v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5, help="encoder LR")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--limit", type=int, default=None, help="TRAIN rows (smoke runs)")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="stage-A style: train the Yiddish heads only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="mid-epoch val every N optimizer steps (0 = epoch end only)")
    ap.add_argument("--test-peek", type=int, default=50,
                    help="test rows evaluated for visibility only -- never trained on")
    ap.add_argument("--select-on", default="val_char",
                    choices=("val_char", "val_loss", "test_char"),
                    help="checkpoint selection signal for `best`. `val_char` is "
                         "val char_acc with val loss as tie-break (val is "
                         "near-saturated, so char_acc alone ties constantly and "
                         "makes selection arbitrary); `test_char` selects on the "
                         "test-peek split -- a real signal but it makes the test "
                         "set no longer held out, so the number it reports stops "
                         "being an honest estimate.")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.init)
    model = PhonikudModel.from_pretrained(args.init)
    labels = MirrorLabels(model.config)
    assert model.menaked.yi_nikud_cls is not None, "checkpoint has no Yiddish mirror heads"
    print(f"warm start: {args.init}  "
          f"({sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(labels.yi_nikud_classes)} nikud classes)", flush=True)

    if args.freeze_encoder:
        model.freeze_encoder_only()
        model._frozen_encoder = True
    else:
        model.unfreeze_all()
        if device == "mps":
            # MPS SDPA cannot do dropout; zero it rather than crash mid-run.
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
            for attr in ("attention_probs_dropout_prob", "hidden_dropout_prob"):
                if hasattr(model.config, attr):
                    setattr(model.config, attr, 0.0)
    model.to(device)

    max_chars = args.max_length - 32
    t0 = time.perf_counter()
    train_ex = read_jsonl(args.data / "train.jsonl", max_chars, args.limit)
    val_ex = read_jsonl(args.data / "val.jsonl", max_chars, args.val_limit)
    test_ex = read_jsonl(args.data / "test.jsonl", max_chars, args.test_peek or None)
    print(f"data: train {len(train_ex)} chunks / val {len(val_ex)} / "
          f"test-peek {len(test_ex)}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    # GUARD: the collator places labels via return_offsets_mapping, which only a
    # fast tokenizer provides. transformers 5.x loads this checkpoint with the
    # slow BertTokenizer, offsets silently vanish, ~everything becomes IGNORE,
    # and an unfrozen encoder then destroys itself chasing a handful of labels
    # (this exact failure burned a full RunPod run). Fail loudly instead.
    if not getattr(tokenizer, "is_fast", False):
        sys.exit("FATAL: tokenizer is not fast (no offset mapping) — "
                 "pin transformers to a version that loads BertTokenizerFast "
                 "for this checkpoint (4.56.x works).")
    _probe = Collator(tokenizer, labels, args.max_length)(val_ex[: min(64, len(val_ex))])
    _placed = int((_probe.nikud != IGNORE).sum())
    _expected = sum(sum(1 for m in ex.marks if m is not None)
                    for ex in val_ex[: min(64, len(val_ex))])
    if _expected and _placed < 0.5 * _expected:
        sys.exit(f"FATAL: label collapse — collator placed {_placed} supervised "
                 f"labels where the dataset has {_expected}. Tokenizer/offset "
                 f"mismatch; refusing to train.")

    train_loader = make_loader(train_ex, tokenizer, labels, args.batch_size, True,
                               args.max_length, args.num_workers)
    val_loader = make_loader(val_ex, tokenizer, labels, args.batch_size, False,
                             args.max_length, args.num_workers)
    test_loader = make_loader(test_ex, tokenizer, labels, args.batch_size, False,
                              args.max_length, args.num_workers)

    head_names = ("menaked.yi_nikud_cls", "menaked.yi_shin_cls", "menaked.yi_rafe_cls")
    head_params, enc_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if any(name.startswith(h) for h in head_names) else enc_params).append(p)
    groups = [{"params": head_params, "lr": args.head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr})
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * args.warmup), total_steps)
    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    log = log_path.open("a", encoding="utf-8")

    def emit(rec):
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()

    emit({"event": "start", "args": {k: str(v) for k, v in vars(args).items()},
          "device": device, "train_chunks": len(train_ex), "total_steps": total_steps})

    print("baseline (warm-start weights, before any update):", flush=True)
    base_val = evaluate(model, val_loader, device, loss_fct)
    base_test = evaluate(model, test_loader, device, loss_fct)
    print(f"  val  {base_val}\n  test {base_test}", flush=True)
    emit({"event": "baseline", "val": base_val, "test": base_test})

    # ---- checkpoint selection -------------------------------------------------
    # `best` is chosen on a TOTALLY ORDERED key, highest wins. Selecting on val
    # char_acc alone is a no-op on this dataset: the val episodes were in round
    # 4's training data and carry the same supervision, so char_acc is saturated
    # and identical epoch to epoch, and every `>` comparison loses. The tie-break
    # on val loss keeps the choice meaningful when accuracy stops moving.
    def select_key(v: dict, t: dict) -> tuple:
        if args.select_on == "val_char":
            return (v["char_acc"], -v["loss"])
        if args.select_on == "val_loss":
            return (-v["loss"],)
        return (t["char_acc"], -t["loss"])

    if args.select_on == "test_char":
        print("WARNING: --select-on test_char selects checkpoints on the test "
              "split; its reported accuracy is no longer a held-out estimate.",
              flush=True)

    best: tuple | None = None
    best_meta: dict | None = None

    def consider(epoch: int, step: int, v: dict, t: dict) -> bool:
        nonlocal best, best_meta
        key = select_key(v, t)
        if best is not None and key <= best:
            return False
        best, best_meta = key, {"epoch": epoch, "step": step, "val": v,
                                "test_peek": t, "select_on": args.select_on,
                                "select_key": list(key)}
        save(model, tokenizer, args.init, args.out / "best", best_meta)
        return True

    step = 0
    run_t0 = time.perf_counter()
    first_losses, last_losses = [], []
    model.train()
    if args.freeze_encoder:
        model.bert.eval()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        running, nb = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            batch = move(batch, device)
            out = model(batch.input)
            loss = head_loss(out, batch, loss_fct)
            (loss / args.grad_accum).backward()
            lv = float(loss.detach())
            running += lv
            nb += 1
            if len(first_losses) < 20:
                first_losses.append(lv)
            last_losses.append(lv)
            if len(last_losses) > 20:
                last_losses.pop(0)

            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if args.log_every and step % args.log_every == 0:
                    el = time.perf_counter() - run_t0
                    print(f"  e{epoch} step {step}/{total_steps} "
                          f"loss {running / max(nb, 1):.4f} "
                          f"({el:.0f}s, {el / max(step, 1):.2f} s/step)", flush=True)
                    emit({"event": "step", "epoch": epoch, "step": step,
                          "loss": round(running / max(nb, 1), 4),
                          "elapsed_s": round(el, 1)})
                    running, nb = 0.0, 0

                if args.eval_every and step % args.eval_every == 0:
                    v = evaluate(model, val_loader, device, loss_fct)
                    t = evaluate(model, test_loader, device, loss_fct)
                    print(f"  [step {step}] val {v} | test-peek {t}", flush=True)
                    emit({"event": "eval", "epoch": epoch, "step": step,
                          "val": v, "test_peek": t})
                    consider(epoch, step, v, t)

            if args.max_hours and (time.perf_counter() - run_t0) > args.max_hours * 3600:
                print("max-hours reached, stopping", flush=True)
                break

        v = evaluate(model, val_loader, device, loss_fct)
        t = evaluate(model, test_loader, device, loss_fct)
        dt = time.perf_counter() - ep_t0
        print(f"epoch {epoch} done in {dt:.0f}s | val {v} | test-peek {t}", flush=True)
        emit({"event": "epoch", "epoch": epoch, "step": step, "seconds": round(dt, 1),
              "val": v, "test_peek": t})

        save(model, tokenizer, args.init, args.out / f"epoch{epoch}",
             {"epoch": epoch, "step": step, "val": v, "test_peek": t})
        save(model, tokenizer, args.init, args.out / "last",
             {"epoch": epoch, "step": step, "val": v, "test_peek": t})
        if consider(epoch, step, v, t):
            print(f"  new best ({args.select_on}) {best}", flush=True)

        if args.max_hours and (time.perf_counter() - run_t0) > args.max_hours * 3600:
            break

    total = time.perf_counter() - run_t0
    summary = {
        "event": "done",
        "seconds": round(total, 1),
        "steps": step,
        "sec_per_step": round(total / max(step, 1), 3),
        "sec_per_train_chunk": round(total / max(len(train_ex) * args.epochs, 1), 4),
        "select_on": args.select_on,
        "best_select_key": list(best) if best else None,
        "best_epoch": (best_meta or {}).get("epoch"),
        "best_step": (best_meta or {}).get("step"),
        "best_val": (best_meta or {}).get("val"),
        "best_test_peek": (best_meta or {}).get("test_peek"),
        "best_val_char_acc": (best_meta or {}).get("val", {}).get("char_acc"),
        "first20_mean_loss": round(sum(first_losses) / max(len(first_losses), 1), 4),
        "last20_mean_loss": round(sum(last_losses) / max(len(last_losses), 1), 4),
        "baseline_val": base_val,
    }
    print(json.dumps(summary, indent=2), flush=True)
    emit(summary)
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
