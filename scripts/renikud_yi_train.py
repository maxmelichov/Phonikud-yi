#!/usr/bin/env python3
"""ReNikud-yi: per-letter Yiddish G2P on the phonikud-yi encoder.

The model of Melichov, Kolani & Alper (2026) ported to Yiddish: every Hebrew
letter predicts a (consonant, vowel, stress) triple through three coupled
heads on a char-level BERT — here the 24-layer encoder of phonikud-yi v6
(``models/phonikud_yi_v6/best``, its ``bert.*`` weights), which has already
seen the corpus as a pointing model. The heads are new. Loss is cross-entropy
on labelled letters only (IGNORE = -100 elsewhere), so unvouched words give
no gradient and are learned from context.

Data: scripts/renikud_yi_prepare.py (data/renikud_yi/{train,val,test}.jsonl,
labels.json). Rows are chunked at word boundaries to --max-chars.

Metrics, per epoch on val and test: letter accuracy per head, joint letter
accuracy, and WORD accuracy — a word counts when every labelled letter's
triple is right, i.e. its IPA is reproduced exactly. Selection on val word
accuracy.

Usage:
  python scripts/renikud_yi_train.py --data data/renikud_yi --init models/phonikud_yi_v6/best \\
      --out models/renikud_yi_v1 --epochs 3 [--limit 300]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phonikud" / "model"))

IGNORE = -100


# ------------------------------------------------------------------ data

@dataclass
class Example:
    text: str
    cons: list[int]
    vowel: list[int]
    stress: list[int]
    rid: str


def read_rows(path: Path, max_chars: int, limit: int | None = None) -> list[Example]:
    out: list[Example] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            text, c, v, s = r["text"], r["cons"], r["vowel"], r["stress"]
            # split at word boundaries so no word straddles two chunks
            start = 0
            while start < len(text):
                end = min(start + max_chars, len(text))
                if end < len(text):
                    cut = text.rfind(" ", start, end)
                    if cut > start:
                        end = cut
                out.append(Example(text[start:end], c[start:end], v[start:end], s[start:end], r["id"]))
                start = end
                while start < len(text) and text[start] == " ":
                    start += 1
            if limit and len(out) >= limit:
                break
    return out[:limit] if limit else out


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, items: list[Example]):
        texts = [it.text for it in items]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length,
                       return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        B, S = enc["input_ids"].shape
        cons = torch.full((B, S), IGNORE, dtype=torch.long)
        vow = torch.full((B, S), IGNORE, dtype=torch.long)
        strs = torch.full((B, S), IGNORE, dtype=torch.long)
        for b, it in enumerate(items):
            for t, (a, z) in enumerate(enc["offset_mapping"][b].tolist()):
                if z - a == 1 and a < len(it.cons) and it.cons[a] != IGNORE:
                    cons[b, t] = it.cons[a]
                    vow[b, t] = it.vowel[a]
                    strs[b, t] = it.stress[a]
        enc.pop("offset_mapping")
        return enc, cons, vow, strs, items


# ------------------------------------------------------------------ model

class ReNikudYi(nn.Module):
    def __init__(self, encoder: nn.Module, n_cons: int, n_vowel: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cons_head = nn.Linear(h, n_cons)
        self.vowel_head = nn.Linear(h + n_cons, n_vowel)
        self.stress_head = nn.Linear(h + n_cons + n_vowel, 2)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        hidden = self.dropout(hidden)
        c = self.cons_head(hidden)
        v = self.vowel_head(torch.cat([hidden, c], -1))
        s = self.stress_head(torch.cat([hidden, c, v], -1))
        return c, v, s


def load_encoder(init: Path, device):
    """The v6 pointing model's BERT body, weights from its safetensors."""
    from transformers import AutoTokenizer, BertConfig, BertModel
    from safetensors.torch import load_file
    cfg = BertConfig.from_pretrained(init)
    enc = BertModel(cfg, add_pooling_layer=False)
    state = load_file(str(init / "model.safetensors"))
    bert_state = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    missing, unexpected = enc.load_state_dict(bert_state, strict=False)
    missing = [m for m in missing if not m.startswith("pooler")]
    print(f"encoder loaded from {init}: {len(bert_state)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    tok = AutoTokenizer.from_pretrained(init)
    return enc.to(device), tok


# ------------------------------------------------------------------ eval

@torch.no_grad()
def evaluate(model, loader, device, use_amp) -> dict:
    model.eval()
    hit = {"cons": 0, "vowel": 0, "stress": 0, "joint": 0}
    n = 0
    words_ok = words_n = 0
    for enc, cons, vow, strs, items in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            c, v, s = model(enc["input_ids"], enc["attention_mask"])
        pc, pv, ps = c.argmax(-1).cpu(), v.argmax(-1).cpu(), s.argmax(-1).cpu()
        mask = cons != IGNORE
        n += int(mask.sum())
        okc, okv, oks = (pc == cons) & mask, (pv == vow) & mask, (ps == strs) & mask
        hit["cons"] += int(okc.sum()); hit["vowel"] += int(okv.sum()); hit["stress"] += int(oks.sum())
        joint = okc & okv & oks
        hit["joint"] += int(joint.sum())
        # word accuracy: a run of labelled tokens between unlabelled ones is a word
        for b in range(mask.shape[0]):
            m = mask[b].tolist(); j = joint[b].tolist()
            t = 0
            while t < len(m):
                if not m[t]:
                    t += 1
                    continue
                u = t
                ok = True
                while u < len(m) and m[u]:
                    ok = ok and j[u]
                    u += 1
                words_n += 1
                words_ok += int(ok)
                t = u
    return {"letters": n, **{k: round(100 * v / max(1, n), 2) for k, v in hit.items()},
            "word_acc": round(100 * words_ok / max(1, words_n), 2), "words": words_n}


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "data/renikud_yi")
    ap.add_argument("--init", type=Path, default=REPO / "models/phonikud_yi_v6/best")
    ap.add_argument("--init-run", type=Path, default=None,
                    help="continue from a previous ReNikud-yi run (its best/encoder + best/heads.pt): stage 2 of a curriculum")
    ap.add_argument("--out", type=Path, default=REPO / "models/renikud_yi_v1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=480)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda" and not args.no_amp
    labels = json.loads((args.data / "labels.json").read_text(encoding="utf-8"))
    n_cons, n_vowel = len(labels["consonants"]), len(labels["vowels"])

    if args.init_run:
        from transformers import AutoTokenizer, BertModel
        enc = BertModel.from_pretrained(args.init_run / "best" / "encoder", add_pooling_layer=False).to(device)
        tok = AutoTokenizer.from_pretrained(args.init_run / "best" / "encoder")
        model = ReNikudYi(enc, n_cons, n_vowel).to(device)
        heads = torch.load(args.init_run / "best" / "heads.pt", map_location=device)["heads"]
        model.load_state_dict(heads, strict=False)
        print(f"stage 2: continuing from {args.init_run}", flush=True)
    else:
        enc, tok = load_encoder(args.init, device)
        model = ReNikudYi(enc, n_cons, n_vowel).to(device)
    collate = Collator(tok, args.max_length)
    train = read_rows(args.data / "train.jsonl", args.max_chars, args.limit)
    val = read_rows(args.data / "val.jsonl", args.max_chars, args.val_limit)
    test = read_rows(args.data / "test.jsonl", args.max_chars, args.val_limit)
    print(f"chunks train {len(train):,} val {len(val):,} test {len(test):,}", flush=True)

    def loader(rows, shuffle):
        return torch.utils.data.DataLoader(rows, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.encoder_lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    steps_total = max(1, math.ceil(len(train) / args.batch_size) * args.epochs)
    warm = int(steps_total * args.warmup)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, steps_total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train_log.jsonl").open("a", encoding="utf-8")
    ev0 = {"val": evaluate(model, loader(val, False), device, use_amp), "test": evaluate(model, loader(test, False), device, use_amp)}
    print(f"epoch 0  val word {ev0['val']['word_acc']}%  test word {ev0['test']['word_acc']}%  (untrained heads)", flush=True)
    best = -1.0
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0; k = 0
        for enc_b, cons, vow, strs, _ in loader(train, True):
            enc_b = {kk: vv.to(device) for kk, vv in enc_b.items()}
            cons, vow, strs = cons.to(device), vow.to(device), strs.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                c, v, s = model(enc_b["input_ids"], enc_b["attention_mask"])
            loss = (ce(c.float().view(-1, n_cons), cons.view(-1)) + ce(v.float().view(-1, n_vowel), vow.view(-1))
                    + ce(s.float().view(-1, 2), strs.view(-1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run += loss.item(); k += 1
            if step % args.log_every == 0:
                print(f"  ep{epoch} step {step}/{steps_total} loss {run / k:.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min", flush=True)
                log.write(json.dumps({"event": "step", "epoch": epoch, "step": step, "loss": run / k}) + "\n")
                run = 0.0; k = 0
        ev = {"val": evaluate(model, loader(val, False), device, use_amp), "test": evaluate(model, loader(test, False), device, use_amp)}
        print(f"epoch {epoch}  val: word {ev['val']['word_acc']}% joint {ev['val']['joint']}% (c {ev['val']['cons']} v {ev['val']['vowel']} s {ev['val']['stress']})  "
              f"test: word {ev['test']['word_acc']}% joint {ev['test']['joint']}%  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        log.write(json.dumps({"event": "epoch", "epoch": epoch, "step": step, **ev}) + "\n"); log.flush()
        torch.save({"heads": {kk: vv for kk, vv in model.state_dict().items() if not kk.startswith("encoder.")},
                    "labels": labels, "args": vars(args) | {"data": str(args.data), "init": str(args.init), "out": str(args.out)}},
                   args.out / "last_heads.pt")
        model.encoder.save_pretrained(args.out / "last_encoder")
        if ev["val"]["word_acc"] > best:
            best = ev["val"]["word_acc"]
            (args.out / "best").mkdir(exist_ok=True)
            torch.save({"heads": {kk: vv for kk, vv in model.state_dict().items() if not kk.startswith("encoder.")},
                        "labels": labels, "epoch": epoch, "val": ev["val"], "test": ev["test"]}, args.out / "best" / "heads.pt")
            model.encoder.save_pretrained(args.out / "best" / "encoder")
            tok.save_pretrained(args.out / "best" / "encoder")
            print(f"   new best val word acc {best}% -> {args.out / 'best'}", flush=True)
    log.write(json.dumps({"event": "done", "best_val_word_acc": best, "seconds": time.time() - t0}) + "\n")
    log.close()
    print(f"done. best val word accuracy {best}%")


if __name__ == "__main__":
    main()
