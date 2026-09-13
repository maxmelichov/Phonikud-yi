#!/usr/bin/env python3
"""Export a ReNikud-yi run (BERT body + the three per-letter heads) to ONNX.

The graph takes (input_ids, attention_mask) and returns consonant, vowel and
stress logits per token. The runtime metadata the bundle needs — the char
vocab in true id order, the special ids, the two class lists — is embedded in
the model, the way the pointing export carries its own (yiddish_nikud.py
reads it from there and never from vocab.txt, whose ids are off by one).

  python scripts/export_renikud_onnx.py --run models/renikud_yi_audio --out models/renikud_yi_audio/onnx
  python scripts/export_renikud_onnx.py ... --int8    # also write onnx_int8/ (dynamic quantisation)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REPO / "models/renikud_yi_audio"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    import numpy as np
    import onnx
    import torch
    from renikud_yi_eval import load_model
    run = Path(args.run)
    out = Path(args.out or run / "onnx")
    out.mkdir(parents=True, exist_ok=True)
    model, tok, labels = load_model(run, "cpu")
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids, attention_mask)

    w = Wrapper(model)
    sample = tok(["מיט א פאר יאר צוריק, האב איך געוואוינט אין וויליאמסבורג."], return_tensors="pt")
    path = out / "model.onnx"
    torch.onnx.export(w, (sample["input_ids"], sample["attention_mask"]), str(path),
                      input_names=["input_ids", "attention_mask"], output_names=["cons_logits", "vowel_logits", "stress_logits"],
                      dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "cons_logits", "vowel_logits", "stress_logits")},
                      opset_version=args.opset, dynamo=False)
    m = onnx.load(str(path), load_external_data=True)
    itos = [tok.convert_ids_to_tokens(i) for i in range(len(tok))]
    meta = {"vocab": json.dumps(itos, ensure_ascii=False), "consonants": json.dumps(labels["consonants"], ensure_ascii=False),
            "vowels": json.dumps(labels["vowels"], ensure_ascii=False), "cls_id": str(tok.cls_token_id), "sep_id": str(tok.sep_token_id),
            "pad_id": str(tok.pad_token_id), "unk_id": str(tok.unk_token_id), "max_len": "512", "run": run.name}
    del m.metadata_props[:]
    for k, v in meta.items():
        p = m.metadata_props.add(); p.key, p.value = k, v
    onnx.save(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = run / "best_encoder" / name
        if src.exists():
            (out / name).write_bytes(src.read_bytes())
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sents = ["מיט א פאר יאר צוריק, האב איך געוואוינט אין וויליאמסבורג.", "די רבי'ס חתונה איז געווען שיין! ער האט געזאגט שלום עליכם",
             "עפּעס דאַרף מען זאָגן כאָטשיק עפּעס אַ קלייניקייט עפּעס אַ מין רמז."]
    mism, maxd = 0, 0.0
    for s in sents:
        enc = tok([s], return_tensors="pt")
        with torch.no_grad():
            t = w(enc["input_ids"], enc["attention_mask"])
        o = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})
        for a, b in zip(t, o):
            maxd = max(maxd, float(np.abs(a.numpy() - b).max()))
            mism += int((a.numpy().argmax(-1) != b.argmax(-1)).sum())
    size = sum(f.stat().st_size for f in out.glob("model.onnx*")) / 1e6
    print(f"exported {path} ({size:.0f} MB)  verify: max |diff| {maxd:.2e}, argmax mismatches {mism}")
    if args.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = out.parent / "onnx_int8"
        q.mkdir(exist_ok=True)
        quantize_dynamic(str(path), str(q / "model.onnx"), weight_type=QuantType.QInt8)
        mq = onnx.load(str(q / "model.onnx"))
        del mq.metadata_props[:]
        for k, v in meta.items():
            p = mq.metadata_props.add(); p.key, p.value = k, v
        onnx.save(mq, str(q / "model.onnx"))
        for name in ("labels.json",):
            (q / name).write_bytes((out / name).read_bytes())
        sq = ort.InferenceSession(str(q / "model.onnx"), providers=["CPUExecutionProvider"])
        mq_ = 0
        for s in sents:
            enc = tok([s], return_tensors="pt")
            o = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})
            oq = sq.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})
            mq_ += sum(int((a.argmax(-1) != b.argmax(-1)).sum()) for a, b in zip(o, oq))
        print(f"int8: {sum(f.stat().st_size for f in q.glob('model.onnx*')) / 1e6:.0f} MB, argmax mismatches vs fp32 on {len(sents)} sentences: {mq_}")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
