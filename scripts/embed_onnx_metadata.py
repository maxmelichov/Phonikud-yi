#!/usr/bin/env python
"""Stamp the runtime metadata into an exported ONNX model.

scripts/infer_onnx.py is sidecar-free by design: everything it needs (char
vocab, the three head class lists, special-token ids) lives in the .onnx
custom metadata. scripts/export_onnx.py does not write that metadata — v3's
was embedded by hand last session and the step was never scripted. This
closes the gap.

The metadata is DERIVED, never trusted from a file: the vocab comes from the
export dir's own tokenizer via convert_ids_to_tokens (data/vocab.txt is off
by one vs the true id space — the pitfall in CLAUDE.md), and the class lists
are read from a reference model that already carries them (default: v3),
whose label space the finetunes share by construction (warm start, same
heads). A mismatch in head sizes aborts.

Usage:
  .venv/bin/python scripts/embed_onnx_metadata.py --onnx models/phonikud_yi_v5/v5.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "models" / "phonikud_yi_v3_gpu" / "v3.onnx" / "model.onnx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True,
                    help="export dir containing model.onnx + tokenizer files")
    ap.add_argument("--reference", type=Path, default=REFERENCE,
                    help="model.onnx whose class-list metadata is copied")
    args = ap.parse_args()

    model_path = args.onnx / "model.onnx"
    ref = ort.InferenceSession(str(args.reference),
                               providers=["CPUExecutionProvider"])
    meta = dict(ref.get_modelmeta().custom_metadata_map)

    # vocab + special ids from THIS export's tokenizer, never from the ref
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.onnx))
    n = len(tok)
    vocab = tok.convert_ids_to_tokens(list(range(n)))
    meta["vocab"] = json.dumps(vocab, ensure_ascii=False)
    for k, tid in (("cls_id", tok.cls_token_id), ("sep_id", tok.sep_token_id),
                   ("pad_id", tok.pad_token_id), ("unk_id", tok.unk_token_id)):
        meta[k] = str(tid)

    # head-size sanity: the model's output dims must match the class lists
    sess = ort.InferenceSession(str(model_path),
                                providers=["CPUExecutionProvider"])
    outs = {o.name: o.shape[-1] for o in sess.get_outputs()}
    sizes = sorted(v for v in outs.values() if isinstance(v, int))
    want = sorted(len(json.loads(meta[k])) for k in
                  ("nikud_classes", "shin_classes", "rafe_classes"))
    if sizes != want:
        raise SystemExit(f"head sizes {sizes} != class lists {want}; "
                         f"reference metadata does not fit this model")

    m = onnx.load(str(model_path), load_external_data=False)
    del m.metadata_props[:]
    for k, v in sorted(meta.items()):
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(model_path))
    print(f"embedded {len(meta)} metadata keys into {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
