#!/usr/bin/env python3
"""Build the portable label-stack bundle for another machine (e.g. the TTS box).

The bundle is this directory's modules + the engine + its eight generated
tables + the phonikud-yi v9 pointing export + the ReNikud-yi context model, laid out so ``yiddish_nikud`` finds the
model beside itself and ``yiddish_labels`` finds the engine beside itself.
Nothing in it needs torch, transformers or a network -- only onnxruntime and
numpy.

The build REFUSES to ship a stack that does not pass selftest.py, run inside
the assembled tree (not this repo), so the artifact is verified in the layout
it will actually be used in.

Usage:
    .venv/bin/python src/make_bundle.py                    # -> dist/phonikud-yi-engine.zip
    .venv/bin/python src/make_bundle.py --out /tmp/x       # elsewhere
    .venv/bin/python src/make_bundle.py --with-dataset     # + yiddish_tts_dataset_v2.tsv (51 MB)
    .venv/bin/python src/make_bundle.py --no-model         # code+tables only (2 MB)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NAME = "phonikud-yi-engine"

TABLES = ("gold_lexicon.py", "audio_pe_lk.py", "audio_vowel_lk.py",
          "audio_endorsed_lk.py", "homograph_lk.py", "sefaria_pointed_lk.py",
          "printed_respelling_lk.py", "model_pointed_lk.py",
          "stress_overrides.py")
MODULES = ("yiddish_labels.py", "yiddish_nikud.py", "yiddish_renikud.py", "selftest.py", "README.md")
# The letter aligner ReNikud-yi's decode needs lives with the training scripts.
SCRIPT_MODULES = ("yi_align.py",)
# v9 (2026-09-13): retrain9 = the attested tier with ReNikud-yi as second
# witness (coverage 83.8%); beats v8 36:12 (p=0.0007) on the audio yardstick
# and is flat-to-better on gold pointing (docs/xeus_finetune.md §28). v8 beat
# v6 48:4 the same way (§18). PHONIKUD_YI_MODEL overrides.
MODEL_SRC = pathlib.Path(os.environ["PHONIKUD_YI_MODEL"]) if os.environ.get("PHONIKUD_YI_MODEL") else REPO / "models" / "phonikud_yi_v9" / "v9.onnx"
DATASET = REPO / "data" / "corpus" / "yiddish_tts_dataset_v2.tsv"
# ReNikud-yi (2026-09-11): the context model for rule-path words, docs
# §19/§26 — 94.5% agreement with the audio where the rule engine has 88.0%.
# The int8 dynamic-quantised export reads identically to fp32 on the held-out
# episodes (renikud_bundle_check.py) at a quarter of the size. Override with
# PHONIKUD_YI_RENIKUD_SRC; --no-renikud ships the engine without it.
RENIKUD_SRC = pathlib.Path(os.environ["PHONIKUD_YI_RENIKUD_SRC"]) if os.environ.get("PHONIKUD_YI_RENIKUD_SRC") else REPO / "models" / "renikud_yi_audio" / "onnx_int8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "dist")
    ap.add_argument("--with-dataset", action="store_true",
                    help="include yiddish_tts_dataset_v2.tsv (51 MB)")
    ap.add_argument("--no-model", action="store_true",
                    help="skip the 1.1 GB v5 export (code + tables only)")
    ap.add_argument("--no-renikud", action="store_true",
                    help="skip the ReNikud-yi export (the engine then reads rule-path words by rule alone)")
    ap.add_argument("--skip-selftest", action="store_true",
                    help="build even if the assembled tree fails its checks")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / NAME
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "data" / "lexicons").mkdir(parents=True)

    for mod in MODULES:
        shutil.copy2(HERE / mod, stage / mod)
    shutil.copy2(REPO / "yiddish_g2p.py", stage / "yiddish_g2p.py")
    for mod in SCRIPT_MODULES:
        shutil.copy2(REPO / "scripts" / mod, stage / mod)
    for tbl in TABLES:
        shutil.copy2(REPO / "data" / "lexicons" / tbl, stage / "data" / "lexicons" / tbl)
    if not args.no_model:
        if not (MODEL_SRC / "model.onnx").exists():
            raise SystemExit(f"no v5 export at {MODEL_SRC}; pass --no-model to skip")
        shutil.copytree(MODEL_SRC, stage / "onnx_yiddish_v9")
    if not args.no_renikud:
        if not (RENIKUD_SRC / "model.onnx").exists():
            raise SystemExit(f"no ReNikud-yi export at {RENIKUD_SRC}; run scripts/export_renikud_onnx.py --int8 or pass --no-renikud")
        shutil.copytree(RENIKUD_SRC, stage / "onnx_renikud_yi")
    if args.with_dataset:
        if not DATASET.exists():
            raise SystemExit(f"{DATASET} missing; run scripts/retag_tts_dataset.py")
        shutil.copy2(DATASET, stage / DATASET.name)

    # verify IN THE STAGED LAYOUT -- the repo's own paths must not be in play
    if not args.skip_selftest and not args.no_model:
        proc = subprocess.run([sys.executable, "selftest.py"], cwd=stage,
                              capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit("selftest failed in the staged bundle; not shipping")

    zip_path = args.out / f"{NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for path in sorted(stage.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            zf.write(path, path.relative_to(args.out))
    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n{zip_path}  {size_mb:.0f} MB\nsha256 {sha256(zip_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
