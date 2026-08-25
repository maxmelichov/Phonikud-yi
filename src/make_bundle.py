#!/usr/bin/env python3
"""Build the portable label-stack bundle for another machine (e.g. the TTS box).

The bundle is this directory's modules + the engine + its seven generated
tables + the phonikud-yi v5 export, laid out so ``yiddish_nikud`` finds the
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
          "model_pointed_lk.py", "stress_overrides.py")
MODULES = ("yiddish_labels.py", "yiddish_nikud.py", "selftest.py", "README.md")
MODEL_SRC = REPO / "models" / "phonikud_yi_v6" / "v6.onnx"
DATASET = REPO / "data" / "corpus" / "yiddish_tts_dataset_v2.tsv"


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
    for tbl in TABLES:
        shutil.copy2(REPO / "data" / "lexicons" / tbl, stage / "data" / "lexicons" / tbl)
    if not args.no_model:
        if not (MODEL_SRC / "model.onnx").exists():
            raise SystemExit(f"no v5 export at {MODEL_SRC}; pass --no-model to skip")
        shutil.copytree(MODEL_SRC, stage / "onnx_yiddish_v6")
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
