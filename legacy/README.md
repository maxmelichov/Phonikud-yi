# legacy/

Superseded code, kept for reference only. Nothing in the live pipeline imports
from here, and this directory is not on any import path.

- `yiddish_nikud_onnx_v1.py` — the first diacritizer wrapper, aimed at the
  `data/onnx_yiddish` export (no longer present). Replaced by
  `src/yiddish_nikud.py`, which targets the phonikud-yi **v5** export. It used
  to sit at the repo root, where its name collided with the live module; the
  `sys.path` ordering guard in `src/yiddish_labels.py` and `src/selftest.py`
  exists because of that collision and is kept as a defence for bundles.
