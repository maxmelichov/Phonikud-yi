# Yiddish label stack

Text → nikud → IPA, with the deployment guard that keeps a half-installed
stack from silently producing wrong labels. Needs only **onnxruntime + numpy**
(no torch, no transformers, no network).

Two ways to use it, same code:

- **In this repo** — `src/` finds the engine, the `data/` tables and the v5
  export at their repo paths automatically.
- **On another machine** — `python src/make_bundle.py` produces
  `dist/phonikud-yi-engine.zip`, a flat self-contained copy. The builder runs
  `selftest.py` *inside the staged tree* and refuses to ship a bundle that
  fails, so the artifact is verified in the layout it will be used in.

```bash
.venv/bin/python src/make_bundle.py                 # 1.1 GB, code+tables+v5
.venv/bin/python src/make_bundle.py --with-dataset  # + yiddish_tts_dataset_v2.tsv
.venv/bin/python src/make_bundle.py --no-model      # 2 MB, G2P only
```

## 1. Verify before you use it

```bash
python3 selftest.py        # must print ALL CHECKS PASSED
```

In the repo: `.venv/bin/python src/selftest.py`.

## 2. Use it

```python
from yiddish_labels import text_to_ipa, text_to_nikud, text_to_nikud_batch

text_to_nikud("מיט א פאר יאר צוריק")   # 'מִיט אַ פּאָר יאָר צוּרִיק'
text_to_ipa("מיט א פאר יאר צוריק")     # 'mit a pˈur jur ʦirˈik'
```

Batch when generating a dataset (`text_to_nikud_batch`) — ~4,200 chars/sec on
CPU, and it holds one ONNX session instead of one per call.

**Import `yiddish_labels`, not `yiddish_g2p` directly.** See §4.

## 3. THE INSTRUCTION THAT MATTERS MOST

**Never read yiddish24's stored `nikud`/pointing column.** That column is the
old labelling: it disagrees with itself (האט pointed 18 different ways, האבן
31) and marks פאר as פֿאַר "far" even in אַ פּאָר יאָר "a few years". It is
the direct cause of the released voice mixing dialects — hut/hot/hat, vus/vos.
Any loader still reading it (e.g. `load_yiddish24_wav`) keeps ~90% of the
Yiddish data on the broken labels no matter what else you change.

Generate labels with this stack instead, or use the pre-generated
`yiddish_tts_dataset_v2.tsv` (join on `id`) from the labels bundle.

Corollary: **do not score this model against that column.** v5 deliberately
disagrees with it on exactly the words it got wrong. Higher agreement = worse.

## 4. Why `yiddish_labels.py` exists (the trap it closes)

`yiddish_g2p.py` loads its knowledge from seven generated tables in `./data`.
Every loader swallows a missing file and returns `{}` — deliberate, so the
engine survives a table being regenerated. The cost: an incomplete deployment
emits plausible-looking IPA with **zero** native verdicts and **zero** audio
corrections, silently.

| word | with tables | tables missing |
| --- | --- | --- |
| פעקל | `pɛkl` | `fɛkl` |
| יארצייט | `jˈurʦajt` | `jˈarʦajt` |
| האף | `huf` | `haf` |

Both columns look like reasonable Yiddish; only the left one is right.
`yiddish_labels.verify()` runs at import, asserts every table loaded, and
spot-checks readings only the tables can produce — so a bad deploy fails
immediately instead of surfacing months later in a listening test.

## 5. Contents

```
yiddish_labels.py     front door + deployment guard  ← import this
yiddish_nikud.py      diacritizer wrapper, points at v5
selftest.py           run this first
make_bundle.py        build the portable zip (repo only)
yiddish_g2p.py        the G2P engine — in the bundle; in the repo it stays at the root
data/*.py             7 generated tables the engine needs (2.1 MB)
onnx_yiddish_v5/      phonikud-yi v5 export — in the bundle; in the repo it is
                      models/phonikud_yi_v5/v5.onnx (or $PHONIKUD_YI_MODEL)
```

**A name collision to know about:** an older `yiddish_nikud.py` aimed at the
superseded `onnx_yiddish` export used to sit at the repo root (it now lives in
`legacy/`, off the import path) — importing that one regenerates exactly the
labels this stack replaces.
`yiddish_labels.py` forces `src/` ahead of the repo root on `sys.path` so the
right module always wins; that is why you import `yiddish_labels` first.

## 6. Where the labels' authority comes from

Fixed chain, highest first — a lower tier never overrides a higher one:

1. **native verdicts** (Chezky) — 502 gold words, byte-identity enforced by a
   test gate in the source repo; nothing may move one
2. **corpus audio** — PhoneticXeus over 900 episode chunks: 77 פ letters
   corrected f→p, 42 komets vowels corrected to /u/, 105 rescued Hebrew
   readings refuted and barred from training
3. **published pointing** (Sefaria) → 4. **model guesses** — both LOW
   confidence, always queued for human review

v5 was finetuned on labels repaired under that chain, including the 793
training rows where אַ פאר had been labelled "far".

## 7. Known limits

- Rows containing digits, Latin text or URLs are quarantined by the engine's
  strict policy (~11% of corpus rows) — **skip them for training**, don't patch.
- The G2P is deterministic; the nikud model is contextual. If they disagree on
  a word, the G2P's tables win (they hold the native verdicts).
- Words still marked LOW confidence in the source repo are the human-review
  queue, not errors — but they are the least certain readings here.

## 8. Regenerating (source repo, not this box)

```bash
.venv/bin/python scripts/run_corpus_v3.py --limit 0    # engine IPA + QA gates
.venv/bin/python scripts/retag_tts_dataset.py          # rebuild the TSV
```
