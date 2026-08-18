# Phonikud-yi — grapheme-to-phoneme for Hasidic Yiddish

A deterministic **text → nikud → IPA** stack for contemporary Hasidic
(Unterland / Central) Yiddish, built to relabel the yiddish24 podcast corpus
(264 episodes, ~197 h, 1.83 M word tokens) for TTS training.

An adaptation of **Phonikud** ([arXiv 2506.12311](https://arxiv.org/abs/2506.12311),
*"Phonikud: Overcoming Phonetic Underspecification for Hebrew Text-To-Speech"*),
with one methodological change: the supervision signal here is **native-speaker
verdicts plus actual pronounced audio**, not a morpho-phonological analyzer.

---

## The problem

Yiddish orthography is a hybrid:

- The **Germanic component** (~75 % of running text) is spelled essentially
  phonemically — every syllable nucleus gets a vowel letter (`א ע ו י יי ײַ ױ`).
  Rules handle it, once you know the dialect's vowel classes.
- The **loshn-koydesh component** (Hebrew/Aramaic-origin words) keeps its
  *historical Hebrew* spelling, unvocalized, while the Yiddish pronunciation has
  drifted centuries away from it. `שבת` → `ʃabəs`, `משפחה` → `miʃpuxə`,
  `חתונה` → `xasənə`, `בעל-הבית` → `baləbos`. Letter-by-letter transliteration
  produces garbage (`שבת` → `ʃbs`).

That is the phonetic-underspecification problem Phonikud solves for Hebrew.
In Yiddish it is concentrated in the loshn-koydesh lexicon — so that is what
this repo attacks, and it attacks it with evidence rather than with guesses.

## Status

| | |
|---|---|
| G2P rules | 27, all executable (`docs/yiddish_phoneme_set.md`, 148 asserted examples) |
| Gold lexicon | 510 rows / 509 primaries, native-verified, **byte-identity enforced** |
| Corpus coverage | 1.82 M tokens: 64.3 % HIGH, 17.6 % MED, 18.1 % LOW confidence |
| Quarantined | 5.8 % of running tokens (digits, Latin, URLs) |
| Nikud model (phonikud-yi v5) | 99.94 % word-level in-distribution, **78.31 %** on a 409-word OOD test |
| TTS dataset | `data/corpus/yiddish_tts_dataset_v2.tsv` — 23,666 rows, 20,898 with an IPA label (~173 h) |

LOW confidence is the **human-review queue**, not an error rate: a word the
evidence chain cannot settle ships the model's contextual guess rather than
silence, tagged so it stays visible.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install requests python-dotenv onnxruntime numpy
brew install ffmpeg          # only needed for audio chunking
```

Verify before you use it — this is not optional, see [*Why the guard exists*](#why-the-guard-exists):

```bash
.venv/bin/python src/selftest.py        # must print ALL CHECKS PASSED
```

Then:

```python
import sys; sys.path.insert(0, "src")
from yiddish_labels import text_to_ipa, text_to_nikud, text_to_nikud_batch

text_to_nikud("מיט א פאר יאר צוריק")   # 'מִיט אַ פּאָר יאָר צוּרִיק'
text_to_ipa("מיט א פאר יאר צוריק")     # 'mit a pˈur jur ʦirˈik'
```

Use `text_to_nikud_batch` when labelling a corpus (~4,200 chars/sec on CPU; one
ONNX session instead of one per call).

For another machine: `.venv/bin/python src/make_bundle.py` builds
`dist/phonikud-yi-engine.zip` — a flat, self-contained copy needing only
onnxruntime + numpy. The builder runs the selftest *inside the staged tree* and
refuses to ship a bundle that fails.

### Why the guard exists

`yiddish_g2p.py` loads its knowledge from seven generated tables in `data/lexicons/`, and
every loader tolerates a missing file on purpose (so a table can be regenerated
in place). The cost: an incomplete deployment emits plausible-looking IPA with
**zero** native verdicts and **zero** audio corrections, silently.

| word | tables present | tables missing |
| --- | --- | --- |
| פעקל | `pɛkl` | `fɛkl` |
| יארצייט | `jˈurʦajt` | `jˈarʦajt` |
| האף | `huf` | `haf` |

Both columns look like reasonable Yiddish; only the left one is right.
`yiddish_labels.verify()` runs at import, asserts every table loaded, and
spot-checks readings only the tables can produce — so a bad deploy fails loudly
instead of surfacing months later in a listening test. **Import
`yiddish_labels`, never `yiddish_g2p` bare.**

## Architecture

### The engine — `yiddish_g2p.py` (single file)

Routing per token, strict order:

```
abbreviations → multiword/MWE → gold lexicon → legacy lexicons → rule path
```

A token the rules cannot voice (unpointed loshn-koydesh) enters the **rescue
chain** — `_AUDIO_ENDORSED` → `_HOMOGRAPH_LK` → `_SEFARIA_POINTED` →
`_MODEL_POINTED` — each emitting at LOW confidence with a distinct `reason`, so
it stays in the verification queue instead of pretending to be settled.

`g2p_token(word)` returns `route` (lexicon/rule/fallback), `confidence`
(HIGH/MED/LOW) and `reason` alongside the IPA.

Two pointed-Hebrew registers: `read_pointed_wh` (Whole-Hebrew, for quoted
pesukim — shuruk stays [u], final kometz-hey [u]) vs. the merged register
(embedded LK — shuruk → i, final kometz-hey → ə). `scripts/register_policy.py`
picks the primary per word; a merged reading that loses a consonant or vowel vs.
Whole-Hebrew is defective and can never ship, whatever the audio says.

### The tables — generated but committed

`data/lexicons/` holds `gold_lexicon.py`, the five `*_lk.py` modules and
`stress_overrides.py`. They are build products
that are checked in on purpose: the engine must stay deterministic and
self-contained (no network, no model at import). Each has a builder in
`scripts/` — regenerate through the builder, never hand-edit the output.

### The audio-evidence layer

`docs/audio_evidence.md` is the reference. In short: episode audio →
PhoneticXeus → folded onto the closed phone inventory → positionally aligned
against the engine's own reading → per-slot votes in an append-only shared pool
(`data/audio_lexicon/pe_sweep_tags.jsonl`).

Votes become verdicts only after three filters — **surprising** vs. the
recognizer's own base rate, at a grapheme the **spelling leaves open**, and not
already **ruled on by gold**. Survivors ship at MED confidence; everything
contested goes to a queue file for a native speaker, not into the engine.

### The pointing model — phonikud-yi

A 306 M char-BERT with three heads predicts Hasidic nikud in context and feeds
the last rescue link. Retrainable:
`scripts/prepare_retrain_dataset_v2.py` (stamps verified readings as diacritics,
masks the rest) → `scripts/train_phonikud_yi.py` (warm start, hard
label-collapse guard) → `scripts/eval_phonikud_yi.py`. ONNX export via
`scripts/export_onnx.py` + `scripts/infer_onnx.py` (CPU, ~50 ms/sentence).

### The verification loop — how accuracy actually improves

```
corpus run → frequency-sorted LOW batches → native verdicts → gold lexicon
          → rebuild tables → LOW share drops → repeat
```

Audio evidence and published pointing feed the same loop one tier down.

## The authority chain

Fixed, highest first. A lower tier never overrides a higher one.

1. **Native verdicts** (gold CSV) — byte-identity enforced by a test gate;
   nothing may move a gold primary.
2. **Corpus audio** (PhoneticXeus) — and only at graphemes the spelling leaves
   open (spec §4: `א`, `פ`, `יי`, `וי`, shuruk-`ו`). Elsewhere the letter
   decides, so an audio deviation is a *process*, not evidence.
3. **Published pointing** (Sefaria) — LOW confidence, always queued.
4. **Model guesses** — LOW confidence, always queued.

When audio contradicts gold the conflict becomes a **question for the native
reviewer**, never a silent flip.

## Invariants — do not "fix" these back

- **Gold byte-identity.** `hebrew_to_ipa(word, stress=True)` reproduces all 509
  gold primaries exactly. `scripts/test_g2p_gold.py` is the hard gate.
- **Citation forms, not surface forms.** Labels encode what a word *is*, not
  what fast speech does to it. `האט` stays `hut` though the recognizer hears
  *hat* in 41 % of clips; `איז` stays voiced. Reduction and devoicing are
  predictable processes and must never be folded into the lexicon.
- **Closed phone inventory.** Vowels `a aː ɛ ə i u ɔ ej aj ɔj oʊ`; consonants
  `b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ`; marks `ˈ` (immediately before
  the stressed vowel) and `ː` (only in `aː`). Nothing else may ever be emitted;
  a corpus-wide gate enforces it.
- **No-drop policy.** No Hebrew word is ever silently dropped — unsettled words
  get a LOW-confidence guess. Only non-Hebrew tokens (digits, Latin) quarantine.
- **v3 reversals of v2.** Devoicing is OFF everywhere (`iz`, `zuɡt` stay
  voiced; only voicing-ward assimilation survives); syllabic finals take no
  epenthetic schwa (`zuɡn`, not `zuɡən`); notation is `aj`/`ɔj`, not `aɪ`/`ɔɪ`;
  the `־ער` default is `ɛr` with a closed lexical ir-list.
- **Never read yiddish24's stored nikud column.** It disagrees with itself
  (`האט` pointed 18 different ways, `האבן` 31) and marks `פאר` as `פֿאַר` *far*
  even in `אַ פּאָר יאָר` *"a few years"*. It is the direct cause of the released
  voice mixing dialects. Generate labels with this stack, or join
  `yiddish_tts_dataset_v2.tsv` on `id`. Corollary: do **not** score the model
  against that column — higher agreement means worse.

## Commands

The test suites are plain scripts, not pytest. Everything runs from the repo
root with the venv python.

```bash
# gates — all seven must pass before anything ships
.venv/bin/python scripts/test_g2p.py             # core engine regressions
.venv/bin/python scripts/test_g2p_spec.py        # spec behaviours (2 documented XFAILs)
.venv/bin/python scripts/test_g2p_gold.py        # 509/509 gold byte-identity — the hard gate
.venv/bin/python scripts/test_rules_doc.py       # executes docs/yiddish_phoneme_set.md
.venv/bin/python scripts/test_xeus_map.py        # PhoneticXeus → Yiddish phone-map coverage
.venv/bin/python scripts/test_audio_evidence.py  # audio table integrity + sweep verdict logic
.venv/bin/python scripts/test_g2p_wh.py          # Whole-Hebrew / merged register readers
.venv/bin/python src/selftest.py                 # tables loaded, canaries correct, v5 loads

# corpus
.venv/bin/python scripts/run_corpus_v3.py --limit 0   # full run + QA gates a–d (a few minutes)
.venv/bin/python scripts/run_corpus_v3.py             # 2,000-row quick check
.venv/bin/python scripts/retag_tts_dataset.py         # rebuild the TSV from the corpus run

# evaluation against a native speaker
.venv/bin/python scripts/build_eval_sheet.py                  # stratified 30-sentence sheet
.venv/bin/python scripts/score_eval_sheet.py <returned.tsv>   # WER / PER / stress

# audio-evidence loop (hours of local GPU; no network, no APIs)
.venv/bin/python scripts/xeus_sweep_all.py --plan-only
.venv/bin/python scripts/xeus_sweep_all.py --max-chunks 1500
.venv/bin/python scripts/audio_calibrate.py
.venv/bin/python scripts/build_audio_pe_lexicon.py
.venv/bin/python scripts/build_audio_vowel_lexicon.py

# portable bundle
.venv/bin/python src/make_bundle.py --with-dataset
```

`.env` at the repo root, for the scripts that need it:

```
AI_GATEWAY_API_KEY=...
AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1
RUNPOD_API_KEY=...
```

## Layout

```
yiddish_g2p.py            the G2P engine (single file, ~4.3 k lines)
src/                      the deployment front door — import this
  yiddish_labels.py         text → nikud → IPA + the load guard
  yiddish_nikud.py          diacritizer wrapper (phonikud-yi v5)
  selftest.py               run before trusting a deployment
  make_bundle.py            builds dist/phonikud-yi-engine.zip
scripts/                  builders, gates, corpus runs, audio sweeps, training
  test_*.py                 the seven gates
  build_*.py                regenerate the committed data/lexicons/ tables
  xeus_*.py                 the PhoneticXeus audio-evidence pipeline
docs/
  yiddish_phoneme_set.md    the 27 rules — executable, run by test_rules_doc.py
  audio_evidence.md         how recordings become lexicon verdicts
  PROJECT_HISTORY.md        what was done, decided, and why
  paper_draft.md            academic write-up of the system
  xeus_to_yiddish_map.md    recognizer inventory → closed Yiddish inventory
data/
  lexicons/                 generated tables the engine loads
    gold_lexicon.py           native verdicts, compiled
    *_lk.py                   the four rescue tables + stress_overrides.py
  gold/g2p_gold_v3.csv      authority #1 — the native verdict CSV itself
  spec/                     g2p_spec_v3.md (authoritative), the xeus phone map
  corpus/                   yiddish_tts_dataset_v2.tsv, episodes + audio manifest,
                            canonical pointing
  annotations/              per-episode LLM annotation shards
  audio_lexicon/            audio-evidence vote pool, confusion, calibration
  candidates/               mined LK / MWE / homograph candidate lists
  stress/                   stress-eval sample, cache, report, review queue
  eval/                     native-speaker evaluation sheets
  review/                   what goes out to a human reviewer, and what came back
  scratch/                  one-off run logs (model bakeoffs, RunPod state)
phonikud_yi/              AI-Gateway client + ffmpeg chunking (annotation era)
scraper/                  yiddish24 episode scraper
legacy/                   superseded code, not on any import path
```

Ignored and regenerable (not in git): `data/audio/`, `data/chunks/`,
`data/phonemized/`, `data/retrain*/`, `models/`, `dist/`, `phonikud/`.

## Pitfalls that have burned real runs

- `yiddish_g2p._ROUTE_CACHE` caches per-token routing. **Clear it after any
  lexicon mutation** in measurement scripts, or you measure stale answers.
- **Write generated tables with `repr()`**, never hand-quoted f-strings —
  Yiddish keys carry apostrophes (`מורא'דיקע`, `אויפ'ן`) that close the literal
  early and make the module unparsable.
- Table loaders degrade on an **absent** file (deliberate) but **raise** on a
  file that exists and fails to load. Do not restore the old catch-all: a
  SyntaxError once silently emptied a table and shipped it.
- A raw majority vote over PhoneticXeus output is meaningless (it reports `ʦ`
  as *s* 62 % of the time). Always score against `data/audio_lexicon/confusion.tsv`.
- phonikud-yi training/inference needs **transformers==4.56.2**; 5.x loads the
  slow tokenizer, offsets vanish, and supervision silently collapses.
- On MPS, dropout crashes fused attention — the trainer's guards (frozen encoder
  stays in eval; dropout zeroed when unfrozen) are load-bearing.
- `vocab.txt` in exported checkpoints is off by one vs. the true tokenizer id
  space — build id maps from the tokenizer, never from raw `vocab.txt` lines.
- `phonikud/` is a **vendored upstream clone with its own `.git`**, gitignored;
  the actual trainer lives in `phonikud/model/src/train/`. It is not in this
  repo's history — back it up before editing.
- RunPod GPU runs go through `scripts/runpod_ctl.py`. **Terminate pods when done.**

## Docs are executable

`docs/yiddish_phoneme_set.md` is parsed and run by `scripts/test_rules_doc.py` —
every example row asserts live engine output. Change behaviour and the doc's
examples must change in the same commit, or the gate fails. Add a rule, and it
needs examples there.

## Open work

- Return and score the 30-sentence native evaluation sheet (`data/eval/`) for a
  measured WER/PER number.
- Apostrophe morpheme seams (`תורה'ס`, `משיח'ן`) — ~250 types on the LK path.
- Single-letter geresh abbreviations (`ד'`, `ס'`, `ח'`) currently quarantined.
- A Yiddish numeral reader (~3 k digit tokens quarantined).
- Adjudicate the pending audio-vs-gold conflicts with the native reviewer.
