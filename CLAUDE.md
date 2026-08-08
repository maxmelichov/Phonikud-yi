# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A grapheme-to-phoneme (G2P) system for contemporary Hasidic (Unterland/Central) Yiddish, built for
TTS training on the yiddish24/ivelt podcast corpus (1.83M tokens). `yiddish_g2p.py` is the engine;
everything else feeds, tests, or consumes it. The README's architecture section describes the older
pre-v3 engine — trust `data/g2p_spec_v3.md` (the authoritative spec), `docs/yiddish_phoneme_set.md`,
and `docs/PROJECT_HISTORY.md` over the README.

## Commands

Everything runs with the venv python from the repo root. The test suites are **plain scripts, not
pytest** — pytest is not installed.

```bash
.venv/bin/python scripts/test_g2p.py          # core engine regressions
.venv/bin/python scripts/test_g2p_spec.py     # spec behaviors (2 documented XFAILs are expected)
.venv/bin/python scripts/test_g2p_gold.py     # 500/500 gold byte-identity — the hard gate
.venv/bin/python scripts/test_rules_doc.py    # executes docs/yiddish_phoneme_set.md examples
.venv/bin/python scripts/test_xeus_map.py     # PhoneticXeus→Yiddish phone-map coverage
.venv/bin/python scripts/test_audio_evidence.py  # audio-pe table integrity + sweep verdict logic
.venv/bin/python scripts/test_g2p_wh.py       # Whole-Hebrew / merged register readers

.venv/bin/python scripts/run_corpus_v3.py --limit 0   # full-corpus run + QA gates a–d (few min)
.venv/bin/python scripts/run_corpus_v3.py             # 2000-row quick gate check
```

Quick engine probe:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from yiddish_g2p import hebrew_to_ipa, g2p_token; print(hebrew_to_ipa('וואס איז דאס', stress=True)); print(g2p_token('שבת'))"
```

## Non-negotiable invariants

- **Gold byte-identity**: `hebrew_to_ipa(word, stress=True)` must reproduce all 500 primaries of
  `g2p_gold_v3 - g2p_gold_v3.csv.csv` exactly. No change may move a gold primary — these are
  native-speaker (Chezky) verdicts, authority #1.
- **Authority order**: gold CSV > audio evidence (PhoneticXeus) > published pointing (Sefaria) >
  model guesses. Never let a lower tier override a higher one; recognizer/audio evidence does not
  outrank an explicit native verdict.
- **Closed phone inventory** (spec §1): vowels `a aː ɛ ə i u ɔ ej aj ɔj oʊ`, consonants
  `b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ`, marks `ˈ` (immediately before the stressed
  vowel) and `ː` (only in aː). Nothing else may ever be emitted; gate (b) enforces it corpus-wide.
- **No-drop policy** (2026-08-08 user directive): no Hebrew word is ever silently dropped. A word
  the evidence chain cannot settle gets phonikud-yi's contextual guess at LOW confidence rather
  than silence. Only non-Hebrew tokens (digits, Latin) may quarantine.
- **v3 phonology decisions that reversed v2** (do not "fix" them back): devoicing OFF everywhere
  (`iz`, `zuɡt` stay voiced; only voicing-ward assimilation survives); syllabic finals take no
  epenthetic schwa (`zuɡn`, not `zuɡən`); notation is `aj/ɔj`, not `aɪ/ɔɪ`; the ־ער default is
  `ɛr` with a closed lexical ir-list.

## Architecture

### The engine (`yiddish_g2p.py`, single file)

Routing per token, strict order: abbreviation table (acronym-words → letter-name-words →
letter-name spell-out) → multiword/MWE table → gold lexicon → legacy lexicons → rule path. A token
the rules can't voice (unpointed loshn-koydesh) enters the **rescue chain**:
`_AUDIO_ENDORSED` → `_HOMOGRAPH_LK` → `_SEFARIA_POINTED` → `_MODEL_POINTED`, each emitting at LOW
confidence with a distinct `reason` so it stays in the verification queue.

The five `data/*_lk.py` modules and `data/gold_lexicon.py` are **generated but committed** — the
engine must stay deterministic and self-contained (no network, no model at import). Each has a
builder in `scripts/` (`build_gold_lexicon.py`, `build_sefaria_lexicon.py`,
`build_homograph_lexicon.py`, `build_model_guess_lexicon.py`); regenerate via the builder, never
hand-edit the generated file.

Two pointed-Hebrew reading registers: `read_pointed_wh` (Whole-Hebrew, for quoted pesukim —
shuruk stays [u], final kometz-hey [u]) vs the merged register (embedded LK — shuruk→i, final
kometz-hey→ə). `scripts/register_policy.py` decides which is primary per word; a merged reading
that loses a consonant or vowel vs WH is defective and can never ship, regardless of audio votes.

### Per-token metadata

`g2p_token(word)` → `route` (lexicon/rule/fallback), `confidence` (HIGH/MED/LOW), `reason`.
LOW = defaulted ambiguous grapheme or rescued reading; this is the verification work queue, not
noise. `run_corpus_v3.py` emits the §12 record format plus LOW_CONF/OOV triage logs under
`data/phonemized/v3/`.

### The verification loop (how accuracy improves)

Corpus run → frequency-sorted LOW batches (`data/verification_batch_v4.*`) → native verdicts fold
into the gold lexicon → rebuild → LOW share drops. Audio evidence comes from PhoneticXeus
(`scripts/xeus_*.py`): episode MP3s in `data/audio/` are transcribed to universal IPA, folded onto
the closed inventory (`scripts/xeus_map.py`), and aligned per word to vote on contested readings.

### The pointing model (phonikud-yi)

`models/phonikud_yi_v3_gpu/best` (306M char-BERT, three heads) predicts Hasidic nikud in context;
its guesses feed the last rescue link and it is retrainable: `scripts/prepare_retrain_dataset_v2.py`
(stamps verified readings as diacritics, masks the rest) → `scripts/train_phonikud_yi.py` (warm
start; has a hard label-collapse guard) → `scripts/eval_phonikud_yi.py`. ONNX export:
`scripts/export_onnx.py` + `scripts/infer_onnx.py` (CPU, ~50 ms/sentence).

## Pitfalls that have burned real runs

- `yiddish_g2p._ROUTE_CACHE` caches per-token routing. **Clear it after any lexicon mutation** in
  measurement scripts, or you measure stale answers.
- Training/inference for phonikud-yi requires **transformers==4.56.2**; 5.x loads the slow
  tokenizer, offsets vanish, and supervision silently collapses (the trainer now aborts on this,
  but keep the pin).
- On MPS, dropout crashes fused attention — the trainer handles it (frozen encoder stays in eval;
  dropout zeroed when unfrozen); don't remove those guards.
- `phonikud/` is a **vendored upstream clone, gitignored, with its own .git** — the actual trainer
  lives in `phonikud/model/src/train/`. Back it up before editing; it is not in this repo's history.
- `vocab.txt` in exported checkpoints is off by one vs the true tokenizer id space — build id maps
  from the tokenizer, never from raw vocab.txt lines.
- Large data (`data/retrain*/`, `data/phonemized/`, `data/yiddish_tts_dataset.tsv`,
  `data/pointed_sources/raw/`, models) is gitignored and regenerable; don't commit it.
- RunPod GPU runs: `scripts/runpod_ctl.py` (key in `.env`). Terminate pods when done.

## Docs are executable

`docs/yiddish_phoneme_set.md` is parsed and run by `test_rules_doc.py` — every example row asserts
live engine output. When you change behavior, update the doc's examples in the same change or the
gate fails; when you add a rule, give it examples there.



# Readable Summarization

When summarizing a document:

1. Start with a one-sentence TL;DR.
2. State the document's main purpose or argument.
3. Organize the summary with descriptive headings.
4. Use short paragraphs and bullets where appropriate.
5. Preserve important numbers, caveats, assumptions, and disagreements.
6. Remove repetition, filler, generic introductions, and obvious statements.
7. Prefer plain, natural language over academic or bureaucratic wording.
8. Do not add information that is not in the source.
9. Clearly distinguish:
   - What the source explicitly says
   - What can reasonably be inferred
   - What remains uncertain
10. End with the practical takeaway or “why this matters.”

Default format:

## TL;DR
One or two clear sentences.

## Main points
- 3–7 important points, explained briefly.

## Important details
Include evidence, numbers, definitions, or caveats that affect interpretation.

## Takeaway
Explain what the reader should remember or do.

Target length: 10–20% of the original unless the user specifies otherwise.
Write for an intelligent, busy reader. Be concise, specific, and easy to scan.