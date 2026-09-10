# yiddish24 training dataset

Built by `scripts/prep_dataset.py` from the 121 transcribed yiddish24 files: audio is
cut at the Yiddish Labs phrase timestamps, greedily merged toward 10 s clips,
never across a paragraph break.

| | clips | hours |
|---|---|---|
| all clips | 5,143 | 12.2 |
| ASR train / dev / test | 4,589 / 82 / 163 | 10.1 / 0.18 / 0.38 |
| TTS train / dev / test | 3,669 / 71 / 137 | 7.9 / 0.16 / 0.31 |

## Files

- `clips/16k/*.wav` 16 kHz mono PCM16, for ASR. `clips/24k/*.wav` 24 kHz mono, for TTS.
- `clips.tsv` master table: every clip with source file, start/end, orthographic
  text, IPA, share of tokens read at LOW confidence, digit/Latin flags, split, and
  the `asr_ok` / `tts_ok` verdicts. Re-filter from this rather than rerunning.
- `parakeet_{split}.jsonl` NeMo manifest: `audio_filepath`, `duration`, `text` (orthographic).
- `qwen3tts_{split}.jsonl`: `audio`, `text` (IPA), `orth_text`, `speaker` (`cat<id>`), `duration`.
- `tts_{split}.csv` LJSpeech-style `clip_id|orth_text|ipa`.

## Filters

- Any clip with digits or Latin letters is excluded from both (225 clips). Spell
  numbers out in `clips.tsv` and re-emit if you want them back.
- ASR: 0.5 to 20 s. TTS: 1 to 15 s and at most 30% LOW-confidence tokens
  (971 clips fail this; lower the bar as the lexicon gets reviewed).

## Splits

Speakers are held out whole for the shiurim: categories 237, 236, 276 are test,
229 and 167 are dev. The bulletin (cat 57) is one studio voice and is split
90/5/5 by item so it appears in every split. Change `TEST_CATS`, `DEV_CATS`
at the top of the script.

## Before training

- Parakeet: build a SentencePiece tokenizer on the train text first
  (`scripts/tokenizers/process_asr_text_tokenizer.py` in NeMo) since the stock
  vocab has no Hebrew script.
- Qwen3-TTS: the `text` column is IPA from `../lexicon/lexicon_merged.tsv`; the
  orthography rides along in `orth_text` if you prefer to train on graphemes.
- Alignment: clip ends are the next phrase's start plus 50 ms. Spot-check a few
  clips per speaker; if tails are cut, raise `END_PAD` and rerun.
