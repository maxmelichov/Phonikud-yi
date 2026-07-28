# Phonikud for Yiddish

An adaptation of **Phonikud** ([arXiv 2506.12311](https://arxiv.org/abs/2506.12311), *"Phonikud:
Overcoming Phonetic Underspecification for Hebrew Text-To-Speech"*) to **Hasidic (Central/Poylish)
Yiddish**, for training a local Yiddish TTS system.

## The problem

Yiddish orthography is a hybrid:

- The **Germanic component** (~75% of running text) is spelled essentially phonemically — every
  syllable nucleus gets a vowel letter (`א ע ו י יי ײַ ױ`). A rule-based G2P handles it well.
- The **loshn-koydesh component** (Hebrew/Aramaic-origin words) keeps its *historical Hebrew*
  spelling, unvocalized, while its Yiddish pronunciation has drifted centuries away from it.
  `שבת` → `ʃabəs`, `משפחה` → `miʃpuxə`, `חתונה` → `xasənə`, `בעל-הבית` → `baləbos`.
  Letter-by-letter transliteration produces garbage (`שבת` → `ʃbs`).

This is *exactly* the phonetic-underspecification problem Phonikud solves for Hebrew — nikud alone
does not determine stress, mobile shva, or prefix boundaries. In Yiddish the underspecification is
concentrated in the loshn-koydesh lexicon, so that is what we attack.

Phonikud's key methodological move — **pseudo-label a large corpus, then hand-fix the ~1K most
frequent words** — is reproduced here as: *transcribe real Hasidic speech with a multimodal LLM,
diff it against the rule engine, and mine the divergences as lexicon candidates*. Our supervision
signal is better than Phonikud's in one respect: it comes from **actual pronounced audio**, not
from a morpho-phonological analyzer.

## Architecture

### Phase 1 — Rule engine + audio-mined lexicon (implemented)

`yiddish_g2p.py` is a three-stage engine:

| Stage | What it does |
| --- | --- |
| 1. Orthography | Loshn-koydesh lexical swap (`שבת` → `שאָבעס`), Hasidic contractions, silent-`ה` patch |
| 2. Latin base | Context-aware Hebrew-script → Latin transliteration (`_word_to_latin`) |
| 3. Phonology | Latin → Central Yiddish IPA (`ey`→`aɪ`, `ay`→`aː`, `o`→`u`, `u`→`i`) |

Its weakness is Stage 1: the lexicon is small and hand-written. Out-of-lexicon Hebrew-origin words
fall through to Stage 2 and get transliterated letter-by-letter — wrong.

The pipeline in this repo grows that lexicon from data:

```
data/audio/*.mp3
    │  phonikud_yi/segment.py   (ffmpeg segment muxer, ~30s mono 16kHz chunks)
    ▼
data/chunks/<episode_id>/chunk_%05d.mp3
    │  scripts/annotate_audio.py  (Gemini via Vercel AI Gateway, strict JSON)
    ▼
data/annotations/<episode_id>.jsonl   {chunk_idx, start_s, end_s, text_yi, ipa, confidence, notes}
    │  scripts/mine_lk_lexicon.py  (word-align text_yi vs ipa, diff against hebrew_to_ipa)
    ▼
data/lk_candidates.tsv                 word ⇥ rule_ipa ⇥ observed_ipa ⇥ count
    │  human review of the top N (the Phonikud "fix the top 1K words" step)
    ▼
_LOSHN_KOYDESH in yiddish_g2p.py
```

The miner only uses chunks where the Yiddish token count equals the IPA token count (a cheap,
high-precision alignment filter), and only emits words that (a) diverge from the rule output and
(b) score as loshn-koydesh under a heuristic: Hebrew-only letters (`ת ח`), low vowel-letter
density, LK morphology (`־ה`, `־ות`, `־ים`), optional Hebrew wordlist membership, minus Germanic
giveaways (`אַ אָ ײַ ױ וו יי`, `גע־` prefix).

Real sample output (3 chunks of one episode):

```
פרשת    rule=frʃs     obs=parʃəs
חתנ'ס   rule=xsns     obs=xasənəs
פחד     rule=fxd      obs=paxat
קדושת   rule=kdiʃs    obs=kdʊʃəs
```

### Phase 2 — Trainable "enhanced respelling" head (planned)

Phonikud freezes DictaBERT-large-char-menaked and trains a small 2-layer MLP head (hidden 256,
ReLU) on the char-level encoder outputs to predict *enhanced diacritics*: stress, mobile shva as
`/e/`, and prefix boundaries. Training: ~5M lines, ~6 epochs, batch 256, lr 5e-3, 5% val.

The Yiddish analogue: freeze a char-level Hebrew-script encoder and train a small head that
predicts, per character, an **enhanced respelling / phoneme-class tag** — which turns
`שבת` into the phonetic respelling `שאָבעס` that Stage 1 would have supplied from the lexicon.
This generalises the mined lexicon to unseen loshn-koydesh words instead of memorising them.

- Backbone candidates: `dicta-il/dictabert-large-char-menaked` (already char-level Hebrew script;
  Yiddish shares the script), or a small char-level transformer trained from scratch.
- Labels: pairs of (Hebrew-script word, observed IPA) from `data/annotations/`, projected back
  onto the respelling alphabet. Chunks with token-count alignment give free word-level supervision.
- Fallback order at inference: curated lexicon → mined lexicon → learned head → Stage 2 rules.

Not implemented yet — no ML dependencies are installed. The corpus produced by Phase 1 is the
training set.

### Phase 3 — TTS training (planned)

Train a local TTS on the IPA produced by Phases 1–2, mirroring the paper's Piper / StyleTTS2 setup.

- Corpus: the same ~270 episodes, chunked, with `ipa` as the phoneme sequence.
- Compute: RunPod (`RUNPOD_API_KEY` in `.env`).
- Validate the phoneme inventory against the model's vocab with
  `yiddish_g2p.validate_ipa_vocab(ipa, char_to_id)`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install requests python-dotenv
brew install ffmpeg          # required for chunking (ffmpeg + ffprobe on PATH)
```

`.env` at the repo root:

```
AI_GATEWAY_API_KEY=...
AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1
RUNPOD_API_KEY=...
```

### Models

Verified against `GET /models` on the gateway. The slugs `google/gemini-3.1-pro` and
`google/gemini-flash-3.6` **do not exist**; the real ones, hardcoded in `phonikud_yi/gateway.py`:

| Role | Slug |
| --- | --- |
| bulk annotation | `google/gemini-3.6-flash` (`MODEL_FLASH`) |
| high-quality passes | `google/gemini-3.1-pro-preview` (`MODEL_PRO`) |

Two gateway quirks, both handled in `phonikud_yi/gateway.py`:

1. **Audio parts must use the OpenAI `file` / `file_data` data-URL shape.** The classic
   `{"type": "input_audio", ...}` and `{"type": "audio_url", ...}` parts both return
   `400 Invalid input` for `google/gemini-*`.
2. **Gemini 3.x are reasoning models** — reasoning tokens are billed against `max_tokens`, so a
   small budget returns empty content. The client floors `max_tokens` at 1024.

## Running

```bash
# 0. verify everything (rule engine + gateway; --audio also tests a base64 audio call)
.venv/bin/python scripts/smoke_test.py --audio

# 1. annotate audio  (resumable — re-running skips chunks already in the output file)
.venv/bin/python scripts/annotate_audio.py --limit 1 --max-chunks 3   # try it out
.venv/bin/python scripts/annotate_audio.py                            # everything
.venv/bin/python scripts/annotate_audio.py --episode 161701 --model google/gemini-3.1-pro-preview

# 2. mine the loshn-koydesh lexicon
.venv/bin/python scripts/mine_lk_lexicon.py
.venv/bin/python scripts/mine_lk_lexicon.py --min-count 2 --min-score 0.6
.venv/bin/python scripts/mine_lk_lexicon.py --include-known   # audit existing entries too

# 3. review data/lk_candidates.tsv, then add confirmed entries to
#    _LOSHN_KOYDESH in yiddish_g2p.py as  "<hebrew spelling>": "<yiddish respelling>"
```

Optional: drop a newline-separated Hebrew wordlist at `data/hebrew_wordlist.txt` to make the
loshn-koydesh detector exact rather than heuristic.

## Layout

```
phonikud_yi/
  gateway.py            AI Gateway client: retries/backoff, text + base64 audio messages, loose JSON parsing
  segment.py            ffmpeg CLI chunking (chunk_mp3, duration_s, have_ffmpeg)
scripts/
  annotate_audio.py     manifest → chunks → Gemini → data/annotations/<id>.jsonl
  mine_lk_lexicon.py    annotations → data/lk_candidates.tsv
  smoke_test.py         rule engine samples + gateway verification
yiddish_g2p.py          the three-stage rule engine (hebrew_to_ipa / hebrew_to_latin)
data/
  audio_manifest.jsonl  {id, path, bytes, mp3_url}   (written by the scraper)
  episodes.jsonl        episode metadata             (written by the scraper)
  audio/<id>.mp3
  chunks/<id>/chunk_%05d.mp3
  annotations/<id>.jsonl
  lk_candidates.tsv
```
