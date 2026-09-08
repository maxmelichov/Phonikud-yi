# Fine-tuning PhoneticXeus into a Yiddish phone recognizer

How the speech-to-phones model used for audio evidence (`docs/audio_evidence.md`)
was fine-tuned on the corpus, using **only words a native speaker has verified**,
and how it is measured. Results are at the end.

Scripts: `scripts/xeus_ft_common.py` (shared), `xeus_ft_text.py` (local),
`xeus_ft_prepare.py` / `xeus_ft_train.py` / `xeus_ft_eval.py` (GPU pod),
`xeus_yi_decode.py` (inference + dictionary), `xeus_ft_runpod.sh` (orchestration).

---

## 1. Why

PhoneticXeus is a universal phone recognizer. Pooled over the corpus it has an
accent (`docs/audio_evidence.md` §4): it hears ʦ as *s* 62% of the time, aj as
*i* 50%, ej as *ɛ* 50%, and weakens every voiced sibilant. Every audio verdict
so far has had to model that accent statistically before trusting a vote. A
recognizer that hears Hasidic Yiddish as Hasidic Yiddish removes that layer.

## 2. What counts as certain

The training label for a clip is a pronunciation; the clip is only usable if
that pronunciation is *known to be right*. One source qualifies:

> a row of `data/gold/g2p_gold_v3.csv` whose `source` is `chezky-verified`
> or `chezky-approved` — **413 word types**.

Excluded on purpose:

- the 97 `claude-annotated` gold rows — a model's guess that nobody has checked;
- every reading the engine derives by rule, from Sefaria pointing, from the
  printed index, or from earlier audio votes — all of them are exactly the
  things this recognizer is supposed to be evidence *for*, so none of them may
  be its teacher.

Those 413 types cover **59.9% of corpus tokens** (1.09M of 1.83M), so the
restriction costs nothing in volume. Non-certain words are still present in
every chunk: the engine's reading of them is used to *place* the certain
words in time (§3), never as a label.

175 of the 413 have more than one accepted pronunciation (`di | də`,
`hut | hɔt`). Rather than guess, the variant the pretrained model finds most
likely for that clip is chosen at data-preparation time (CTC likelihood over
the clip's frames, other words fixed), so the label matches what the speaker
actually did.

## 3. Cutting the audio: forced alignment

CTC needs a label for the whole clip, and a 30-second corpus chunk is 86 words
of which only some are certain. So the clips are made, not taken:

1. The pretrained model's frame log-probabilities are computed for the chunk.
2. `torchaudio.functional.forced_align` aligns them to the chunk's full phone
   string (certain words: gold; the rest: engine), giving every word a start
   and end frame (20 ms frames).
3. Every **maximal run of consecutive certain words** becomes one segment. It
   is cut at the run's edges, padded by up to 60 ms but never past halfway to
   the neighbouring word, so no foreign phone bleeds in. Runs longer than 8 s
   are split at word boundaries; segments under 0.2 s, or whose aligned frames
   average below 0.25 posterior, are dropped.
4. The pretrained model's greedy reading of the same frames, folded onto the
   inventory by `scripts/xeus_map.py`, is stored beside the label as the
   **baseline** — so every later comparison is on identical audio.

Chunks are visited rarest-word-first and a per-type quota (500 training
segments) stops frequent words drowning the rest.

## 4. Splits

Two held-out sets, chosen before any audio was cut (`data/xeus_ft/split.json`,
seed 20260907):

| split | what is held out | measures |
|---|---|---|
| `val_words` | **60 word types** (never seen in training in any clip; mid-frequency, 14 of them loshn-koydesh, half carrying the phones in §1) | whether the model learned phones or memorised word shapes |
| `val_eps` | **14 episodes** (any word) | transfer to recordings it never heard |

Any segment from a held-out episode is `val_eps`; any other segment containing
a held-out word type is `val_words`; training uses neither. The corpus is one
host, so `val_eps` is recording-, not speaker-, generalisation.

## 5. The model

PhoneticXeus = CNN frontend → linear → 19 E-Branchformer blocks (d=1024,
570M params) → 428-way CTC head. The encoder was trained with inter-CTC
*conditioning*: intermediate posteriors from that head are fed back into later
blocks. Hence the surgery:

- the pretrained 428-way head **stays, frozen** — removing or resizing it would
  change what every later block sees;
- a new **35-way Yiddish head** (`<blank>` + the 34 phones of
  `docs/yiddish_phoneme_set.md` §1, stress excluded) is added beside it on the
  final output and carries the loss. It is **warm-started** from the pretrained
  rows: a diphthong's row is the mean of its two elements' rows (aj ← a, ɪ),
  an affricate's is the pretrained affricate row (ʦ ← t͡s). At epoch 0 the
  model is the pretrained recognizer folded onto the inventory;
- the CNN frontend and the lowest 6 encoder blocks are frozen; blocks 6–18,
  the encoder's non-block layers and the new head train.

AdamW, lr 2e-5 (encoder) / 5e-4 (head), 300-step warm-up then cosine, bf16
autocast, CTC loss with `zero_infinity`, batches of ≤48 padded seconds,
length-bucketed. The checkpoint with the best `val_words` PER is kept.

Stress is not predicted. CTC stress placement is unreliable and stress is the
G2P's job; the recognizer answers *which phones*, the lexicon answers *where
the stress is*.

## 6. Metrics

All paired, baseline vs fine-tuned, on the same segments:

- **PER** (phone error rate, edit distance / reference length) per split;
- **exact-match rate** of whole segments;
- **recall on the phones the pretrained model mishears** (ʦ aj ej z ʃ u f ɔj oʊ ʒ),
  with what each was heard as;
- **per-held-out-type exact match** on single-word clips.

## 7. The dictionary (`xeus_yi_decode.py`)

`data/xeus_ft/dictionary.json` holds the 413 certain words and their
pronunciations — nothing the engine merely guessed. The decoder uses it two
ways:

- **`--words`** — the transcript is known and the question is *how* each word
  was said: every dictionary variant of every word is scored by CTC likelihood
  over the clip and the best one is reported per word, with the margin in
  nats. This is speech-to-pronunciation, and the tool for homographs and
  multi-reading words.
- **`--snap`** — no transcript: decode freely, then segment the phone string
  into dictionary words by edit-distance dynamic programming. Spans no word
  claims are returned as bare phones, flagged — the recognizer's output
  becomes a checkable claim about words without ever being forced onto a
  pronunciation nobody verified.

Words reach the dictionary through the engine's lexicon key (nikud stripped,
finals and YIVO ligatures folded), mirrored in `xeus_ft_common.lexicon_key`
and verified identical to the engine over all 92,617 surface forms in the
corpus.

## 8. Running it

```
.venv/bin/python scripts/xeus_ft_text.py            # local, 13 s
CLOUD=SECURE scripts/xeus_ft_runpod.sh up            # pod + deps + 5.5 GB audio
scripts/xeus_ft_runpod.sh prepare                    # forced-align, cut, label
scripts/xeus_ft_runpod.sh train --epochs 4
scripts/xeus_ft_runpod.sh eval
scripts/xeus_ft_runpod.sh fetch && scripts/xeus_ft_runpod.sh down
```

Traps met on the way: community-cloud pods can sit "RUNNING" with no machine
for 15 minutes (use `CLOUD=SECURE`); macOS `rsync` is openrsync and rejects
`--info=progress2`; the vendored model code imports `typeguard` without
declaring it; `nohup … &` inside a plain `ssh` call holds the session open.

## 9. Results (run of 2026-09-07, RTX 3090, 37 min of training, ~$1)

Data actually used: **79,448 training segments, 21.0 h** of certain-word audio
cut from 23,666 chunks; 4,000 `val_words` clips (1.4 h) and 4,000 `val_eps`
clips (0.92 h). 348 of the 413 certain types reached training (the rest are
held out or have no clip that survived the filters).

Trainable: 392.7M of 575M parameters. Best checkpoint: epoch 4, chosen by
`val_words` PER. Each epoch took 9.3 min at 2.87 steps/s.

### Before / after, same 8,000 clips

| split | PER before | PER after | exact before | exact after |
|---|---|---|---|---|
| `val_words` (60 unseen word types) | 0.517 | **0.336** | 2.1% | **7.1%** |
| `val_eps` (14 unseen episodes) | 0.502 | **0.244** | 4.8% | **35.4%** |

Epoch by epoch (`val_words` / `val_eps` PER): 0.652/0.692 at epoch 0 (the
warm-started head before any gradient — worse than the fold baseline, as a
mean-of-rows initialisation is crude), 0.347/0.270 after epoch 1,
0.349/0.255, 0.336/0.245, 0.336/0.244. Most of the gain is in the first epoch.

### The phones the pretrained model could not hear (`val_words` recall)

| phone | n | before | after | before heard as | after heard as |
|---|---|---|---|---|---|
| aj | 1,592 | 0.000 | **0.769** | i 885, a 311 | aj 1225, ∅ 223 |
| ɔj | 523 | 0.000 | **0.730** | i 202, ɔ 112 | ɔj 382, ∅ 72 |
| oʊ | 157 | 0.000 | **0.535** | ɔ 129 | oʊ 84, ɔ 31 |
| ej | 260 | 0.000 | **0.496** | ɛ 173 | ej 129, ∅ 71, aj 26 |
| ʦ | 582 | 0.110 | **0.658** | s 313, t 102 | ʦ 383, ∅ 115 |
| z | 2,545 | 0.306 | **0.726** | s 890 | z 1848, s 57 |
| ʃ | 1,079 | 0.529 | **0.780** | s 244 | ʃ 842, s 20 |
| u | 2,251 | 0.490 | **0.669** | ə 290, a 270 | u 1505, ∅ 364 |
| f | 967 | 0.789 | 0.776 | — | — |

Every Yiddish diphthong went from **zero** recall — the fold map never
produces one — to 50–77%, on word types the model never saw. The sibilant
voicing confusion (z→s 890 times before) is essentially gone (57). What
remains is mostly *deletion* (∅): the fine-tuned model drops a phone it is
unsure of rather than substituting the wrong one, which is the better failure
for a voting pipeline.

### Per-word exact match on single-word clips (57 held-out types)

Macro 0.161 → 0.167; 14 types improved, 13 regressed. Large gains on words
carrying the phones above (`דריי` 0→70%, `ביזנעס` 0→65%, `טעקסט` 12→65%,
`עסן` 0→90%, `שיינע` 0→13% of 100 clips); real regressions on some short
words (`מער` 74→11%, `בילד` 75→6%, `כי` 100→17%). §10 looks at those.
Single-word clips are the hardest case — 0.2–0.4 s of audio, often
function words reduced in running speech — so exact match there is a
stricter test than the PER, and the sample per word is small.

Files: run 1's reports are under `data/xeus_ft/run1/` (`ckpt/before/eval.md`,
`ckpt/eval.md`, `ckpt/train_log.jsonl`, `ckpt/lattice_eval.json`,
`prepare_stats.json`). Its weights were lost to a fetch/move race on
2026-09-08 and are not needed: run 2 (§11) matches or beats it on every
measure and is `data/xeus_ft/ckpt/best`.

## 10. What the regressions are (diagnosis on the held-out single-word clips)

Decoding the clips of the regressed words shows two systematic errors, not
noise:

| word | gold | fine-tuned says (n) | before said |
|---|---|---|---|
| מער | `m ɛ r` | `m i r` **17/19** | `m ɛ r` 14/19 |
| בילד | `b i l t` | `b ɛ l t` 6, `b ɛ l` 3, `b i l t` 2 | `b i l t` 8 |
| שיינע | `ʃ aj n ə` | `ʃ aj n` **57/86**, `ʃ aj n n` 15, `ʃ aj n ə` 4 | `ʃ ɛ i n ə` 33 |
| כי | `k i` | `k i n` 3, `k i` 3 | `k i` 6 |
| דריי | `d r aj` | `d r aj` **14/19** | `d r a i` 7 |
| עסן | `ɛ s n` | `ɛ s n` **8/9** | `ɛ s ə n` 5 |

1. **ɛ ↔ i in short stressed syllables**, in both directions (`מער`→*mir*,
   `בילד`→*bɛlt*). The Hasidic stressed vowels here are close in quality and
   0.27–0.6 s clips give the model little context; the pretrained model had
   these right, so this is a bias the fine-tune introduced, most likely from
   the frequency skew of `i` in the training labels (מיר, איז, די…).
2. **Final unstressed ə is dropped** (`שיינע`→*ʃajn*). The diphthong — the
   thing the fine-tune was for — is right in 79 of 86 clips; the schwa is
   missing in 75. The likeliest cause is on the data side: segment edges are
   padded by at most 60 ms and never past halfway to the next word, and the
   forced alignment often hands a word-final schwa's frames to the following
   word, so many training clips end before their schwa. The model learned
   that a clip ends without one.

Neither touches the diphthong / affricate / sibilant gains, which are what
the audio-evidence pipeline needed. Both are fixable in a second run without
new labels: looser end-padding when the next word's onset is a consonant
(2), and either class-balanced sampling or a small ɛ/i-focused hold-out for
model selection (1). A second run costs ~$1 and 40 minutes.

Practical reading for now: trust the fine-tuned model's *diphthong class*,
*affricate* and *sibilant voicing* votes; treat an ɛ/i vote in a monosyllable
and a missing word-final ə as uninformative.

## 11. Run 2: the ear as its own aligner, and augmentation

Same held-out word types and episodes; the clips were **re-cut** with three
changes, so the two runs are not on byte-identical audio (the pretrained
baseline on run 2's clips is 0.547, on run 1's 0.517 — the new cuts include
the schwa tails and are harder). Chain 3 (§12) scores both ears on identical
clips.

What changed:

1. **Aligner and variant chooser = the run-1 model**, in the Yiddish phone
   space. Its alignment-confidence scores are not on the pretrained model's
   scale (a fine-tuned CTC model is peaky; median per-word posterior 0.02 vs
   0.56), so the absolute 0.25 filter became a relative one: drop the 10% of
   each split with the lowest score.
2. **End-padding up to the next word's onset** (≤120 ms) instead of half the
   gap, so a word-final ə handed to the neighbour by the alignment stays in
   the clip.
3. **Waveform augmentation** during training: one speed factor per batch
   (0.9/1.0/1.1), per-clip gain ±6 dB, white noise at 10–40 dB SNR half the
   time. Six epochs, encoder lr 1.5e-5, 500-step warm-up. 57 min.

Data: 78,180 training clips, 20.9 h; 3,600 clips per validation split.

### Before / after — clean, and with noise added

| split | PER before | PER after | exact before | exact after |
|---|---|---|---|---|
| `val_words` clean | 0.547 | **0.336** | 1.8% | 6.9% |
| `val_words` @ 15 dB SNR | 0.675 | **0.360** | 0.2% | 5.5% |
| `val_eps` clean | 0.548 | **0.272** | 4.0% | **34.7%** |
| `val_eps` @ 15 dB SNR | 0.723 | **0.301** | 0.1% | **29.0%** |

This is the robustness result. Adding noise at 15 dB costs the pretrained
recognizer 13–18 PER points and takes its exact-match rate to zero; it costs
the fine-tuned model 2.4–2.9 points. The fine-tuned model in noise is better
than the pretrained model on clean audio by a wide margin.

### Phones (val_words recall, before → after)

aj 0.00→**0.79**, ɔj 0.00→**0.64**, oʊ 0.00→0.39, ej 0.00→**0.48**,
ʦ 0.10→**0.67**, z 0.30→**0.72**, ʃ 0.52→**0.82**, u 0.40→0.63,
**ɛ 0.56→0.70, i 0.36→0.67** (the run-1 confusion is now tracked and both
improved over the pretrained model), **ə 0.74→0.62** — the schwa is still
the phone the fine-tune loses: 937 of 3,915 are deleted. The longer
end-padding did not cure it; the deletion is now in the model's habit, not
only in the clips, and needs a training-side fix (e.g. a blank-penalty at
decode time, or weighting ə in the loss).

Per-word exact match on single-word clips: macro 0.110 → 0.128; 13 improved,
8 regressed (`מער` still regresses, 0.75→0.19; `צי` 0.18→0.91,
`ביזנעס` 0→0.31).

Verdict: run 2 matches run 1 on clean held-out words (0.336 both), is far
more robust to noise, and is the checkpoint to use. Training curves for both
runs flatten after epoch 1–2; the remaining errors are not going to come
from more epochs.

## 12. The lattice: word accuracy by candidate level

`scripts/xeus_lattice.py`. The ear's frame posteriors are scored against a
candidate list and the best candidate is taken, per word, left to right — the
ear only has to *discriminate*. Three nested candidate sets:

- **graph** — every reading the spelling can legally have (open slots of
  spec §4 branched on the gold reading, plus ɛ/ə reduction and final
  devoicing);
- **menu** — what was actually heard for the word across the 78,180 training
  clips: for each clip the graph candidate the ear scores best, counted
  (`data/xeus_ft/menu.json`, 348 words; e.g. `פאר`: far 2,850 · fur 107 ·
  fɔr 39 · par 20). Corpus evidence filtered through the graph;
- **gold** — the word's native-verified readings.

Word accuracy on the held-out clips, run-2 ear (run-1 ear on the identical
clips in parentheses):

| candidates | unseen **words**, single-word clips (n=405) | unseen **episodes**, single-word clips (n=730) | unseen words, all words (n=16,435) | unseen episodes, all words (n=11,264) |
|---|---|---|---|---|
| free (generate) | 13.8% (12.8) | 64.7% (49.7) | — | — |
| graph | 73.1% (70.6) | 91.1% (86.7) | 80.7% (80.0) | 85.6% (84.2) |
| menu | **95.3%** (95.1) | **95.2%** (92.3) | 89.6% (88.9) | 90.0% (88.9) |
| gold | 95.3% (95.1) | 98.6% (96.6) | 96.6% (95.7) | 97.4% (96.5) |

The same ear goes from 14% to 95% on words it has never heard, with nothing
about the acoustic model changed — the task changed from generation to
multiple choice. The menu, mined from audio, is worth +22 points over the
graph on unseen words and closes the whole gap to gold there; on unseen
episodes gold still adds 3.4 points, which is what the remaining menu
distractors cost. Run 2 is better than run 1 at every level on identical
clips, most visibly in free decoding on unseen episodes (49.7 → 64.7).

Where the remaining single-word errors live (run 2, unseen words): 240 are
pure generation errors the graph fixes, 90 are graph distractors the menu
removes, 19 are cases where even the gold list does not help — the ear
cannot tell two verified readings apart, or the label is wrong. On unseen
episodes: 270 / 41 / 25, plus 31 where the menu carries a distractor that
gold does not.

## 13. Whisper for Yiddish (`ivrit-ai/yi-whisper-large-v3`), zero-shot

`scripts/whisper_yi_probe.py`, 20 held-out chunks, RTX 3090, 10.6 s per
30-s chunk: **WER 0.503** against the Yiddish Labs transcripts after nikud
stripping, final folding and punctuation removal. Inspection says a large
share of that is orthography, not recognition — `מיטן` vs `מיט'ן`, YIVO
vs Hasidic spellings, the ivrit.ai training data's conventions — so the
number is an upper bound on its real error. It is not usable zero-shot as
the *which word* front end for the lattice; it would need fine-tuning on the
10 h of aligned Hasidic transcripts plus a spelling normaliser on both sides
before it is worth measuring again.

## 14. Decode-side gains without retraining (`scripts/xeus_beam.py`)

Two decoders compared on all 3,600 held-out clips per split, run-2 ear:

| decoder | unseen words PER | exact | single-word acc. | ə recall | unseen episodes PER | exact | single-word acc. | ə recall |
|---|---|---|---|---|---|---|---|---|
| greedy | 0.333 | 6.9% | 13.6% | 0.62 | 0.272 | 34.5% | 64.7% | 0.76 |
| greedy + blank penalty 1 | 0.322 | 7.5% | 14.6% | 0.67 | 0.276 | 34.1% | 64.8% | 0.79 |
| **dictionary-guided prefix beam (8)** | **0.321** | **10.4%** | **21.0%** | 0.63 | **0.267** | **37.6%** | **67.4%** | 0.77 |
| beam + blank penalty 1 | 0.327 | 9.4% | 19.0% | 0.66 | 0.285 | 35.2% | 66.0% | 0.79 |

The prefix beam keeps the best eight phone prefixes and, through a trie of
Chezky's pronunciations, rewards a prefix that completes a known word and
taxes one that walks off the trie — the acoustic probabilities are never
edited. It lifts single-word accuracy on unseen words from 13.6% to 21.0%
and whole-clip exact match on unseen episodes from 34.5% to 37.6%, at no
cost in PER. It is now the default free decoder in `xeus_yi_decode.py`
(`--greedy` restores the old one).

The blank penalty recovers some of the dropped schwas (+4–5 points of ə
recall) but pays for it elsewhere once combined with the beam, and past 1.0
it hurts everything. The schwa is therefore still a training-side item: the
posteriors under-weight ə, and a decode-time thumb on the scale can only
trade one error for another.

## 15. The menu as a review surface

`data/xeus_ft/menu.json` rendered as a page (one card per word: Chezky's
readings, every reading the ear heard with its count and share, ✓ on his):
https://claude.ai/code/artifact/467f6927-f41f-4ecc-8b3c-a40d243a37cd — also
`data/xeus_ft/menu.tsv`. Cards where the most-heard reading is not his are
the review queue: **16 of 348**, listed with shares in
`data/xeus_ft/chezky_review_queue.tsv` (23 rows once secondary readings over
25% are included). The largest are reduction in function words (`דער` heard
*dər* in 28% of 5,840 clips, `דעם` *dəm* 30%) and a handful of genuine
questions for him (`ניו` *nji* 70%, `סארט` *surt* 47%, `אפאר` *afar* 44%);
the `זעהט/זעה/זעהן` → *aj* rows are most likely the ear's ej confusion (ej
recall 0.48) rather than his label. A ruling on a card is a dictionary edit;
the menu rebuilds from the corpus in ~10 minutes on the pod.

## 16. Run 3 (schwa oversampling): a negative result

Hypothesis: the dropped word-final ə (§11) is a data-balance problem. Test:
re-cut with the run-2 ear, then continue training from run 2 with every clip
containing a word-final ə repeated three times (18,289 clips → 114,236
training rows, 33 h), encoder lr 8e-6, head lr 2e-4, augmentation on.

Result on the third cut of the held-out clips (run 2 itself scores 0.357 /
0.308 there):

| epoch | unseen words PER | unseen episodes PER | ə recall | oʊ recall |
|---|---|---|---|---|
| 0 (= run 2) | 0.357 | 0.308 | 0.62 | 0.39 |
| 1 | 0.375 | 0.309 | 0.61 | 0.06 |
| 2 | 0.385 | 0.308 | 0.59 | 0.06 |

Worse on the target metric, no gain on ə, and oʊ collapsed. Model selection
kept epoch 0, so the run returned run 2 unchanged; `ckpt_schwa/` holds only
its reports. Oversampling moved the model's priors without teaching it the
schwa — the deletion is a blank-vs-ə decision inside the CTC objective, not a
shortage of examples. The fix to try next, before any further GPU time, is at
the loss: a fixed penalty on the blank logit during training so ə has to win
its own frames, or a frame-level auxiliary loss on the ə frames from the
forced alignment. Cost of this run: ~$1.20.

The run-2 checkpoint (`data/xeus_ft/ckpt/best`) remains the model to use,
with the dictionary-guided beam decoder (§14).

## 17. Audio attestation: the ear decides the rule-path words

`scripts/xeus_attest_text.py` + `scripts/xeus_attest.py`. The pointing model
(phonikud-yi) has only ever been supervised on words the engine could vouch
for — lexicon HIGH/MED, 65.8% of tokens (retrain3). The rest, **635k tokens /
83k types read by rule at LOW/MED**, never had a label. With the run-2 ear
every one of those occurrences can be decided: forced-align the chunk, take
the token's frames, score the spelling's legal readings (the §12 graph,
branched from the engine's reading) in one batched CTC call, keep the best
and its margin.

Result (A6000, 103 min, sharing the GPU): **576,017 tokens scored**; the
engine's reading kept on 357,671 (62%), changed on 218,346 (38%); 164,975
decided at ≥ 2 nats, 24,306 at ≥ 4. What changes, at ≥ 2 nats, is the
Hasidic sound system asserting itself over the rule engine's defaults:
ɛ→ə 7,689 (unstressed reduction), u→i 4,365 and a→u 2,495 (the א/ו vowel
classes), aː→aj 1,453, oʊ→ɔj 1,079 (the אויפ־ prefix), ej→aj 1,078,
d→t 1,219 (final devoicing), f→p 898 (`געפאניקט`).

Two datasets are built from it, locally, in under a minute each:

- **retrain8** (`prepare_retrain_dataset_v8.py`, for the nikud model):
  111,366 tokens stamped from their own clip's decision, 135,186 from a
  type-level reading (6,001 types agreeing at ≥ 85% over ≥ 5 clips), each
  pointing read back through `reconcile`. Coverage **65.8% → 79.3%** of
  tokens; `test.jsonl` untouched.
- **renikud_yi** (`renikud_yi_prepare.py`, §18): 162,656 occurrence-level
  and 220,114 type-level audio labels on top of gold + lexicon; **84.9%** of
  tokens labelled (63.9% without audio).

## 18. ReNikud for Yiddish

Melichov, Kolani & Alper (2026) cast Hebrew G2P as per-letter classification
— every letter predicts (consonant, vowel, stress) on a char-BERT — trained
on (text, IPA) pairs pseudo-labelled from audio by a phoneme ASR. The Yiddish
port keeps the frame and changes two things:

1. **The aligner** (`scripts/yi_align.py`): Yiddish writes its vowels, so a
   vowel letter takes consonant ∅ and a vowel, a Germanic consonant letter a
   consonant and vowel ∅, a loshn-koydesh letter both; digraphs
   (וו יי וי זש טש דזש טס תש) put the phone on the first letter; the LK
   vowel-before-consonant pattern (רוח rˈiəx) and final devoicing are
   allowed. It places **98.5%** of Chezky's readings and 98.3% of corpus
   types; what it refuses is what should be ignored (spelled-out
   abbreviations).
2. **The labels** (`scripts/renikud_yi_prepare.py`): not a free ASR
   transcript but the lattice decision — gold, then lexicon, then the ear
   choosing among the spelling's *legal* readings against the clip.
   Unvouched words are IGNORE (-100), never guessed.

Model (`scripts/renikud_yi_train.py`): the phonikud-yi v6 encoder (24-layer
char BERT, d=1024) with three coupled heads. 3 epochs, 66 min on an A6000
shared with attestation.

### The measurement that matters

The trainer's word accuracy is agreement with the labels and saturates
(99.65% on the test episode, no-audio model) — those are word types the model
saw. `scripts/renikud_yi_eval.py` scores the test episode's **rule-path words
with an audio decision at ≥ 2 nats (n = 517)**, which no label ever covered,
against what the audio says:

| system | rule-path words, agreement with audio | gold words |
|---|---|---|
| frozen rule engine (production) | **88.4%** | 99.97% |
| ReNikud-yi, no audio labels | 46.0% | 96.85% |
| ReNikud-yi, audio labels | _training_ | |

Without audio the per-letter model memorises the labelled types and does not
recover the rules the engine has by hand; 88.4% is the bar. The with-audio
model's number lands in this table when its run finishes.
