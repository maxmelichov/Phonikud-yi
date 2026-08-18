# Audio evidence: turning recordings into lexicon verdicts

How the corpus audio is used to decide pronunciations the writing system leaves
open, and — equally important — how it is prevented from deciding the ones it
cannot. Read `docs/xeus_to_yiddish_map.md` first for the phone-folding layer;
this document covers everything above it.

---

## 1. Why audio at all

Unpointed Hasidic Yiddish does not write the information the G2P needs. For
`פאר` the page is silent on /f/ vs /p/; for `א` it is silent between a, ɔ and
u. Only two things in the world can settle such a word:

1. a native speaker — the gold CSV, 509 words, authority #1
2. the recording — what a speaker actually produced

Native verdicts do not scale: 509 words in four rounds over a week. The corpus
holds ~250 episodes. Audio is the only source of evidence that scales, so the
question is not *whether* to use it but how to use it without letting the
recognizer's errors into the lexicon.

## 2. The pipeline

```
episode MP3  ──ffmpeg──>  16 kHz mono slice of one dataset row
                              │
                    PhoneticXeus (428-symbol universal IPA)
                              │
                    fold to the closed v3 inventory      scripts/xeus_map.py
                              │
        Needleman-Wunsch align against the engine's own reading
                              │
        per-word record: engine phones, heard phones, PER-SLOT pairing
                              │
                    data/audio_lexicon/pe_sweep_tags.jsonl
```

The tag pool is **shared and append-only**: every sweep writes into it, and
every folder reads from it, so one transcription run improves all downstream
verdicts. Chunks already present are skipped, which makes every sweep
resumable.

**Positional alignment is the load-bearing detail.** Earlier tooling recorded
only a per-word agreement score, which cannot answer "is *this* פ a p?".
`tag_positional()` keeps the aligned heard phone at every engine slot, so a
verdict can be attached to one grapheme rather than to a whole word.

## 3. Choosing what to listen to

`scripts/xeus_sweep_all.py` picks chunks greedily by **token gain**: how many
LOW-confidence *tokens* a chunk would newly bring to ≥3 clips. Type-greedy
selection was tried first and is wrong — LOW is 51k types but only 346k
tokens, so ranking by types spends the compute on hapaxes while the words a
listener actually hears go unchecked. Measured over the same run: token-greedy
reaches 73% of LOW tokens where type-greedy reaches 7% of types.

Gains are recomputed lazily (a heap entry is re-scored when it surfaces), so
selection stays exact as coverage fills in without rescoring every candidate
each round.

**Current coverage (2,403 chunks transcribed):** 81% of LOW tokens heard at
least once, 73% heard three or more times.

## 4. The recognizer is not a witness — it has an accent

Pooled over every tagged slot, PhoneticXeus does this to the inventory
(`data/audio_lexicon/confusion.tsv`, regenerate with `scripts/audio_calibrate.py`):

| engine phone | what the recognizer reports |
|---|---|
| ʦ | s 62%, t 16%, **ʦ only 10%** |
| aj | i 50%, ɛ 19%, ə 15% |
| ej | ɛ 50%, ə 36% |
| z | s 44%, z 35% |
| ʃ | ʃ 60%, s 31% |
| u | u 35%, a 21%, ə 21% |
| f | f 72%, p 11% |

A raw majority vote over these outputs would conclude that Yiddish has no
affricates, no diphthongs and no voiced sibilants. Any use of audio evidence
must model this first.

## 5. Three filters, in order

A verdict is folded only if it survives all three. Nothing here is a
word-list; each filter is a general test.

### Filter 1 — is it surprising for this word?

`scripts/audio_calibrate.py` computes the base rate P(heard | engine) from the
table above, then scores each word-slot's votes against it with a binomial
tail probability (in log space — frequent words have hundreds of clips and
`math.comb` overflows). ʃ→s at 31% is the recognizer's habit; a word whose ʃ
is heard as s in 9 of 10 clips is surprising, one at 4 of 10 is not.

Of 26,560 strongly-voted slots, 80% agree with the engine outright and 2,442
of the disagreements are statistically surprising.

### Filter 2 — does the spelling even leave it open?

Surprise is not enough, and this is where a purely statistical approach fails.
`האט` is heard as *hat* in 1,833 of 4,424 clips — highly surprising, and
completely wrong as a label: it is unstressed-vowel reduction in fast speech,
not a different word. Same for `איז`→*is* and `געזאגט`→*gezukt*, which are
final devoicing, a process v3 deliberately excludes from citation forms.

The discriminator is the orthography. **ז can only be /z/; ג can only be /ɡ/.**
Audio deviation at a letter the spelling determines is a predictable process,
not lexical evidence, and is discarded. Audio is admitted only where the
writing system is genuinely undecided — the spec §4 ambiguity sets:

| grapheme | open between |
|---|---|
| א | a · ɔ · u · aː |
| פ | f · p |
| יי | aj · aː · ej |
| וי | ɔj · oʊ |
| ו (shuruk) | i · u |

This filter cuts 2,442 candidates to **412**, and the 2,030 it rejects are
exactly the expected inventory of processes: a→ə, i→ə, ʃ→s, d→t, b→v.

### Filter 3 — has a human already ruled?

Gold outranks audio, always. 37 of the 412 touch a gold word; they are **not
folded** and go to the native-verification queue instead, with the vote counts
attached. That is how `זוכט` (gold *zixt*, audio *zukht* 48/60) reaches Chezky
as a question rather than silently flipping.

**375 candidates remain eligible**, of which 204 are words the engine is
currently guessing at (LOW/MED).

## 6. What ships, and at what confidence

| table | built by | contents | route |
|---|---|---|---|
| `data/lexicons/audio_pe_lk.py` | `build_audio_pe_lexicon.py` | 77 words whose f-default the audio refuted unanimously | MED, `audio-pe` |
| `data/lexicons/audio_vowel_lk.py` | `build_audio_vowel_lexicon.py` | 136 alef-default words corrected to a clean-target vowel | MED, `audio-vowel` |
| `data/lexicons/audio_endorsed_lk.py` | `xeus_verify_hebrew.py` | 107 Hebrew readings whose corpus pointing the audio confirms | LOW, `pointed-audio-endorsed` |

All three are consulted **after** every gold and legacy lexicon and **before**
the rule path, and an explicit פּ/פֿ point in the text still overrides them.
MED, not HIGH: the evidence is acoustic, not native, so these words remain
visible for review.

Contested and rejected material is written to queue files rather than dropped:
`vowel_queue.tsv` (932 slots where bias could explain the vote),
`lk_sweep_votes.tsv` (1,159 AUDIO-OK / 234 SUSPECT Hebrew readings),
`calibrated.tsv` (every slot with its p-value and verdict).

## 7. Feeding the pointing model

`scripts/prepare_retrain_dataset_v2.py` stamps the audio tables as training
targets for phonikud-yi, and — critically — **excludes the 234 SUSPECT
readings from supervision entirely**. Teaching the model a reading the audio
refutes is worse than leaving the word unsupervised. Verified: 2,057 tokens of
suspect types appear in the training data, zero of them stamped.

`scripts/relabel_evidence_conflicts.py` handles the reverse case — a label that
is already supervised but contradicts a higher-authority verdict. The stamping
passes only fill *unsupervised* tokens, so a wrong v1 label survives every
pass; that hole is why 793 `א פאר` tokens stayed labelled *far* through two
retrains. Repairs are validated by read-back (the candidate re-pointing must
strip to identical letters and read as the evidence IPA), never assumed.

## 8. What audio cannot do

Stated plainly, because two rounds of this project got it wrong:

- **It cannot overrule a native speaker.** Filter 3 exists for this.
- **It cannot decide reduced vs citation forms.** The recordings show what was
  said in fast speech; the labels encode what the word is. `האט` is *hut* no
  matter how often it surfaces as *hat*.
- **It cannot settle a letter the spelling determines.** Filter 2.
- **It cannot cover the tail.** 82% of word types have fewer than 5 corpus
  occurrences; most will never reach a 3-clip threshold. Those stay with the
  rules, the book pointing, or the model guess.

## 9. Reproducing

```bash
# transcribe more corpus audio into the shared pool (resumable)
.venv/bin/python scripts/xeus_sweep_all.py --max-chunks 1500
.venv/bin/python scripts/xeus_sweep_all.py --plan-only     # projection only

# recompute the recognizer profile and per-slot verdicts
.venv/bin/python scripts/audio_calibrate.py

# regenerate the committed tables, then gate
.venv/bin/python scripts/build_audio_pe_lexicon.py
.venv/bin/python scripts/build_audio_vowel_lexicon.py
.venv/bin/python scripts/xeus_lk_sweep.py --report-only
.venv/bin/python scripts/test_audio_evidence.py
.venv/bin/python scripts/run_corpus_v3.py --limit 0
```

`test_audio_evidence.py` (3,210 checks) is the gate: table integrity, the
authority order, re-derivation of every vote bar from the raw counts, and
unit tests of the verdict logic itself. It runs inside corpus QA gate (c).
