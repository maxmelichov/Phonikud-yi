# Hasidic Yiddish G2P — Project History

How the pipeline got from a rule-sketch engine to a native-verified,
audio-validated, context-aware G2P system. Covers Aug 6–14, 2026 — the
v2→v3 sprint, the no-drop rescue chain, corpus-wide audio evidence, and the
v4/v5 pointing models — built on the pre-existing engine, scraper, nikud model, and TTS
dataset. Written as the record of what was done, what was decided, and why.

---

## 0. Starting point (before Aug 6)

- `yiddish_g2p.py`: a three-stage engine (Hebrew script → Latin base → IPA)
  tuned against the yiddish24 corpus and one native reviewer, with rule-based
  stress. No vowel-class system: **all** `o` surfaced as [u], no [ej]/[oʊ]/[ɔ],
  no postlexical layer.
- `models/phonikud_yi`: a Hasidic nikud model (dictabert-char fine-tune,
  4 training rounds) + distilled ONNX student, trained on Gemini-annotated
  pointing.
- `data/yiddish_tts_dataset.tsv`: 23,666 rows / 1.83M tokens of transcribed
  yiddish24 audio; ~250 episode MP3s on disk.

## 1. The v2 spec implementation (Aug 6)

**Input:** a 13-page phonemization guide (Unterland/Hasidic koine): Weinreich
vowel-class chain shifts, four lexically-ambiguous graphemes, LK nikud table,
stress rules, minimal-pair test set.

**Method:** first multi-agent workflow (7 Opus agents): gap analysis + spec
test-suite authoring in parallel → implementation → 3 adversarial verifiers →
fix round. The verifiers earned their keep: 22 confirmed defects the first
implementation pass missed, including a genuine regression (the affricate
normalizer fusing legitimate d+ʃ clusters, corrupting ~98 tokens/400 rows) and
a vowel-deletion bug making every א+וו word (אוועק, אוונט) vowelless.

**Delivered:** class 41 [ɔ] split from 12/13 [u]; class 25 [ej]; class 54 [oʊ]
split from 42/44 [ɔɪ]; final devoicing + regressive voicing assimilation;
LK/WH layers; 186-case spec suite. Conflicts with the audio-verified lexicon
logged, not overwritten (`data/spec_conflicts.md`).

## 2. Testing accuracy for real (Aug 6)

Spec-suite green ≠ accurate, so three instruments were built:

1. **Engine diff** (HEAD vs new) over the full corpus: 21.5% of running text
   changed, decomposed by change type; surfaced 5 issues no suite caught
   (ר׳→[rɛp] devoicing, רוב misclassification, aroyf- gap…).
2. **Native reviewer loop**: diff → frequency-sorted question sheets → Chezky
   verdicts (the format that became the gold CSV).
3. The groundwork for **audio validation** (later: PhoneticXeus).

## 3. Spec v3 + the gold lexicon (Aug 6–7)

**Input:** Corpus Pipeline Spec v3 — authoritative, incorporating four native-
verification rounds, with `g2p_gold_v3.csv` (500 words, 403 native-settled).
v3 **reversed** major v2 decisions: devoicing OFF everywhere (only voicing-ward
assimilation stays), notation aɪ→aj, syllabic finals with no epenthetic schwa,
the ־ער system (default ɛr, lexical ir-list, ej never before r), closed phone
inventory, routing/confidence/QA-gate architecture.

**Baseline measured first:** old engine vs gold = 38.4% exact.

**Method:** two-phase workflow (6 Opus agents): core phonology → gold-seeded
lexicon + routing + per-token metadata (`g2p_token`: layer/route/confidence) +
corpus runner with QA gates → 3 verifiers → fix. Second round fixed 14 more
findings — the biggest: unlexiconed loshn-koydesh (5% of corpus) was being
**emitted as well-shaped garbage** into training data (hkdiʃ, mʦrim); now
quarantined per §6.3.

**Result:** gold 500/500 byte-identical; all suites green; full-corpus QA
gates pass. Fingerprint `f8a48918093a`. Corpus health: 64% lexicon (HIGH) /
30% rule (MED) / 19.2% LOW / 5.8% quarantined. Rule-only diagnostic: 46.8% —
with the misses concentrated exactly in the four graphemes the spec declares
rule-unresolvable, i.e. the system is verification-limited, not rule-limited.

**Docs made executable:** `docs/yiddish_phoneme_set.md` (inventory + 23 rules)
is parsed and run by `scripts/test_rules_doc.py` — 104 examples against the
live engine, so documentation cannot silently drift. Writing it caught a real
bug (trailing quotes leaking into IPA).

## 4. Audio grounding with PhoneticXeus (Aug 7)

**Model:** `changelinglab/PhoneticXeus` — universal phone recognition
(self-conditioned CTC on the XEUS encoder), 16 kHz mono audio → IPA, no
language parameter, runs locally (2.3 GB checkpoint, MPS).

**Pipeline built:**
- `scripts/xeus_map.py` + `data/xeus_to_yiddish.tsv`: all 424 vocab symbols
  folded onto the closed Yiddish inventory (415 mapped / 9 deliberate drops /
  0 illegal), conservative by design (monophthong [e]→ɛ so only true
  diphthongs vote for marked classes). Guarded by `test_xeus_map.py`.
- `scripts/xeus_tag.py`: ffmpeg slice → transcribe → fold → Needleman-Wunsch
  align against the G2P prediction → per-word `heard` phones + agreement.
- `scripts/xeus_select_jobs.py` / `xeus_run_jobs.py` / `xeus_vote.py`:
  targeted job selection (contested gold words, top LOW_CONF, homographs),
  batch tagging, and vote aggregation.

**Scale run:** 1,215 chunks across ~250 episodes → **99,576 word tags** →
variant votes on 166 contested gold words (audio backs the primary 51%; the
rest split into real devoicing evidence — zukt 481:28 — running-speech
reduction, and known recognizer bias) and 762 consistent vowel votes,
including genuine lexicon catches (יארצייט→jurʦaːt, שטאט→ʃtut, אדם→udəm).
Audio evidence was folded into the verification batch, never applied
unilaterally (audio informs, the native speaker decides).

## 5. The four follow-up tracks (Aug 7)

One workflow (6 agents), all audited:

1. **Devoiced variants**: every voiced-final primary gets an auto devoiced
   variant (iz→is) for forced-alignment voting; primaries untouched
   (re-proven 500/500). User decision: ג spells /ɡ/ → primaries stay voiced.
2. **Line-yield policy**: strict vs `--emit-partial` measured on all 23,666
   rows — 8.2% vs **93.0% token yield (11.3×)**. Settled policy: segment
   partial lines at elision points, train on clean spans ≥4 tokens →
   `scripts/make_training_segments.py` → **83,266 spans / 1.67M tokens**.
3. **Chezky batch v4**: `data/verification_batch_v4.{csv,md}` — 400 words,
   frequency-sorted, grouped by question type, engine-derived candidate
   readings, audio-vote annotations on 42 rows. Projected: LOW_CONF
   19.18% → 10.84% when verified.
4. **Audio job list** (feeds §4's scale run).

## 6. Retraining phonikud-yi for context (Aug 7)

**Why:** the four ambiguous graphemes + homographs are resolved by nikud in
context; the G2P already reads nikud as overriding evidence. The v3 lexicon
created something new: verified readings for 64% of running tokens →
back-convertible to diacritics → real supervision for the pointing model.

**Dataset** (workflow: recon → build → train-setup → audit → fix):
- Recon found the actual trainer (vendored `phonikud/model/src/train/`,
  gitignored) and the canonical pointing convention — including the critical
  subtlety that **class-41 [ɔ] is written unpointed** (a komets would flip it
  to [u]).
- `scripts/prepare_retrain_dataset.py`: 23,100 train rows / 258 episodes /
  1.09M supervised tokens (61%), loss-masked elsewhere; test = 409 canonical
  rows of episode 100313, fully excluded from training; every supervised span
  round-trips to the raw text.
- The audit caught 5 issues pre-launch, most importantly that train targets
  and test gold were in **different pointing conventions** — fixed with a
  strict engine-derived table plus a computed eval ceiling (91.0 char / 77.0
  word on the old-convention test).

**Training** (three failed starts, each with a durable fix):
1. Local MPS run crashed at step 1001 — `evaluate()` re-enabled dropout in the
   frozen encoder; MPS SDPA can't do dropout. Fix: mode restoration + dropout
   zeroing on MPS.
2. Resume crashed copying a checkpoint file onto itself. Fix: same-file guard.
3. RunPod run trained on garbage — transformers 5.x loaded the slow tokenizer,
   offsets vanished, supervision collapsed to 198 chars, the unfrozen encoder
   destroyed itself (test 11.5%). Fix: version pin + a **hard label-collapse
   guard** in the trainer (abort if <50% of expected labels get placed).
4. Clean run: RTX 3090 (community, $0.22/hr), full unfrozen fine-tune, bf16,
   5,772 steps, ~1.9h. Total RunPod spend: **$0.59**.

**Evaluation** (the two-convention story, exactly as the ceiling predicted):

| Target | baseline | v2 |
|---|---|---|
| Old-convention canonical gold (char/word) | 76.9 / 51.4 | 75.8 / 49.3 |
| Engine-verified stamped targets (char/word) | 98.3 / 94.3 | **99.8 / 99.2** |
| Downstream letter-safety | 99.97 | 99.97 |

v2 moved toward the system of record: word-level pointing error vs the
verified convention cut **5.7% → 0.8% (7×)**. Caveat recorded: the stamped
test shares the training convention; independent generalization is scored
against Chezky's v4 verdicts when they return. **Decision: ship v2**
(`models/phonikud_yi_v2_gpu/best`), round4 kept as fallback.

---

## 7. No-drop: the Hebrew rescue chain (Aug 8)

**Directive:** "no drop can happen guess. train the phonikud-yi on that hebrew.
make it guess." Unpointed loshn-koydesh that no lexicon knew was being
quarantined — 5.82% of running tokens emitted nothing at all.

**Built:** a four-link rescue chain below the rule path, each link emitting at
LOW confidence with its own `reason` so the word stays in the verification
queue rather than passing as settled:
`_AUDIO_ENDORSED` (107 readings the audio confirms) → `_HOMOGRAPH_LK` (215
context-scored) → `_SEFARIA_POINTED` (~3,460 from published pointing) →
`_MODEL_POINTED` (7,574 phonikud-yi guesses).

Plus the §7.5 abbreviation ladder: acronym-words (רש"י → rˈaʃi) before
letter-name words (מ"ם → mɛm) before letter-by-letter spell-out
(תשפ"ה → tuf ʃin paj haj).

**Result:** quarantine 5.82% → **0.33%**, and what remains is only digits,
Latin and orphan letters — a number-normalizer gap, not a Hebrew gap.

## 8. The LK architecture (Aug 8)

Implemented from the user's own design: MWE tokenization (38 corpus-mined
collocations, כלל ישראל / זכרונו לברכה / בעל הבית fused before routing), a
register-aware diacritic bridge (Whole-Hebrew vs merged reading, with defect
rules — a merged reading that loses a consonant or vowel against WH can never
ship regardless of audio votes), and an LK-root + Germanic-suffix stemmer
(פשטלעך → pʃatləx from gold פשט + לעך).

The defect check earned itself immediately: `תצוה` had regressed to `tʦˈai`,
losing its /v/, resurrected by a noisy two-clip audio vote.
`merged_drops_a_consonant()` — with affricate decomposition, after a
false positive of my own on ת+ש → ʧ — now blocks that class outright.

## 9. ONNX, and the accuracy question (Aug 8)

v3 exported to ONNX and debugged through four layers: missing runtime
metadata, wrong class source, the BERT `[CLS]` offset convention, and finally
**`vocab.txt` being off by one against the true tokenizer id space** — the
pitfall now recorded in CLAUDE.md. Verified byte-identical to torch; ~51 ms
per sentence on CPU.

Asked "how accurate is it really", the honest answer was: gold 100% by
construction (64% of tokens), rules-only 47.2% type-level, corpus WER
*estimated* 5–13% with no measured number. That gap is still open and is the
single highest-value half-hour of human time left in the project.

## 10. far/par: one homograph, three layers (Aug 10)

A TTS sample was mispronouncing `א פאר יאר` ("a few years") as *a far yor*.
The bug existed in three places at once, which is why it had survived:

1. **The engine** — `פאר` is gold *far*; nothing fired for the noun reading.
   Fixed with an `א פאר → a pˈur` multiword entry, anchored by the
   chezky-verified gold row `אפאר → apˈur`, plus a new guard so an explicit
   פּ/פֿ point outvotes the point-stripped lexicon key (`פּאָר` no longer
   reads *far*).
2. **The Gemini nikud prompt** — listed פֿאַר in its pasekh examples and never
   demanded a choice, so every פאר came out as the preposition.
3. **The training data** — 793 tokens labelled *far*, and *already marked
   supervised*, so every evidence-stamping pass skipped them. Fixed by
   `relabel_evidence_conflicts.py`, which repairs labels that contradict a
   higher-authority verdict, validating each by read-back.

Measured in the old dataset: 27% of word types are pointed inconsistently
(האט 15 ways, פאר 36, האבן 34) — the direct cause of the released voice
mixing dialects.

## 11. Corpus-wide audio evidence (Aug 10–12)

The largest piece of work in this stretch; full methodology in
`docs/audio_evidence.md`.

- **pe sweep** (600 chunks): every f-default word voted on. 82 flip
  candidates, 77 folded into `data/audio_pe_lk.py` — כאפן *xapn*, פסוקים
  *psˈikim*, דאפלט *daplt* (12–0), plus Hasidic names (ראפשיצער, פשעווארסק)
  and loanwords the f-default mangled.
- **LK sweep**: all 6,169 rescued/fallback Hebrew types scored. 1,159
  AUDIO-OK, **234 SUSPECT** — the latter barred from training supervision.
- **General sweep** (1,500 chunks, token-weighted greedy): brought the pool to
  2,403 chunks, 81% of LOW tokens heard at least once, 73% heard 3+ times.
  Vowel table 42 → **136** entries.
- **Calibration** (`audio_calibrate.py`): measured what the recognizer does to
  every phone (ʦ→s 62%, aj→i 50%, ej→ɛ 50%) and scored each slot against its
  own base rate. Combined with the orthographic-ambiguity filter, 26,560
  strong verdicts reduce to 375 eligible corrections and 37 gold conflicts
  routed to Chezky.

Two bugs surfaced and were fixed here, both worth remembering: the table
builders wrote keys with hand-quoted f-strings, so a Yiddish apostrophe
(מורא'דיקע) broke the generated module — and the engine's seven loaders
**caught that SyntaxError and returned `{}`**, silently converting a corrupt
table into no table. Absent-file degradation stays; a table that exists and
will not load now raises.

## 12. phonikud-yi v4 and v5 (Aug 10–11)

Two short RunPod finetunes (RTX 3090, $0.92 total, pods terminated):

| | val char-acc | held-out episode |
|---|---|---|
| v3 baseline | 99.82 | 77.89 |
| **v4** (audio-corrected labels) | 99.93 | 78.24 |
| **v5** (+ 793 far/par repairs) | **99.94** | **78.31** |

v5 is the first model that points `אַ פּאָר יאָר` and `פֿאַר דער קהילה`
correctly in the same sentence. Probing showed it generalizes to invented
sentences, but only within the frame it was taught: `צוויי פאר שיך` still
comes out *far*. The lesson recorded — the model learns exactly the pattern
the labels encode, so broadening coverage is a labelling problem, not a
training problem.

Held-out test scoring carries a caveat: the test episode's labels predate the
audio corrections, so on corrected words the old key penalises the right
answer. Flat test-peek is the expected shape.

## 13. Packaging (Aug 11)

`src/` — an importable label stack with a **deployment guard**. The engine's
tables degrade silently by design; `yiddish_labels.verify()` asserts at import
that all seven loaded and that canary readings only those tables can produce
(פעקל → pɛkl, יארצייט → jˈurʦajt, האף → huf) actually appear. A half-installed
deployment now fails immediately instead of emitting plausible IPA with zero
native verdicts.

`src/make_bundle.py` builds `dist/phonikud-yi-engine.zip` and **refuses to ship
unless `selftest.py` passes inside the staged tree**. Also fixed here: an
import collision where the repo root's older `yiddish_nikud.py` (aimed at the
superseded export) shadowed the v5 wrapper.

Dataset handoff: `yiddish_tts_dataset_v2.tsv` — 23,666 rows, 20,895 fully
labelled, nikud from v5 (letter-identity validated per row), IPA from the
engine's strict corpus run. 93% of word types now carry one consistent IPA,
up from 82%.

## 14. Native verdict rounds (Aug 10, Aug 12)

Two WhatsApp rounds from Chezky, folded straight into gold:

- **Aug 10:** פאמיליע and פראגעס are *f* not *p* (pinning readings the engine
  already had, so the TTS voice — not the engine — was at fault); **האף is
  *huf*, not *haf*** — a real engine bug in the alef-default class.
- **Aug 12** (correction sheet, 9 words): גאס *gaːs*, פריינד *fraːnd*,
  פארגעסן *faːrɡˈɛsn*, אפשאצן ***ˈupʃaʦn*** (exactly the word flagged as
  suspicious in the TTS audio probe days earlier), פראביר *prɔbˈir*,
  פאדלאגע *pˈɔdlɔɡə*, געקאכט *ɡəkˈɔxt*. געפילט already matched inside the
  closed inventory.

**Not folded: לערנען.** Gold row 274 records him verifying *lˈirnən* with the
hint "l-ear-nen"; his Aug 10 message said "Lear not leer"; the Aug 12 sheet
says "lernen". Two native verdicts disagree, so the primary stays until he
settles it — and the answer decides the whole ־ער class, not one word.


---

## Verification methodology (used throughout)

Every substantive change went through the same shape:
**measure baseline → implement → adversarial verify (independent agents told
to refute, re-running everything themselves) → fix → re-gate.** The
adversarial passes found real, shipped-quality-level bugs in *every* round
(22, 14, 5+1, 5 findings respectively). Standing gates, all runnable:

| Gate | Command | Status (Aug 14) |
|---|---|---|
| Core regressions | `scripts/test_g2p.py` | 102/102 |
| Spec behaviors | `scripts/test_g2p_spec.py` | 300/302 (+2 documented XFAIL) |
| Gold byte-identity | `scripts/test_g2p_gold.py` | 509/509 |
| Executable docs | `scripts/test_rules_doc.py` | 23 rules / 122 examples |
| Phone-map coverage | `scripts/test_xeus_map.py` | 415 mapped / 9 drops |
| Whole-Hebrew readers | `scripts/test_g2p_wh.py` | 74/74 |
| Audio-evidence layer | `scripts/test_audio_evidence.py` | 3,210 checks |
| Corpus QA (a–d) | `scripts/run_corpus_v3.py --limit 0` | all pass |

(Plain scripts — run with `.venv/bin/python`, not pytest.)

## Key decisions log

| Decision | Choice | Basis |
|---|---|---|
| Devoicing | primaries voiced, devoiced auto-variants | ג spells /ɡ/ (user); audio says surface devoicing is real (zukt 481:28) → variants for alignment voting |
| Authority order | gold CSV > v3 spec > v2 guide > code comments | user-declared; spec's own "lexicon is the dialect" |
| Ambiguous graphemes | lexicon-only, never etymology at runtime | spec §4; rule-only diagnostic confirms |
| Quarantine | unlexiconed LK never emitted; lines segmented at holes | §6.3 + 11.3× yield measurement |
| Audio votes | inform batches, never edit lexicon directly | §12 loop; recognizer bias documented |
| Pointing convention | class-41 [ɔ] unpointed; engine-derived strict table | recon of canonicalize_pointing.py |
| Context handling | retrain nikud model, not new architecture | nikud already overrides in G2P; 1.09M supervised tokens available |
| No-drop | every Hebrew word gets a LOW-confidence rescued reading; only digits/Latin quarantine | user directive Aug 8; quarantine 5.82% → 0.33% |
| Audio vs orthography | audio admitted ONLY at spec §4 ambiguous graphemes (א פ יי וי ו); elsewhere the letter decides | ז is /z/, ג is /ɡ/ — deviation there is devoicing/reduction, not evidence |
| Audio vs recognizer bias | verdicts scored against the recognizer's own base rate, not raw majority | ʦ→s 62%, aj→i 50% — a raw vote removes affricates and diphthongs from the language |
| Audio vs gold | audio never overrides a native verdict; conflicts are queued as questions | 37 gold conflicts routed to Chezky rather than folded |
| Citation vs surface | labels encode the underlying word, not fast-speech realisation | האט stays *hut* though heard *hat* in 41% of clips; reduction is predictable, lexicon is not |
| Suspect readings | audio-refuted readings are excluded from training supervision, not just flagged | 234 SUSPECT types, 2,057 tokens, zero stamped |
| Corrupt generated tables | absent file degrades silently (deliberate); a file that exists and fails to load raises | a SyntaxError silently emptied a table and shipped it |
| Bundle integrity | the portable zip is built only if selftest passes inside the staged tree | tables degrade silently; a half-install must fail loudly |

## Artifact map

**Engine & tables** `yiddish_g2p.py`; generated-but-committed:
`data/gold_lexicon.py` (509), `audio_pe_lk.py` (77), `audio_vowel_lk.py` (136),
`audio_endorsed_lk.py` (107), `homograph_lk.py` (215), `sefaria_pointed_lk.py`
(3,460), `model_pointed_lk.py` (7,574), `stress_overrides.py`

**Gates** `scripts/test_g2p.py`, `test_g2p_spec.py`, `test_g2p_gold.py`,
`test_rules_doc.py`, `test_xeus_map.py`, `test_g2p_wh.py`,
`test_audio_evidence.py`, `run_corpus_v3.py`

**Docs** `data/g2p_spec_v3.md` (authoritative), `docs/yiddish_phoneme_set.md`
(executable), `docs/audio_evidence.md`, `docs/xeus_to_yiddish_map.md`,
`data/spec_conflicts.md`, this file

**Audio pipeline** `scripts/xeus_map.py`, `xeus_tag.py`, `xeus_pe_sweep.py`,
`xeus_lk_sweep.py`, `xeus_sweep_all.py`, `audio_calibrate.py`,
`xeus_verify_hebrew.py`, `xeus_vote.py`; pool
`data/audio_lexicon/pe_sweep_tags.jsonl`; reports `confusion.tsv`,
`calibrated.tsv`, `vowel_queue.tsv`, `lk_sweep_votes.tsv`

**Table builders** `build_gold_lexicon.py`, `build_audio_pe_lexicon.py`,
`build_audio_vowel_lexicon.py`, `build_sefaria_lexicon.py`,
`build_homograph_lexicon.py`, `build_model_guess_lexicon.py`

**Pointing model** `scripts/prepare_retrain_dataset_v2.py`,
`relabel_evidence_conflicts.py`, `train_phonikud_yi.py`,
`eval_phonikud_yi.py`, `export_onnx.py`, `embed_onnx_metadata.py`,
`infer_onnx.py`; `models/phonikud_yi_v5/` (+ `v5.onnx`)

**Consumers** `src/` (importable stack + deployment guard + `make_bundle.py`),
`scripts/retag_tts_dataset.py`, `data/yiddish_tts_dataset_v2.tsv`

## State & next steps (as of Aug 14, 2026)

**Where the system stands**

- Gold 509 words, byte-identity enforced; all eight gates green.
- Quarantine 0.33%; LOW-confidence 18.95% of running tokens (51,036 types).
- Audio: 2,403 chunks transcribed; 73% of LOW tokens heard 3+ times.
- phonikud-yi v5 shipped, ONNX-exported with embedded metadata.
- `yiddish_tts_dataset_v2.tsv`: 20,895 fully-labelled rows, 93% of word types
  carrying one consistent IPA.

**Critical path — not a G2P task**

Retrain the TTS voice on `yiddish_tts_dataset_v2.tsv`. Every improvement since
Aug 10 is invisible until that run happens. The training box must also stop
reading yiddish24's stored `nikud` column (`load_yiddish24_wav`), which keeps
~90% of the data on the old inconsistent labels regardless of what dataset
arrives.

**Open, in priority order**

1. **Measure.** There is still no measured accuracy number — 5–13% WER is an
   estimate. 30 random sentences scored word-by-word by a native converts it
   into a number and says whether further G2P work is worth doing.
2. **Fold the 375 calibrated audio corrections** (204 where the engine is
   merely guessing are pure gain); queue the rest.
3. **Chezky's queues**: 37 gold-vs-audio conflicts, 932 contested vowel slots,
   234 suspect Hebrew readings, 6 borderline פ words — frequency-sorted.
4. **לערנען** — the ־ער class hangs on it.
5. **Quantifier frame** (`צוויי פאר שיך`): v5 learned `א פאר` only. Broaden by
   labelling, not by hand-written rules.
6. Longer term: a direct text→IPA model on audio-derived targets
   (renikud-style) — the sweeps are its prerequisite and are half built.
