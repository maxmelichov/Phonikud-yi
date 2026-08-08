# Hasidic Yiddish G2P — Project History

How the pipeline got from a rule-sketch engine to a native-verified,
audio-validated, context-aware G2P system. Covers Aug 6–7, 2026 (the v2→v3
sprint), built on the pre-existing engine, scraper, nikud model, and TTS
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

## Verification methodology (used throughout)

Every substantive change went through the same shape:
**measure baseline → implement → adversarial verify (independent agents told
to refute, re-running everything themselves) → fix → re-gate.** The
adversarial passes found real, shipped-quality-level bugs in *every* round
(22, 14, 5+1, 5 findings respectively). Standing gates, all runnable:

| Gate | Command | Status |
|---|---|---|
| Core regressions | `scripts/test_g2p.py` | 52/52 |
| Spec behaviors | `scripts/test_g2p_spec.py` | 237/239 (+2 documented XFAIL) |
| Gold byte-identity | `scripts/test_g2p_gold.py` | 500/500 |
| Executable docs | `scripts/test_rules_doc.py` | 23 rules / 104 examples |
| Phone-map coverage | `scripts/test_xeus_map.py` | 415 mapped / 9 drops |
| Corpus QA (a–d) | `scripts/run_corpus_v3.py` | all pass, fp `f8a48918093a` |

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

## Artifact map

Engine & tests: `yiddish_g2p.py`, `data/gold_lexicon.py`,
`scripts/test_g2p*.py`, `scripts/test_rules_doc.py`, `scripts/run_corpus_v3.py`
Docs: `data/g2p_spec_v3.md`, `docs/yiddish_phoneme_set.md`,
`docs/xeus_to_yiddish_map.md`, `data/spec_conflicts.md`, this file
Corpus outputs: `data/phonemized/v3/` (tokens, lines, segments, LOW_CONF/OOV
logs, line-policy report)
Audio: `scripts/xeus_*.py`, `data/xeus_to_yiddish.tsv`,
`data/audio_lexicon/xeus_tags_*.jsonl`, `xeus_votes_*.tsv`
Verification: `data/verification_batch_v4.{csv,md}`,
`data/audio_lexicon/xeus_strong_nonbatch.tsv`
Retrain: `scripts/prepare_retrain_dataset.py`, `scripts/train_phonikud_yi.py`,
`scripts/eval_phonikud_yi.py`, `scripts/point_text.py`, `data/retrain/`,
`models/phonikud_yi_v2_gpu/`

## State & next steps (as of Aug 7, 2026)

- ~80% of running tokens verified-or-rule-solid; LOW_CONF 19.2% against the
  2% phoneme-freeze threshold.
- **Critical path: batch v4 to Chezky** → fold verdicts → LOW_CONF ~11% →
  batch v5 (audio-flagged words ready) → ~6–7% → freeze range in ~3 rounds.
- On v4 return: score phonikud-yi v2's contextual predictions against the
  verdicts (independent generalization test); next lexicon build + retrain.
- Quarantined 5.8% (quoted Hebrew) recoverable via §6 nikud fallback against
  pointed sources — yield, not accuracy; doesn't block the freeze.
- Then: freeze phonemes, generate TTS training data from the 1.67M-token
  span set.
