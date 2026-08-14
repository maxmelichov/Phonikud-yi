# Audio-Supervised Grapheme-to-Phoneme Conversion for Hasidic Yiddish: Overcoming Phonetic Underspecification

**Draft — paper framing of the working system.** Engine: `yiddish_g2p.py` ·
Spec: `data/g2p_spec_v3.md` · Gold benchmark: `g2p_gold_v3 - g2p_gold_v3.csv.csv`

> This draft reuses the R01–R23 rule tables verbatim from
> `docs/yiddish_phoneme_set.md`, which remains the executable specification
> run by `scripts/test_rules_doc.py`. **That file is the source of truth for
> the tables; edit examples there, not here.** Numbers in this draft are the
> measured values as of 2026-08-14, cross-checked against
> `docs/PROJECT_HISTORY.md`.

## Abstract

Grapheme-to-phoneme (G2P) conversion for Hasidic (Unterland) Yiddish is
blocked by severe phonetic underspecification: the orthography leaves several
graphemes open between unrelated phonemes (א ∈ {a, ɔ, u, aː}; unpointed
פ ∈ {f, p}; יי ∈ {aj, aː, ej}; וי ∈ {ɔj, oʊ}), and the language freely embeds
unvocalized Hebrew-Aramaic (loshn-koydesh) items whose written form carries no
vowels at all. We present a G2P framework that emits fully specified IPA with
lexical stress over a closed 11-vowel / 24-consonant inventory. A
deterministic rule engine and a native-speaker gold lexicon form the base;
the open readings are then resolved by weak audio supervision: a universal
phone recognizer pseudo-labels the podcast corpus, character-level alignment
attaches each acoustic vote to a single grapheme slot, and a statistical
calibration filter admits a vote only when it is surprising under the
recognizer's own measured confusion profile *and* the orthography genuinely
leaves the slot open. Unlike pipelines that absorb spoken-norm reductions,
ours explicitly rejects them to preserve canonical citation forms. A
contextual pseudo-vocalization model fine-tuned on the audio-vetted labels
resolves homographs in context, and a hierarchical fallback chain reduces the
out-of-vocabulary quarantine rate from 5.82% to 0.33% of corpus tokens.

## 1. Introduction

TTS training data requires labels that are both fully specified (every vowel,
plus stress) and internally consistent (one word, one citation form). Hasidic
Yiddish orthography provides neither: the same letter sequence פאר writes the
preposition /far/ and the noun /pur/ "pair", and high-frequency function words
leave their vowel entirely to the reader. Two prior lines of work motivate our
design. *Phonikud* (Hebrew TTS) resolves underspecification with a contextual
diacritizer feeding a deterministic G2P; *ReNikud* obtains supervision from
audio via phoneme-ASR pseudo-labeling with character-level alignment. We adopt
both mechanisms but invert ReNikud's goal: where its pseudo-labels capture the
spoken norm, our filters are designed to *reject* fast-speech reduction and
devoicing, because TTS labels must encode what a word is, not what rapid
speech does to it (§4.3).

The pipeline is: bare text → contextual diacritization (`phonikud-yi`) →
deterministic G2P → IPA. Authority is strictly ordered: expert
native-speaker verdicts (the gold lexicon) > audio evidence > published
pointing (Sefaria) > model guesses; a lower tier can never override a higher
one, and audio–gold conflicts are queued for human adjudication rather than
merged.

## 2. The phonetic inventory (closed set)

## 1. The phone inventory (closed set)

Nothing outside this set may ever appear in engine output.
(`scripts/run_corpus_v3.py` gate b enforces this corpus-wide.)

### Vowels

| Phone | Description | Example word | IPA |
|---|---|---|---|
| `a` | open central, short | מאכן | `maxn` |
| `aː` | same quality, long (class 34: flattened *ay*) | היינט | `haːnt` |
| `ɛ` | open-mid front (stressed ע) | קען | `kɛn` |
| `ə` | schwa — every unstressed ɛ | אבער | `ˈɔbər` |
| `i` | close front (also the native ו-vowel) | גוט | `ɡit` |
| `u` | close back (class 12/13 א, LK kometz) | וואס | `vus` |
| `ɔ` | open-mid back (class 41 א) | דארט | `dɔrt` |
| `ej` | class-25 lengthened e | וועג | `vejɡ` |
| `aj` | default יי | צוויי | `ʦvaj` |
| `ɔj` | default וי | שוין | `ʃɔjn` |
| `oʊ` | class-54 וי (lexical list) | הויז | `hoʊz` |

### Consonants

`b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ`

### Marks

- `ˈ` — primary stress, placed **immediately before the stressed vowel**; at most one per word; never on monosyllables.
- `ː` — length, only ever in `aː`.

## 3. Morphophonological rule architecture

The engine routes each token in strict order — abbreviation tables →
multiword expressions → gold lexicon → legacy lexicons → audio-derived
tables → rule path — and attaches to every output a route, a confidence tier
(HIGH/MED/LOW) and a machine-readable reason; the LOW tier *is* the
verification work queue. Layer codes: G Germanic · L loshn-koydesh ·
E English loan · A abbreviation · N name · X other. The rule tables below are
the executable examples (asserted verbatim by `scripts/test_rules_doc.py`).

### R01 — Routing order: lexicon before rules

Gold/lexicon entries win over every rule; 500/500 gold primaries reproduce
byte-identically (`scripts/test_g2p_gold.py`, QA gate d).

| Input | Expected | Note |
|---|---|---|
| `וואס` | `vus` | gold lexicon, class 12/13 |
| `דארט` | `dɔrt` | gold lexicon, class 41 |
| `גאר` | `ɡur` | lexicon beats the rule default (rule would say *ɡar*) |

### R02 — Ambiguous א: lexicon decides a / ɔ / u; default `a` + LOW confidence

| Input | Expected | Note |
|---|---|---|
| `מאל` | `mul` | lexicon: class 12/13 |
| `וואך` | `vɔx` | lexicon: class 41 |
| `בלאט` | `blat` | not in lexicon → default a, logged LOW_CONF |

### R03 — יי: default `aj`; word-initial → `ji`; class-34 lexical list → `aː`

| Input | Expected | Note |
|---|---|---|
| `דריי` | `draj` | default aj |
| `ייד` | `jid` | word-initial יי = ji |
| `היינט` | `haːnt` | class-34 aa-list |
| `ביים` | `bam` | lexical: fused clitic, aː shortened |

### R04 — וי: default `ɔj`; class-54 lexical list → `oʊ`

| Input | Expected | Note |
|---|---|---|
| `שוין` | `ʃɔjn` | default ɔj |
| `אויב` | `ɔjb` | default ɔj, no devoicing |
| `הויז` | `hoʊz` | oʊ-list |
| `לויט` | `loʊt` | oʊ-list |
| `ארויס` | `arˈoʊs` | productive arous-/ous- prefix |

### R05 — Unpointed פ: default `f`; after ש always `p`; p-list lexical

| Input | Expected | Note |
|---|---|---|
| `שפילן` | `ʃpiln` | שפ → ʃp, rule |
| `פונקט` | `pinkt` | p-list |
| `פלאץ` | `plaːʦ` | p-list (+ class-34 aː) |
| `פעקל` | `pɛkl` | audio-pe: PhoneticXeus corpus vote p=23/f=0 |
| `כאפן` | `xapn` | audio-pe: khapn, p=12/f=1 |
| `פֿעקל` | `fɛkl` | a written rafe still outranks the audio table |

### R06 — Native vov-vowel rule: ו → `i`

| Input | Expected | Note |
|---|---|---|
| `גוט` | `ɡit` | |
| `שול` | `ʃil` | rule path |
| `קומט` | `kimt` | |

### R07 — Voicing policy: NO devoicing anywhere; voicing-ward assimilation ON

| Input | Expected | Note |
|---|---|---|
| `איז` | `iz` | no final devoicing |
| `זאגט` | `zuɡt` | no cluster devoicing |
| `טאג` | `tuɡ` | no final devoicing |
| `בריוו` | `briv` | no final devoicing |
| `מסביר` | `mazbˈir` | voiceless→voiced before voiced obstruent |
| `מוסדות` | `mˈɔjzdəs` | voicing-ward assimilation |
| `צוויי` | `ʦvaj` | /v/ is NOT a voicing trigger |

### R08 — Syllabic finals: -n -l -m after a consonant get no epenthetic vowel

| Input | Expected | Note |
|---|---|---|
| `זאגן` | `zuɡn` | not *zuɡən* |
| `מאכן` | `maxn` | |
| `וויסן` | `visn` | |

### R09 — Unstressed ɛ → ə (after stress assignment)

| Input | Expected | Note |
|---|---|---|
| `אבער` | `ˈɔbər` | ער unstressed → ər |
| `עפעס` | `ˈɛpəs` | stressed ɛ kept, unstressed reduced |
| `נעמען` | `nˈɛmən` | homograph primary (take) |

### R10 — Stress: monosyllables unmarked; one mark, before the vowel

Words whose only nucleus is a single written vowel (syllabic finals don't
count) carry no `ˈ`.

| Input | Expected | Note |
|---|---|---|
| `מאכן` | `maxn` | monosyllable by v3 counting |
| `וואך` | `vɔx` | |
| `ווערטער` | `vˈɛrtər` | polysyllable: marked |

### R11 — Unstressed prefixes ge- ba- be- far- der- tse- → stress the next nucleus

| Input | Expected | Note |
|---|---|---|
| `געזאגט` | `ɡəzˈuɡt` | |
| `פארוואס` | `farvˈus` | |
| `באקומען` | `bakˈimən` | |

### R12 — Directional a(r)- words → second nucleus

| Input | Expected | Note |
|---|---|---|
| `ארויס` | `arˈoʊs` | |
| `ארויף` | `arˈoʊf` | |
| `אראפ` | `arˈup` | |
| `אוועק` | `avˈɛk` | |
| `אהיים` | `ahˈajm` | |

### R13 — Loshn-koydesh: penult retraction unless the lexicon says otherwise

| Input | Expected | Note |
|---|---|---|
| `שבת` | `ʃˈabəs` | |
| `מלחמה` | `milxˈumə` | |
| `ישראל` | `jisrˈuəl` | |
| `ניגונים` | `niɡˈinim` | |

### R14 — The ־ער system: default `ɛr`; lexical ir-list; `ej` never before r

| Input | Expected | Note |
|---|---|---|
| `ווער` | `vɛr` | default ɛr |
| `מער` | `mɛr` | default ɛr |
| `שווער` | `ʃvir` | ir-list |
| `הערן` | `hirn` | ir-list |
| `לערנען` | `lˈirnən` | ir-list |
| `וועג` | `vejɡ` | ej away from r |
| `טעג` | `tejɡ` | ej away from r |
| `געבן` | `ɡejbn` | ej away from r |
| `דעם` | `dejm` | lexical raising before m |
| `נאכדעם` | `nuxdˈejm` | lexical raising before m |

### R15 — ה: silent after a vowel before consonant/word-end; [h] at onset; final ה after consonant → ə

| Input | Expected | Note |
|---|---|---|
| `זעהן` | `zejn` | silent ה |
| `געהאט` | `ɡəhˈat` | onset h |
| `שירה` | `ʃˈirə` | feminine -ה → ə |
| `עבודה` | `avˈɔjdə` | feminine -ה → ə |

### R16 — א silent before a vowel-ו; = `a` before וו

| Input | Expected | Note |
|---|---|---|
| `וואו` | `vi` | |
| `אונז` | `inz` | |
| `אוועק` | `avˈɛk` | א before וו = a |

### R17 — Suffix spellings: ־ליך → `ləx`; ־יג → `iɡ` (no devoicing)

| Input | Expected | Note |
|---|---|---|
| `ערליך` | `ˈɛrləx` | rule |
| `הערליך` | `hˈɛrləx` | |
| `נאטירליך` | `natˈirləx` | rule, loan stem |
| `אייביג` | `ˈajbiɡ` | ־יג keeps ɡ |

### R18 — Clitic splits: ס' מ' כ' (and apostrophe-less סא/מא/כא + known word)

| Input | Expected | Note |
|---|---|---|
| `ס'איז` | `siz` | |
| `כ'האב` | `xɔb` | |
| `מ'קען` | `mˈɛkən` | |

### R19 — Abbreviations (mid-word gershayim never take the rule path)

| Input | Expected | Note |
|---|---|---|
| `ר'` | `rɛb` | |
| `ה'` | `haʃˈɛm` | |
| `שליט"א` | `ʃlˈitə` | |
| `זצ"ל` | `zaʦˈal` | |
| `יו"ט` | `jˈɔntəf` | |
| `ב"ה` | `bˈurəx haʃˈɛm` | |

A gershayim token that the abbreviation table does not know is resolved in
three steps, never by the rule path:

1. **word-pronounced acronym** (§7.5a) — an established acronym nobody spells
   out gets its word reading from `_ACRONYM_WORDS`. Editorial reading, so
   `route=lexicon` but `confidence=LOW`.
2. **letter-name token** (§7.5b) — the token *is* a letter's name
   (`_LETTER_NAME_WORDS`), so it reads as that one name.
3. **letter-name fallback** (§7.5) — anything left is spelled out character by
   character, the way a reader copes with an acronym he does not recognize.

| Input | Expected | Note |
|---|---|---|
| `רש"י` | `rˈaʃi` | 1 word-pronounced acronym (1157 occ; never `rajʃ ʃin jid`) |
| `חז"ל` | `xazˈal` | 1 |
| `רמב"ם` | `rˈambam` | 1 |
| `מ"ם` | `mɛm` | 2 the letter mem's name, not `mɛm mɛm` |
| `יו"ד` | `jid` | 2 the letter yud's name |
| `תשפ"ה` | `tuf ʃin paj haj` | 3 unknown gematria year -> spelled out |
| `כ"ה` | `xuf haj` | 3 unrecognized date -> spelled out |

### R20 — Multiword table; compound-internal reduced forms

| Input | Expected | Note |
|---|---|---|
| `בית מדרש` | `bis-mˈɛdrəʃ` | |
| `בית־מדרש` | `bis-mˈɛdrəʃ` | makef form |
| `בית` | `bajs` | bare form keeps bajs |
| `א פאר` | `a pˈur` | article + pair-noun; the fused spelling אפאר is gold apˈur |
| `א פאר יאר` | `a pˈur jur` | the MWE fires inside running text |

### R21 — Homographs: emit the primary; token-level hooks

| Input | Expected | Note |
|---|---|---|
| `שטייט` | `ʃtajt` | primary (stands); ʃtaːt is the variants-column alternate |
| `נעמען` | `nˈɛmən` | primary (take); nˈejmən alternate |
| `בעל` | `baːl` | primary (LK); bɛl alternate |
| `געוואלט` | `ɡəvˈɔlt` | primary (wanted) |
| `אויף` | `oʊf` | standalone |
| `אויפן` | `afn` | fused/reduced token rule |
| `פאר` | `far` | primary (preposition); after the article אַ the MWE reads a pˈur (R20) |
| `פּאָר` | `pur` | the writer's own dagesh outvotes the point-stripped gold key |

### R22 — The rescue chain: no Hebrew word is ever dropped

A would-be-quarantined LK word runs through three rescue tables in strict
priority order — hearing beats books, books beat guesses. Every rescue emits
`route='rule'`, `confidence='LOW'` with its own reason, so rescued words stay
in the verification queue and any native verdict instantly outranks them:

1. `data/audio_endorsed_lk.py` — reason `pointed-audio-endorsed`; the corpus's
   unverified pointing confirmed against episode audio (PhoneticXeus).
2. `data/sefaria_pointed_lk.py` — reason `sefaria-pointed`; a single agreed
   pointing in the verified published editions (Sefaria MAM / Torat Emet).
3. `data/model_pointed_lk.py` — reason `model-pointed-guess`; phonikud-yi v3
   (97% held-out accuracy on evidence-backed Hebrew) pointed the word in
   sentence context. The no-drop policy: a good guess beats silence, and it is
   never the raw consonant skeleton.

Both pointing-based rescues (2 and 3) are REGISTER-AWARE
(`scripts/register_policy.py`). A pointing is read as EMBEDDED loshn-koydesh by
default — `read_pointed_merged()`, shuruk/kubuts take the Yiddish u->i shift and
a final komets-hey is `[ə]` — because that is what a Hebrew word is doing inside
a Yiddish sentence. The Whole-Hebrew register (`read_pointed_wh()`, §7.1) is the
QUOTATION register and is kept only where the evidence says the word is quoted:
episode audio fits it better, or >= 70% of the type's corpus tokens sit inside a
run of >= 3 consecutive loshn-koydesh tokens. The losing register ships as a
variant. 8,664 of the 11,034 entries in the two tables are read in the
merged register, 5,951 of them with a reading the Whole-Hebrew reader would
not have given.

| Input | Expected | Note |
|---|---|---|
| `צדקה` | `ʦdˈukə` | 1 audio-endorsed |
| `חסד` | `xˈɛsəd` | 2 sefaria: `חֶסֶד` — merged-register |
| `זכות` | `zxis` | 2 sefaria: `זְכוּת` — merged-register, shuruk -> i; audio agrees |
| `מחלוקת` | `maxalˈɔjkɛs` | 3 model guess (was quarantined as *mxliks*); WH kept — the merged reader retracts the stress to *maxˈalɔjkəs* |
| `תהילים` | `təhˈilim` | 3 model guess |
| `דבר` | `dˈuvur` | 3 model guess — never the skeleton *dbr* |

Only tokens outside the phone system entirely still quarantine (§2.7 —
digits, Latin script, phone numbers), pending the number normalizer:

| Input | Expected | Note |
|---|---|---|
| `845-554-0338` | `∅` | non-Hebrew: withheld until the number reader exists |

### R23 — Normalization: quotes/geresh unified and stripped; final letters folded

| Input | Expected | Note |
|---|---|---|
| `ישראל"` | `jisrˈuəl` | surrounding quote stripped |
| `ס׳איז` | `siz` | Hebrew geresh = apostrophe |


---

## 4. Weak audio supervision via ASR pseudo-labeling

Native verdicts do not scale (509 words over four elicitation rounds); the
corpus holds ~250 episodes of audio. We therefore derive supervision from a
428-symbol universal phone recognizer (PhoneticXeus), under controls that keep
its errors out of the lexicon.

### 4.1 Phonetic discretization

Universal IPA output is deterministically folded onto the closed inventory
(`scripts/xeus_map.py`; see `docs/xeus_to_yiddish_map.md`). Folding is
conservative: bare monophthongs map to short vowels, so marked classes
(aː, ej, oʊ) can only be selected by strong evidence, never by fold artifacts.

### 4.2 Character-level phonemic alignment

Word-level agreement scores cannot answer "is *this* פ a /p/?". Each
recognizer transcript is Needleman-Wunsch-aligned against the engine's own
reading of the same word, and the aligned heard phone is stored **per slot**
(`tag_positional()`), so a verdict attaches to one grapheme rather than a
whole word. Tags accumulate in a shared, append-only pool
(`data/audio_lexicon/pe_sweep_tags.jsonl`); chunk selection is greedy by
**token** gain with lazy re-scoring — token-weighted selection reaches 73% of
LOW-confidence tokens at ≥3 clips where type-weighted selection reached 7% of
types on the same budget. Current coverage: 2,403 chunks; 81% of LOW tokens
heard at least once.

### 4.3 Three admission filters

The recognizer has an accent. Pooled over every tagged slot
(`data/audio_lexicon/confusion.tsv`):

| engine phone | recognizer reports |
|---|---|
| ʦ | s 62%, t 16%, ʦ only 10% |
| aj | i 50%, ɛ 19%, ə 15% |
| ej | ɛ 50%, ə 36% |
| z | s 44%, z 35% |
| ʃ | ʃ 60%, s 31% |

A raw majority vote would conclude the language has no affricates, diphthongs
or voiced sibilants. A pseudo-label is therefore admitted only if it survives,
in order:

1. **Statistical calibration** (`scripts/audio_calibrate.py`): model the
   confusion profile as the null hypothesis P(heard | engine) and score each
   word-slot's votes with a log-space binomial tail. ʃ→s at the 31% base rate
   is recognizer habit; 9-of-10 clips is surprising. Of 26,560 strongly-voted
   slots, 80% agree with the engine outright; 2,442 disagreements are
   statistically surprising.
2. **Orthographic openness**: surprise is not evidence where the spelling
   decides. האט is heard as *hat* in 1,833/4,424 clips — a maximally
   surprising, completely wrong label: it is unstressed-vowel reduction, not a
   different word. Since ז can only be /z/ and ג only /ɡ/, deviations at
   determined letters are discarded as process, and audio is admitted only at
   the open graphemes of the abstract. This cuts 2,442 candidates to 412; the
   2,030 rejects are exactly the expected process inventory (a→ə, ʃ→s, d→t).
3. **Expert precedence**: 37 of the 412 touch a gold word; they are queued for
   native adjudication with vote counts attached, never folded.

What ships (all at MED confidence, visible for review, and always outranked by
an explicit rafe/dagesh in the text): 77 audio-refuted f-defaults
(`data/audio_pe_lk.py`), 136 vowel corrections toward the recognizer-clean
target /u/ (`data/audio_vowel_lk.py`), 107 audio-endorsed Hebrew pointings.
Contested material goes to queue files (932 vowel slots; 234 SUSPECT Hebrew
readings), not into the engine.

## 5. Context-aware pseudo-vocalization (`phonikud-yi`)

Homographs make static lexicons insufficient, so disambiguation is framed as
pseudo-vocalization: a 306M character-level model predicts Hasidic diacritics
in context, and the deterministic engine reads the pointed result. Training
labels are stamped from the authority chain (1.09M supervised tokens, 61% of
corpus text; loss is masked elsewhere), with two safeguards: the 234
audio-SUSPECT readings are excluded from supervision entirely (verified: 0 of
their 2,057 corpus tokens stamped), and already-supervised labels that
contradict higher-authority evidence are repaired by validated re-pointing
(793 פאר labels), since stamping passes only fill unsupervised tokens.

The current model (v5) scores 99.94% word-level on the held-in validation
distribution and **78.31%** on a 409-word out-of-distribution canonical test
set — the honest number for unseen material — while correctly separating
אַ פּאָר יאָר from פֿאַר דער קהילה within one sentence.

## 6. Evaluation and continuous integration

Every corpus run enforces: (a) zero ill-shaped outputs, (b) zero symbols
outside the closed inventory, (c) the deterministic suites
(`test_g2p.py`, 302-case `test_g2p_spec.py`, 3,210-check
`test_audio_evidence.py`, executable-doc `test_rules_doc.py`), and (d)
byte-identity on all 509 gold primaries. The fallback chain of R22 reduced
token quarantine from 5.82% to 0.33%; 93% of word types now carry one
consistent IPA across the corpus (82% before audio supervision).

## Limitations

Audio evidence cannot overrule a native speaker, cannot distinguish citation
from reduced forms on its own (the orthographic filter does that), and cannot
cover the frequency tail: 82% of word types occur fewer than five times and
will never meet a 3-clip threshold — those remain with the rules, published
pointing, or the contextual model at LOW confidence.
