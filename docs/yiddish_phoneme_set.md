# Hasidic Yiddish Phoneme Set & G2P Rules (v3)

Engine: `yiddish_g2p.py` · Spec: `data/g2p_spec_v3.md` · Gold: `g2p_gold_v3 - g2p_gold_v3.csv.csv`

**This document is executable.** Every example row below is run against the live
engine by `scripts/test_rules_doc.py`, which parses the `### R…` sections and
asserts `hebrew_to_ipa(input, stress=True) == expected` (`∅` = empty output,
i.e. the token is quarantined). A rule with no passing example fails the build.

---

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

---

## 2. Rules

Each rule has an ID, the behavior, and verified examples. Layer codes:
G Germanic · L loshn-koydesh · E English loan · A abbreviation · N name · X other.

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

## 3. QA gates (every corpus run — `scripts/run_corpus_v3.py`)

| Gate | Meaning |
|---|---|
| (a) | zero vowel-less / ill-shaped outputs emitted |
| (b) | zero symbols outside §1 |
| (c) | `test_g2p.py` + `test_g2p_spec.py` pass |
| (d) | gold_v3 reproduces byte-identically (`test_g2p_gold.py`) |

Per-token record: `word, ipa_primary, variants, layer, route, confidence`
(HIGH=lexicon, MED=unambiguous rule, LOW=defaulted א/פ or LK-fallback).
