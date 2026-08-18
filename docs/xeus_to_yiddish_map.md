# PhoneticXeus → Yiddish phoneme conversion

Maps the 428-symbol universal-IPA vocabulary of PhoneticXeus
(`changelinglab/PhoneticXeus`) onto the closed Yiddish v3 inventory
(`docs/yiddish_phoneme_set.md` §1), so recognizer output and G2P output are
directly comparable phone-for-phone.

- Code: `scripts/xeus_map.py` (`fold_phone_string`, `map_transcript`)
- Full generated table: `data/spec/xeus_to_yiddish.tsv` (415 mapped, 9 dropped)
- Coverage guard: `scripts/test_xeus_map.py` — every vocab symbol must map
  into the inventory or be a documented drop; nothing may map outside it.
- Consumers: `scripts/xeus_tag.py` (word tagging), `scripts/xeus_report.py`

## Design rules

1. **Many-to-one, conservative.** A monophthong [e] folds to `ɛ` (not `ej`) and
   [o] to `ɔ` (not `oʊ`); only clear diphthongs ([eɪ], [oʊ], [aɪ], [ɔɪ], [aʊ])
   vote for the diphthong classes. This biases audio votes *against* the marked
   classes, so a diphthong vote is strong evidence.
2. **Length is dropped except `aː`** — the only length contrast the v3 set has
   (class 34). [iː uː ɛː ɔː] fold to their short phones; [eː]→`ej`, [oː]→`oʊ`
   (quality, not length, is the cue there).
3. **Diacritics and stress are stripped.** CTC stress placement is unreliable;
   secondary articulation (ʰ ʲ ʷ nasalization…) folds into the base phone.
4. **Exotics fold to the nearest Yiddish phone** rather than vanishing, so a
   stray recognizer frame still casts a plausible vote: implosives → plain
   stops, ɸ→f, ħ/ɧ→x, ɥ→j, ɴ→ŋ, lateral fricatives→l.
5. **Deliberate drops (9):** ʔ (+ʔʲ), ʕ, and the click series — no Yiddish
   correlate; dropping beats a wrong vote.

## Notable folds

| Universal | Yiddish | Rationale |
|---|---|---|
| ɪ, y, ɨ, ʏ | i | no length/tenseness contrast in class 31/32 |
| ʊ, ɯ, ʉ | u | |
| e, ø, œ | ɛ | conservative: only [eɪ]/[eː] vote `ej` |
| o, ɒ | ɔ | conservative: only [oʊ]/[oː]/[aʊ] vote `oʊ` |
| ɐ, ʌ, ɑ, æ, ä | a | |
| ɜ, ɘ, ɤ, ɵ, ɞ | ə | |
| aɪ/ai/ɑɪ/ʌɪ | aj | |
| eɪ/ei/ɛɪ/eː | ej | |
| ɔɪ/ɔi/oɪ/oi | ɔj | |
| oʊ/ou/aʊ/au/əʊ/oː | oʊ | |
| aː/ɑː | aː | the one kept length mark |
| ɹ ɾ ʀ ʁ ɻ ɽ | r | all rhotics are one phone (§10.3: realization is the acoustic model's job) |
| w ʋ ʍ ɰ | v | |
| χ ç ɣ ħ ɧ | x | |
| ʂ ɕ | ʃ · ʐ ʑ → ʒ | |
| θ → s · ð → z | | closest sibilant |
| ts→ʦ tʃ/tɕ/ʈʂ→ʧ dʒ/dʑ→ʤ | | dz → z (no ʣ in the v3 set) |
| ɲ ɳ → n · ɱ → m · ɴ → ŋ | | place folding |
| c→k ɟ→ɡ q→k ɢ→ɡ · implosives ɓ ɗ ʄ ɠ → b d ʤ ɡ | | |

## Caveats when reading audio votes

- The recognizer systematically weakens sibilant voicing and affricates
  (ʃ→s, ʦ→s, z→s) — discount those confusions; they are model noise, not
  dialect evidence (measured on episode 100313: see `scripts/xeus_report.py`).
- Unstressed function words genuinely reduce in running speech (mən→mə,
  ix→əx); the G2P emits citation forms by design (§10.5 no sandhi), so those
  mismatches are expected and are not G2P errors.
- Single-clip votes are weak; aggregate ≥3 occurrences per word before folding
  anything into the lexicon (same consensus rule as
  `data/audio_lexicon/lexicon.jsonl`).
