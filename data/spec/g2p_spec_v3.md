# Hasidic Yiddish G2P — Corpus Pipeline Spec v3 (IMPLEMENTATION)

Status: authoritative. Supersedes the engine's current behavior and, where they conflict, the v2 guide (PDF). Incorporates all four native-verification rounds (Aug 5–6, 2026). Seed lexicon: `data/gold/g2p_gold_v3.csv` (403/500 native-settled). Target: deterministic IPA for the full ivelt corpus (92,613 types / 1.83M tokens).

## 1. Phone inventory & notation (closed set — nothing else may appear in output)

Vowels: `a aː ɛ ə i u ɔ ej aj ɔj oʊ`
Consonants: `b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ`
Marks: `ˈ` (immediately before the stressed vowel) · `ː` (only in aː)

Conventions (locked):
- One stress mark per word; none on monosyllables.
- Unstressed ɛ → ə (applied after stress assignment).
- Syllabic finals: -n -l -m after a consonant get NO epenthetic vowel: `zuɡn`, not `zuɡən`.
- Every output must contain at least one vowel symbol per 3 consonant symbols. Violations must never be emitted — route to the OOV log instead (§10).

## 2. Text normalization (before anything else)

1. Unicode NFC. Strip all nikud/cantillation (U+0591–U+05C7) for the lookup key — but if the token had nikud, retain the pointed form as a side-channel for the LK fallback (§6).
2. Unify geresh/gershayim: ׳→', ״→". Strip surrounding punctuation and quotes (ישראל" → ישראל).
3. Split on makef ־ and hyphen; process parts separately unless the whole string matches the multiword table (§8).
4. Final letters ך ם ן ף ץ → base forms internally (position is still known).
5. Clitic split: leading ס' מ' כ' ר' ה' — detach, map ס'→s(ə), מ'→m(ə), כ'→x, then process the remainder as a word; ר' and ה' route to the abbreviation table (§8). Also split the apostrophe-less clitic when the token starts סא/מא/כא + a known word (סאיז → s+iz).
6. Tokens containing " mid-word → abbreviation table; no rule path.
7. Latin letters/digits → number & foreign-token normalizer (out of scope; log).

## 3. Routing — strict order per token

1. abbreviation table → use it (primary variant; see §9)
2. multiword table → use it
3. gold/full lexicon hit → use it
4. LK detector fires? → LK path §6
5. English-loan list/pattern? → loan path §7
6. else → Germanic rules §5

LK detector: any of ח, כּ, שׂ, תּ, ת; suffix ־ות; word on the known merged-LK list; or the "Hebrew spelling shape" heuristic — fewer than 1 vowel-letter (א ע י ו יי וי) per 3 consonants. Words the detector catches that aren't in the lexicon are the #1 quality risk — see §6 fallback.

## 4. The lexicon is the dialect

Four graphemes are lexically ambiguous — rules alone cannot resolve them (empirically: 81% of naked-rule errors):

| Grapheme | Options | Default (when not in lexicon) | Lexicon carries |
|---|---|---|---|
| א vowel | a / ɔ / u | a + flag LOW_CONF | class 12/13 (u): zugn, yur, vus…; class 41 (ɔ): dort, vokh, kop, got, volt… |
| יי | aj / aː | aj (word-initial יי = ji: ייד jid) | class 34 aː list: maan, tsaat, shraabn, vaal… |
| וי | ɔj / oʊ | ɔj | oʊ list (û-class 54 only): ous, arous, houz, mouz, moul, touznt, lout, ouf, arouf, drousn, krout, shtount, hout, boukh, pouer. **Not** oukh — אויך is ɔjx (44). **טויב** is a homograph: toʊb dove vs tɔjb deaf — keep both; primary tɔjb. Its §9 noun-slot test is wired (bare טויב/טויבן directly after די → toʊb; the adjective inflects there: די טויבע), still no context MODEL. English *ou/ow* cousins are a diagnostic gut-check, not an inference lookup. |
| פ unpointed | f / p | f (after ש always p) | p-list: plaats, pinkt, praaz, pushit, plaan, pin… |

Never guess these from etymology at runtime. Defaults exist only so the pipeline never stalls; every default application on א gets logged for lexicon triage (§10).

## 5. Germanic rule path (default letter table)

Digraphs first, longest match: דזש→ʤ, טש→ʧ, וו→v, יי→aj (initial→ji), וי→ɔj, זש→ʒ, שפ→ʃp.

| Letter | Phone | Notes |
|---|---|---|
| א | a | silent word-initially before י; silent anywhere before a vowel-ו (וואו→vi, אונז→inz); before וו = a (אוועק) |
| ב ג ד ה ז ט כ/ך ל מ נ ס ק ר ת | b ɡ d h z t x l m n s k r s | ה silent only after a vowel and before a consonant or word-end (זעהן→zejn); real [h] at syllable onset (געהאט) |
| ו | i | the native vov-vowel rule |
| י | i | consonantal j word-initially before a vowel letter or between vowels |
| ע | ɛ | reduces to ə when unstressed (§11) |
| פ / ף | f | see §4 p-list |
| צ/ץ | ʦ | |
| ש | ʃ | |

Suffix spellings: ־ליך → ləx (meyglekh), ־יג → iɡ (aybig — no devoicing).

The ־ער system (native-verified, replaces everything earlier):
- Default: ɛr — covers er, der, ver, mer, verter, ersht, and all r+cluster words.
- The ir-list (class-25 stems before r): ʃvir, virn, virt, hirn, lirnen, ɡəhˈirt — lexicon.
- ejr never occurs before r. Away from r, the ey-class keeps ej: zejn, ɡəvˈejn, vejɡ, tejɡ, ɡejbn, bejtn, jejdn, brˈejnɡən, mejɡ, kˈejɡən.
- Raising before m (lexical, not a rule): dejm, ejm (אים/איהם/עם-as-him), nuxdˈejm.

## 6. LK path

1. Merged-LK lexicon first (shabes, tojrə, pajsəx, kojəx, milxˈumə…, all of gold_v3's L-layer).
2. Nikud fallback: fetch pointing from a pointed source (Tashma index / Sefaria / the token's own side-channel nikud), then apply:

| Nikud | Phone | Nikud | Phone |
|---|---|---|---|
| kometz (both) | u | cholam | ɔj |
| pasekh | a | chirik | i |
| tsere | aj | shuruk/kubuts | i |
| segol | ɛ (→ ej in a stressed open syllable: nˈejfiʃ, mˈajləx-type) | shva na | ə or ∅ |

Pasekh+guttural sequences → aː (mˈaːsə, ʃu, ʃˈaːlə). Stress: penult retraction (ʃˈabəs, jisrˈuəl, ʦadˈikim); plural shift stored per pair (bˈuxər → buxˈirim). ־יו → uv; ת without dagesh = s, with dagesh = t (stam, tɔjxˈɛxə).

3. No pointing found → log OOV-LK, emit nothing to the training set for that token (a flagged schwa-filled approximation may go to a quarantine file). Vowel-less consonant strings are forbidden outputs.

## 7. English-loan path

Route by list (tromp, ɡuɡl, kar, imˈejl, biznəs, ofis, akˈaunt, nju, pin, bel, link, kɔst, sɔrt, dɔlər…) — map through the loan adapter, not the Yiddish letter table. Yiddishized loans keep their verified surprises: kɔmpˈajn ("comp-ine"), problˈejm. New unrecognized all-consonant-plausible English shapes → log for triage.

## 8. Abbreviations & multiword table

ר' → rɛb · ה' → haʃˈɛm · שליט"א → ʃlˈitə · זצ"ל → zaʦˈal · ז"ל → zal · זי"ע → zxisˈɔj jˈuɡin ulˈajni · יו"ט → jˈɔntəf · ב"ה → bˈurəx haʃˈɛm · extend as encountered.
Multiword: בית־מדרש / בית מדרש → bis-mˈɛdrəʃ (בית alone = bajs). Compounds get their own entries; reduced forms fire only inside the compound.

A gershayim token the table above misses is resolved in three steps (never the rule path):
7.5a word-pronounced acronyms — established acronyms nobody spells out (רש"י → rˈaʃi, חז"ל → xazˈal, רמב"ם → rˈambam, של"ה → ʃlu, תרי"ג → tarjˈaɡ, …). Tabled, but the readings are editorial, so route=lexicon at LOW confidence.
7.5b letter-name tokens — the token *is* a letter's name (מ"ם → mɛm, כ"ף → xuf, יו"ד → jid, ו"ו → vuv), read as that one name.
7.5 letter-name fallback — everything else is spelled out character by character (תשפ"ה → tuf ʃin paj haj, כ"ה → xuf haj), which is what a reader does with an acronym he does not recognize.

## 9. Homographs & register pairs (context hooks)

Emit the primary unless a disambiguator fires; all variants go to a variants column for forced-alignment voting later.

| Spelling | Primary | Alternate | Disambiguator |
|---|---|---|---|
| שטייט | ʃtajt (stands/says) | ʃtaːt (slowly, =שטאַט) | adverb position (after זייער/adj slot) → ʃtaːt |
| נעמען | nɛmən (take) | nejmən (names) | plural-noun context |
| בעל | baːl (LK) | bɛl (bell) | LK collocations (בעל־...) |
| געוואלט | ɡəvˈɔlt (wanted) | ɡəvˈald (exclamation) | interjection punctuation |
| הלל | halˈɛl (prayer) | hilˈɛl (name) | name context |
| עם | ejm (him) | am (Hebrew) | quoted-Hebrew context |
| אויף / אויפן | oʊf standalone | afn fused/reduced | token = אויפן → afn |
| בית | bajs | bis- | only inside בית־מדרש |
| טויב / טויבן | tɔjb (deaf) | toʊb (dove) | bare form directly after די, no intervening punctuation → toʊb (wired) |

## 10. Postlexical rules (ordered) — the settled voicing policy

1. Voicing-ward assimilation ON: voiceless obstruent → voiced before a voiced obstruent (mazbˈir, mˈɔjzdəs, zɡˈilə-type).
2. Devoicing OFF — everywhere. No final devoicing (kind, ɔjb, hub, iz, ruv, jid) and no devoicing-ward assimilation (zuɡt, ɡəzˈuɡt, ʃraːbt). Lexicalized devoiced forms (jˈankəf as variant, ʃkˈɔjəx) live in the lexicon only.
3. Syllabic n/l/m formation (no vowel inserted); place assimilation is the acoustic model's job — don't encode.
4. Unstressed ɛ → ə.
5. No cross-word sandhi in the phone string. Ever.

## 11. Stress assignment

1. Lexicon stress wins.
2. Monosyllable → no mark.
3. Unstressed prefixes ge- ba- be- far- der- tse- → stress the next nucleus (ɡəzˈuɡt, farvˈus, bakˈimən).
4. Directional a(r)- words → second nucleus (arˈoʊs, arˈup, ahˈajm, avˈɛk).
5. LK → penult retraction unless the lexicon says otherwise (haraxmˈɔn is final-stressed — lexical).
6. Loans → per-word list (prəzidˈɛnt, kɔmpˈajn, iˈran).
7. Everything else → initial.

## 12. Output format & the iteration loop

Per token emit: `word, ipa_primary, variants, layer(G/L/E/A/N/X), route(lexicon/rule/fallback), confidence(HIGH=lexicon | MED=unambiguous rule | LOW=defaulted א/פ or LK-fallback)`.

QA gates every run: (a) zero vowel-less outputs; (b) zero symbols outside §1; (c) the minimal-pair regression set passes; (d) the gold_v3 lexicon reproduces byte-identically.

Iteration loop: run corpus → diff report vs previous run, token-weighted, with per-category top examples → Chezky marks up the docx with rhyme-words → verdicts fold into the lexicon with provenance → LOW_CONF log sorted by frequency becomes the next verification batch. Target: drive the LOW_CONF share of running tokens under 2%, then freeze phonemes and start TTS data generation.
