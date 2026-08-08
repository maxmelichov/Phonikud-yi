# Spec vs. audio-verified lexicon — logged conflicts

Spec: *Contemporary Hasidic Yiddish — Phonemization Guide (G2P Spec)*, 13 pp.
Engine: `yiddish_g2p.py`. Suites: `scripts/test_g2p.py`, `scripts/test_g2p_spec.py`.

**Rule applied:** the spec itself says *lexicon beats rule / heuristic* (§3). Where a
`_WORD_LATIN` entry is marked as verified against a native Boro Park/Williamsburg
speaker (2026-08-05) and the spec disagrees, the audio-verified entry is KEPT and the
divergence is recorded here. Where the spec adds behavior the entry does not cover
(e.g. postlexical devoicing), the spec is followed on top of the entry.

Each conflict is also pinned as an `XFAIL` case in `scripts/test_g2p_spec.py`, so it
stays visible and would flip to `FIXED` if the lexicon is ever revised.

## A. Class 25 [ej] — "e before r" words the reviewer pinned to [i] or [a]

The reviewer's note (`yiddish_g2p.py`, "e before ר, Satmar/Hungarian"): *"e before r
has split into three different sounds depending on the specific word and the speaker's
family background — ee (shveer, veert, heern), a (barg, vark), ay (zayer). Don't make
it a global rule."* This is why class 25 was implemented as a **lexicon block only**,
with no `ער` heuristic (spec §3 offers one; it is deliberately not encoded).

| # | Word | Spec says | Lexicon says (kept) | Engine output |
|---|------|-----------|---------------------|---------------|
| 1 | שווער | `shveyr` [ʃvejr] (25) | `shvir` | `ʃvir` |
| 2 | הערן | `heyrn` [hejrən] (25) | `hirn` | `hirən` |
| 3 | געהערט / דערהערט | `geheyrt / derheyrt` (25) | `gehirt / derhirt` | `ɡəhˈirt / dərhˈirt` |
| 4 | ווערט / ווערן | `veyrt / veyrn` (25) | `virt / virn` | `virt / virən` |
| 5 | ערד | `eyrd` (25) → [ejrt] | `ird` | `irt` |
| 6 | בערג | `berg` (21) → [bɛrk] | `barg` (class 11) | `bark` |
| 7 | ווערק | `verk` (21) | `vark` (class 11) | `vark` |
| 8 | comparative of שווער | `shveyrer` (§9: 25-stems keep [ej]) | follows #1 → `shvirer` | `ʃvˈirər` |

Note items 5, 6 now also carry §4.1 final devoicing (`ird`→[irt], `barg`→[bark]); that
part of the spec IS applied, only the vowel is lexicon-pinned.

## B. Other divergences (not audio conflicts — scope / frequency decisions)

| Word / feature | Spec says | Engine does | Why |
|---|---|---|---|
| מעשה | (silent) | `maase` [maasə], not `mayse` | Audio-verified; spec does not contradict, retained. |
| אן | `un` [un] "without" (12/13, §8) | `an` | Same unpointed spelling as the indefinite article *an*, which is far more frequent in running text. Homograph, lexically unresolvable; the article reading ships. |
| ער (stressed) | `eyr` [ejr] (25, §8) | `ɛr` | `ער` is overwhelmingly the unstressed clitic pronoun; it stays in `_CLITICS`. XFAIL in the spec suite. |
| ־נג / ־נק | [ŋ(k)], §4.7 | `nɡ` / `nk` (`laŋ` → `lank`) | `ŋ` deliberately out of scope for this pass, along with nasal place assimilation (`[ŋ̩] [m̩]`), dark `ɫ` and r-variant coloring — one `l`, one `r`, and the engine's existing `ə`+nasal convention for syllabic nasals. |
| ווייס | `vays` (24) / `vaas` (34) | `vaɪs` | Identical spelling, lexical only; the spec agrees it is unresolvable. The verb reading ships (§11.3). |
| מערן, מלך, ווייס | two readings each | one reading | §7.3/§7.4 need the layer router (§13), which is out of scope. |
| רוּחַ, כֹּחַ | pasekh genuvah inserts `@` before the guttural | `rixa`, `kɔɪxa` | Not implemented on the *pointed* path; the unpointed lexicon forms (`koyekh`, `moyekh`) are already correct. XFAIL. |
| Cheshvn | derived by §4.2 (`sh`→`zh` before `v`) | stored as `khezhvn` in `_WORD_LATIN` | A voicing `/v/` would also give `*dzvay` for צוויי and `*midzve` for מצווה. `/v/` is a target of assimilation but not a trigger; the one spec example needing it is a frozen month name, so it is lexical. |

## B2. Audit pass 2026-08-06 — where the engine deviates from an auditor's
## expected string (the defect itself was fixed in every case)

| Word | Auditor expected | Engine now | Why the difference |
|---|---|---|---|
| אוונט | `[uvənt]` | `[uvnt]` | Syllabic nasals are `ə`+nasal **word-finally only** (`latin_to_ipa`, the `\b`-anchored rule that must not be widened — see its comment). Matches the pre-existing `אווענט` entry. |
| שטערנס | `[ʃtejrəns]` | `[ʃtejrns]` | Same rule; the `n` is not word-final. The finding's real content — class 25 `[ej]` across the paradigm — is fixed. |
| אוועקגעלייגט | `[ˈavɛkɡəlaːkt]` | `[ˈavəkɡəlaɪkt]` | Two engine-wide conventions, neither in scope: unstressed `ɛ` reduces to `ə` everywhere (`reduce_unstressed`), and לייגן is class 22/24 `[aɪ]` here — the auditor's own romanization says *laygt* (= spec ay = `[aɪ]`), so the `[aː]` in their IPA looks like a slip. |
| אפגעטון | `[ˈɔpɡəton]` | `[ˈɔpɡətin]` | The óp- prefix **is** fixed (was `[ˈavɡətin]`). The stem vowel stays `i`: it is spelled with ו, and spec §3's one "genuinely clean rule" is that ו in a native word is always `i`. |
| חסיד | `[ˈxusid]` | `[ˈxusit]` | §4.1 final devoicing, which the spec applies to names too (*Yankev* → **Yankef**); same treatment as `דָּוִד` → `duvit` in the suite. |
| קודש | `[kidʃ]` | `[kitʃ]` | The auditor's expectation was "as at HEAD". The deleted phoneme — the real bug — is restored; the `d`→`t` is §4.2 regressive devoicing before `ʃ`, which HEAD only avoided because the affricate fuser had already eaten the cluster. |

**§4.2 narrowed: only fricatives/affricates voice regressively.** Every voicing
example in the spec has a fricative target (*shabesdik*→*shabezdik*,
*Cheshvn*→*Chezhvn*, the aroys- seam); a voiced **plosive** target appears
nowhere, and assuming one destroyed the separable prefixes at the compound seam
(óp-getun → `[ɔbɡə-]`, avék-gelaygt → `[avɛɡɡə-]`, tsurík-gekimen → `[ʦiriɡɡə-]`).
Devoicing is unrestricted. Cost: `באוואוסטזיין` is now `[bavistzaɪn]` rather than
`[bavizdzaɪn]` — the `t` no longer voices before `z`.

**Degemination added.** Assimilation regularly produces a doubled phone at a
morpheme seam (*ge-red-t* → `[ɡərɛtt]`); Yiddish has no geminates and a repeated
phone is not a legal TTS sequence, so identical adjacent obstruents collapse.

**`_restates_point` refinement reverted, measured.** Requiring a consonant's
point and a following pointed א/ע to *agree* before suppressing the consonant's
vowel is correct for Whole-Hebrew (it is what makes יִשְׂרָאֵל *Yisruel* rather
than *isreyl*) and wrong for the pointing this engine actually consumes: the
Hasidic nikud model writes פָּאַר, דָאֹס, אַזָאַ, אָמָאַל, מוֹצָאֵי with two
*different* marks for one vowel, and the strict rule inserted a spurious vowel in
15 of 468 corpus rows (`par` → `*puar`). ישראל is lexicalised instead.

**`בעל-הבית` stress override changed** from `ba-le-BUS` to `BA-le-bus`. The old
value was not audio-verified and contradicted spec §6.2, which lists *BAlebus*
beside *SHAbes*, *TOYre*, *CHAsene*. Not a lexicon-vs-spec conflict — just wrong.

**`ארויסגיין` stress moved** to `arˈoʊzɡaɪn`. §6.1 writes *aróusgayn*, and the
bare word was already `arˈoʊs`; separable-prefix stress is now a per-prefix
nucleus index rather than a blanket 0.

## C. Cases corrected in the test files

- `scripts/test_g2p.py`: `זָאגְט` `zuɡt` → `zukt` (annotated `# spec:`) — §4.2.
- `scripts/test_g2p_spec.py` `קומען`: `kimən` → `kimɛn`. Segmental cases run at
  `stress=False`, where the suite's own convention keeps unstressed `ɛ`
  (cf. `לערנען` → `lejrnɛn` in the same block); `[ə]` there is a `stress=True` result.
- `scripts/test_g2p_spec.py` `דָּוִד`: `duvid` → `duvit`. §4.1 devoicing applies to
  names — the spec's own example is *Yankev* → **Yankef**.
- `scripts/test_g2p_spec.py` `ארויסגיין`: `ˈaroʊsɡaɪn` → `ˈaroʊzɡaɪn`. §4.2 applies at
  the prefix seam, the same juncture type as *shabes*+*dik* → *shabezdik*.
