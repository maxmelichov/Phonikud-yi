# Vowel-label audit of the audio attestation (attest_v9.jsonl)

Analysis only, no models. Script: `scripts/vowel_label_audit.py`; counts in `data/eval/vowel_audit.json`; review queue in `data/eval/vowel_review_queue.tsv`.

Input: 576,017 rule-path occurrences, 88,841 word types (`w`). Witness 1 = run-2 ear (`chosen`, `ear_margin`, graph of §12 capped at 16 candidates ranked by slots changed). Witness 2 = ReNikud-yi int8 (`renikud`, `renikud_margin`, `agree` = whole-word equality with the ear). ReNikud missing on 5,664 rows; ReNikud's reading has a different length from the ear's (its free reading, outside the graph) on 19,721.

**Slot classes** (from `xeus_lattice._OPEN_CLASSES`): alef a/ɔ/u/aː, vov i/u, yy aj/aː/ej, vy ɔj/oʊ, ɛ/ə, pe f/p, final devoicing (b/d/ɡ/v/z word-finally). A class is *open* on a row when the engine reading contains one of its phones (an engine `u` is open in both alef and vov, an engine `aː` in alef and yy). A substitution is attributed to the class that contains both phones; substitutions outside every class (ReNikud's free readings, e.g. `ear i / rn ɔj` 2,388, `ear a / rn i` 5,698) are counted as `other`.

## 1. Per slot class: change rate, agreement, decision rules

Rows where the class is open. *ear≠engine* = the ear changed a phone of this class. *word-agree* = `agree` (whole word). *class-agree* = ReNikud's reading does not differ from the ear's inside this class (it may differ elsewhere). Decision rules are whole-word: **ear2** = ear_margin ≥ 2; **v9** = agree ∧ ear_margin ≥ 0.5; **strict** = agree ∧ ear_margin ≥ 2 ∧ renikud_margin ≥ 1. *cap16* = share of rows where the ear's 16-candidate cap was hit (not every graph combination was scored).

| class | n open | ear≠engine | word-agree | class-agree | ear2 | v9 | strict | cap16 |
|---|---|---|---|---|---|---|---|---|
| alef a/ɔ/u/aː | 263,234 | 26.5% | 58.4% | 81.4% | 15.6% | 47.0% | 13.8% | 46.2% |
| vov i/u | 307,169 | 17.8% | 69.4% | 89.9% | 29.0% | 60.0% | 27.3% | 27.3% |
| yy aj/aː/ej | 87,762 | 12.2% | 75.2% | 99.0% | 43.2% | 69.7% | 40.1% | 28.3% |
| vy ɔj/oʊ | 50,381 | 12.6% | 78.7% | 99.2% | 42.6% | 71.3% | 40.3% | 35.6% |
| ɛ/ə | 252,660 | 27.8% | 67.4% | 87.3% | 20.7% | 55.9% | 19.0% | 29.6% |
| pe f/p | 79,514 | 18.7% | 59.0% | 83.4% | 11.7% | 47.0% | 10.3% | 51.1% |
| final devoicing | 32,385 | 43.5% | 52.8% | 80.6% | 12.2% | 41.2% | 10.4% | 32.4% |
| **all rows** | 576,017 | 37.9% | 70.8% | — | 28.6% | 60.9% | 26.9% | 23.2% |

### 1b. Class-level agreement with ReNikud-yi, binned by ear_margin (n in brackets)

| class | <0.5 | 0.5-1 | 1-2 | 2-4 | >4 |
|---|---|---|---|---|---|
| alef a/ɔ/u/aː | 68.8% (84,271) | 78.6% (60,920) | 89.2% (77,019) | 96.7% (39,695) | 96.5% (1,329) |
| vov i/u | 78.9% (73,684) | 85.3% (58,325) | 93.3% (86,118) | 98.8% (82,144) | 99.5% (6,898) |
| yy aj/aː/ej | 96.6% (13,555) | 98.3% (12,480) | 99.4% (23,840) | 99.9% (29,553) | 99.9% (8,334) |
| vy ɔj/oʊ | 97.8% (8,558) | 98.9% (7,659) | 99.3% (12,706) | 99.8% (12,610) | 99.9% (8,848) |
| ɛ/ə | 76.2% (68,628) | 84.4% (55,654) | 92.1% (76,192) | 97.9% (48,653) | 98.6% (3,533) |
| pe f/p | 74.5% (25,609) | 81.8% (19,677) | 89.0% (24,897) | 96.5% (9,057) | 97.8% (274) |
| final devoicing | 72.2% (11,310) | 78.4% (7,953) | 86.1% (9,155) | 95.9% (3,774) | 99.0% (193) |
| whole-word agree, all rows | 41.5% (136,699) | 58.7% (110,164) | 78.7% (164,179) | 94.7% (140,669) | 97.6% (24,306) |

### 1c. When the ear overrides the engine, whom does ReNikud-yi side with?

Among rows where the ear changed this class. *with ear* = ReNikud matches the ear inside the class; *with engine* = ReNikud matches the engine inside the class; the rest is a third reading. Last column: when the ear *kept* the engine's phone, how often ReNikud agrees.

| class | ear changed | RN with ear | RN with engine | changed at ≥2 | RN with ear at ≥2 | ear kept: RN with ear |
|---|---|---|---|---|---|---|
| alef a/ɔ/u/aː | 69,791 | 44.5% | 48.1% | 3,969 | 74.9% | 94.7% |
| vov i/u | 54,697 | 56.2% | 41.3% | 5,544 | 86.2% | 97.2% |
| yy aj/aː/ej | 10,695 | 94.3% | 5.4% | 2,545 | 99.2% | 99.7% |
| vy ɔj/oʊ | 6,369 | 99.2% | 0.8% | 1,077 | 98.9% | 99.2% |
| ɛ/ə | 70,336 | 72.5% | 25.5% | 7,850 | 91.0% | 93.0% |
| pe f/p | 14,837 | 51.2% | 47.3% | 1,065 | 80.0% | 90.8% |
| final devoicing | 14,077 | 75.7% | 24.3% | 1,340 | 93.7% | 84.4% |

### 1d. Direction of the ear's changes (all / at ≥ 2 nats) and of the ear–ReNikud splits

- **alef a/ɔ/u/aː** — ear vs engine: a→u 46,633 (2,495), u→a 8,291 (360), a→ɔ 6,977 (607), aː→a 2,563 (138), u→ɔ 2,391 (171), aː→u 1,562 (121). Ear vs ReNikud: ear u / rn a 29,553, ear a / rn u 8,956, ear ɔ / rn a 4,979, ear a / rn ɔ 1,932, ear u / rn ɔ 1,527, ear ɔ / rn u 979.
- **vov i/u** — ear vs engine: u→i 31,077 (4,365), i→u 24,428 (1,188). Ear vs ReNikud: ear u / rn i 25,872, ear i / rn u 6,552.
- **yy aj/aː/ej** — ear vs engine: aː→aj 7,872 (1,453), ej→aj 2,193 (1,078), aj→aː 465 (11), aj→ej 157 (8), ej→aː 21 (0), aː→ej 9 (0). Ear vs ReNikud: ear aː / rn aj 404, ear ej / rn aj 175, ear aj / rn ej 125, ear aj / rn aː 115, ear aː / rn ej 2, ear ej / rn aː 1.
- **vy ɔj/oʊ** — ear vs engine: oʊ→ɔj 6,396 (1,079), ɔj→oʊ 27 (1). Ear vs ReNikud: ear oʊ / rn ɔj 262, ear ɔj / rn oʊ 124.
- **ɛ/ə** — ear vs engine: ɛ→ə 64,057 (7,689), ə→ɛ 9,238 (346). Ear vs ReNikud: ear ɛ / rn ə 19,149, ear ə / rn ɛ 13,356.
- **pe f/p** — ear vs engine: f→p 10,766 (898), p→f 4,187 (169). Ear vs ReNikud: ear f / rn p 7,123, ear p / rn f 6,004.
- **final devoicing** — ear vs engine: d→t 9,954 (1,219), v→f 1,322 (37), ɡ→k 1,079 (31), z→s 1,072 (48), b→p 650 (5). Ear vs ReNikud: ear d / rn t 2,511, ear k / rn ɡ 1,061, ear f / rn v 754, ear s / rn z 673, ear p / rn b 585, ear t / rn d 236.
- **outside the graph** (ReNikud's free reading): ear a / rn i 5,698, ear i / rn a 4,309, ear i / rn ɔj 2,388, ear a / rn ə 1,734, ear u / rn ə 1,388, ear a / rn aj 1,301, ear ə / rn i 1,126, ear b / rn v 934.

**The וי slot in numbers.** Engine reads oʊ on 6,592 tokens; the ear keeps oʊ on 250 of them (3.8%), 8 at ≥ 2 nats; ReNikud-yi reads oʊ on 32; both witnesses keep oʊ together on 2. Corpus-wide the ear's output has 276 oʊ against 53,416 ɔj; ReNikud's has 138 oʊ.

## 2. What the run-2 probe (`data/xeus_ft/ear3/probe_att_best.json`) says about each class

Run 2 scoring true reading against the one-slot swap on gold clips (unseen words / unseen episodes): **oʊ 0.47 (29/62) / 0.75 (6/8)**, **ɔj 0.93 / 0.93**, **ə 0.79 / 0.90**, **ɛ 0.91 / 0.85**. No probe exists for alef, vov, yy, pe or devoicing.

- **vy ɔj/oʊ**: the ear is a coin flip when the truth is oʊ and 93% when it is ɔj, so its 6,396 oʊ→ɔj rewrites are what a biased witness would produce whether or not oʊ was there; about half of any true oʊ tokens are being written over. The 99% ReNikud agreement here is *not* independent evidence: ReNikud-yi was trained on the ear's audio labels (§18/§21), and it produces oʊ on 138 tokens in the whole corpus. Treat every oʊ→ɔj decision as unverified; the 6,592 engine-oʊ tokens (1,504 types) are a human task, not a witness task.
- **ɛ/ə**: run 2 is right 79–90% on ə and 85–91% on ɛ per clip, i.e. 10–20% per-slot noise. ReNikud sides with the ear on 72.5% of the ear's 70,336 ɛ/ə changes overall and on 91.0% of the 7,850 changes at ≥ 2 nats; the split is asymmetric (ear ɛ / RN ə 19,149 vs ear ə / RN ɛ 13,356). With two witnesses at ≈85–90% each and correlated through training, the v9 rule (agree ∧ ≥ 0.5) is plausibly ≈95% right on this class, the strict rule higher; the human should see only the disputed high-frequency types (queue rows like אזעלכע, פעלט, ענטפערט).
- **ɔj (as the ɔj/oʊ contrast) and yy/aj**: 0.93 on ɔj and class-agreement 99.0–99.2% on yy and vy means the ear and ReNikud almost never split on these; the question is only the oʊ direction above.
- **alef, vov, pe, devoicing**: unprobed. The corpus itself gives the warning: on the ear's changes ReNikud sides with the engine about as often as with the ear (alef 48.1% vs 44.5%, pe 47.3% vs 51.2%, vov 41.3% vs 56.2%), and the agreement curve only crosses 95% at ear_margin ≥ 2 (alef 96.7%, pe 96.5%, devoicing 95.9%). Below 2 nats these classes are where the two witnesses disagree most, and the 16-candidate cap bites hardest on alef (46.2%) and pe (51.1%) rows, so some of the ear's choices there were made among a truncated list.

## 3. Label yield under each decision rule

Tokens = occurrences decided; types (≥1) = types with at least one decided token; type-level = the §17 retrain8 rule (≥ 5 decided clips and ≥ 85% of them on one reading), with the tokens it would stamp (all occurrences of those types).

| rule | tokens decided | share | types with ≥1 | types with ≥5 | type-level types | tokens covered by type-level |
|---|---|---|---|---|---|---|
| ear ≥ 2 alone | 164,975 | 28.6% | 24,338 | 4,093 | 3,784 | 271,609 |
| agree ∧ ear ≥ 0.5 (v9) | 350,777 | 60.9% | 44,048 | 8,486 | 8,322 | 394,480 |
| agree ∧ ear ≥ 2 ∧ RN ≥ 1 | 154,988 | 26.9% | 20,066 | 3,815 | 3,800 | 295,824 |

## 4. Review queue for Chezky (`data/eval/vowel_review_queue.tsv`)

Types where the ear and ReNikud-yi disagree on a vowel-slot class with ear_margin ≥ 1 and renikud_margin ≥ 1 on ≥ 3 occurrences and ≥ 10% of the type's occurrences: 622 types qualify. 31 of them are already in `chezky_disagreements_top60.tsv` or `pointing_v9_eval_disagreements.tsv` (606 words on those lists) and were dropped; the top 150 by corpus frequency were written. Columns: word, freq (rule-path occurrences), n_disagree_m1, engine (majority engine reading), ear (majority ear reading, share of all occurrences), renikud (same), slot_class (classes in dispute, with counts), example episode/chunk#word-index with that occurrence's ear reading, ReNikud reading and both margins.

Head of the queue: סא (1,095; ear s u 54% vs RN s i 99%), נו (1,005), אזעלכע (1,001; ɛ/ə), טאן (457; ear t u n 68% vs RN t i n 100%), מוזיק (428), זעקס, החיים, נא, מחבר, אחד, הנאה, אלקים, ענטפערט. The first 30 rows cover 8,503 occurrences, all 150 cover 15,769 (2,462 of them qualifying disagreements); disputed classes across the 150: alef 55, vov 48, ɛ/ə 45, yy 2, vy 0.

## 5. Reading

1. **Solid without a human**: yy aj/aː/ej and the ɔj side of וי — the two witnesses agree inside the class on 99% of open rows, 97–98% even below 0.5 nats; the ear's aː→aj (7,872) and ej→aj (2,193) rewrites are the Hasidic system, ReNikud confirms 94% of them.
2. **Solid at ≥ 2 nats, noisy below**: ɛ/ə and vov i/u — class agreement 92–93% at 1–2 nats, 98–99% at ≥ 2; the v9 rule decides 56–60% of these rows, the strict rule 19–27%. Both witnesses share a training lineage, so agreement overstates independence by an unknown amount; a 100-token spot check per class would calibrate it.
3. **Needs the human**: alef a/ɔ/u/aː, pe f/p and final devoicing below 2 nats — ReNikud sides with the engine as often as with the ear on the ear's changes, 46–51% of alef/pe rows were decided among a truncated candidate list, and these classes carry 264k/80k/32k open rows; the queue's alef/vov rows (סא, טאן, החיים, הנאה, דאם, כ'אב) are the cheapest place to learn which witness is right.
4. **Unverifiable by either witness**: oʊ — run 2 is a coin flip on true oʊ, ReNikud learned the ear's prior, and together they erased oʊ from 6,592 engine tokens down to 2 jointly-kept ones; the 99% agreement on וי is one witness counted twice. Hand-label the 1,504 engine-oʊ types (or a frequency-ranked slice) before any oʊ label enters training.
5. **Yield**: ear ≥ 2 alone gives 164,975 tokens / 24,338 types (type-level: 3,784 types, 271,609 tokens); v9 gives 350,777 / 44,048 (8,322 / 394,480); strict gives 154,988 / 20,066 (3,800 / 295,824). v9 more than doubles the token yield over ear-alone for a whole-word agreement that runs 92–99% at ≥ 1 nat on every class except alef/pe/devoicing (86–89%); strict costs 6% of the ear-alone tokens and buys almost nothing the ≥ 2 bar had not already bought (agreement at ≥ 2 is 96–99% everywhere).
