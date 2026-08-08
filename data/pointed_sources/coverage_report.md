# Pointed-source coverage of the v3 quarantine

Source: Sefaria verified pointed texts (Tanakh = *Miqra according to the
Masorah*, CC-BY-SA; Mishnah + Siddur Ashkenaz = *Torat Emet 357*, Public
Domain). Index: `data/pointed_sources/pointed_index.jsonl`.

Quarantine: **14,732 types / 100,827 tokens** (`data/pointed_sources/quarantine_full_snapshot.tsv`, produced by `run_corpus_v3.py --limit 0`).

## Coverage (phonemic fold)

Candidate pointings are compared after `phonemic_fold()`, which collapses
differences that do not change the Ashkenazi reading (gemination dagesh
outside bet/kaf/pe/tav, qamats qatan, the holam dot over a male vav) and
which the two editions write inconsistently. Cantillation is already
stripped at index time.

| bucket | types | % types | tokens | % tokens |
|---|---:|---:|---:|---:|
| (a) exactly one pointing | 2,901 |  19.7% | 26,092 |  25.9% |
| (b) dominant pointing (>=80%) | 758 |   5.1% | 18,135 |  18.0% |
| (c) multiple conflicting pointings | 1,250 |   8.5% | 18,348 |  18.2% |
| (d) no hit in index | 9,823 |  66.7% | 38,252 |  37.9% |

### Same buckets, comparing pointings exactly as printed

The gap between the two tables is pure edition convention, i.e. conflicts
that need no linguistic decision.

| bucket | types | % types | tokens | % tokens |
|---|---:|---:|---:|---:|
| (a) exactly one pointing | 2,443 |  16.6% | 20,301 |  20.1% |
| (b) dominant pointing (>=80%) | 785 |   5.3% | 15,860 |  15.7% |
| (c) multiple conflicting pointings | 1,681 |  11.4% | 26,414 |  26.2% |
| (d) no hit in index | 9,823 |  66.7% | 38,252 |  37.9% |

**Any hit:** 4,909 types (33.3%) / 62,575 tokens (62.1%).

**Directly usable (a + b):** 3,659 types (24.8%) / 44,227 tokens (43.9%).

## Top 40 unresolved conflicts (bucket c), by quarantine token frequency

Genuine ambiguity: these need a homograph decision (usually context), not
just a source. Alternatives shown folded.

| word | freq | pointings (count) |
|---|---:|---|
| מה | 531 | מַה (599), מָה (254), מֶה (85) |
| חתם | 527 | חֹתָם (3), חָתֻם (1) |
| לך | 451 | לְךָ (752), לָךְ (544), לֵךְ (83), לֶךְ (8) |
| מצות | 446 | מִצְוַת (67), מִצְות (56), מַצות (30) |
| זכר | 309 | זָכָר (103), זֵכֶר (30), זְכֹר (25), זָכַר (13) |
| חפץ | 307 | חָפֵץ (49), חֵפֶץ (18) |
| קרבן | 301 | קָרְבַּן (68), קָרְבָּן (65), קֻרְבַּן (1) |
| שכר | 297 | שָׂכָר (50), שְׂכַר (19), שֵׁכָר (13), שָׂכַר (5) |
| משנה | 289 | מִשְׁנֶה (13), מִשְׁנֵה (4), מְשַׁנֶה (4), מִשְׁנָה (4) |
| נס | 286 | נֵס (25), נָס (17), נַס (1), נֹס (1) |
| מחיה | 224 | מִחְיָה (24), מְחַיֵה (20), מְחַיֶה (17) |
| בר | 222 | בַר (43), בָר (11), בֹר (1) |
| חי | 220 | חַי (169), חָי (39), חֵי (12) |
| הלכות | 198 | הֲלָכות (6), הִלְכות (2), הֹלְכות (1) |
| שמות | 189 | שֵׁמות (31), שְׁמות (29), שַׁמות (3) |
| חזון | 185 | חָזון (15), חֲזון (8) |
| מסכת | 174 | מִסֻכֹּת (2), מַסֶכֶת (2), מַסֵכַת (1) |
| לשם | 173 | לְשֵׁם (132), לְשָׁם (30), לֶשֶׁם (4), לַשֵׁם (4) |
| חיות | 172 | חַיות (10), חֵיוַת (7), חָיות (1) |
| תולדות | 169 | תולָדות (3), תולְדות (2) |
| דבר | 165 | דָבָר (361), דְבַר (296), דִבֶּר (217), דַבֵּר (76) |
| שליח | 154 | שְׁלִיחַ (5), שָׁלִיחַ (4) |
| חס | 148 | חָס (2), חַס (2) |
| מתן | 142 | מַתַּן (11), מַתָּן (6), מֻתָן (1) |
| חזק | 137 | חֲזַק (24), חָזָק (20), חַזֵק (7), חָזַק (5) |
| ישמח | 135 | יִשְׂמַח (27), יְשַׂמַח (8), יִשמַח (7), יִשְׂמָח (4) |
| קל | 134 | קַל (33), קָל (9) |
| מדבר | 133 | מְדַבֵּר (34), מִדְבָּר (26), מִדְבַּר (24), מִדַבֵּר (19) |
| שמחת | 133 | שִׂמְחַת (10), שִׂמַחְתָּ (2), שמַחְתָּ (1) |
| חלה | 131 | חַלָה (27), חָלָה (16), חֹלֶה (7), חִלָה (2) |
| מכל | 121 | מִכָּל (309), מִכֹּל (114), מִכּל (1) |
| נשמות | 120 | נְשַׁמות (4), נְשָׁמות (2) |
| שלח | 114 | שָׁלַח (83), שְׁלַח (25), שַׁלַח (16), שֹׁלֵחַ (16) |
| חטא | 113 | חֵטְא (44), חָטָא (38), חֹטֶא (1), חֲטֹא (1) |
| קרח | 113 | קֹרַח (37), קָרֵחַ (14), קֵרֵחַ (7), קָרַח (2) |
| מצה | 111 | מַצָה (16), מִצָה (8) |
| מחשבות | 110 | מַחְשְׁבות (26), מַחֲשָׁבות (10) |
| חיי | 100 | חַיַי (17), חַיֵי (16), חַיָי (11), חֱיִי (5) |
| בהמות | 99 | בְהֵמות (11), בַהֲמות (5) |
| תם | 98 | תָם (30), תֹם (13), תַם (4) |

## What the misses are (bucket d)

| kind | types | tokens | note |
|---|---:|---:|---|
| no Hebrew letter (Latin, digits, phone numbers) | 909 | 4,479 | out of scope for a Hebrew source |
| abbreviation (geresh / gershayim) | 1,790 | 8,983 | needs an expansion table, not pointing |
| Hebrew word, hit after stripping a clitic prefix | 1,237 | 4,188 | reachable by prefix-aware lookup |
| Hebrew word, genuinely absent | 5,887 | 20,602 | mostly Talmudic/Aramaic and modern coinages |

Total bucket (d): 9,823 types / 38,252 tokens.

Prefix-aware lookup (strip one or two of ו/ב/כ/ל/מ/ש/ה and retry) is the single biggest available gain and is left to the consumer, since the prefix must be re-vocalized by the engine rather than read off the source.

## Top 40 misses (bucket d), by quarantine token frequency

| word | freq |
|---|---:|
| רש"י | 1,157 |
| מחלוקת | 341 |
| תיבות | 317 |
| חומש | 310 |
| קורח | 269 |
| 845-554-0338 | 260 |
| doch | 246 |
| חז"ל | 239 |
| רמב"ם | 235 |
| קשיות | 210 |
| השפעות | 202 |
| תוספות | 180 |
| חילוק | 174 |
| תפילין | 169 |
| קדושת | 155 |
| מדרגה | 153 |
| שייכות | 147 |
| חיד"א | 145 |
| תירוץ | 145 |
| חיזוק | 142 |
| ל"ג | 142 |
| ד' | 130 |
| תיקון | 130 |
| אייבערשטנס | 126 |
| חוזק | 126 |
| זלמן | 122 |
| מרמז | 120 |
| שאלות | 115 |
| תהילים | 112 |
| תפילות | 109 |
| 5 | 108 |
| חידושים | 108 |
| 40 | 107 |
| ע"ה | 104 |
| 50 | 101 |
| 20 | 100 |
| גשמיות | 100 |
| רמב"ן | 100 |
| טענות | 95 |
| מוח | 94 |
