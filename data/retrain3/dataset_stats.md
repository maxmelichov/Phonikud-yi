# Retrain dataset v3 — chain-attested tier stamped in

Merger over v2 (`data/retrain2/`): tokens v2 left unsupervised are
stamped with a corpus-attested pointing of their type whenever the
frozen engine vouches for the type's reading (route `lexicon`,
confidence HIGH/MED) and the pointing reads back as an allowed
reading (ipa_primary + listed/gold variants) under the convention
read-back (`reconcile`, unrepaired). One target per type: the most
explicitly pointed survivor, ties by attested frequency, then
codepoint order. `test.jsonl` is a byte-for-byte copy of v2's
(asserted).

## Headline

| metric | count |
| --- | ---: |
| tokens newly supervised (chain-attested) | 189791 |
| rows changed | 21389 |
| types stamped | 391 |

## Coverage (train+val, all tokens)

| | supervised | total | share |
| --- | ---: | ---: | ---: |
| v2 (before) | 1013167 | 1827568 | 55.44% |
| v3 (after) | 1202958 | 1827568 | 65.82% |

## Types skipped (counted, never guessed)

| reason | types |
| --- | ---: |
| `route-not-lexicon-high-med` | 75157 |
| `homograph-type` | 174 |
| `suspect-key` | 178 |
| `no-attested-pointing-survives` | 209 |
| `no-allowed-reading` | 0 |
| letters-misfit (types with >=1 misfit occurrence) | 0 |

Occurrence-level letter misfits skipped: 0 tokens.

## Top 20 stamped types by token count

| type (lexicon key) | target | tokens stamped | unsupervised before |
| --- | --- | ---: | ---: |
| האט | הָאָט | 43203 | 43203 |
| דעמ | דֶעם | 19622 | 19622 |
| אימ | אִים | 11631 | 11631 |
| אויפ | אַויף | 6817 | 6817 |
| האבנ | האָבְן | 6238 | 6238 |
| אנ | אַן | 6221 | 6221 |
| רבי | רֶבִּי | 5909 | 5909 |
| רב | רָב | 5145 | 5145 |
| אזא | אַזאַ | 4811 | 4811 |
| האב | האָבֿ | 4532 | 4532 |
| קיינ | קֵיין | 3983 | 3983 |
| געזענ | גֶעזען | 2964 | 2964 |
| אויכ | אוֹיךְ | 2768 | 2768 |
| איינער | אֵיינֶער | 2486 | 2486 |
| זעט | זעט | 2256 | 2256 |
| געבנ | גֶעבְנְ | 1877 | 1877 |
| אויפנ | אויפְֿן | 1695 | 1695 |
| וועגנ | ווֶעגְן | 1589 | 1589 |
| זענ | זען | 1556 | 1556 |
| פרשת | פַּרְשַׁת | 1542 | 1542 |
