# Eval: `models/phonikud_yi_v8/best`

Test set: `data/retrain2/test.jsonl` -- 409 rows (episode 100313, held out of train and val).
Inference 10.1s. Letter-unsafe rows: 0. Token-misaligned rows: 0.

## 1. Character accuracy (supervised positions only)

| metric | value |
|---|---|
| `supervised_chars` | 12058 |
| `char_acc` | 75.58 |
| `marked_char_acc` | 74.74 |
| `dagesh_acc` | 96.89 |
| `rafe_acc` | 98.9 |
| `shin_acc` | 99.1 |
| `ceiling_char_acc` | 91.04 |
| `char_headroom_to_ceiling` | 15.46 |

**Ceiling 91.04%** -- see the section below. This model is at 83.02% of it.

## 2. Word-level exact pointing

48.28% of 2,997 supervised words get *every* Hebrew character exactly right.

Ceiling is 77.04%, so this is 62.67% of what is reachable (28.76 points of headroom).

## 3. Downstream (frozen `yiddish_g2p`)

| comparison | tokens | agreement |
|---|---|---|
| `vs_engine_verified` | 3,209 | 99.94% |
| `vs_gold_all` | 4,845 | 90.94% |
| `vs_gold_fallback` | 20 | 85.0% |
| `vs_gold_lexicon` | 3,184 | 99.94% |
| `vs_gold_rule` | 1,641 | 73.55% |

`vs_engine_verified` = predicted pointing vs the engine's own reading on lexicon-route tokens; the gold lexicon is keyed unpointed, so <100% means the pointing broke letters or token boundaries.

## Per-vowel accuracy

| gold vowel | n | acc |
|---|---|---|
| NO_MARK | 7,739 | 78.41% |
| pasekh | 1,041 | 80.31% |
| komets | 934 | 63.38% |
| sheva | 758 | 79.95% |
| segol | 695 | 86.62% |
| khirik | 537 | 93.11% |
| holam | 190 | 70.0% |
| tsere | 122 | 80.33% |
| hataf-pasekh | 37 | 70.27% |
| hataf-segol | 5 | 100.0% |

## Top vowel confusions

| gold | predicted | n |
|---|---|---|
| NO_MARK | sheva | 545 |
| NO_MARK | segol | 363 |
| NO_MARK | komets | 265 |
| NO_MARK | pasekh | 259 |
| komets | NO_MARK | 237 |
| NO_MARK | khirik | 149 |
| pasekh | NO_MARK | 136 |
| sheva | NO_MARK | 123 |
| komets | pasekh | 86 |
| segol | NO_MARK | 71 |
| holam | NO_MARK | 54 |
| NO_MARK | holam | 44 |

## The ceiling on this test set

The retrain targets and this gold are two different pointings of
the same convention family, so the metric is capped below 100%.
Feeding every gold-supervised test token through the pipeline's own
stamper (`prepare_retrain_dataset.Builder.token`) -- i.e. exactly
what a model that fits the training objective perfectly emits --
and scoring it with the metric above gives:

| quantity | value |
|---|---|
| gold-supervised test tokens | 2,997 |
| stamper agrees with gold | 915 |
| stamper disagrees with gold | 688 |
| stamper does not supervise (gold credited) | 1,394 |
| **ceiling char_acc** | **91.04%** |
| **ceiling word-exact** | **77.04%** |

8.96% of supervised characters are therefore unreachable by construction. Read every delta on this
test set against the ceiling, not against 100%: the retrain can move
char accuracy by at most 15.46 points and word-exact
by at most 28.76 from here.


## Samples

- gold: בְּשֵׁם ה' נַעֲשֶׂה וְנַצְלִיחַ.
  pred: בְּשֵׁם הּ' נַעֲשֶׂה וְנַצְלִיחַ.
- gold: בָּרוּךְ ה' מֶען האַלט שוין דאָ פּרשַׁת מִשְׁפָּטִים תְּשַׁפ"ד לפ"ק.
  pred: בָּרוּךְ הּ' מֶען הַאַלְט שׁוֹין דאָ פַּרְשַׁת מִשְׁפָּטִים תַּשְׁפָּ"ד לפְ"ק.
- gold: הַיינְט קֶען מֶען זאָגן בסייעתא דִּשְׁמַיָּא מיט דֶעם שֶׁעפֶּערְס הִילְף אַז מֶען האַלט שוין אַןעהאַלטן צייט דִּינְסְטָאָג בַּיינַאכט לְכָבוֹד שַׁבַּת רֹאשׁ חוֹדֶשׁ אֲדָר.
  pred: הַייַנְט קֶען מֶען זאָגְן בִּסיַיְעָתָא דִּשְׁמַיּאָ מִיט דֶעם שֶׁעפֶּֿערְס הִילְף אַז מֶען הַאַלְט שׁוֹין אַןֶעהאַלְטְן צַייַט דִינְסְטָאָג בַּייַנַאַכְט לִכְבוֹד שַׁבָּת רֹאשׁ חוֹדֶשׁ אדָר.

---

## Reading these numbers

* **Never read char/word accuracy against 100%.** The ceiling above is
  the score of the pipeline's own stamper on this gold; the difference
  is a disagreement between two pointings, not something training can
  learn away.
* **`vs_engine_verified` is a letter-safety gate, not a quality metric.**
  `yiddish_g2p.lexicon_key()` strips nikud, so on lexicon-route tokens the
  engine's reading is pointing-independent by construction; anything below
  100% means the model corrupted letters or word boundaries.
* **The signal to move is `vs_gold_rule` and `vs_gold_fallback`.** Those
  are the tokens where the pointing actually drives the phonemization,
  i.e. the OOV words the runtime dictionary cannot cover. They are also
  not capped by the convention gap the way the char metric is.

## Model selection

Val (`data/retrain/val.jsonl`) is near-saturated before a single gradient
step -- the val episodes were in round 4's training data and their
supervision comes from the same lexicon -- so val `char_acc` ties epoch to
epoch and cannot rank checkpoints on its own. `scripts/train_phonikud_yi.py`
therefore selects `best` on `--select-on` (default `val_char` = val
char_acc with val loss as tie-break) and records `best_select_key`,
`best_val` and `best_test_peek` in `run_summary.json`. `--select-on
test_char` selects on the test-peek split instead: a stronger signal, but
it spends the held-out set, so the number it reports stops being an
honest estimate.

