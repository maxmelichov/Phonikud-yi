# Retrain dataset v7 — homograph unmask at the scorer decide bar

Merger over v3 (`data/retrain3/`): v3 train rows are the input;
already-supervised tokens are never restamped; `val.jsonl` and
`test.jsonl` are byte-for-byte copies of v3's (asserted).

v6 unmasked audio-decided homograph occurrences at `MARGIN_MIN=0.08`.
v7 uses `0.05`, the scorer's own decide bar in
`scripts/xeus_score_homographs.py`. Type-level homograph stamps stay
forbidden.

## Headline

| metric | v6 (0.08) | v7 (0.05) |
| --- | ---: | ---: |
| confident votes (cleared the bar) | 893 | 1982 |
| skip_low_margin | 1089 | 0 |
| newly unmasked on retrain3 | 0 | 965 |
| skip_already_supervised | 778 | 778 |
| rows changed | 0 | 660 |

Newly unmasked on retrain3 is the *additional* supervision this pass
adds. Occurrences already unmasked at 0.08 (and carried through v2/v3)
count as `skip_already_supervised`.

## וי IPA (not rewritten)

Phonikud trains on pointing, not IPA, so this pass did not restamp וי
targets. Eval-time G2P now uses the sibling lexicon fix: default וי → ɔj,
oʊ only via lexicon exceptions, gold אויך `ɔjx` (variant `oʊx` kept).
גרויס / טויט / בלויז already ɔj. אנגעהויבן left as ˈunɡəhɔjbn.

## Top newly unmasked types (v7)

| type | tokens |
| --- | ---: |
| שכר | 18 |
| שקר | 17 |
| שמות | 15 |
| לחם | 15 |
| חלה | 13 |
| כסף | 11 |
| מכת | 11 |
| בשר | 10 |
| חדש | 10 |
| בשלח | 10 |
| לשמה | 10 |
| שקל | 9 |
| שמן | 8 |
| הלכות | 8 |
| רבה | 8 |
| דבר | 8 |
| נקבה | 8 |
| בקר | 8 |
| שבר | 8 |
| מטל | 8 |

