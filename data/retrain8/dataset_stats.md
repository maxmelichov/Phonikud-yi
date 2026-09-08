# Retrain dataset v8 — audio-attested tier stamped in

Merger over v3 (`data/retrain3/`): tokens v3 left unsupervised (rule-path LOW/MED) are
stamped with a corpus-attested pointing of their type whenever the fine-tuned Yiddish ear,
scoring the spelling's legal readings against the clip of that token, decides the reading
(occurrence margin ≥ 2.0 nats, or a type reading agreed by ≥ 5 occurrences
at ≥ 85% with margin ≥ 1.0) and the pointing reads back to it under
`reconcile`. `test.jsonl` is a byte-for-byte copy of v3's (asserted).

## Headline

| metric | count |
| --- | ---: |
| attest records | 576,017 |
| tokens stamped, occurrence-level | 111,366 |
| tokens stamped, type-level | 135,186 |
| types with a type-level reading | 6,001 |
| rows changed | 22,406 |

## Coverage (train+val, all tokens)

| | supervised | total | share |
| --- | ---: | ---: | ---: |
| v3 (before) | 1,202,958 | 1,827,568 | 65.82% |
| v8 (after) | 1,449,510 | 1,827,568 | 79.31% |

## Skipped (counted, never guessed)

| reason | tokens |
| --- | ---: |
| `skip_multi_heb_token` | 38 |
| `skip_no_target_occurrence` | 43,335 |
| `skip_no_target_type` | 73,501 |
| `skip_reading_length_mismatch` | 254 |
| `type_reading_no_surviving_form` | 11,661 |
| `undecided` | 254,675 |
