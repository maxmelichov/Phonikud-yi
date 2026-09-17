# Stress from prosody: can the ear's forced alignment recover the engine's stress?

Script: `scripts/stress_prosody_probe.py`. Ear: run 2 (`data/xeus_ft/ckpt/best`). Clips: `data/xeus_ft/run3` all splits (train + val_words + val_eps), 1–4 words per clip, every word with >= 2 vowels whose engine reading (`yiddish_labels.text_to_ipa`) has the same vowel count as the gold target is a case. Label = the vowel the engine stresses. Split is word-type-disjoint by lexicon key (md5 hash, 25 % of types held out).

| | cases | clips | word types |
|---|---|---|---|
| train | 3876 | 3359 | 133 |
| test  | 919 | 895 | 30 |
| Gemini-disagreement slice (excluded from train) | 679 | 652 | 18 |

Engine stresses ə in 31 / 4795 cases (should be 0): {'נאכדעם': 23, 'נאָכדעם': 8} — the engine reads these with a full vowel where the gold has ə, so the transferred index lands on a schwa.

## Raw correlates (train cases): stressed vs unstressed vowels, non-schwa only

| | n | mean duration (frames) | median duration | mean RMS | median F0 rel. clip (semitones) |
|---|---|---|---|---|---|
| stressed | 3845 | 4.94 | 4 | 0.1421 | 0.66 |
| unstressed | 1035 | 4.65 | 3 | 0.1064 | 0.00 |

## Word-level accuracy on held-out word types (label = engine stress)

| model | correct | n | accuracy |
|---|---|---|---|
| first syllable | 828 | 919 | 90.1 % |
| last syllable | 60 | 919 | 6.5 % |
| penultimate | 777 | 919 | 84.5 % |
| first non-schwa | 889 | 919 | 96.7 % |
| longest vowel | 712 | 919 | 77.5 % |
| loudest vowel | 674 | 919 | 73.3 % |
| highest F0 | 685 | 919 | 74.5 % |
| duration only | 712 | 919 | 77.5 % |
| energy only | 674 | 919 | 73.3 % |
| F0 only | 685 | 919 | 74.5 % |
| dur + energy + F0 (acoustic only) | 741 | 919 | 80.6 % |
| position only | 777 | 919 | 84.5 % |
| position + vowel identity (no audio) | 915 | 919 | 99.6 % |
| acoustic + vowel identity | 906 | 919 | 98.6 % |
| all | 908 | 919 | 98.8 % |

## Hard subsets of the held-out cases

Vowel identity solves most words because the unstressed syllable holds ə. The audio question only bites where it does not.

| subset | n | types | first syllable | penultimate | longest vowel | loudest vowel | highest F0 | duration only | energy only | F0 only | dur + energy + F0 (acoustic only) | position + vowel identity (no audio) | acoustic + vowel identity | all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all held-out | 919 | 30 | 90.1 % | 84.5 % | 77.5 % | 73.3 % | 74.5 % | 77.5 % | 73.3 % | 74.5 % | 80.6 % | 99.6 % | 98.6 % | 98.8 % |
| no ə in the word | 106 | 6 | 71.7 % | 71.7 % | 76.4 % | 73.6 % | 78.3 % | 76.4 % | 73.6 % | 78.3 % | 83.0 % | 96.2 % | 87.7 % | 89.6 % |
| engine stress not on first vowel | 91 | 3 | 0.0 % | 34.1 % | 40.7 % | 31.9 % | 51.6 % | 40.7 % | 31.9 % | 51.6 % | 46.2 % | 100.0 % | 100.0 % | 100.0 % |
| no ə AND stress not first | 30 | 1 | 0.0 % | 0.0 % | 66.7 % | 50.0 % | 70.0 % | 66.7 % | 50.0 % | 70.0 % | 76.7 % | 100.0 % | 100.0 % | 100.0 % |
| >= 2 non-ə vowels | 158 | 8 | 81.0 % | 48.1 % | 80.4 % | 76.6 % | 79.7 % | 80.4 % | 76.6 % | 79.7 % | 84.2 % | 97.5 % | 91.8 % | 93.0 % |

## Learned weights (acoustic only, standardised)

| feature | weight |
|---|---|
| f0_max | +0.530 |
| f0_rel_clip | -0.442 |
| dur_max | +0.361 |
| f0_rank | -0.287 |
| f0_rel_word | +0.278 |
| rms_rel | +0.210 |
| log_rms | +0.116 |
| rms_rank | +0.079 |
| dur_rank | -0.077 |
| dur_rel | +0.076 |
| rms_max | -0.058 |
| log_dur | +0.045 |

## By number of vowels (held-out types)

| vowels | n | first syllable | no audio (pos + vowel) | acoustic + vowel | all |
|---|---|---|---|---|---|
| 2 | 806 | 92.6 % | 99.5 % | 98.4 % | 98.6 % |
| 3 | 113 | 72.6 % | 100.0 % | 100.0 % | 100.0 % |

## Learned weights (all features, standardised)

| feature | weight |
|---|---|
| v_ɛ | +1.188 |
| v_ə | -1.109 |
| is_schwa | -1.109 |
| v_aj | +1.089 |
| is_last | -0.648 |
| f0_max | +0.484 |
| v_ej | +0.477 |
| v_ɔ | +0.378 |
| is_first | -0.377 |
| f0_rank | -0.376 |
| v_i | -0.343 |
| is_penult | -0.239 |
| rms_rel | +0.210 |
| v_a | -0.209 |

## Type-level (majority vote over a type's clips, held-out types)

| model | types (>= 3 clips) correct | n | accuracy |
|---|---|---|---|
| acoustic + vowel identity | 30 | 30 | 100.0 % |
| all | 30 | 30 | 100.0 % |

## 5-fold type-disjoint cross-validation (pooled over all types)

The single split holds out only 30 types. Here every type is held out exactly once (fold = md5(key) mod 5), so the hard subsets carry every polysyllabic gold type the dictionary has.

| subset | n | types | random | first syllable | penultimate | first non-schwa | longest vowel | loudest vowel | highest F0 | duration only | energy only | F0 only | dur + energy + F0 (acoustic only) | position only | position + vowel identity (no audio) | acoustic + vowel identity | all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 4795 | 163 | 47.8 % | 71.5 % | 74.3 % | 85.8 % | 68.1 % | 65.0 % | 66.9 % | 68.1 % | 65.0 % | 66.9 % | 72.5 % | 74.6 % | 91.9 % | 93.9 % | 93.4 % |
| 2 vowels | 4166 | 139 | 50.0 % | 76.7 % | 76.7 % | 88.2 % | 71.5 % | 66.2 % | 67.8 % | 71.5 % | 66.2 % | 67.8 % | 74.3 % | 76.7 % | 94.3 % | 93.8 % | 93.4 % |
| 3 vowels | 611 | 22 | 33.3 % | 37.2 % | 60.4 % | 71.5 % | 45.2 % | 57.6 % | 60.2 % | 45.2 % | 57.6 % | 60.2 % | 59.9 % | 60.4 % | 76.6 % | 95.6 % | 94.4 % |
| 4+ vowels | 18 | 2 | 25.0 % | 33.3 % | 0.0 % | 33.3 % | 55.6 % | 55.6 % | 66.7 % | 55.6 % | 50.0 % | 66.7 % | 66.7 % | 66.7 % | 66.7 % | 77.8 % | 72.2 % |
| no ə in the word | 822 | 41 | 49.7 % | 43.8 % | 42.2 % | 43.8 % | 56.7 % | 55.5 % | 59.1 % | 56.7 % | 55.5 % | 59.1 % | 60.9 % | 42.2 % | 73.4 % | 70.6 % | 68.6 % |
| >= 2 non-ə vowels | 1131 | 53 | 45.1 % | 42.7 % | 44.7 % | 42.7 % | 53.7 % | 58.5 % | 62.2 % | 53.7 % | 58.4 % | 62.2 % | 62.4 % | 45.8 % | 68.6 % | 77.0 % | 74.9 % |
| engine stress not on first vowel | 1368 | 49 | 45.1 % | 0.0 % | 27.0 % | 50.4 % | 40.2 % | 49.6 % | 52.3 % | 40.2 % | 49.5 % | 52.3 % | 53.7 % | 27.9 % | 85.2 % | 89.1 % | 87.7 % |
| no ə AND stress not first | 462 | 20 | 50.0 % | 0.0 % | 0.0 % | 0.0 % | 51.3 % | 50.4 % | 53.9 % | 51.3 % | 50.4 % | 53.9 % | 56.7 % | 0.0 % | 77.5 % | 76.0 % | 71.9 % |
| stress not first, >= 2 non-ə vowels | 648 | 27 | 45.1 % | 0.0 % | 24.5 % | 0.0 % | 46.0 % | 53.5 % | 57.6 % | 46.0 % | 53.4 % | 57.6 % | 58.3 % | 26.4 % | 73.6 % | 81.8 % | 78.9 % |

Type-level (majority vote over each type's clips, types with >= 3 clips), CV predictions:

| model | types correct | n | accuracy | of which stress-not-first types correct / n |
|---|---|---|---|---|
| first syllable | 111 | 158 | 70.3 % | 0 / 47 |
| first non-schwa | 133 | 158 | 84.2 % | 22 / 47 |
| longest vowel | 115 | 158 | 72.8 % | 16 / 47 |
| dur + energy + F0 (acoustic only) | 128 | 158 | 81.0 % | 29 / 47 |
| acoustic + vowel identity | 147 | 158 | 93.0 % | 41 / 47 |
| position + vowel identity (no audio) | 144 | 158 | 91.1 % | 40 / 47 |
| all | 146 | 158 | 92.4 % | 40 / 47 |

## The Gemini-disagreement slice

Word types where the Gemini judge (`data/stress/stress_eval_cache.jsonl`, `ours_ok=false`, `correct_index` != engine's index at judging time, confidence >= 0.5) put the stress elsewhere than the engine. Classifiers were trained without these types. `engine` = the engine's index NOW (overrides applied since the judging); types whose current engine index already equals Gemini's are listed separately.

Two sub-slices. (a) **still open**: the engine's current index != Gemini's, so the classifier chooses between them. (b) **override applied**: an override since moved the engine onto Gemini's index; the classifier chooses between the OLD rule index (`ours_cache`) and Gemini's = the current engine's. Only types whose cached syllable count equals the gold vowel count.

### all

| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |
|---|---|---|---|---|---|---|---|---|
| open | 24 | 5 | 19 | 0 | 2 | 0 | 2 | 0 |
| applied | 655 | 499 | 156 | 0 | 16 | 13 | 3 | 0 |

### acoustic + vowel identity

| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |
|---|---|---|---|---|---|---|---|---|
| open | 24 | 10 | 14 | 0 | 2 | 0 | 2 | 0 |
| applied | 655 | 487 | 161 | 7 | 16 | 13 | 3 | 0 |

### dur + energy + F0 (acoustic only)

| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |
|---|---|---|---|---|---|---|---|---|
| open | 24 | 6 | 10 | 8 | 2 | 0 | 2 | 0 |
| applied | 655 | 380 | 265 | 10 | 16 | 10 | 6 | 0 |

### duration only

| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |
|---|---|---|---|---|---|---|---|---|
| open | 24 | 5 | 14 | 5 | 2 | 0 | 2 | 0 |
| applied | 655 | 402 | 247 | 6 | 16 | 10 | 6 | 0 |

### energy only

| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |
|---|---|---|---|---|---|---|---|---|
| open | 24 | 10 | 3 | 11 | 2 | 1 | 0 | 1 |
| applied | 655 | 301 | 343 | 11 | 16 | 8 | 8 | 0 |

### F0 only

| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |
|---|---|---|---|---|---|---|---|---|
| open | 24 | 4 | 14 | 6 | 2 | 0 | 2 | 0 |
| applied | 655 | 297 | 345 | 13 | 16 | 6 | 10 | 0 |

### position + vowel identity (no audio)

| slice | clips | prosody = Gemini | prosody = rule | neither | types | types→Gemini | types→rule | types→neither |
|---|---|---|---|---|---|---|---|---|
| open | 24 | 4 | 20 | 0 | 2 | 1 | 1 | 0 |
| applied | 655 | 578 | 77 | 0 | 16 | 14 | 2 | 0 |

### Per type (model: all)

| slice | word | vowels | rule idx | Gemini idx | prosody idx | clips | side |
|---|---|---|---|---|---|---|---|
| applied | אזוי | a ɔj | 0 | 1 | 1 | 51 | Gemini |
| applied | אזא | a a | 0 | 1 | 1 | 46 | Gemini |
| applied | אוועק | a ɛ | 0 | 1 | 1 | 41 | Gemini |
| applied | ארום | a i | 0 | 1 | 0 | 41 | rule |
| applied | אראפ | a u | 0 | 1 | 1 | 41 | Gemini |
| applied | ארויס | a ɔj | 0 | 1 | 1 | 41 | Gemini |
| applied | אריין | a aj | 0 | 1 | 1 | 41 | Gemini |
| applied | כדי | ə aj | 0 | 1 | 1 | 40 | Gemini |
| applied | אמאל | a u | 0 | 1 | 1 | 40 | Gemini |
| applied | צוריק | i i | 0 | 1 | 0 | 40 | rule |
| applied | פּראָבלעם | ɔ ɛ | 0 | 1 | 1 | 40 | Gemini |
| applied | צוזאמען | i a ə | 0 | 1 | 1 | 40 | Gemini |
| applied | גייען | aj ə | 1 | 0 | 0 | 40 | Gemini |
| applied | מסביר | a i | 0 | 1 | 0 | 40 | rule |
| applied | ניגונים | i i i | 0 | 1 | 1 | 39 | Gemini |
| applied | ארויף | a oʊ | 0 | 1 | 1 | 34 | Gemini |
| open | אנגעהויבן | u ə ɔj | 0 | 2 | 0 | 20 | rule |
| open | וויפיל | i i | 0 | 1 | 0 | 4 | rule |

## Findings

1. **Prosody alone reproduces the engine's stress weakly.** 5-fold type-disjoint CV over 4795 cases / 163 types: duration only 68.1 %, energy only 65.0 %, F0 only 66.9 %, all three 72.5 % — against first-syllable 71.5 %, first-non-ə 85.8 % and random 47.8 %. The ear's vowel spans are 3–5 frames (60–100 ms) and CTC-peaky, so duration is coarse; the raw correlates go the right way (stressed vowels are longer, louder, higher) but the separation is small.
2. **Vowel identity does the work, not audio.** Position + vowel identity with no audio at all: 91.9 %; adding the acoustics: 93.9 % (all features 93.4 %). In the gold lexicon the unstressed syllable is nearly always ə (and ə is never stressed by the engine except the transferred-index anomaly נאכדעם, engine nuxdˈejm vs gold nuxdəm).
3. **Where it bites (no ə in the word, n=822, 41 types, random 49.7 %)**: acoustic only 60.9 %, vs first syllable 43.8 %, no-audio model 73.4 %, acoustic + vowel 70.6 %. Non-initial stress with no ə (n=462, 20 types): acoustic only 56.7 % vs random 50.0 % and 0 % for every position rule. So the audio carries a real but weak stress signal (~+7–10 points over chance per clip), and it is the only signal available on exactly the words where the rule stage fails.
4. **On the Gemini-disagreement types** (the a-initial adverbs etc., 16 types with the override already applied): the acoustic-only classifier sides with Gemini on the majority of clips and 10 / 16 types; with vowel identity 13 / 16. Two types are still open (אנגעהויבן, וויפיל) and prosody sides with the rule on both. Per-clip prosody is noisy; the per-type majority over ~40 clips is what moves.
5. **Caveat on n**: the certain gold lexicon has only 163 polysyllabic types with an engine reading of matching vowel count, 49 of them with non-initial stress. Clip counts are large (4,795) but the type count is what the split is over, and the within-type clips are mostly the same few speakers.

