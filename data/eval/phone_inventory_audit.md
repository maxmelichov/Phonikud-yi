# Phone inventory audit (2026-09-18)

Question: does the closed inventory (23 consonants, 11 vowels, stress on 10 of them) have every
Hasidic Yiddish sound, and nothing superfluous? Three sources: the engine's use of each symbol over
the attested corpus (1.83 M words, 6.95 M phones), the lattice ear's open-slot decisions on held-out
gold (which contrasts it hears), and the ear's free decode on gold_val_eps (substitutions).

## Usage (engine phones over the corpus)

Every symbol is used except **ŋ**, which the engine never emits (-ינג loans come out as `inɡ`:
farkˈinɡ, bˈildinɡ). Rarest real symbols: ʤ 1,008 (0.01 %), ʒ 3,076 (0.04 %), ʧ 10,865 (0.16 %),
oʊ 19,379 (0.28 %; the ear keeps only 10,892 of them), p 34,921 (0.5 %), ɔ 54,008 (0.78 %),
aː 54,725 (0.79 %), ej 64,702 (0.93 %).

## What the ear hears apart (held-out gold, open-slot alternatives offered; share it prefers the alternative)

| contrast | offered | ear flips | strongly (≥ 3 nats) |
|---|---:|---:|---:|
| ɔj vs oʊ | 270 | 0.0 % | 0 |
| aj vs ej | 576 | 1.2 % | 0 |
| aj vs aː | 729 | 2.7 % | 3 |
| i → u | 6,067 | 7.0 % | 23 |
| f ↔ p | 386 | 7–19 % | 9 |
| a ↔ ɔ | 1,451 | 13–15 % | 41 |
| ɛ ↔ ə | 9,253 | 14–22 % | 199 |
| z → s | 892 | 17.7 % | 13 |
| u → i | 2,953 | 25.8 % | 87 |
| ɔ ↔ u | 957 | 27–38 % | 76 |
| u → a | 565 | 42.5 % | 35 |

Free decode (no graph) confirms the same picture: ɔ → u 7 % of ɔ tokens, aː → aj 8.7 %, ɛ ↔ i/ə 3–5 %,
u → i 3.7 %; the diphthongs and most consonants are clean (k 83 %, ʃ 77 %, ʧ 88 % recall);
h is the weakest consonant (46 % recall, 38 % deleted — it is breathy and short in this speaker).

## Verdict

**Nothing is missing that the corpus needs.** The three places where Hasidic (Central) Yiddish has a
contrast the inventory folds:
- vowel length **i / iː** and **u / uː** (Central Yiddish is phonemic here: zin/ziːn, hunt/huːnt).
  Folded on purpose — the labels never carried length, minimal pairs are rare in running speech and
  TTS learns the durations from context. Adding it would need new labels for every long vowel.
- **ŋ** is in the inventory but never produced; the engine writes n+ɡ. Harmless (a dead class),
  and correct for Yiddish itself; an English -ing loan is the only place it would matter.
- English loan sounds (w, θ, ð, ɹ, æ) are mapped into the inventory (v, t/d or s/z, r, a/ɛ) — the
  right call for a Yiddish voice.

**Nothing is superfluous, but two symbols are weakly supported by the audio:**
- **ɔ** (0.78 %): the ear confuses it with u on a third of the offers and with a on 15 %. It is a
  real category (ɔbər, hɔb/hub variants, the ɔ of ɔj) but the least separable vowel. Keep; do not
  expand its use.
- **oʊ** (0.28 %, and the ear discards 44 % of the engine's): never preferred over ɔj on 270
  offers, 20 % free recall on its 10 gold tokens. It exists for English loans (ʃoʊ, ɡoʊ) and one
  lexeme family (ארויף). Keep for the loans; treat every engine oʊ in a Yiddish word as a
  candidate for ɔj.
- **ʒ / ʤ** are rare (0.05 % together) but well recognised (loanwords: ʒurnal, ʤab). Keep.

The gradient contrasts (ɛ/ə reduction, a/ɔ/u for א, i/u for ו) are exactly the open slots; they
are phonemic distinctions the orthography leaves open, so they must stay separate symbols — the
ear v3 run (open-slot negatives in training) is what sharpens them.
