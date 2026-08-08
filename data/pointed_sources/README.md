# Verified pointed-Hebrew sources

Fully-vowelized (nikud-bearing) Hebrew from **published, editorially verified**
editions. This is rescue source #2 for the loshn-koydesh quarantine, and it
ranks *below* the audio evidence in `data/audio_endorsed_lk.py`: audio outranks
book pointing.

Distinct from the corpus `text_pointed` column, which is model-produced and
unverified. Everything here is human-edited published text.

## What is here

| path | contents |
|---|---|
| `raw/*.json` | Sefaria API v3 responses, exactly as downloaded |
| `pointed_index.jsonl` | unpointed form -> pointed forms + counts (built) |
| `coverage_report.md` | quarantine coverage measurement (built) |
| `quarantine_full_snapshot.tsv` | reference quarantine set, see below |

## Provenance

Retrieved 2026-08-07 from the Sefaria API,
`https://www.sefaria.org/api/v3/texts/<ref>?version=hebrew&return_format=text_only`.

| corpus | files | edition | license |
|---|---|---|---|
| Tanakh, 39 books | `raw/Genesis.json` … `raw/II_Chronicles.json` | **Miqra according to the Masorah** (MAM), a digital edition of the Aleppo Codex and related manuscripts, ed. Dovi, Hebrew Wikisource | CC-BY-SA |
| Mishnah, 63 tractates | `raw/Mishnah_*.json` | **Torat Emet 357** | Public Domain |
| Siddur Ashkenaz, 449 sections | `raw/Siddur_Ashkenaz.json` | **Torat Emet 357** | Public Domain |

MAM carries full te'amim and marks qamats qatan with its dedicated codepoint
(U+05C7). Torat Emet has nikud but no te'amim, and writes some optional marks
inconsistently (dagesh lene, holam over a male vav, sin dot).

`Siddur_Ashkenaz.json` is not a raw API response — the Siddur is a complex text,
so it was fetched leaf node by leaf node and stored as
`{ref: {versionTitle, license, text}}`.

CC-BY-SA obligations apply to redistribution of the Tanakh text and of anything
derived from it, including `pointed_index.jsonl`. Attribute Sefaria and the MAM
edition, and keep the same license, if this index is published.

## The index

    python scripts/build_pointed_index.py --coverage

`pointed_index.jsonl`, one JSON object per line:

    {"k": "אבותי", "n": 1, "t": 19,
     "p": [["אֲבוֹתַי", 12], ["אֲבותַי", 4], ["אֲבוֹתָי", 3]]}

- `k` — lookup key: NFC, all nikud and te'amim stripped, final letters exactly
  as printed. Maqaf, paseq and sof pasuk split words rather than joining them.
- `n` — 1 for a word, 2/3 for a phrase (`k` and the pointed forms are both
  space-joined).
- `t` — total occurrences.
- `p` — `[pointed form, count]`, most frequent first. Cantillation is stripped
  from the pointed forms; nikud, dagesh, shin/sin dots and qamats qatan are kept
  as printed. A source token bearing no nikud at all is never stored as a
  candidate.

Phrase entries are verse-internal bigrams and trigrams restricted to n-grams
whose every word appears in the quarantine — that is what keeps the file at
~6 MB instead of several hundred. They exist so that a quoted posuk can be
matched as a span, where the surrounding words disambiguate a homograph that
the unigram entry cannot.

`phonemic_fold()` in the builder is a **comparison** key, not stored output. It
collapses pointing differences that do not change the Ashkenazi reading and that
the two editions write inconsistently: gemination dagesh outside bet/kaf/pe/tav,
word-initial dagesh (always lene), qamats qatan, and the holam dot over a male
vav (U+05B9/U+05BA). Contrastive marks — the shin/sin dot, and dagesh in
bet/kaf/pe/tav other than word-initially — are preserved. Use it before treating
two candidate pointings as a real conflict.

## Reference quarantine set

`quarantine_full_snapshot.tsv` is 14,732 types / 100,827 tokens, produced by
`scripts/run_corpus_v3.py --limit 0` on 2026-08-07. It is snapshotted here
because `data/phonemized/v3/quarantine.tsv` reflects whatever the last corpus
run happened to be — a `--limit`'ed run leaves a much smaller file behind — so
it is not a stable denominator for a coverage number.
