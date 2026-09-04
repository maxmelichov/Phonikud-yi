# Yiddish IPA lexicon

Built from the yiddish24 transcripts, the Phonikud-yi G2P engine, and the
printed phonetic index of loshn-koydesh words (`kodesh_words.pdf`, 79 pages).

## Files

| file | rows | what |
|---|---|---|
| `lexicon_merged.tsv` | 12,870 | **Use this one.** One row per word. Engine gold/audio/Sefaria readings win; where the engine had to guess by rule, the printed index reading replaces it. `other_reading` keeps the losing reading so nothing is hidden. |
| `corpus_lexicon.tsv` | 11,191 | Every word type in the 122 transcripts, with frequency and the raw engine output (IPA, variants, route, confidence, reason). |


Index statuses: `clean` (4,363) usable as is; `needs_review` (1,873) a disputed vowel-shift rule fired or the key is a homograph; `plural` (881) the index respelled a plural under a singular head; `conflict` (483) the engine already has a native-verified or audio-backed reading that disagrees with the index.

Phone inventory: vowels `a aː ɛ ə i u ɔ ej aj ɔj`, consonants `b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ`, `ˈ` before the stressed vowel.

## Rebuild

```bash
python build_lexicon.py
```

To re-extract the PDF (PyMuPDF, no poppler needed):

```bash
cd Phonikud-yi
python scripts/ingest_printed_index_fitz.py ../kodesh_words.pdf
python scripts/build_respelling_lexicon.py
```

`Phonikud-yi/` is a copy of https://github.com/maxmelichov/Phonikud-yi with the
regenerated `data/kodesh_index/` staging tables and the rebuilt
`data/lexicons/printed_respelling_lk.py`. The nikud model (ONNX) is not included;
the engine's committed tables cover the rescue chain without it.
