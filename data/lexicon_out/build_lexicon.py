"""Build the Yiddish IPA lexicon.

Inputs
  1. yiddish24 transcripts (../yiddish24/**/*.yl.txt)       -> corpus word list
  2. Phonikud-yi engine (yiddish_g2p.py + its tables)        -> IPA per word
  3. staged phonetic index (data/kodesh_index/*.tsv)         -> loshn-koydesh entries

Outputs (this folder)
  corpus_lexicon.tsv      every word type in the transcripts with IPA, route, confidence
  kodesh_index_lexicon.tsv every entry of the printed phonetic index with standard + hasidic IPA
  lexicon_merged.tsv      one row per word: index reading where it exists, else engine reading
"""
import os, re, sys, csv, glob
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", ".."))
TRANS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "..", "yiddish24")  # folder of *.yl.txt transcripts
sys.path.insert(0, REPO)
import yiddish_g2p as g

# ---------- 1. corpus words ----------
TS = re.compile(r"<\|[^|]*\|>|⟦#\d+⟧")
TOK = re.compile(r"[א-תיִ-ﭏְ-ׇ]+(?:['\"״׳][א-ת]+)*")
freq = Counter()
n_files = 0
for f in glob.glob(os.path.join(TRANS, "*", "*.yl.txt")):
    txt = TS.sub(" ", open(f, encoding="utf-8").read())
    n_files += 1
    for m in TOK.finditer(txt):
        freq[m.group(0)] += 1
print(f"{n_files} transcripts, {sum(freq.values())} tokens, {len(freq)} types")

# ---------- 2. engine over corpus ----------
rows = []
conf = Counter(); routes = Counter()
for w, n in freq.most_common():
    try:
        r = g.g2p_token(w)
    except Exception as e:
        rows.append([w, n, "", "", "error", "", str(e)[:80], ""]); continue
    conf[r["confidence"]] += 1; routes[r["route"]] += 1
    rows.append([w, n, r["ipa_primary"], "|".join(r.get("variants") or []), r["route"],
                 r["confidence"], r["reason"], r.get("layer", "")])
with open(os.path.join(HERE, "corpus_lexicon.tsv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f, delimiter="\t", lineterminator="\n")
    wr.writerow(["word", "count", "ipa", "variants", "route", "confidence", "reason", "layer"])
    wr.writerows(rows)
print("confidence:", dict(conf)); print("routes:", dict(routes))
tok_by_conf = Counter()
for r in rows: tok_by_conf[r[5]] += r[1]
print("tokens by confidence:", dict(tok_by_conf))

# ---------- 3. printed index ----------
IDX = os.path.join(REPO, "data", "kodesh_index")
idx_rows = []
def load(name, status_default):
    p = os.path.join(IDX, name)
    if not os.path.exists(p): return
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            status = r.get("status") or status_default
            note = r.get("reason", "")
            if "owner_tier" in r:
                note = f"conflict: engine {r['owner_tier']} says {r['owner_ipa']}"
            idx_rows.append([r["word_key"], r["hebrew_as_printed"], r["phonetic_as_printed"],
                             r["standard_ipa"], r["hasidic_ipa"], r["shift_rules_fired"],
                             name.replace(".tsv", ""), status, note])
for name, st in [("singles.tsv", "clean"), ("multiwords.tsv", "clean"), ("abbreviations.tsv", "clean"),
                 ("quarantine_plurals.tsv", "plural"), ("conflicts.tsv", "conflict")]:
    load(name, st)
idx_rows.sort(key=lambda r: (r[0], r[1]))
with open(os.path.join(HERE, "kodesh_index_lexicon.tsv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f, delimiter="\t", lineterminator="\n")
    wr.writerow(["word_key", "hebrew_as_printed", "phonetic_respelling", "ipa_standard_yiddish",
                 "ipa_hasidic", "shift_rules", "source_table", "status", "note"])
    wr.writerows(idx_rows)
print("index entries:", len(idx_rows), Counter(r[7] for r in idx_rows))

# ---------- 4. merged ----------
# Authority: engine HIGH/MED (gold, audio, Sefaria, rules) beats the printed index;
# the index (clean single-word rows only, never the plural quarantine) replaces
# any reading the engine had to derive by rule or fallback. Both readings stay visible.
idx_by_key = {}
for r in idx_rows:
    if r[7] == "clean" and r[6] == "singles" and r[0] not in idx_by_key:
        idx_by_key[r[0]] = r
merged = {}
for r in rows:
    w = r[0]; k = g.lexicon_key(w)
    i = idx_by_key.get(k)
    idx_ipa = i[4] if i else ""
    if i and r[4] in ("rule", "fallback") and len(w) > 1:
        merged[w] = [w, r[1], i[4], "printed-index", "MED", r[2], i[2]]
    else:
        merged[w] = [w, r[1], r[2], r[4], r[5], idx_ipa, i[2] if i else ""]
for k, i in idx_by_key.items():
    if i[1] not in merged and " " not in i[1]:
        merged[i[1]] = [i[1], 0, i[4], "printed-index", "MED", "", i[2]]
with open(os.path.join(HERE, "lexicon_merged.tsv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f, delimiter="\t", lineterminator="\n")
    wr.writerow(["word", "corpus_count", "ipa", "source", "confidence", "other_reading", "phonetic_respelling"])
    wr.writerows(sorted(merged.values(), key=lambda r: (-r[1], r[0])))
print("merged:", len(merged), Counter(r[3] for r in merged.values()))
