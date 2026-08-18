#!/usr/bin/env python3
"""Build Chezky verification batch v4 from the v3 corpus triage logs."""
import csv, re, sys
from pathlib import Path
from collections import Counter

ROOT = Path("/Users/maxm/Documents/Phonikud-yi")
sys.path.insert(0, str(ROOT))
V3 = ROOT / "data/phonemized/v3"
from yiddish_g2p import hebrew_to_ipa, normalize_surface

DAGESH, RAFE, PATAH, KAMATZ = "ּ", "ֿ", "ַ", "ָ"
MARK = re.compile(r"[֑-ׇ]")
LETTER = re.compile(r"[א-תװ-ײ]")

def rows(p):
    with open(p, encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t")

freq = Counter()
for p in ("tokens.tsv", "quarantine.tsv", "low_conf.tsv", "oov_lk.tsv"):
    for r in rows(V3 / p):
        try: freq[r["word"]] = max(freq[r["word"]], int(r["freq"]))
        except (KeyError, ValueError, TypeError): pass
for r in rows(V3 / "alef_defaults.tsv"):
    freq[r["word"]] = max(freq[r["word"]], int(r["applications"]))

TOKENS_SEEN = 1832439          # 1,725,714 emitted + 106,725 quarantined
LOW_TOKENS = sum(int(r["freq"]) for r in rows(V3 / "low_conf.tsv"))   # 351,418

gold_words = set()
for r in csv.DictReader(open(ROOT / "data/gold/g2p_gold_v3.csv", encoding="utf-8")):
    w = r["word"].strip()
    gold_words |= {w, normalize_surface(w), MARK.sub("", w)}

HEB_ONLY = re.compile(r"^[א-תװ-ײ֑-ׇ־\-]+$")
def ok_word(w):
    return (w and len(w) <= 22 and HEB_ONLY.match(w)
            and not re.search(r"[0-9A-Za-z\"'׳״]", w)
            and LETTER.search(w))

cand = {}
def add(w, reasons):
    cand.setdefault(w, set()).update(reasons)

low_freq = {}
for r in rows(V3 / "low_conf.tsv"):
    low_freq[r["word"]] = int(r["freq"]); add(r["word"], r["reason"].split(","))
for r in rows(V3 / "oov_lk.tsv"):
    add(r["word"], r["reason"].split(","))
for r in rows(V3 / "alef_defaults.tsv"):
    add(r["word"], ["alef-default"])

pool = [w for w in cand
        if not ({w, normalize_surface(w), MARK.sub("", w)} & gold_words)
        and ok_word(w)
        and cand[w] & {"alef-default", "pe-default", "lk-fallback"}]
pool.sort(key=lambda w: (-freq[w], w))
BATCH = pool[:400]

# ---------- romanization ---------------------------------------------------
CONS = [("ʦ","ts"),("ʧ","tsh"),("ʤ","dzh"),("ʃ","sh"),("ʒ","zh"),("x","kh"),
        ("ŋ","ng"),("j","y"),("ɡ","g")]
VOW = [("ej","ey"),("aj","ay"),("ɔj","oy"),("oʊ","ou"),("aː","a"),("ɛ","e"),
       ("ə","e"),("ɔ","o")]
V = set("aeiouy")

def romanize(ipa):
    if not ipa: return ""
    s = ipa
    for a, b in VOW: s = s.replace(a, b)
    for a, b in CONS: s = s.replace(a, b)
    s = s.replace("ː", "")
    out = []
    for ch in s:
        if ch != "ˈ":
            out.append(ch); continue
        # uppercase from the start of the coming syllable's onset
        j = len(out)
        while j > 0 and out[j-1] not in V: j -= 1
        out.append("\x00")           # placeholder marking stress start
        out.insert(j, "\x01")
    s = "".join(out)
    # apply: text between \x01 and end of the following vowel run -> uppercase
    res, i = [], 0
    while i < len(s):
        if s[i] == "\x01":
            j = s.index("\x00", i); k = j + 1
            while k < len(s) and s[k] not in V: k += 1
            while k < len(s) and s[k] in V: k += 1
            res.append((s[i+1:j] + s[j+1:k]).upper()); i = k
        else:
            res.append(s[i]); i += 1
    return "".join(res).replace("\x00", "").replace("\x01", "")

def ipa_of(w):
    try: return hebrew_to_ipa(w, stress=True, quarantine=False)
    except Exception: return ""

# ---------- א classification ----------------------------------------------
def units(w):
    """[(index, letter, marks)] over the word."""
    out, i = [], 0
    while i < len(w):
        if LETTER.match(w[i]) or not MARK.match(w[i]):
            j = i + 1
            while j < len(w) and MARK.match(w[j]): j += 1
            out.append((i, w[i], w[i+1:j])); i = j
        else:
            i += 1
    return out

def alef_slots(w):
    """Classify every unpointed א: 'silent' | 'final' | 'vowel'."""
    us = units(w)
    letters = [k for k, (_, ch, _) in enumerate(us) if LETTER.match(ch)]
    slots = []
    for k, (idx, ch, marks) in enumerate(us):
        if ch != "א" or marks: continue
        nxt = us[k+1][1] if k + 1 < len(us) else ""
        prev = us[k-1][1] if k > 0 else ""
        if len(letters) > 1 and k == letters[0] and nxt in ("י", "ו", "ײ", "ױ"):
            slots.append((idx, "silent"))
        elif len(letters) > 1 and k == letters[-1]:
            slots.append((idx, "final"))
        else:
            slots.append((idx, "vowel"))
    return slots

def point_alefs(w, kinds, mark):
    out, off = w, 0
    for idx, kind in alef_slots(w):
        if kind not in kinds: continue
        out = out[:idx+off+1] + mark + out[idx+off+1:]; off += len(mark)
    return out

def swap_final_vowel(rom, new):
    m = re.search(r"([aeiouy]+)$", rom, re.I)
    return (rom[:m.start()] if m else rom) + new

def schwa_fill(rom):
    DIG = ("sh","kh","ts","zh","ng","dz","ay","ey","oy")
    out = []
    for i, c in enumerate(rom):
        out.append(c)
        nxt = rom[i+1] if i + 1 < len(rom) else ""
        if (c.isalpha() and c.lower() not in V and nxt.isalpha()
                and nxt.lower() not in V and (c + nxt).lower() not in DIG):
            out.append("e")
    return "".join(out)

GA, GU, GO = "a as in 'father'", "u as in 'put'", "o as in 'law'"

# ---------- build rows -----------------------------------------------------
out_rows = []
for w in BATCH:
    rs = cand[w]
    slots = [k for _, k in alef_slots(w)]
    ipa = ipa_of(w); rom = romanize(ipa)
    qs, cl, subs = [], [], []

    if "vowel" in slots:
        a = romanize(ipa_of(point_alefs(w, {"vowel"}, PATAH)))
        u = romanize(ipa_of(point_alefs(w, {"vowel"}, KAMATZ)))
        opts = [x for x in (a, u) if x]
        if len(opts) == 2 and a != u:
            cl += opts
            qs.append(f"the unpointed א: {a} ({GA}) or {u} ({GU})? "
                      f"— or an o ({GO})?")
        else:
            qs.append("the unpointed א — a, u or o?")
        subs.append("alef-vowel")
    if "final" in slots:
        fi = [i for i, k in alef_slots(w) if k == "final"][0]
        base = romanize(ipa_of(w[:fi] + w[fi+1:]))
        if not base or not re.search(r"[aeiouy]", base, re.I):
            base = None
        e_, a_, u_ = ((base + x) if base else swap_final_vowel(rom, x)
                      for x in ("e", "a", "u"))
        sil = base or rom
        cl += [e_, a_, u_, sil]
        qs.append(f"the א at the end: {e_} (…-e), {a_} (…-a), {u_} (…-u) "
                  f"— or silent ({sil})?")
        subs.append("alef-final")
    if "pe-default" in rs:
        f_ = romanize(ipa_of(re.sub(r"פ(?![֑-ׇ])", "פ" + RAFE, w, count=0)))
        p_ = romanize(ipa_of(re.sub(r"פ(?![֑-ׇ])", "פ" + DAGESH, w, count=0)))
        if f_ and p_ and f_ != p_:
            cl += [f_, p_]
            qs.append(f"the פ: f ({f_}) or p ({p_})?")
        else:
            qs.append("the פ/ף — f or p?")
        subs.append("pe-default")
    if "lk-fallback" in rs:
        fill = schwa_fill(rom.lower())
        opts = [rom] + ([fill] if fill != rom else [])
        cl += opts
        qs.append("our guess: " + (f"{opts[0]} — or {opts[1]}?" if len(opts) > 1
                                   else f"{rom}?") + " How do you say it?")
        subs.append("oov-lk")

    if not qs:   # א only ever a silent vowel-carrier here
        qs.append(f"we say **{rom}** — right?")
        subs.append("alef-silent")

    reason = ("oov-lk" if "lk-fallback" in rs
              else "pe-default" if subs == ["pe-default"] else "alef-default")
    seen, cls = set(), []
    for c in cl:
        if c and c not in seen: seen.add(c); cls.append(c)
    out_rows.append({
        "rank": 0, "word": w, "freq": freq[w], "reason": reason,
        "subtype": "+".join(subs),
        "all_reasons": ",".join(sorted(rs & {"alef-default","pe-default",
                                             "lk-fallback","vowel-ratio",
                                             "bad-phone","clitic"})),
        "engine_ipa": ipa, "engine_guess": rom, "candidates": " | ".join(cls),
        "question": " ".join(qs), "verdict": "",
    })
for i, r in enumerate(out_rows, 1): r["rank"] = i

csv_p = ROOT / "data/verification_batch_v4.csv"
with open(csv_p, "w", encoding="utf-8", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    wtr.writeheader(); wtr.writerows(out_rows)

# ---------- markdown -------------------------------------------------------
covered = sum(low_freq.get(w, 0) for w in BATCH)
new_share = (LOW_TOKENS - covered) / TOKENS_SEEN
SECTIONS = [
 ("oov-lk", lambda r: r["reason"] == "oov-lk",
  "Hebrew / Aramaic words we've never been taught",
  "These are loshn-koydesh words our engine doesn't know. The text gives it no nekudos, so\n"
  "its guess is usually missing vowels — *khzis* for כזית, that kind of thing.\n"
  "**Just write how you say the word.** English spelling is fine; write it the way you'd\n"
  "text it to someone."),
 ("alef-vowel", lambda r: "alef-vowel" in r["subtype"] and r["reason"] != "oov-lk",
  "Unpointed א — *a* or *u*?",
  "Yiddish writes אַ and אָ with a dot, but the news text drops it, so we have to guess.\n"
  "אַ is **a** as in *father*; אָ is the **u** in *put* / *should* (that's the rule you gave us).\n"
  "For each word, mark which one it is — or write your own spelling if it's neither\n"
  "(some of these might be a real **o** as in *law*)."),
 ("alef-final", lambda r: "alef-final" in r["subtype"] and r["reason"] != "oov-lk",
  "א at the end of a word — *-e*, *-a* or *-u*?",
  "A word-final א can be an *-e* (like *gemore*), an *-a*, an *-u* — or nothing at all\n"
  "(as in הוא, where it just sits there). Which is it in each of these?"),
 ("pe-default", lambda r: "pe-default" in r["subtype"],
  "פ / ף — *f* or *p*?",
  "פּ is *p*, פֿ is *f*, and the text writes both as a bare פ. Which one?"),
 ("alef-silent", lambda r: "alef-silent" in r["subtype"],
  "Quick confirmations — א as a silent vowel-carrier",
  "In these the א just carries the vowel that follows (אי = *i*, איי = *ay*, או = *u*).\n"
  "We think we've got them right — please just glance down the column and fix anything\n"
  "that's off. Skipping a row means \"yes, that's right\"."),
]
used, lines = set(), [
 "# Yiddish TTS — verification batch v4 (Chezky)", "",
 f"{len(out_rows)} words. These are the most common words our engine is still guessing at,",
 f"straight off the corpus — together they account for **{covered:,} of the {TOKENS_SEEN:,}",
 f"running words** in everything we've processed. If you get through the whole list, the share",
 f"of the corpus we're unsure about drops from **19.18% to {new_share:.2%}**.", "",
 "Only fill in the last column where our guess is wrong — a blank row means \"you got it right\".",
 "CAPITALS mark the stressed syllable (*KIMendige*, not *kimenDIge*).",
 "The Hebrew/Aramaic section is the one that matters most; the last section is quick yes/nos.", ""]
for key, pred, title, intro in SECTIONS:
    grp = [r for r in out_rows if r["rank"] not in used and pred(r)]
    used |= {r["rank"] for r in grp}
    if not grp: continue
    lines += [f"## {title}", "",
              f"*{len(grp)} words — {sum(r['freq'] for r in grp):,} times in the corpus*", "",
              intro, "",
              "| # | word | count | our guess | the question | YOUR ANSWER |",
              "|---|------|------:|-----------|--------------|-------------|"]
    for r in grp:
        lines.append(f"| {r['rank']} | {r['word']} | {r['freq']:,} | "
                     f"**{r['engine_guess']}** | {r['question']} |  |")
    lines.append("")
lines += ["## If you only have ten minutes", "",
          "Do the first fifteen rows of the Hebrew/Aramaic section — they're the highest-count",
          "words in the whole list, so they buy us the most.", ""]
(ROOT / "data/verification_batch_v4.md").write_text("\n".join(lines), encoding="utf-8")

assert len({r["word"] for r in out_rows}) == len(out_rows)
assert not [r for r in out_rows if {r["word"], normalize_surface(r["word"]),
                                    MARK.sub("", r["word"])} & gold_words]
assert not [r for r in out_rows if re.search(r"[0-9A-Za-z]", r["word"])]
assert used == {r["rank"] for r in out_rows}
print("rows", len(out_rows))
print(Counter(r["reason"] for r in out_rows))
print(Counter(r["subtype"] for r in out_rows).most_common())
print("covered", covered, f"{covered/TOKENS_SEEN:.2%}")
print(f"LOW_CONF 19.18% -> {new_share:.2%}")
print("no-guess rows", sum(1 for r in out_rows if not r["engine_guess"]))
print("freq range", out_rows[0]["freq"], "..", out_rows[-1]["freq"])
