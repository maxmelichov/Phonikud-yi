"""
Hybrid Yiddish G2P: Hebrew-script Yiddish -> IPA for TTS.

ARCHITECTURE: Lexicon routing (spec v3 §3) in front of a three-stage rule path
  (Orthography -> Latin Base -> Central Phonology)

Spec v3 makes the lexicon, not the rules, the dialect: four graphemes (א, יי, וי,
unpointed פ) are lexically ambiguous, and 81% of naked-rule errors come from
them. So hebrew_to_ipa routes every token -- abbreviation table, multiword
table, the native-verified gold lexicon (data/gold_lexicon.py, authority #1),
the legacy merged-LK and high-frequency lists -- and only then falls through to
the rules below. g2p_token exposes that decision per token (route + confidence)
for the §12 iteration loop; scripts/run_corpus_v3.py runs it over the corpus and
enforces the QA gates.

DIALECT: Poylish/Galitzyaner/Modern Hasidic

Handles ALL THREE spelling systems:
  - Unpointed Hasidic spelling (א = vowel a/o, י = vowel i, no diacritics)
  - Pointed YIVO orthography (אַ אָ פּ בֿ ייִ ...)
  - Fully pointed nikud, as produced by the model in this directory, in the
    Hasidic convention where a consonant carries a vowel point even though the
    vowel letter is written too (דֶער, בַּא, גְלַייַך)

Diacritics are read as evidence wherever they appear: they pick /a/ vs /o/ for א,
/p/ vs /f/ for פ, /t/ vs /s/ for ת, /ey/ vs /ay/ for יי, and -- most usefully --
they supply the vowels of Hebrew-origin words that unpointed spelling omits
entirely, so כְּלַל is klal rather than kll. Rules that read a vowel point only
fire when one is present, so unpointed input behaves exactly as it did before.
"""

from __future__ import annotations
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

_HEBREW_CHAR = re.compile(r"[\u0590-\u05FF]")


def _strip_points(text: str) -> str:
    """Remove Hebrew diacritics (nikud) so pointed lexicon keys also match unpointed text."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


# =====================================================================
# THE CLOSED PHONE INVENTORY (spec v3 §1)
#
# Nothing outside this set may ever reach corpus output. Space and "-" are
# separators, not phones: a few lexicon entries are multiword (ב"ה ->
# "bˈurəx haʃˈɛm") or hyphenated compounds (bis-mˈɛdrəʃ), and the QA gate
# checks phones only, after splitting on those.
# =====================================================================
PHONE_VOWELS = ("aː", "ej", "aj", "ɔj", "oʊ", "a", "ɛ", "ə", "i", "u", "ɔ")
PHONE_CONSONANTS = frozenset("bdfɡhjklmnprstvzxʃʒʦʧʤŋ")
PHONE_MARKS = frozenset("ˈ")
# Longest first: aː/ej/aj/ɔj/oʊ are single symbols, and their second character
# (ː e j ʊ) is not independently legal -- e and o never stand alone.
PHONE_SYMBOLS = PHONE_VOWELS + tuple(sorted(PHONE_CONSONANTS)) + tuple(PHONE_MARKS)
PHONE_INVENTORY = frozenset("".join(PHONE_SYMBOLS))
SEPARATORS = " -"


def ipa_phone_violations(ipa: str) -> list[str]:
    """Symbols in ``ipa`` that are outside the §1 closed inventory.

    Tokenizes longest-symbol-first, so a stray bare "e" or "o" (legal only as
    part of ej / oʊ) is reported even though the character occurs in the set.
    """
    bad: set[str] = set()
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS:
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i):
                i += len(sym)
                break
        else:
            bad.add(ipa[i])
            i += 1
    return sorted(bad)


def vowel_consonant_counts(ipa: str) -> tuple[int, int]:
    """(vowel nuclei, consonant phones) in an IPA string, per §1's ratio rule.

    Diphthongs are one vowel each: the tokenizer takes aː/ej/aj/ɔj/oʊ whole, so
    the off-glide is never counted as the consonant /j/.
    """
    vowels = consonants = 0
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS:
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i):
                if sym in PHONE_VOWELS:
                    vowels += 1
                elif sym in PHONE_CONSONANTS:
                    consonants += 1
                i += len(sym)
                break
        else:
            i += 1
    return vowels, consonants


def max_consonant_run(ipa: str) -> int:
    """Longest run of consonant phones with no vowel between them."""
    longest = run = 0
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS:
            run = 0
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i):
                if sym in PHONE_VOWELS:
                    run = 0
                elif sym in PHONE_CONSONANTS:
                    run += 1
                    longest = max(longest, run)
                i += len(sym)
                break
        else:
            i += 1
    return longest


def violates_vowel_ratio(ipa: str) -> bool:
    """§1: "at least one vowel symbol per 3 consonant symbols" — as calibrated
    against the gold, which is authority #1 over the spec's phrasing.

    Read literally (vowels * 3 >= consonants) the rule rejects 14 native-verified
    gold primaries -- mɛnʧn, ʦviʃn, ʃraːbn, pinkt, tɛkst, brɛnɡt, trɔmp ... --
    because v3 §1 also deletes the epenthetic schwa of the syllabic finals, so a
    perfectly ordinary Yiddish monosyllable now carries one vowel and four
    consonant symbols. What the rule is actually guarding against is the failure
    mode named in §6.3: an unpointed loshn-koydesh word emerging as a bare
    consonant string (מלך -> mlx, ספר -> sfr). So the test is:

      * no vowel at all, or
      * a consonant run longer than FOUR.

    AUDIT 2026-08-07: the run bound was three, justified by "no Yiddish syllable
    has a longer run and nothing in the gold produces one". The gold half of that
    is true (0 of 500 primaries exceed three) but it is not evidence for the
    generalization, and the corpus disproves it: ˈajbərʃtn (r-ʃ-t-n), dˈarfsti
    (r-f-s-t), fˈinktləx, ˈɛkstra, ˈajɡntləx are ordinary Germanic words with a
    four-consonant run, and quarantining them dropped 871 tokens / 347 types per
    5k corpus rows for nothing. Four still catches every real offender, because
    the failure mode this guards -- an unpointed LK word emerging as a bare
    consonant string -- is caught by the vowels == 0 clause above it.
    """
    for word in re.split(r"[ -]+", ipa):
        if not word.strip():
            continue
        vowels, consonants = vowel_consonant_counts(word)
        if consonants and vowels == 0:
            return True
        if max_consonant_run(word) > 4:
            return True
    return False


# =====================================================================
# TEXT NORMALIZATION (spec v3 §2)
# =====================================================================
_FINAL_FOLD = {"ך": "כ", "ם": "מ", "ן": "נ",
               "ף": "פ", "ץ": "צ"}
_FINAL_FOLD_TABLE = str.maketrans(_FINAL_FOLD)
# The YIVO ligatures are spelling variants of the two-letter digraphs the corpus
# writes out, and no gold key uses them, so the lookup key folds them together:
# pointed זײַן must reach the same entry as unpointed זיין.
_LIGATURE_FOLD = {"ײ": "יי", "ױ": "וי", "װ": "וו"}
_LIGATURE_FOLD_TABLE = str.maketrans(_LIGATURE_FOLD)
_GERESH_CHARS = re.compile(r"[׳ʼ‘’`]")
_GERSHAYIM_CHARS = re.compile(r"[״“”]")
_DASH_CHARS = re.compile(r"[־‐‑‒–—]")
# Punctuation that may sit around a token. The apostrophe and the gershayim are
# deliberately absent: they are load-bearing inside abbreviations (ר', שליט"א)
# and are stripped in a second, later pass once the abbreviation table has had
# its look (§2.2, §2.6).
_EDGE_PUNCT = " \t\n\r.,!?;:()[]{}<>«»„‚‹›…׃״“”\"*/\\|-"


def normalize_surface(text: str) -> str:
    """§2.1-2.2: NFC, unify geresh/gershayim, unify makef and dashes to '-'."""
    text = unicodedata.normalize("NFC", text)
    text = _DASH_CHARS.sub("-", text)
    text = _GERESH_CHARS.sub("'", text)
    return _GERSHAYIM_CHARS.sub('"', text)


def split_affixes(token: str) -> tuple[str, str, str]:
    """(leading punctuation, core, trailing punctuation) of a normalized token."""
    core = token.strip(_EDGE_PUNCT)
    if not core:
        return "", token, ""
    start = token.index(core)
    return token[:start], core, token[start + len(core):]


def lexicon_key(word: str) -> str:
    """§2.1/§2.4: the whole-token lookup key -- nikud stripped, finals folded."""
    bare = _strip_points(normalize_surface(word))
    return bare.translate(_FINAL_FOLD_TABLE).translate(_LIGATURE_FOLD_TABLE)


# =====================================================================
# DIACRITICS
#
# Input may be unpointed (Hasidic press), partially pointed (YIVO) or fully
# pointed -- the last being what the nikud model in this directory produces, in
# the Hasidic convention where a consonant carries a vowel point even though the
# vowel letter (א/ע/י/ו) is written too. Every rule below that reads a vowel
# point is gated on that point being present, so unpointed input still follows
# the original heuristics unchanged.
# =====================================================================
DAGESH = "\u05bc"
RAFE = "\u05bf"
SHIN_DOT = "\u05c1"
SIN_DOT = "\u05c2"
SHEVA = "\u05b0"
HIRIQ = "\u05b4"
TSERE = "\u05b5"
PATAH = "\u05b7"
QAMATS = "\u05b8"
HOLAM = "\u05b9"
QUBUTS = "\u05bb"

# Ashkenazi / Central Yiddish readings of the Hebrew vowel points. Sheva is
# silent here; latin_to_ipa re-inserts a schwa where a syllable needs one.
#
# SPEC \u00a75: komets (gadol AND katan, incl. final \u05b8\u05d4) rides diaphoneme class 12/13
# and surfaces as [u] -- Duvid, Shabus, shlitu, chuchme, kul. In the Latin layer
# that class is written "oo" (see the label key on _LATIN_TO_IPA); plain "o" is
# now reserved for class 41 [\u0254]. Writing komets as "o" here would silently flip
# every pointed loshn-koydesh word from [u] to [\u0254].
_POINT_TO_LATIN: dict[str, str] = {
    SHEVA: "",
    "\u05b1": "e",   # hataf segol
    "\u05b2": "a",   # hataf patah
    "\u05b3": "oo",  # hataf qamats -> class 12/13 [u]
    HIRIQ: "i",
    TSERE: "ey",
    "\u05b6": "e",   # segol
    PATAH: "a",
    QAMATS: "oo",    # komets -> class 12/13 [u]
    HOLAM: "oy",     # cholam -> class 42/44 [\u0254\u026a]
    QUBUTS: "u",     # shuruk / kubuts -> class 51/52 [i] (near-exceptionless)
    "\u05c7": "oo",  # qamats qatan -> class 12/13 [u]
}

# Character class matching any Hebrew point/accent, for diacritic-tolerant regexes.
_MARKS_CLASS = "[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7]*"


def _tolerant(word: str) -> str:
    """Regex source matching ``word`` with optional diacritics after each letter."""
    return "".join(re.escape(c) + _MARKS_CLASS for c in _strip_points(word))


def _split_units(word: str) -> list[tuple[str, str]]:
    """Split into (base character, attached combining marks) pairs."""
    units: list[list[str]] = []
    for ch in unicodedata.normalize("NFD", word):
        if unicodedata.category(ch) == "Mn":
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(base, marks) for base, marks in units]


def _vowel_point(marks: str) -> str:
    """The vowel point among ``marks``, or '' if there is none."""
    for mark in marks:
        if mark in _POINT_TO_LATIN:
            return mark
    return ""


# =====================================================================
# STAGE 1: ORTHOGRAPHY & LEXICON (Loshn-Koydesh -> phonetic respelling)
# =====================================================================
_LOSHN_KOYDESH = {
    # --- Verified against a native Boro Park/Williamsburg speaker, 2026-08-05.
    # These are respellings into Yiddish orthography, not IPA: the single engine
    # then reads them with the ordinary Germanic rules. Loshn-koydesh cannot be
    # derived from the diacritics -- סְפָרִים and מִשְׁפָּחָה carry the identical
    # komets on the identical pe and take different vowels -- so it has to be a
    # lexicon, and this is where his answers live.
    "שבת": "שאַבעס",        # shabes -- was שאָבעס (komets), but שַׁבָּת has a
                            # PATACH: "if the engine says shubes it is treating
                            # that patach like a komets". 2,115 uses.
    "שבתים": "שאַבאָסים",   # shabosim
    "מעשה": "מײַסע",      # mˈaːsə -- "mayse is Polish/Russian; maase is pure
                            # Hungarian/Williamsburg Heimish". The vowel is the
                            # single long aː (v3 §6 pasekh+guttural), which the
                            # engine spells ײַ; the old מאַאַסע read as two a's.
    "גמרא": "געמאָרע",      # gemure -- the gm cluster is too harsh to swallow
    "פרשת": "פּאַרשעס",     # parshes mishputim -- final vowel reduces
    # CONSTRUCT STATE REDUCTION. Standalone בית is bais, but joined to a noun
    # the /aɪ/ reduces to /ɪ/ or /ɛ/ -- bis-medresh, bes-din, bes-oylem. Longest
    # key wins in _LK_PATTERN, so the compounds are matched before bare בית.
    "בית המדרש": "ביסמעדרעש",
    "בית מדרש": "ביסמעדרעש",
    "בית דין": "בעסדין",
    "בית עולם": "בעסוילעם",
    "בית הכנסת": "בעסאַקנעסעס",
    "בית": "בייס",          # standalone: bais (plain yy -> aɪ; ײַ would
                            # flatten to aː, which is the Germanic rule)
    "עולם": "אוילעם",       # oylem
    "רבי": "רעבע",          # rebbe (a Hasidic leader); "reb" before a name
    "פסוק": "פּאָסיק",      # pusik -- short u. (poysek = a halachic authority)
    "יצחק": "ייִצכאָק",     # yitskhuk -- proper names keep the full vowel
    "תשובה": "טשיווע",      # tsheeve -- deep dialect, the u becomes ee
    "שלום": "שאָלעם",       # shoolem -- LONG oo, see the note on _LATIN_TO_IPA
    "ברכה": "בראָכע",       # bruche -- short u
    "נשמה": "נעשאָמע",      # neshume -- the n-sh cluster keeps a short e
    "ספרים": "ספֿאָרים",    # sfoorim -- LONG oo
    "חסידים": "כאַסידעם",   # khasidem
    "כבוד": "קאָוועד",      # kuved -- short u
    "חכמה": "כאָכמע",       # khukhme -- short u
    "מנהג": "מינהעג",       # minheg
    "משפחה": "מישפּאָכע",   # mishpuche -- short u
    "את": "עס",             # es
    "מצוה": "מיצווע",       # mitsve
    "שמחה": "סימכע",        # simkhe
    "תפלה": "טפֿילע",       # tfile
    "חיים": "כײים",         # Khayim -- loshn-koydesh KEEPS the sharp ay
                            # diphthong; only Germanic words flatten it to aː
                            # (הײַנט haant). Same for דיין dayen, מקיים mkayem.
    # === Religion, Holidays, and Time ===
    "יום-טובֿ": "יאנטעוו",  # unpointed on purpose: yontef is class 41 [ɔ]
                            # (§7.3), and a komets would now read [u]. The
                            # respelling is picked up by _WORD_LATIN.
    "יום-טובֿים": "יאָנטויווים",
    "פּסח": "פּייסעך",
    "חנוכּה": "כאַניקע",
    "פורים": "פּורים",
    "ראש-השנה": "ראָשעשאָנע",
    "יום-כּיפּור": "יאָנקיפּער",
    "סוכּות": "סוקעס",
    "שבועות": "שוווּעס",
    "תּישעה-באָב": "טישעבאָוו",
    "חודש": "כוידעש",
    "חדשים": "כאַדאָשים",
    "אדר": "אָדער",
    "תּמיד": "טאָמיד",
    "זמן": "זמאַן",
    "שעה": "שאָ",
    "שעות": "שאָעס",

    # === Family, People, and Roles ===
    "משפּחה": "מישפּאָכע",
    "חתן": "כאָסן",
    "כּלה": "קאַלע",
    "מחותּן": "מעכוטן",
    "מחותּנים": "מעכוטאָנים",
    "חבֿר": "כאַווער",
    "חבֿרים": "כאַוויירים",
    "בעל-הבית": "באַלעבאָס",
    "בעלת-הבית": "באַלעבאָסטע",
    "רב": "ראָוו",
    "רבנים": "ראַבאָנים",
    "רבי": "רעבע",
    "צדיק": "צאַדיק",
    "תּלמיד": "טאַלמיד",
    "חכם": "כאָכעם",
    "חכמים": "כאַכאָמים",
    "גנבֿ": "גאַנעוו",
    "גנבֿים": "גאַנאָווים",
    "שדכן": "שאַדכן",
    "שוחט": "שויכעט",
    "קצב": "קאַצעוו",
    "רוצח": "ראָטסייעך",
    "נבֿיא": "נאָווי",
    "עולם": "אוילעם",

    # === Jewish Life, Texts, and Practice ===
    "תּורה": "טוירע",        # toyre (was טויערע -> tˈɔɪərə, a spurious schwa)
    "ספֿר": "סייפער",
    "ספֿרים": "ספֿאָרים",
    "מעשׂה": "מײַסע",      # mˈaːsə, not mayse -- see the note above
    "מעשׂיות": "מײַסעס",
    "פּירוש": "פּיירעש",
    "הגדה": "האַגאָדע",
    "תּפֿילה": "טפֿילע",
    "ברכה": "בראָכע",
    "ברכות": "בראָכעס",
    "מצווה": "מיצווע",
    "עבֿירה": "אַוויירע",
    "חתונה": "כאַסענע",
    "לוויה": "לעווײַע",
    "ברית": "בריס",
    "בית-מדרש": "בעסמעדרעש",
    "כּשר": "קאָשער",
    "טריף": "טרייף",
    "משיח": "מאָשיעך",
    "גלות": "גאָלעס",
    "נשמה": "נעשאָמע",

    # === Concepts, Attributes, and Objects ===
    "אמת": "עמעס",
    "שׂכל": "סייכל",
    "חכמה": "כאָכמע",
    "פּנים": "פּאָנעם",
    "חן": "כיין",
    "סוד": "סאָד",
    "סודות": "סוידעס",
    "קול": "קאָל",
    "קולות": "קוילעס",
    "חלום": "כאָלעם",
    "מזל": "מאַזל",
    "סכּנה": "סאַקאָנע",
    "טעות": "טאָעס",
    "כּוח": "קויעך",
    "מלחמה": "מילכאָמע",
    "שלום": "שאָלעם",
    "חוצפּה": "כוטספּע",
    "טעם": "טאַם",
    "כּעס": "קאַס",
    "ספֿק": "סאָפֿעק",
    "תּכלית": "טאַכלעס",
    "הצלחה": "האַצלאָכע",
    "פּרנסה": "פּאַרנאָסע",
    "חורבן": "כורבן",
    "צורה": "צורע",
    "טבֿע": "טעווע",
    "רחמנות": "ראַכמאָנעס",

    # === Common Phrases ===
    "אפֿילו": "אַפֿילע",
    "ודאַי": "וואַדע",
    "דווקא": "דאַווקע",
    "כּמעט": "קימאַט",
    "אפֿשר": "עפֿשער",
    "למשל": "לעמאָשל",
    "פּשוט": "פּושעט",
    "ממש": "מאַמעש",
    "תּיכּף": "טייקעף",
    "סך-הכּל": "סאַכאַקל",
    "מזל-טובֿ": "מאַזלטאָוו",

    # === English loanwords ===
    "שאַקי": "שייקי",
    "שאקי": "שייקי",

    # === §5 / §6.2 frozen merged-LK forms (spec romanization in the comment) ===
    # These are listed verbatim in the spec and are NOT derivable from the nikud
    # table: the post-tonic syllable reduces to e where the raw rules keep u/i.
    "קידוש": "קידעש",       # §5 shuruk row / §6.2 KIdesh
    "חמש": "כימעש",         # §5 kubuts row chimesh (חֻמָּשׁ; the komets is
                            # post-tonic and reduces, so it is not [u] here)
    "עדות": "אײדעס",        # §5 tsere row aydes
    "שידוך": "שידעך",       # §6.2 SHIdech
    "תענית": "טאָנעס",      # §7.3 merged tunes (the WH reading is taanis)
    "מתיבתא": "מעסיפֿטע",   # §4.2 mesivta -> mesifte
    "מסיבתא": "מעסיפֿטע",
    "בוחר": "באָכער",       # §6.2 BUcher
    "בחורים": "באַכירעם",   # §6.2 baCHIrem (plural stress shift)
    "בעלי-בתים": "באַלעבאַטים",  # §6.2 baleBAtim
    "ישראל": "ייִסראָעל",   # §5 chirik row Yisruel

    # === AUDIT 2026-08-07: top OOV-LK types by corpus frequency ===============
    # §6.3 now withholds an unpointed LK word that no lexicon knows, which is
    # right but costs coverage, and these were the most expensive individual
    # losses in the quarantine and OOV-LK logs. Each is a settled, unambiguous
    # merged-LK reading, so it is cheaper to list it than to withhold it.
    "מלך": "מיילעך",        # §11.14 maylech (merged), vs the WH mɛlɛx
    "שם": "שעם",            # shem -- 945 tokens, was the vowel-less ʃm
    "מדרש": "מעדרעש",       # medresh -- matches bis-mˈɛdrəʃ (§8)
    "שלמה": "שלוימע",       # Shloyme
    "לברכה": "ליווראָכע",   # livruche (זכרונו לברכה) -- was lbrxh, 1,158 tokens
    "חלילה": "כאַלילע",     # kholile -- was xlilh, 420 tokens
    "עבודה": "אַוווידע",    # avoyde -- was ˈɛbidh, 555 tokens
    "אמונה": "עמונע",       # emine -- was ˈaminh
    "שירה": "שירע",         # shire -- was ʃirh
    "רגע": "רעגע",          # rege -- was the MED-confidence rɡə
    # NOT listed: מנין. The Yiddish reading is mˈinjən, and the respelling layer
    # cannot spell a consonantal yud in that position (it would come out
    # mˈiniən, a syllable too many). It stays on the OOV-LK list for Chezky.
}

# Merged-LK entries above are the CASUAL reading (§7.3). When the input arrives
# explicitly pointed it is Whole-Hebrew — a quoted posuk, a citation — and the
# §5 nikud table is the better evidence, so the merged respelling is skipped and
# the points are read directly: מלך is maylekh unpointed but מֶלֶךְ is melekh.
#
# עולם was removed from this set after the end-to-end pointing probe
# (data/verification/e2e_pointing.md): with model-supplied nikud the hatch fired
# 5 times and 3 were regressions — עולם הבא and פּאַר דעם עולם ("the crowd") are
# merged-Yiddish [ˈɔjləm] even when the pointing model happens to point them.
# A quoted-פסוק עוֹלָם loses its [u], which is the cheaper error.
_WH_WHEN_POINTED = frozenset({"מלך"})

# The LK lexicon is keyed on the POINT-STRIPPED form, which is the right default
# (a word must match however it arrives) but cannot express a pair that is
# distinguished by the points alone. קִדּוּשׁ 'kidesh' (§5 shuruk row) and
# קָדוֹשׁ 'kudoysh' are the same six letters unpointed, so this one entry is
# matched on the pointed spelling directly, before the main swap. The unpointed
# spelling קידוש carries its own yud and is an ordinary lexicon entry.
_LK_POINTED: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<![\w֐-׿])קִדּ?וּ?שׁ?(?![\w֐-׿])"),
     "קידעש"),
]

# Keyed on the unpointed form so a word matches whether it arrives unpointed,
# YIVO-pointed or fully pointed by the nikud model.
_LK_BARE: dict[str, str] = {_strip_points(_k): _v for _k, _v in _LOSHN_KOYDESH.items()}

_LK_PATTERN = re.compile(
    r"(?<![\w\u0590-\u05FF])("
    + "|".join(_tolerant(k) for k in sorted(_LK_BARE, key=len, reverse=True))
    + r")(?![\w\u0590-\u05FF])"
)


def _lk_replace(match: re.Match) -> str:
    raw = match.group(1)
    bare = _strip_points(raw)
    if bare in _WH_WHEN_POINTED and _vowel_point(raw):
        return raw
    return _LK_BARE.get(bare, raw)

# =====================================================================
# STAGE 1.5: HIGH-FREQUENCY WORD LEXICON (unpointed spelling -> Latin base)
# =====================================================================
_WORD_LATIN: dict[str, str] = {
    # --- mid-word א, class 12/13 -> engine 'oo' = [u] (spec §2, §3) ---
    # The א ambiguity is lexical only (spec §3): the SAME grapheme is class
    # 12/13 [u] here and class 41 [ɔ] in the block further down. Untagged
    # mid-word א keeps the engine's old default 'a' -- the spec says growth of
    # this list, not a rule, is the fix.
    "וואס": "voos",
    "דאס": "doos",
    "דא": "doo",
    "נאר": "nor",           # gold nɔr
    "נאך": "nookh",
    "נאכדעם": "nookhdeem",  # v3 §5 raising before m: nuxdˈejm
    "נאכער": "nookher",
    "נאכאמאל": "nookhamool",
    "זאל": "zool",
    "זאלן": "zooln",
    "זאלסט": "zoolst",
    "זאג": "zoog",
    "זאגן": "zoogn",
    "זאגט": "zoogt",
    "זאגסט": "zoogst",
    "געזאגט": "gezoogt",
    "אנזאגן": "oonzoogn",
    "יא": "yoo",
    "יאר": "yoor",
    "יארן": "yoorn",
    # האב / האבן / האסט are class 41 [ɔ], NOT 12/13 [u]: the spec's §4.8
    # cliticization example is spelled "hob ikh" -> [ˈhɔbəx], and the paradigm
    # mate האט is already hot. They were the only members of the paradigm on
    # reading A of the ambiguous א (spec §3).
    "האב": "hob",
    "האבן": "hobn",
    "האסט": "host",
    "וואלט": "volt",        # gold vɔlt
    "וואלטן": "voltn",
    "מאל": "mool",
    "אמאל": "amool",
    "קיינמאל": "keynmool",
    "געווארן": "gevorn",    # gold ɡəvˈɔrn
    "פארוואס": "farvoos",
    "שלאפן": "shloofn",
    "שלאף": "shloof",
    "גארנישט": "goornisht",
    "לאמיר": "lomir",       # gold lˈɔmir
    "אווענט": "oovnt",
    "אוונט": "oovnt",       # 12/13 uvnt -- the unpointed Hasidic spelling, which
    "אוונטן": "oovntn",     # is what actually occurs in text
    "טאג": "toog",
    "פרייטאג": "fraytoog",
    "דאך": "dokh",          # gold dɔx

    # --- e before ר, Satmar/Hungarian (Williamsburg, Kiryas Joel) ---
    # Deliberately NOT a rule. The native reviewer was explicit: "e before r has
    # split into three different sounds depending on the specific word and the
    # speaker's family background -- ee (shveer, veert, heern), a (barg, vark),
    # ay (zayer). Don't make it a global rule, it will break too many words.
    # Put these specific high-frequency words into your exception dictionary."
    # So the default stays /ɛ/ and only these are listed.
    "שווער": "shvir",       # shvir -- heavy / father-in-law
    "שווערער": "shvirer",   # comparatives keep the stem vowel
    "הערן": "hirn",         # hirn
    "געהערט": "gehirt",     # ɡəhˈirt
    "דערהערט": "derhirt",
    "ווערט": "virt",        # virt -- becomes / worth
    "ווערן": "virn",
    # v3 §5 removed ערד ird / בערג barg / ווערק vark: the ir-list is closed
    # (ʃvir, virn, virt, hirn, lirnen, ɡəhˈirt) and every other r+cluster word
    # takes the ɛr default, which the bare rule path already produces.

    # --- class 25 [ej], AWAY FROM r only (v3 §5) -------------------------
    # v3: "ejr never occurs before r" -- the default before r is ɛr, so מער mɛr
    # and ווער vɛr are NOT listed here any more, and the ee-before-r paradigm
    # expansion (kleer/shteer/keer/treer) is gone entirely. What survives is the
    # gold-verified ej list, none of which has r after the vowel.
    "זען": "zeen",          # zejn 'see'
    "זעהן": "zeen",         # ה silent after a vowel (v3 §5)
    "זעה": "zee",           # zej
    "זעהט": "zeet",         # zejt
    "זעט": "zeet",
    "געזען": "gezeen",      # ɡəzˈejn
    "וועג": "veeg",         # vejɡ -- no final devoicing in v3
    "וועגן": "veegn",       # vejɡn
    "וועגס": "veegs",
    "טעג": "teeg",          # tejɡ
    "מעג": "meeg",          # mejɡ
    "געבן": "geebn",        # ɡejbn
    "בעטן": "beetn",        # bejtn
    "יעדן": "yeedn",        # jejdn
    "ברענגען": "breengen",  # brˈejnɡən
    "קעגן": "keegn",        # kˈejɡn
    "לערנען": "lirnen",     # v3 §5 ir-list: lˈirnən (was leernen)
    "לערנט": "lirnt",
    "לערנן": "lirnen",
    "געלערנט": "gelirnt",
    "לעבן": "leebn",        # lejbn 'live / life'
    "לעבט": "leebt",
    "מעגליך": "meeglekh",   # mˈejɡləx
    "שפעטער": "shpeeter",   # ʃpˈejtər
    "געווען": "geveen",     # ɡəvˈejn
    # v3 §5 raising before m (lexical, not a rule)
    "דעם": "deem",          # dejm
    "אים": "eem",           # ejm
    "איהם": "im",       # gold_v3 primary is im for this spelling
    "עם": "eem",

    # --- class 41: short o, never lengthened -> engine 'o' = [ɔ] (spec §2) ---
    "אבער": "ober",
    "אדער": "oder",
    "דארט": "dort",
    "דארטן": "dortn",
    "וואך": "vokh",
    "וואכן": "vokhn",
    "האט": "hot",
    "געוואלט": "gevolt",
    "אראפ": "aroop",   # gold arˈup
    "פארט": "fort",
    "אפ": "op",
    "אפט": "oft",
    "מארגן": "morgn",
    "קאפ": "kop",
    "גאט": "got",
    "אקס": "oks",
    "נאז": "noz",
    "גראב": "grob",
    "אוודאי": "avade",
    "גאר": "goor",          # gold ɡur
    "ווארט": "vort",        # gold vɔrt
    "ווערטער": "verter",
    "נאמען": "noomen",      # gold nˈumən
    "אפאר": "apoor",        # gold apˈur -- initial פ is /p/ here
    "אביסל": "abisl",

    # --- gold_v3 corrections to the legacy list ---------------------------
    # The gold module overrides these at runtime whatever they say, but
    # hebrew_to_latin and the rule path are used on their own (nikud tooling,
    # OOV triage, compounds built off these stems), so the Latin layer is
    # brought into line with the gold primary rather than left contradicting it.
    "ביים": "bam",          # gold bam (baːm is the listed alternate)
    "יוסף": "yosef",        # gold jˈɔsəf -- class 41, not the WH jˈɔjsəf
    "יום": "yoym",          # gold jɔjm
    "על": "al",             # gold al -- the LK preposition, not Germanic ɛl
    "דורך": "dorekh",       # gold dˈɔrəx -- ɔ + an epenthetic ə, not dirx
    "עפעס": "epes",         # gold ˈɛpəs -- the פ is /p/ here (§4 p-list)

    # --- class 54: MHG ū -> engine 'ou' = [oʊ], split off from oy (42/44).
    # The oykh(44)/boukh(54) and broyt(44)/hout(54) pairs are the spec's
    # top-listed transcription-error source, so these are lexical, not a rule.
    "הויז": "houz",         # hous -> [hoʊs] after devoicing
    "הויזער": "houzer",
    "הויט": "hout",
    "בויך": "boukh",
    "מויל": "moul",
    "טויזנט": "touznt",
    "טויזנטער": "touznter",
    "זויער": "zouer",
    "בויען": "bouen",
    "געבויט": "gebout",

    # --- יי = 'ay' ---
    "זיין": "zayn",
    "ביי": "bay",
    "ביידע": "bayde",
    "ביידן": "baydn",
    "דריי": "dray",
    "היינט": "haynt",
    "ווייל": "vayl",
    "צייט": "tsayt",
    "צייטן": "tsaytn",
    "אריין": "arayn",
    "מיין": "mayn",
    "מיינע": "mayne",
    "דיין": "dayn",
    "דיינע": "dayne",
    "זיינע": "zayne",
    "זייט": "zayt",
    "סיי": "say",
    "מייל": "mayl",
    "ווייטער": "vayter",
    "ניי": "nay",
    "נייע": "naye",
    "ביישפיל": "bayshpil",
    "פיין": "fayn",
    "וויין": "vayn",
    "גלייך": "glaykh",
    "ניין": "nayn",         # §8/§11.6 naan 'no' (34), vs ayn 'one' (24)
    "שרייבן": "shraybn",
    "שרייבט": "shraybt",
    "געשריבן": "geshribn",
    "ווייט": "vayt",
    "ווייטע": "vayte",
    # Listed verbatim in the spec's §2 class-34 column (vaab, blaabn) or written
    # with pasekh-tsvey-yudn in YIVO, i.e. StY ay -> Hasidic aa. They were
    # falling through to the default ay [aɪ] reading while their sister words
    # (זייט, ווייל, דיין, שרייבט) were already pinned.
    "ווייב": "vayb",
    "ווייבל": "vaybl",
    "ווייבער": "vayber",
    "בלייבן": "blaybn",
    "בלייבט": "blaybt",
    "בלייבסט": "blaybst",
    "געבליבן": "geblibn",
    "גלייבן": "glaybn",
    "גלייבט": "glaybt",
    "געגלייבט": "geglaybt",
    "הייזער": "hayzer",

    # --- v3 initial yud: gold has jid / jidn / jˈidiʃə for the א-spelled forms,
    # alongside the productive word-initial יי -> ji rule (ייד jid) below ---
    "איד": "yid",
    "אידן": "yidn",
    "אידיש": "yidish",
    "אידישע": "yidishe",
    "אידישער": "yidisher",
    "אידישן": "yidishn",
    "אידישקייט": "yidishkayt",

    # --- initial פ = 'p' & High-Freq Loanwords ---
    "פרעזידענט": "prezident",
    "פראצענט": "protsent",
    "פראבלעם": "problem",
    "פראבלעמען": "problemen",
    "פלאץ": "plats",
    "פונקט": "punkt",
    "פרעסע": "prese",
    "פאליטיק": "politik",
    "פאליטישע": "politishe",
    "פארטיי": "partey",
    "פרובירן": "prubirn",
    "פרובירט": "prubirt",
    "פלוצלינג": "plutsling",  # TTS audio corroborates /p/ (heard pliʦli-)
    "פראגראם": "program",
    "מאדעל": "model",
    "פאליציי": "politsey",
    "פלאן": "plan",
    "פאפיר": "papir",
    "פארק": "park",
    "פאסט": "post",

    # --- loanwords / names ---
    "טראמפ": "tromp",
    "אבאמא": "obama",
    "קאפי": "kofi",
    "שאפ": "shop",
    "אודיאו": "odyo",
    "אמעריקע": "amerike",
    "אמעריקאנער": "amerikaner",
    "דעמאקראטן": "demokratn",
    "דעמאקראטיש": "demokratish",
    "רעפובליקאנער": "republikaner",
    "קאנגרעס": "kongres",
    "יארק": "york",
    "דאלאר": "dolar",
    "מיליאן": "milyon",
    "ביליאן": "bilyon",
    "סאו": "so",
    "אקעי": "okey",
    "ניו": "nyu",

    # --- Hebrew-origin function words & specific Hasidic contractions ---
    "בערך": "berekh",
    "בכלל": "bikhlal",
    "חלק": "kheylek",
    "מסביר": "masbir",
    "כדי": "kedey",
    "בעצם": "betsem",       # was "beetsem"; "ee" is now the class-25 digraph
    "ממילא": "memeyle",
    "מסתמא": "mistame",
    "אגב": "agev",
    "בקיצור": "bekitser",
    "חוץ": "khuts",
    "שייך": "shayekh",
    "ענין": "inyen",
    "עניינים": "inyonim",
    "פשוטע": "pushete",
    "לבנה": "levone",
    "כח": "koyekh",
    "כוחות": "koykhes",
    "כלל": "klal",
    "כל": "kool",       # §5 kometz -> u: kul
    "הכל": "hakl",
    "סך": "sakh",
    "מח": "moyekh",
    "בדרך": "bederekh",
    "דרך": "derekh",
    "מזרח": "mizrekh",
    "מערב": "mayrev",
    "משה": "moyshe",
    "במשך": "bemeshekh",
    "משך": "meshekh",
    "מצב": "matsev",
    "חברה": "khevre",
    "שטח": "shetekh",
    "שטחים": "shtokhim",
    "גוף": "guf",
    "רוב": "rov",
    "מהלך": "mahalekh",
    "נפש": "nefesh",
    "סוף": "sof",
    "צד": "tsad",
    "צדדים": "tsdodim",
    "שכל": "seykhl",
    "תוך": "tokh",
    "פרשה": "parshe",
    "סדר": "seyder",
    "הסבר": "hesber",
    "משל": "moshl",
    "לכל": "lekool",
    "אוו": "ov",
    "וויבאלד": "vibald",

    # --- Hasidic pronunciation of common words ---
    "אויך": "oukh",         # v3 §4 oʊ-list
    "אויף": "ouf",          # v3 §9: oʊf standalone, afn fused
    "אויפן": "afn",

    # --- §8 frozen loshn-koydesh names / months (spec: store, don't derive) ---
    "יעקב": "yankev",       # Yankef after §4.1 devoicing
    "חשוון": "khezhvn",     # Chezhvn -- the zh is frozen in the month name
    "ישיבה": "yeshive",
    "סייפער": "seefer",     # ← _LOSHN_KOYDESH respelling of ספר. §6.2 SEYfer is
                            # class 25 [ej]; plain tsvey-yudn would read [aɪ],
                            # and the engine's class-25 label is "ee".
    "יאנטעוו": "yontev",    # ← _LOSHN_KOYDESH respelling of יום-טוב; class 41
                            # [ɔ], so it is written unpointed and routed here
                            # rather than through a komets (which is now [u]).
}


# --- v3 §5: the class-25-before-r paradigm expansion is GONE ----------------
# It generated klejrn / ʃtejrn / kejrn / trejrn from an "ee before r" reading of
# the ער grapheme. v3 settles that grapheme the other way: "ejr never occurs
# before r", default ɛr. The dict is kept empty so the expansion loop below and
# g2p_fingerprint keep their shape without reintroducing the forms.
_CLASS25_PARADIGM: dict[str, str] = {}

# Yiddish inflectional endings, as (Hebrew spelling, Latin spelling).
_INFLECTIONS: list[tuple[str, str]] = [
    ("ן", "n"), ("ט", "t"), ("סט", "st"), ("נס", "ns"), ("ס", "s"),
    ("ע", "e"), ("טע", "te"), ("נדיק", "ndik"), ("עדיק", "edik"),
    ("נען", "nen"), ("נט", "nt"), ("נסט", "nst"),
]

for _stem, _latin in _CLASS25_PARADIGM.items():
    for _heb_end, _lat_end in [("", "")] + _INFLECTIONS:
        _WORD_LATIN.setdefault(_stem + _heb_end, _latin + _lat_end)


# Sub-word substitutions for Loshn-Koydesh bases that take Yiddish morphology
_STEM_SUBS: list[tuple[str, str]] = [
    # §6.2 CHUsid -> chSIdim. The plural חסידים has its own whole-word entry;
    # the singular and the derived adjective (חסידישע) had none at all, so the
    # unpointed spelling gave no stem vowel: khsid -> [xsit].
    ("חסיד", "כאָסיד"),
    # PATACH, matching the audio-verified _LOSHN_KOYDESH entry שבת -> שאַבעס.
    # It was a komets here, which now reads [u] and gave shubesdik; the reviewer
    # was explicit that "shubes" means the engine is misreading that patach.
    ("שבת", "שאַבעס"),
    ("אמת", "עמעס"),
    ("חן", "כיין"),
    ("פסק", "פּאַסק"),
    ("פטר", "פּאַטער"),
    ("הרג", "האַרג"),
]

# The same list as VOWEL-tolerant patterns. A plain ``stem in core`` substring
# test sees only the UNPOINTED spelling, so אֱמֶתְדִיג and אמתדיג -- the same
# word -- took different paths: the unpointed one got _STEM_SUBS' audio-matched
# base (ˈɛməzdiɡ, with the ת/ד voicing assimilation) and the pointed one fell
# through to the stemmer's root+suffix concatenation (*ˈɛməsdiɡ). Matching
# through the vowel points keeps one lexeme on one path.
#
# Through the POINTS, not through a letter's identity, and only for bases of
# >= 3 letters:
#   * rafe and the sin dot are excluded outright -- they pin the fricative and
#     make שׂ, so a span that crosses one is a different word (נפֿטר niftar is
#     not a פטר word).
#   * a dagesh is allowed everywhere EXCEPT on פ. Elsewhere it agrees with the
#     base's own reading (שַׁבָּת's בּ is the /b/ of שאַבעס), but פּ/פ is §4's
#     weakest contrast and every collision measured on the corpus is the same
#     one: a Germanic separable prefix that ends in פּ, whose letters then read
#     as an LK base (אַראָפּ+טראַכטן matching פטר, אָפּ+פּסקענען and
#     וויטעפּסקער matching פסק).
#   * the length floor is the same collision argument as _MIN_STEM_ROOT: the one
#     two-letter base, חן, is a frequent letter pair inside pointed Hebrew verbs
#     (וַיִּחַן vaˈixan) where the points spell something else entirely, so it
#     keeps the exact test.
_STEM_MARKS = "[\u0591-\u05bd\u05c1\u05c4\u05c5\u05c7]*"
_STEM_MARKS_NO_DAGESH = "[\u0591-\u05bb\u05bd\u05c1\u05c4\u05c5\u05c7]*"


def _stem_pattern(stem: str) -> re.Pattern:
    if len(stem) < 3:
        return re.compile(re.escape(stem))
    return re.compile("".join(
        re.escape(c) + (_STEM_MARKS_NO_DAGESH if c == "פ" else _STEM_MARKS)
        for c in stem))


_STEM_SUB_RE: list[tuple[re.Pattern, str]] = [
    (_stem_pattern(stem), repl) for stem, repl in _STEM_SUBS
]


def _apply_stem_subs(core: str) -> str:
    """Substitute every _STEM_SUBS base found in ``core``, pointed or not."""
    for pat, repl in _STEM_SUB_RE:
        core = pat.sub(repl, core)
    return core


def _has_stem_sub(core: str) -> bool:
    """Does ``core`` contain a _STEM_SUBS base, pointed or not?"""
    return any(pat.search(core) for pat, _ in _STEM_SUB_RE)

# =====================================================================
# STAGE 2: BASE TRANSLITERATION (context-aware, diacritic-driven)
#
# Words are parsed into (letter, marks) units rather than matched as raw
# substrings, because a letter's marks can arrive in either order -- NFC sorts
# them by combining class, so פּ with a vowel becomes פ + vowel + dagesh and the
# dagesh is no longer adjacent to the letter.
# =====================================================================
_SINGLE_MAP: dict[str, str] = {
    "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ז": "z", "ח": "kh", "ט": "t",
    "כ": "kh", "ך": "kh", "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n",
    "ס": "s", "ע": "e", "פ": "f", "ף": "f", "צ": "ts", "ץ": "ts", "ק": "k",
    "ר": "r", "ש": "sh", "ת": "s",
}

_LATIN_VOWELS = set("aeiou")
_NUCLEUS_START = set("ויױײ")

# Multi-letter consonant clusters, longest first. The vav clusters are spelling
# conventions for /vu/ that would otherwise be read as separate segments.
_CLUSTERS: list[tuple[str, str]] = [
    ("דזש", "dzh"),
    ("דז", "dz"),
    ("זש", "zh"),
    ("טש", "tsh"),
]

# Letters whose stop/fricative value is selected by dagesh vs rafe, as
# (with dagesh, with rafe, unpointed). Yiddish reads a bare ב as /b/ but a bare
# כ/פ/ת as the fricative, which is why the unpointed value is listed separately.
_DAGESH_PAIRS: dict[str, tuple[str, str, str]] = {
    "ב": ("b", "v", "b"),
    "כ": ("k", "kh", "kh"),
    "ך": ("k", "kh", "kh"),
    "פ": ("p", "f", "f"),   # bare פ is /f/, but see _P_BEFORE_LIQUID
    "ף": ("p", "f", "f"),
    "ת": ("t", "s", "s"),
}


def _consonant(ch: str, marks: str) -> str:
    """Latin value of a consonant, honouring dagesh / rafe / sin dot."""
    if ch == "ש":
        return "s" if SIN_DOT in marks else "sh"
    pair = _DAGESH_PAIRS.get(ch)
    if pair is not None:
        hard, soft, bare = pair
        if RAFE in marks:
            return soft
        if DAGESH in marks:
            return hard
        return bare
    return _SINGLE_MAP.get(ch, "")


def _is_feminine_ending(units: list[tuple[str, str]], i: int) -> bool:
    """Whether ``i`` is the vowel of a word-final Hebrew feminine -ה, read as /e/."""
    ch, marks = (units[i] if 0 <= i < len(units) else ("", ""))
    return (
        ch in ("א", "ע", "ו")
        and _vowel_point(marks) in (QAMATS, PATAH)
        and i + 2 == len(units)
        and letter_at(units, i + 1) == "ה"
        and not units[i + 1][1]
    )


def _spells_own_vowel(units: list[tuple[str, str]], i: int) -> bool:
    """Whether the nucleus at ``i`` supplies a vowel without help from a point.

    Used to decide if a preceding consonant's vowel point is redundant. A bare
    single י/ו does not qualify: it is the matres lectionis that spells the
    consonant's own point (בֵית -> beys), not an independent vowel.
    """
    ch, marks = (units[i] if 0 <= i < len(units) else ("", ""))
    if not ch:
        return False
    if ch in ("א", "ע"):
        # A feminine -ה vowel spells the ending, not the stem vowel, so a
        # preceding consonant's point still has to be realised (השפּעה -> hashpoe).
        return bool(_vowel_point(marks)) and not _is_feminine_ending(units, i)
    if ch in ("ײ", "ױ"):
        return True
    if ch == "י":
        return letter_at(units, i + 1) == "י" or bool(_vowel_point(marks))
    if ch == "ו":
        if letter_at(units, i + 1) == "ו":
            return False  # consonantal /v/
        # Only holam / shuruk / kubuts make the vav a vowel of its own. Any
        # other point makes it consonantal /v/ carrying that vowel (see the
        # matching branch in _nucleus), so the PRECEDING consonant's point is
        # still unspelled and must be realised: דָּוִד is Duvid, not *Dvid.
        if _vowel_point(marks) and _vowel_point(marks) not in (HOLAM, QUBUTS):
            return False
        return bool(marks) or letter_at(units, i + 1) == "י"
    return False


def _word_to_latin(word: str) -> str:
    """Transliterate one Hebrew-script word to Latin base with context rules."""
    units = _split_units(word)
    out: list[str] = []
    # The vowel point just realised, if any; a following bare י/ו is then the
    # matres lectionis spelling that same vowel and stays silent. Tracked
    # separately for consonants, because only there does a following bare א/ע
    # restate the vowel (Hasidic דֶער) rather than add one (טאָעס).
    prev_point = ""
    prev_consonant_point = ""
    i = 0
    n = len(units)

    def emitted_any() -> bool:
        return any(t for t in out)

    def last_char() -> str:
        for t in reversed(out):
            if t:
                return t[-1]
        return ""

    def letter(idx: int) -> str:
        return units[idx][0] if 0 <= idx < n else ""

    def marks_of(idx: int) -> str:
        return units[idx][1] if 0 <= idx < n else ""

    while i < n:
        ch, marks = units[i]

        if not _HEBREW_CHAR.match(ch):
            out.append(" " if ch == "-" else ch)
            prev_point = prev_consonant_point = ""
            i += 1
            continue

        bare_run = "".join(letter(j) for j in range(i, n))
        cluster = next((c for c in _CLUSTERS if bare_run.startswith(c[0])), None)
        if cluster is not None:
            out.append(cluster[1])
            prev_point = prev_consonant_point = ""
            i += len(cluster[0])
            continue

        # וו / װ are consonantal /v/; a dagesh on the second vav makes it /vu/.
        if ch == "ו" and letter(i + 1) == "ו":
            out.append("v")
            if DAGESH in marks_of(i + 1):
                out.append("u")
                prev_point, prev_consonant_point = DAGESH, ""
                i += 2
                continue
            point = _vowel_point(marks) or _vowel_point(marks_of(i + 1))
            i += 2
            prev_point = prev_consonant_point = ""
            if point and not _spells_own_vowel(units, i):
                out.append(_POINT_TO_LATIN[point])
                if _POINT_TO_LATIN[point]:
                    prev_point = prev_consonant_point = point
            elif point:
                prev_consonant_point = point  # pending: colors a bare digraph
            continue
        if ch == "װ":
            out.append("v")
            prev_point = prev_consonant_point = ""
            i += 1
            continue

        nucleus = _nucleus(
            units, i, emitted_any(), last_char(), prev_point, prev_consonant_point
        )
        if nucleus is not None:
            latin, size, point_used = nucleus
            # Vowel hiatus guard: two adjacent 'e' vowels (e.g. prefix ge- + root e- in
            # געעפנט, געענדיגט, געעסן, געענטפערט, or ge- + ey- in געאיילט) form separate
            # syllables (ge'efnt -> ɡəˈɛfnt), never the class-25 lengthened 'ee' -> 'ej' digraph.
            if latin and last_char() == "e" and latin.startswith("e"):
                out.append("'")
            out.append(latin)
            prev_point = point_used if latin else ""
            prev_consonant_point = ""
            i += size
            continue

        latin_c = _consonant(ch, marks)
        # §5 digraph table (שפ -> ʃp) and §4 ("פ unpointed ... after ש always
        # p"). Not expressible in _CLUSTERS, which is matched before the dagesh /
        # rafe logic; here an explicit rafe (שפֿ) still wins. Without this every
        # שפ word outside the lexicon came out ʃf -- ʃfiln, ʃfɛt, ʃfrax,
        # ʃfrˈinɡən -- 485 emitted types / 2,625 tokens in the corpus.
        if (
            latin_c == "f"
            and RAFE not in marks
            and letter(i - 1) == "ש"
            and SIN_DOT not in marks_of(i - 1)
        ):
            latin_c = "p"
        # PHONEME-DELETING COLLISION GUARD. The Latin layer is a flat string, so
        # an "h" landing after s / z / k / t silently forms one of latin_to_ipa's
        # digraphs (sh -> ʃ, zh -> ʒ, kh -> x, tsh -> ʧ) and two phonemes become
        # one wrong phoneme: מזוזה -> mziʒ, אתה -> aʃ, עצה -> ɛʧ, תהילים ->
        # ʃˈilim, קהילה -> xilh. The apostrophe is not a phone anywhere in the
        # pipeline (latin_to_ipa skips it outright), so it is a free separator.
        if latin_c.startswith("h") and last_char() and last_char() in "sztk":
            out.append("'")
        # Hebrew soft bet: in pointed loshn-koydesh a bare ב after a pointed
        # consonant reads /v/ (צְבִי -> tsvi, לִבְרָכָה -> livrokhe). Germanic ב
        # sits next to vowel letters and stays /b/ (האָבְן, אָבֶער).
        if (
            latin_c == "b"
            and DAGESH not in marks
            and i > 0
            and units[i - 1][0] not in "אעיוײױ"
            and _HEBREW_CHAR.match(units[i - 1][0])
            and _vowel_point(units[i - 1][1])
        ):
            latin_c = "v"
        # Same rule, second environment (spec §5: "bet/vet ... per dagesh"): a ב
        # that carries a full vowel point of its own and is NOT followed by a
        # vowel letter restating it is pointed Whole-Hebrew, where bare bet is
        # /v/ -- אֲבֵלִים avaylim. Germanic Yiddish always writes the vowel letter
        # (אָבֶער, בֶעסער) or leaves the bet with a sheva (האָבְן), so neither of
        # those is touched.
        if (
            latin_c == "b"
            and DAGESH not in marks
            and _POINT_TO_LATIN.get(_vowel_point(marks))
            and letter(i + 1) not in "אעיוײױ"
        ):
            latin_c = "v"
        point = _vowel_point(marks)
        # Pasekh genuvah (furtive patah): word-final guttural (ח, ע, ה with mappiq)
        # carrying a patah after a non-a vowel emits the vowel BEFORE the consonant.
        # In Yiddish merged register it reduces to 'e' [ə] (e.g. רוּחַ -> riəx, כֹּחַ -> kˈɔjəx,
        # מַשְׁגִּיחַ -> maʃɡˈiəx, תַּפּוּחַ -> tapˈiəx, לוּחַ -> lˈiəx, מַפְתֵּחַ -> maftˈajəx).
        if (
            i == n - 1
            and ch in ("ח", "ע", "ה")
            and point in (PATAH, "ֲ")
            and emitted_any()
            and prev_point not in (PATAH, QAMATS, "ׇ", "ֲ")
            and (last_char() in _LATIN_VOWELS or last_char() == "y")
        ):
            out.append("e")
            out.append(latin_c)
            prev_point = prev_consonant_point = point
            i += 1
            continue

        out.append(latin_c)
        # A vowel point on a consonant is realised unless the next letter already
        # spells that vowel independently, which would double it up.
        prev_point = prev_consonant_point = ""
        if point and not _restates_point(units, i + 1, point):
            vowel = _POINT_TO_LATIN[point]
            # A Hebrew feminine -ה ending is reduced to /e/ in Yiddish, so
            # ברכה is brokhe and נשמה is neshome rather than -o.
            if (
                point in (QAMATS, PATAH)
                and letter(i + 1) == "ה"
                and not marks_of(i + 1)
                and i + 2 == n
            ):
                vowel = "e"
            out.append(vowel)
            if vowel:
                prev_point = prev_consonant_point = point
        elif point:
            # Vowel suppressed because the following letters spell it; keep it
            # visible so a bare digraph can read its quality (זַיין -> zayn,
            # וַוייל -> vayl).
            prev_consonant_point = point
        i += 1

    return "".join(out)


# Points whose vowel is conventionally spelled out with a following matres letter.
_MATRES_FOR = {"י": (HIRIQ, TSERE), "ו": (HOLAM, QUBUTS, DAGESH)}


def _restates_point(units: list[tuple[str, str]], idx: int, point: str) -> bool:
    """Whether the nucleus at ``idx`` merely re-spells the consonant's own point.

    MEASURED, DO NOT "FIX": requiring the two points to AGREE (so that a pointed
    א/ע with a different vowel counts as its own syllable, יִשְׂרָאֵל ->
    Yisruel rather than *isreyl) looks right and is wrong. The Hasidic pointing
    this engine consumes is not internally consistent about which of the pair
    carries which mark -- פָּאַר is komets+pasekh for one /a/, likewise דָאֹס,
    אַזָאַ, אָמָאַל, מוֹצָאֵי -- and the strict version inserted a spurious vowel
    in 15 of 468 corpus rows (par -> *puar). Whole-Hebrew words that genuinely
    need the extra syllable are lexicalised instead (ישראל).
    """
    return _spells_own_vowel(units, idx)


def _nucleus(
    units: list[tuple[str, str]],
    i: int,
    emitted: bool,
    prev_latin: str,
    prev_point: str,
    prev_consonant_point: str = "",
) -> tuple[str, int, str] | None:
    """Resolve a vowel nucleus at ``i`` into (latin, units consumed, point used)."""
    n = len(units)
    ch, marks = units[i]
    nxt, nxt_marks = (units[i + 1] if i + 1 < n else ("", ""))
    point = _vowel_point(marks)

    if ch == "ײ":
        if PATAH in marks or prev_consonant_point == PATAH:
            return ("ay", 1, PATAH)
        # v3 §4: word-initial יי is /ji/, not the aj digraph (ייד -> jid).
        # Anchored on i == 0: after a silent carrier א (אייביג) the digraph is
        # the ordinary nucleus, ˈajbiɡ.
        if i == 0:
            return ("yi", 1, HIRIQ)
        return ("ey", 1, "")
    if ch == "ױ":
        return ("oy", 1, "")

    if ch == "י":
        if nxt == "י":
            # tsvey yudn: a hiriq on either yud is ייִ /yi/, a pasekh marks /ay/
            if HIRIQ in marks or HIRIQ in nxt_marks:
                return ("yi", 2, HIRIQ)
            if PATAH in marks or PATAH in nxt_marks or prev_consonant_point == PATAH:
                return ("ay", 2, PATAH)
            # v3 §4: word-initial יי is /ji/ (ייד -> jid), not aj. Anchored on
            # i == 0 so אייביג (silent carrier א first) stays ˈajbiɡ.
            if i == 0:
                return ("yi", 2, HIRIQ)
            return ("ey", 2, "")
        if point:
            # A word-initial yud before a bare א/ע is consonantal, and the point
            # sitting on it spells the vowel of that letter rather than a nucleus
            # of its own (יָאר -> yor, not *oar). Pointing styles differ on which
            # of the two carries the mark; יאָר takes the other branch already.
            if not emitted and nxt in "אע" and not _vowel_point(nxt_marks):
                return ("y" + _POINT_TO_LATIN[point], 2, point)
            # Any other word-initial pointed yud is likewise consonantal: Hebrew
            # script cannot open a word with a vowel-yud (Yiddish writes אי for
            # that), so יִשְׂרָאֵל is Yisruel and not *Isruel (spec §5, chirik row).
            if not emitted:
                return ("y" + _POINT_TO_LATIN[point], 1, point)
            return (_POINT_TO_LATIN[point], 1, point)
        if prev_point in _MATRES_FOR["י"]:
            return ("", 1, "")  # matres lectionis
        if not emitted and nxt in "אעו":
            return ("y", 1, "")
        if prev_latin in _LATIN_VOWELS:
            return ("y", 1, "")
        if nxt == "ו" and letter_at(units, i + 2) != "ו":
            return ("y", 1, "")
        return ("i", 1, "")

    if ch == "ו":
        # וי is /oy/, but only when the yud is not itself carrying a vowel (לוִי).
        yud_follows = nxt == "י" and not _vowel_point(nxt_marks)
        if HOLAM in marks:
            return ("oy", 2 if yud_follows else 1, HOLAM)
        if DAGESH in marks or QUBUTS in marks:
            return ("u", 1, DAGESH if DAGESH in marks else QUBUTS)
        if _is_feminine_ending(units, i):
            return ("ve", 1, "")
        if point:
            # Any other vowel point makes the vav consonantal /v/ carrying that
            # vowel, so it is not the וי digraph (לוִי is /lvi/, not /loy/).
            return ("v" + _POINT_TO_LATIN[point], 1, point)
        if yud_follows:
            return ("oy", 2, "")
        if prev_point in _MATRES_FOR["ו"]:
            return ("", 1, "")  # matres lectionis
        return ("u", 1, "")

    if ch in ("א", "ע"):
        if _is_feminine_ending(units, i):
            return ("e", 1, "")
        nxt2 = letter_at(units, i + 2)
        if ch == "א" and ((nxt == "י" and nxt2 == "י") or (nxt in ("ײ", "ױ")) or (nxt == "ו" and nxt2 == "י")):
            return ("", 1, "")
        if point:
            # A pointed א directly before a tsvey-yudn digraph contributes NO
            # vowel of its own: the digraph already spells the diphthong. The
            # v2 pointing model writes tsere on the alef of אֵיי (canonical
            # convention: the run-initial shtumer alef is left bare, but the
            # model over-points), and reading both the point AND the digraph
            # doubled the vowel — אֵייבֶּערְשְׁטֶער -> *ajajbərʃtər. Found
            # independently by the leave-one-out and end-to-end probes
            # (data/verification/). The digraph wins; the point is dropped.
            return (_POINT_TO_LATIN[point], 1, point)
        if prev_consonant_point:
            return ("", 1, "")  # the consonant's point already spelled this vowel
        # Word-initial א/ע is a silent vowel-carrier only when the NEXT letter
        # really opens a nucleus. A following double-vav does not: וו is
        # consonantal /v/, so אוועק is avek and אוונט is uvnt -- reading the
        # alef as silent there deleted the word's first vowel outright
        # (אוועק -> *vek, אוונט -> *vnt, with no vowel at all). Spec §5 lists
        # silent alef/ayin, and §12/13 uvnt / §6.1 avék- are both first-vowel
        # words.
        # ע is only silent before a vav-nucleus (עוֹלָם, עוף): before a yud it
        # carries its own vowel (עין ayin), so it is not folded in with א here.
        # v3 §5: א is silent ANYWHERE before a vowel-ו, not only word-initially.
        # וואו is vi (the old ("וואו","vu") cluster gave vu) and אונז is inz.
        if ch == "א" and nxt == "ו" and _opens_nucleus(units, i + 1):
            return ("", 1, "")
        if not emitted and _opens_nucleus(units, i + 1) and (ch == "א" or nxt == "ו"):
            return ("", 1, "")  # silent alef/ayin before a nucleus
        return ("a" if ch == "א" else "e", 1, "")

    # v3 §5: ה is silent after a vowel and before a consonant or word-end
    # (זעהן -> zejn, זעה -> zej, חתונה -> xasənə). It is a real [h] only at a
    # syllable onset, i.e. when a vowel letter follows (געהאט -> ɡəhat).
    if (
        ch == "ה"
        and not marks
        and prev_latin in _LATIN_VOWELS
        and (i == n - 1 or letter_at(units, i + 1) not in "אעיוײױ")
    ):
        return ("", 1, "")

    # Word-final ה after a CONSONANT is the loshn-koydesh feminine ending and
    # reduces to ə (livrˈuxə, ʃˈirə, xalˈilə). It is never a word-final [h]:
    # Yiddish has no such word shape and zero of the 500 gold primaries end in
    # h. The rule above only silences ה after a vowel LETTER, so unpointed LK
    # feminines (לברכה, עבודה, אמונה, שירה, חלילה, משנה, מדינה) were surfacing
    # with a final h -- 25,798 corpus tokens / 2,385 types.
    if (
        ch == "ה"
        and not marks
        and i == n - 1
        and emitted
        and prev_latin not in _LATIN_VOWELS
    ):
        return ("e", 1, "")

    return None


def letter_at(units: list[tuple[str, str]], idx: int) -> str:
    return units[idx][0] if 0 <= idx < len(units) else ""


def _opens_nucleus(units: list[tuple[str, str]], idx: int) -> bool:
    """Whether the letter at ``idx`` opens a vowel nucleus.

    ו normally does (אויף, אונטער), but a DOUBLE vav is the spelling of
    consonantal /v/ and opens no nucleus, which is what makes the א of אוועק a
    real vowel rather than a silent carrier.
    """
    ch = letter_at(units, idx)
    if ch not in _NUCLEUS_START:
        return False
    if ch == "ו" and letter_at(units, idx + 1) == "ו":
        return False
    return True


_TAG_PATTERN = re.compile(r"<\s*[a-zA-Z]+\s*>")
_PUNCT_SPLIT = re.compile(r"^([^\w\u0590-\u05FF]*)([\s\S]*?)([^\w\u0590-\u05FF]*)$")


def strip_tags(text: str) -> str:
    text = _TAG_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _preprocess_hebrew(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"[\u05BE\u2010\u2011\u2012\u2013\u2014]", "-", text)
    text = re.sub(r"[\u05f3\u02bc\u2018\u2019`]", "'", text)
    text = re.sub(r"[\u05f4\u201c\u201d]", '"', text)  # gershayim -> ASCII " (acronyms kept intact)
    # \u00a72.2: strip quotes SURROUNDING a word (\u05d9\u05e9\u05e8\u05d0\u05dc" -> \u05d9\u05e9\u05e8\u05d0\u05dc). Only edge quotes:
    # a gershayim with Hebrew letters on both sides is an abbreviation (\u00a72.6)
    # and must survive to the abbreviation table (\u05e9\u05dc\u05d9\u05d8"\u05d0).
    text = re.sub(r'(?<![\u0590-\u05ff])"(?=[\u0590-\u05ff])', "", text)
    text = re.sub(r'(?<=[\u0590-\u05ff])"(?![\u0590-\u05ff])', "", text)

    # 1. Contractions (diacritic-tolerant: pointed input carries marks on ס/כ/מ)
    text = re.sub(_tolerant("ס") + r"'" + _tolerant("איז"), "סיז", text)
    text = re.sub(_tolerant("כ") + r"'", "איך ", text)
    text = re.sub(_tolerant("מ") + r"'", "מע ", text)
    text = re.sub(_tolerant("ס") + r"'", "עס ", text)

    # \u00A77.5 abbreviation expansion (single geresh, non-gematria). Without this the
    # apostrophe survived to the IPA as a glottal stop -- \u05D4' -> h\u0294, \u05E8\u05F3 -> r\u0294 --
    # and \u0294 is not in the target phone set at all.
    text = re.sub(r"(?<![\u0590-\u05FF])" + _tolerant("\u05E8") + r"'", "\u05E8\u05E2\u05D1", text)
    text = re.sub(r"(?<![\u0590-\u05FF])" + _tolerant("\u05D4") + r"'", "\u05D4\u05D0\u05E9\u05E2\u05DD", text)

    # Drop every remaining apostrophe touching a Hebrew letter, on either side:
    # intra-word (\u05DE\u05D9\u05E8'\u05DF) and word-edge (\u05D1\u05EA\u05E9\u05E8\u05D9', \u05D6\u05D9\u05D9\u05DF') alike.
    text = re.sub(r"(?<=[\u0590-\u05FF])'|'(?=[\u0590-\u05FF])", "", text)

    # 2. Hasidic Silent 'ה' Patch (strips 'ה' before terminal ן, סט, ט)
    text = re.sub(
        rf"([אעיו]{_MARKS_CLASS})ה{_MARKS_CLASS}"
        rf"(?=ן{_MARKS_CLASS}\b|ס{_MARKS_CLASS}ט{_MARKS_CLASS}\b|ט{_MARKS_CLASS}\b)",
        r"\1",
        text,
    )

    # 3. Loshn-Koydesh lexical swap
    for pattern, repl in _LK_POINTED:
        text = pattern.sub(repl, text)
    text = _LK_PATTERN.sub(_lk_replace, text)

    return text


# Class-54 (spec ou = [oʊ]) productive prefixes. אויס־ and ארויס־ are listed by
# the spec as class 54 (ous-, arous-), and they head an open-ended set of
# separable-prefix verbs that no word list can enumerate, so they are rewritten
# on the Latin string. Everything else in class 54 is lexical (_WORD_LATIN):
# אויך oykh (44) is deliberately NOT touched here.
_CLASS54_PREFIXES = (
    ("aroys", "arous"), ("aroyf", "arouf"), ("oys", "ous"),
    # aráan- (spec §6.1) is class 34: the bare word אריין is pinned to "arayn"
    # (= [aː]) but every prefixed verb fell through to the rule path and came
    # out [aɪ], so one morpheme had two vowels depending on the compound.
    ("areyn", "arayn"),
    # אפ- is óp- (41), not *af-: the bare word is pinned to "op" but in
    # compounds the rule path read bare פ as /f/ and default א as /a/.
    ("afge", "opge"),
    # אראפ- is aróp- / arúp-: the bare word is pinned to "aroop" (gold arˈup)
    # but in compounds the rule path defaulted the second א to /a/ and read the
    # bare פ as /f/, which then voiced to /v/ before the ɡ of ge-, so one
    # morpheme had three readings -- ˈaravɡəkimən, ˈarafnəmən, ˈaravɡəfaln
    # against the gold's arˈup (473 emitted types / 1,575 tokens). §10.1 only
    # licenses a voiceless obstruent going to its VOICED COUNTERPART anyway,
    # which for p is b, not v.
    ("araf", "aroop"),
)


# -kaat / -haat: spec §9 makes both class 34 (yidishkaat, gezinthaat,
# frumkaat). The rule path spells them "keyt"/"heyt" (= [aɪ]) and no word list
# can enumerate the suffix, so it is rewritten productively -- which is also
# what finally makes the long-dead "kayt" entry in _NEUTRAL_SUFFIXES reachable.
_KAAT_SUFFIX = re.compile(r"(?<=.)([kh])eyt(n|s|en|er)?$")

# §5 "Suffix spellings: ־ליך → ləx (meyglekh)". The productive path reads ליך as
# l + i + kh and gives lix, which directly contradicts the gold (ˈɛtləxə,
# mˈejɡləx, hˈɛrləx) -- and it did so at MED "unambiguous rule" confidence.
# Only the lexicalised מעגליך was right. The suffix attaches to a consonant, so
# the lookbehind excludes a preceding vowel letter (which would make the "likh"
# the tail of a stem rather than the suffix).
_LEKH_SUFFIX = re.compile(r"(?<=[^aeiou])likh(e|en|er|es|n|s|st)?$")


def _class54_prefix(latin: str) -> str:
    for src, dst in _CLASS54_PREFIXES:
        if latin.startswith(src):
            latin = dst + latin[len(src):]
            break
    latin = _KAAT_SUFFIX.sub(lambda m: m.group(1) + "ayt" + (m.group(2) or ""), latin)
    match = _LEKH_SUFFIX.search(latin)
    if match and _nuclei_spans(latin[: match.start()]):
        latin = latin[: match.start()] + "lekh" + (match.group(1) or "")
    return latin


def hebrew_to_latin(text: str, stem_subs: bool = True) -> str:
    """Hebrew-script Yiddish -> the Latin intermediate layer.

    ``stem_subs=False`` turns off the _STEM_SUBS rewrites. They exist to supply
    vowels the SPELLING does not write, so on input that is fully pointed by a
    published edition they have nothing to add and plenty to break: they would
    overwrite the edition's own points with a Yiddish base spelling (נִפְטָר,
    pointed niftar, rewritten through פטר -> פּאַטער into *nˈipatər). The
    register readers, whose input is pointed Hebrew by contract, pass False.
    """
    tokens = text.split()
    out_tokens: list[str] = []
    for tok in tokens:
        parts = tok.split("-")
        latin_parts: list[str] = []
        for part in parts:
            if not part:
                continue
            m = _PUNCT_SPLIT.match(part)
            lead, core, trail = m.group(1), m.group(2), m.group(3)
            # §2.1: nikud is stripped FOR THE LOOKUP KEY; the pointed form is
            # retained only as a side-channel for the §6 LK fallback. The gate
            # here used to be "... and not _vowel_point(core)", i.e. any vowel
            # point at all disabled the whole word lexicon, so ordinary
            # YIVO-pointed text lost the §4 class-41 ɔ pinning (מאָרגן read as
            # murɡn, גאָט as ɡut, אָפֿט as uft) and the §4 p-list (פאליטיק with a
            # komets read fˈulitik). The lexicon is the dialect (§4); a komets
            # written over a class-41 word does not overrule it.
            bare = _strip_points(core)
            if bare in _WORD_LATIN:
                latin = _WORD_LATIN[bare]
            else:
                if stem_subs:
                    core = _apply_stem_subs(core)
                latin = _word_to_latin(core)
                latin = _class54_prefix(latin)
            latin_parts.append(lead + latin + trail)
        if latin_parts:
            out_tokens.append(" ".join(latin_parts))
    return re.sub(r"\s+", " ", " ".join(out_tokens)).strip()


# =====================================================================
# STAGE 2.5: STRESS ASSIGNMENT
#
# Yiddish was the only language in this project's corpus with NO prosodic marking
# at all -- 0 stress marks per row against 7-12 for he/en/de/it/es/ru -- so the
# acoustic model received no prominence cue and produced flat, uniformly paced
# speech ~25% faster per phoneme than every other language.
#
# Stress is assigned on the Latin string, where syllable nuclei are just vowel
# groups, and the marker passes through latin_to_ipa unchanged (its fallback
# branch appends characters it does not recognise).
#
# CONFIDENCE: the Germanic rules below are well established -- initial stress on
# the stem, with a closed set of unstressed prefixes. Loshn-Koydesh stress is
# irregular and is NOT rule-derived here; those words are handled by
# _STRESS_OVERRIDES and by the Stage 1 lexicon, and the entries carry a
# penultimate default only where that is the attested form. Words outside both
# fall back to initial stress, which is correct for the Germanic core that
# dominates running text but will be wrong for unlisted Hebrew-origin words.
# =====================================================================
STRESS = "ˈ"

# Latin vowel nuclei, longest first so digraphs win. Must stay in sync with the
# label key on _LATIN_TO_IPA: ee/ey before e, oo/ou/oy before o, ay before a.
_NUCLEI = ("ee", "ey", "ay", "oy", "ou", "oo", "a", "e", "i", "o", "u")

# Inseparable prefixes that never take stress: it falls on the following stem
# syllable. Matched only when at least one nucleus follows, so the bare words
# (der, far, tsu ...) are not mis-analysed as prefixed forms.
# v3 §11.3 names exactly six: ge- ba- be- far- der- tse-. tsu-/tser-/ant-/ent-/
# dis- were engine additions and are dropped -- v3 §11.7 sends everything else
# to initial stress.
_UNSTRESSED_PREFIXES = ("ge", "be", "der", "far", "tse", "ba")

# Function words that carry no lexical stress. Marking these would dilute the
# meaning of the marker; espeak leaves the equivalent clitics bare too.
# Every entry must be monosyllabic. ober, oder, iber and unter were listed here
# and came out stress-less (ubɛr, udɛr, ibɛr) -- but they are ÓBER, ÓDER, ÍBER,
# ÚNTER, stressed on the first syllable like any disyllabic Germanic word. That
# was 1.2% of corpus instances emitted with no stress mark at all.
_CLITICS = frozenset({
    "a", "an", "di", "der", "dos", "doos", "dem", "den", "de",
    "in", "im", "un", "az", "tsu", "mit", "fun", "far", "bay", "ba",
    "oyf", "es", "zi", "er", "ix", "mir", "dir",
    "zix", "ze", "do", "doo", "vi", "nor", "noor", "ven",
})

# --- Suffix classes -------------------------------------------------------
# Suffixes are the most reliable automated cue for Yiddish stress, and they fall
# into three behavioural groups. Ordered longest-first so the specific wins.
#
# TONIC: international / Slavic loan suffixes that take the stress themselves.
# "-ir" covers the -irn verb class (regirn, studirn, telefonirn), which is why
# regirung comes out re-GI-rung once the neutral -ung is stripped.
_TONIC_SUFFIXES = (
    "tsyes", "tsees", "tsye", "tsee", "izm", "ist", "ent", "ant", "ur", "ir",
)

# PRE-TONIC ("stress magnets"): the stress lands on the suffix's own first
# nucleus, i.e. ameri-KA-ner, te-o-RI-ye.
_PRETONIC_SUFFIXES = ("aner", "iye")

# NEUTRAL: native Germanic inflection/derivation. These never take stress and do
# not move it off the root, so they are stripped to expose whatever is beneath.
_NEUTRAL_SUFFIXES = (
    "ndik", "shaft", "kayt", "hayt", "keyt", "heyt", "lekh", "dik", "ung",
    "es", "en", "er", "l", "s",
)


# Words whose stress the rules get wrong, as {word: index of stressed vowel}.
#
# An index suffices because the marker sits immediately before the vowel, so no
# syllable-boundary decision is involved -- the ambiguity that once forced these
# to be written out in full (mish|pokhe vs mi|shpokhe) simply does not arise.
#
# Each entry is a specific lexical claim. This is the right place for a native
# reviewer to correct or extend; the rule engine deliberately does not guess at
# Hebrew-origin stress patterns.
_STRESS_OVERRIDES: dict[str, int] = {
    # --- Loshn-Koydesh (Hebrew/Aramaic): historical spelling, mostly penultimate ---
    "mishpookhe": 1,   # mish-PU-khe (komets -> "oo" after the §2-A retag)
    "tsedooke":  1,    # tse-DU-ke
    "meshuge":   1,    # me-SHU-ge
    "mekhutn":   1,    # me-KHU-tn
    "rebetsn":   0,    # RE-be-tsn
    "baleboos":  0,    # BA-le-bus -- spec §6.2 lists BAlebus alongside SHAbes,
    "balebooste": 0,   # TOYre and CHAsene as merged-LK penult/initial
                       # retraction. The old ba-le-BUS pinning was not
                       # audio-verified and contradicted the spec's example list.
    "yeshive":   1,    # ye-SHI-ve
    "khevre":    0,    # KHEV-re
    "shabes":    0,    # SHA-bes
    "yontev":    0,    # YON-tev
    "khasene":   0,    # KHA-se-ne
    "mazltoov":  0,    # MAZL-tuv
    "seykhl":    0,
    "khoydesh":  0,
    "khoolem":   0,
    "afile":     1,    # a-FI-le
    "efsher":    0,
    "asakh":     0,    # gold ˈasax -- initial, not a-SAKH
    # Three-syllable loshn-koydesh: penultimate, which the Germanic default
    # (initial) gets wrong once the word has been respelled into Yiddish.
    "neshoome":  1,    # ne-SHU-me
    "khasidem":  1,    # kha-SI-dem
    "shaboosim": 1,    # sha-BU-sim
    "bishas":    1,
    "beemes":    1,
    "stam":      0,
    "mesifte":   1,    # me-SIF-te (§4.2 mesivta -> mesifte)
    "bakhirem":  1,    # ba-CHI-rem -- §6.2 plural stress shift off BUcher
    "balebatim": 2,    # ba-le-BA-tim -- ditto, off BAlebus
    "hashem":    1,    # ha-SHEM (§7.5 ה' expansion)
    "masbir":    1,    # maz-BIR (gold mazbˈir; LK final stress)    # ha-SHEM (§7.5 ה' expansion)
    # §11.5 penult retraction for the merged-LK entries added 2026-08-07.
    "avoyde":    1,    # a-VOY-de
    "emune":     1,    # e-MU-ne
    "khalile":   1,    # kha-LI-le
    "livrookhe": 1,    # li-VRU-khe

    # --- Unstressed initial vowel: not derivable from prefix or suffix rules,
    # and high-frequency in this corpus, so they are listed explicitly. ---
    "azoy":      1,    # a-ZOY, 1112 occurrences
    "viazoy":    2,
    "aleyn":     1,
    "azoyne":    1,
    "azelkhe":   1,
    "avade":     1,
    "akegn":     1,
    "arum":      1,
    "tsurik":    1,    # ʦirˈik (gold)
    "arous":     1,    # a-ROUS (54)
    "arayn":     1,
    "avek":      1,    # a-VEK -- the bare adverb; §6.1 avék-
    "anider":    1,
    "amool":     1,    # a-MUL
    "aza":       1,    # a-ZA
    "arop":      1,    # a-ROP
    "aroop":     1,    # gold arˈup
    "aheym":     1,    # a-HAJM (v3 §11.4 directional)
    "arouf":     1,
    "arayf":     1,
    "aroyf":     1,
    "abisl":     1,    # a-BI-sl
    "apoor":     1,    # a-PUR
    "aleyns":    1,
    "ahin":      1,
    "aher":      1,
    "atsind":    1,
    # v3 §11.4 covers the BARE directional adverb too. _SEPARABLE_PREFIXES only
    # fires when material follows the prefix, so ארונטער / אדורך were falling to
    # the §11.7 initial default while their own compounds were second-stressed.
    "arunter":   1,
    "adurkh":    1,
    "arouf":     1,

    # --- False prefixes: be-/der- here belong to the root, not a prefix ---
    "beser":     0,    # BE-ser, not be-SER
    "bese":      0,
    "derekh":    0,    # DE-rekh (Hebrew-origin)
    "beged":     0,
    "berye":     0,

    # --- International loanwords whose stress the suffix rules do not reach ---
    "amerike":   1,    # a-ME-ri-ke
    "iran":      1,
    "iraner":    1,
    "kongres":   1,    # kon-GRES
    "kangres":   1,    # as transliterated from קאנגרעס
    "politishe": 1,    # po-LI-ti-she
    "politish":  1,
    "politik":   1,
    "republikaner": 3, # republi-KA-ner
    "demokratn": 2,
    "demakrotn": 2,    # as transliterated from דעמאקראטן
    "milyon":    1,
    "protsent":  1,
}


def _nuclei_spans(word: str) -> list[tuple[int, int]]:
    """(start, end) of each vowel nucleus in a Latin word, left to right."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(word)
    while i < n:
        for nuc in _NUCLEI:
            if word.startswith(nuc, i):
                spans.append((i, i + len(nuc)))
                i += len(nuc)
                break
        else:
            i += 1
    return spans


# v3 §11.2: "monosyllable" is defined on WRITTEN VOWEL NUCLEI, not on phonetic
# syllables. A syllabic final -n/-l/-m adds no vowel symbol (§1), so maxn, zuɡn,
# farn and vɔxn are monosyllables and carry NO stress mark, while ˈarbətn is
# marked because its ע is a written vowel. The old count -- nuclei plus every
# syllabic nasal -- mirrored the ə-insertion that v3 deletes, and produced
# mˈaxn / zˈuɡn / fˈarn against the gold.
def _syllable_count(word: str) -> int:
    """Number of vowel-symbol nuclei; 1 means the word takes no stress mark."""
    return len(_nuclei_spans(word))


# --- Separable (directional) prefixes: these CARRY the stress -----------------
# arop-geyn, avek-forn, unter-shraybn. Only when material follows: bare "arum"
# is a-RUM, but "arumgeyn" is ARUM-geyn.
# The value is WHICH nucleus of the prefix carries the stress. Most are
# monosyllabic or initial-stressed (óus-, ún-, únter-, óp-), but spec §6.1
# writes the polysyllabic converbs with stress on the second syllable --
# aróusgayn, aráankimen, tsurík-gekimen, aníder- -- and the engine used to
# return 0 for every prefix, which contradicted its own bare-word readings
# (ארויס -> arˈoʊs, אריין -> arˈaːn) as soon as anything was suffixed to them.
# avek- keeps 0 because the spec's own example is ávekgelaygt (the bare adverb
# avék is handled by _STRESS_OVERRIDES).
_SEPARABLE_PREFIXES: dict[str, int] = {
    # v3 §11.4: EVERY directional a(r)- prefix stresses its second nucleus, in
    # the compound exactly as in the bare adverb (arˈoʊs, arˈup, ahˈajm, avˈɛk,
    # arˈaːn). The old per-prefix split -- arop/aroyf/avek/arum at 0, the rest
    # at 1 -- made one morpheme stress two different ways depending on whether
    # anything was suffixed to it.
    # "arouf" is what _class54_prefix rewrites ארויפ- to; without it the whole
    # ארויפ- compound family (ארויפגיין, ארויפלייגן, ארויפגעלייגט) fell through
    # to §11.7 initial stress even though the bare word is arˈoʊf.
    "arous": 1, "arayn": 1, "anider": 1, "arunter": 1, "aruf": 1, "arouf": 1,
    "arum": 1, "aroop": 1, "arop": 1, "aroyf": 1, "avek": 1, "aheym": 1,
    "ahin": 1, "aher": 1, "adurkh": 1, "arayf": 1,
    # Non-directional separables keep initial stress (v3 §11.7).
    "tsurik": 1,
    "unter": 0, "iber": 0, "mit": 0, "ous": 0, "oon": 0, "on": 0, "uf": 0,
    "oyf": 0, "ayn": 0,
    "tsuzamen": 1, "farbay": 1, "antkegn": 1,
}

# --- Circumfix validation ----------------------------------------------------
# A grammatical prefix must prove its function: ge- forms past participles, which
# end in -t or -n, and be-/der- form verbs. Requiring a compatible ending stops
# the engine reading root letters as a prefix -- the beser / derekh trap.
_CIRCUMFIX_ENDINGS = {
    # ge- deliberately absent: it forms nouns (geduld, gemitlekh, gesheft) as
    # readily as participles, and gating on -t/-n mis-stressed those. The plain
    # strip-always rule measures 97.8%; its handful of false positives are
    # cheaper to list than to predict.
    "be": ("n", "t", "ndik"),
    "der": ("n", "t", "ndik"),
}

# --- Phonotactic compound seams ----------------------------------------------
# Clusters Yiddish permits ACROSS a compound boundary but not inside one root.
# Finding one means two roots were glued together (bukh|gesheft, briv|marke).
_ILLEGAL_INTERNAL = re.compile(
    r"(kh|sh|ts|s|f|kh|b|d|g|k|p|t|v|z)(g|b|d|k|p|t|m|v|z|sh|ts|kh)"
)
_COMPOUND_SEAMS = (
    "khg", "khb", "khm", "khsh", "sts", "shts", "fm", "fb", "fg", "sg", "sb",
    "tsg", "tsb", "tsm", "khts", "shg", "shb", "zg", "zb", "pg", "pb", "kg",
)


def _compound_split(word: str) -> int | None:
    """Index of a phonotactically impossible internal cluster, if any.

    No dictionary needed: certain clusters simply cannot occur inside a single
    Yiddish root, so their presence marks a seam. Primary stress then belongs to
    the first element, per the Germanic compound rule.
    """
    for seam in _COMPOUND_SEAMS:
        pos = word.find(seam, 1)
        if pos > 0 and _nuclei_spans(word[:pos]) and _nuclei_spans(word[pos:]):
            return pos
    return None


def _splits_nucleus(spans: list[tuple[int, int]], boundary: int) -> bool:
    """Whether a morpheme boundary would fall INSIDE a vowel nucleus.

    "geyen" is g|ey|en, not ge|yen: the ge- prefix rule was cutting the ey
    digraph in half and stressing the wrong syllable (ɡaɪˈɛn for ˈɡaɪən), while
    its sisters zeyen / freyen -- whose first letter is not a prefix -- were
    right. Same trap for the separable prefixes.
    """
    return any(start < boundary < end for start, end in spans)


def _suffix_stress(stem: str, spans: list[tuple[int, int]]) -> int | None:
    """Syllable index dictated by a tonic / pre-tonic suffix, if one applies.

    Both classes resolve to "stress the suffix's first nucleus": for -aner and
    -iye that nucleus is the pre-tonic syllable (ameri-KA-ner), and for -tsye,
    -ent, -izm, -ur it is the suffix itself (informa-TSYE, prezi-DENT).
    """
    for suf in _PRETONIC_SUFFIXES + _TONIC_SUFFIXES:
        if not stem.endswith(suf) or len(stem) == len(suf):
            continue
        start = len(stem) - len(suf)
        # A real suffix begins after a consonant. If a nucleus ENDS exactly where
        # the suffix starts, the "suffix" is really the tail of a vowel sequence:
        # לייענט leyent is l|ey|ent, a native stem, not a -ent loanword, and the
        # tonic reading put the stress on the wrong syllable (laɪˈɛnt).
        if any(nuc_end == start for _, nuc_end in spans):
            continue
        for i, (nuc_start, _) in enumerate(spans):
            if nuc_start >= start:
                return i
    return None


def _strip_neutral(word: str) -> str:
    """Remove one neutral suffix, provided a nucleus-bearing stem survives."""
    for suf in _NEUTRAL_SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            stem = word[: -len(suf)]
            if _nuclei_spans(stem):
                return stem
    return word


def _stressed_syllable(word: str, count: int) -> int:
    """Which syllable of ``word`` takes primary stress (0 = first).

    Order of operations, per the Yiddish stress rulebook:
      1. tonic / pre-tonic suffix on the surface form
      2. strip a neutral suffix, then retry (1) -- exposes regir|ung
      3. strip unstressed prefixes
      4. default to the first syllable of what remains (the root)

    Lexicon lookup happens before this, in add_stress: Loshn-Koydesh stress is
    not recoverable from the orthography and must come from _STRESS_OVERRIDES.
    """
    spans = _nuclei_spans(word)

    # Separable prefixes pull the stress onto themselves, but only when a stem
    # follows -- otherwise the bare adverb (arum, arop) keeps its own stress.
    for pre, pre_idx in _SEPARABLE_PREFIXES.items():
        if word.startswith(pre) and len(word) > len(pre) and not _splits_nucleus(spans, len(pre)):
            tail = word[len(pre) :]
            pre_spans = _nuclei_spans(pre)
            if _nuclei_spans(tail) and pre_spans:
                return min(pre_idx, len(pre_spans) - 1)

    # A compound takes primary stress on its first element.
    seam = _compound_split(word)
    if seam is not None:
        head = word[:seam]
        if _nuclei_spans(head):
            return 0

    idx = _suffix_stress(word, spans)
    if idx is not None:
        return min(idx, count - 1)

    stem = _strip_neutral(word)
    if stem != word:
        idx = _suffix_stress(stem, spans)
        if idx is not None:
            return min(idx, count - 1)

    # Prefix analysis runs on the FULL word, never the neutral-stripped stem:
    # stripping -en from geshen leaves "gesh", where ge- has no root behind it
    # and the prefix rule silently stops firing.
    consumed = 0
    rest = word
    while True:
        for pre in _UNSTRESSED_PREFIXES:
            if rest.startswith(pre):
                if _splits_nucleus(_nuclei_spans(rest), len(pre)):
                    continue  # the "prefix" ends inside a vowel: גייען is ge|yen
                tail = rest[len(pre) :]
                required = _CIRCUMFIX_ENDINGS.get(pre)
                if required and not word.endswith(required):
                    continue  # prefix cannot prove its grammatical role
                if _nuclei_spans(tail):
                    consumed += len(_nuclei_spans(pre))
                    rest = tail
                    break
        else:
            break
    return min(consumed, count - 1)


def add_stress(latin: str, penult: bool = False) -> str:
    """Insert a primary-stress marker before the stressed syllable of each word.

    ``penult`` selects §11.5's loshn-koydesh default (penult retraction:
    ʃˈabəs, jisrˈuəl, ʦadˈikim) instead of the Germanic §11.7 initial default.
    It is used only for the §6.2 nikud path, where the token itself is pointed
    Whole-Hebrew; _STRESS_OVERRIDES still wins over it.
    """
    out: list[str] = []
    for token in re.split(r"(\s+)", latin):
        if not token or token.isspace():
            out.append(token)
            continue
        core = token.strip(".,!?;:\"()'-")
        lead = token[: len(token) - len(token.lstrip(".,!?;:\"()'-"))]
        trail = token[len(lead) + len(core) :]
        spans = _nuclei_spans(core)
        lowered = core.lower()
        # Monosyllables carry no mark: with one nucleus the prominence is not in
        # doubt, so a marker adds no information. This is a Yiddish convention and
        # deliberately differs from Hebrew/English in this corpus, which do mark
        # them -- the shared signal is WHERE the mark sits (before the vowel), not
        # how often it appears.
        if not core or not spans or _syllable_count(lowered) == 1 or lowered in _CLITICS:
            out.append(token)
            continue
        idx = _STRESS_OVERRIDES.get(lowered)
        if idx is None and penult:
            stem_spans = _nuclei_spans(_strip_neutral(lowered)) or spans
            idx = max(len(stem_spans) - 2, 0)
        if idx is None:
            idx = _stressed_syllable(lowered, len(spans))
        idx = min(max(idx, 0), len(spans) - 1)
        # The marker goes immediately BEFORE THE VOWEL, not before the syllable
        # onset. That is the convention every other language in this vocab uses
        # -- measured at 99.7-100% for he/en/de/it/es and 96.4% for ru -- so the
        # model reads a stress mark as "the next vowel is prominent". Marking the
        # onset instead put Yiddish in a different structural position from 90%
        # of the training data.
        at = spans[idx][0]
        out.append(lead + core[:at] + STRESS + core[at:] + trail)
    return "".join(out)


# =====================================================================
# STAGE 3: CENTRAL YIDDISH PHONOLOGY
# =====================================================================
# LATIN LABEL KEY (engine label -> IPA -> spec romanization)
#
# The engine's internal Latin labels are historical and do NOT match the spec's
# romanization. Only the IPA on the right is normative; the labels below are an
# internal encoding and are documented here once so the two can be read together.
#
#   engine   IPA    spec    Weinreich class
#   ------   ----   -----   ---------------
#   a        a      a       11
#   ay       aː     aa      34   (pasekh-tsvey-yudn, flattened)
#   e        ɛ      e       21
#   ee       ej     ey      25   (e lengthened in open syllable)
#   ey       aj     ay      22/24
#   i        i      i       31/32
#   o        ɔ      o       41   (short o, never lengthened)
#   oo       u      u       12/13 (MHG ā; also every komets, §5)
#   oy       ɔj     oy      42/44
#   ou       oʊ     ou      54   (MHG ū: hous, moul, boukh, arous)
#   u        i      i       51/52 (the "vowel written ו is always i" rule)
#
# Scanned in order by latin_to_ipa, so every digraph MUST precede its own first
# letter: oo/ou/oy before o, ee/ey before e, ay before a. _NUCLEI and
# _nuclei_spans keep the same ordering for the stress stage.
_LATIN_TO_IPA: list[tuple[str, str]] = [
    # v3 §1: ʣ is NOT in the closed inventory -- dz stays two phones (d + z).
    ("tsh", "ʧ"), ("dzh", "ʤ"), ("ts", "ʦ"),
    ("kh", "x"), ("sh", "ʃ"), ("zh", "ʒ"),
    # ey and ay must stay DISTINCT. Tsere/ey raises to aɪ (בית -> bais), while
    # pasekh-tsvey-yudn flattens to long aː (הײַנט -> haant, מײַן -> maan,
    # פֿרײַנט -> fraant, חיים -> khaayem). Merging them to aɪ was tried and the
    # native reviewer rejected it outright: "Changing them to haint/main made
    # your engine sound Litvish/YIVO. The Heimish Williamsburg dialect flattens
    # the 'ai' diphthong into a long 'ah' sound."
    # The corpus argues the other way (~85% aj/aɪ) but the corpus IPA is the
    # Gemini-written column, which agrees with this engine only 14.4% of the
    # time -- it is not evidence about the dialect.
    # v3 §1: the diphthongs are written aj / ɔj (not aɪ / ɔɪ). The closed phone
    # inventory in spec v3 lists ej aj ɔj oʊ, so ɪ never appears in output.
    ("ee", "ej"), ("ey", "aj"), ("ay", "aː"),
    ("oy", "ɔj"), ("ou", "oʊ"), ("oo", "u"),
    ("a", "a"), ("o", "ɔ"), ("e", "ɛ"), ("u", "i"), ("i", "i"),
    ("b", "b"), ("v", "v"), ("d", "d"), ("h", "h"), ("z", "z"),
    ("t", "t"), ("l", "l"), ("m", "m"), ("n", "n"), ("s", "s"),
    ("f", "f"), ("p", "p"), ("k", "k"), ("g", "ɡ"), ("r", "r"),
    ("y", "j"), ("-", " "),
]

_APOSTROPHE = re.compile(r"'")

# TIE-BAR FORMS ONLY. The bare sequences (tʃ, dʒ, dz) used to be fused here too,
# but they are legitimate two-phoneme clusters and postlexical() creates them:
# §4.2 devoices the d of קודש kudsh before the ʃ, and normalize_ipa_affricates --
# which hebrew_to_ipa runs AFTERWARDS -- then swallowed the t into an affricate,
# deleting a phoneme (kidʃ -> *kiʧ, xsidʃə -> *xsiʧə, 98 affected tokens in
# 400 corpus rows). latin_to_ipa emits the ligatures directly for the real
# affricates, so the bare alternatives were never needed for engine output.
_AFFRICATE_DECOMPOSE = [
    (re.compile("t͡s", re.I), "ʦ"),
    (re.compile("t͡ʃ", re.I), "ʧ"),
    (re.compile("d͡ʒ", re.I), "ʤ"),
    (re.compile("d͡z", re.I), "dz"),
]


def latin_to_ipa(latin: str) -> str:
    out: list[str] = []
    i = 0
    n = len(latin)
    while i < n:
        ch = latin[i]
        if ch in " \t\n\r":
            out.append(" ")
            i += 1
            continue
        if ch == "'":
            # Never a phone. An apostrophe reaching this far is a leftover
            # geresh; ʔ is outside the target phone set (spec §14).
            i += 1
            continue
        if ch in ".,!?;:\"()":
            out.append(ch)
            i += 1
            continue

        matched = False
        for src, dst in _LATIN_TO_IPA:
            if latin.startswith(src, i):
                out.append(dst)
                i += len(src)
                matched = True
                break

        if not matched:
            # IPA Erasure Bug patched: append the unmatched character
            out.append(ch)
            i += 1

    ipa = "".join(out)

    # v3 §1 / §10.3: syllabic finals -n -l -m after a consonant get NO epenthetic
    # vowel -- zuɡn, not zuɡən; maxn, mˈɛnʧn, ˈarbətn. The \b-anchored ə-insertion
    # regex that used to live here (kept for checkpoint compatibility with the
    # yiddish24 ipa column) is deleted: v3 overrides that compatibility note, and
    # the gold_v3 lexicon writes every one of these bare.

    # Word-final epsilon to schwa reduction (protects single-syllable words)
    ipa = re.sub(r"(\S{2,})ɛ\b", r"\1ə", ipa)
    
    return ipa


# =====================================================================
# STAGE 3.5: POSTLEXICAL PHONOLOGY (spec §4)
#
# Two rules, both fully active in this dialect (unlike StY, where final
# devoicing is absent). They run on the IPA, AFTER the syllabic-nasal schwa
# insertion in latin_to_ipa -- otherwise zogn would look like a ɡ in final
# position and come out *zukn -- and before reduce_unstressed, which only
# touches vowels and so cannot undo them.
#
# Deliberately NOT modelled, by agreement: nasal place assimilation ([ŋ̩] etc.),
# dark l, and r-variant coloring. One l, one r.
# =====================================================================
_DEVOICE = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s", "ʒ": "ʃ",
            "ʤ": "ʧ"}
_VOICE = {v: k for k, v in _DEVOICE.items()}

# Triggers for regressive assimilation -- the SECOND consonant wins.
# /v/ is a voiced obstruent but is NOT a voicing trigger: tsvay and mitsve keep
# their [ʦ] (a voicing /v/ would give *dzvay), while as a TARGET it devoices
# normally (davka -> dafke, mesivta -> mesifte). The one spec example that
# needs a voicing /v/ -- Cheshvn -> Chezhvn -- is a frozen month name and is
# stored in the lexicon instead.
_VOICING_TRIGGERS = frozenset("bdɡzʒʤ")
_DEVOICING_TRIGGERS = frozenset("ptkfsʃxʦʧ")  # v3: targets of §10.1 only
_OBSTRUENTS = _VOICING_TRIGGERS | _DEVOICING_TRIGGERS | frozenset("v")

# Only fricatives and affricates VOICE regressively. Every voicing example in
# the spec has a fricative target (shabesdik -> shabezdik, Cheshvn -> Chezhvn,
# and the aroys- seam), while a voiced plosive target appears nowhere -- and
# assuming one turned the separable prefixes into their own opposites at the
# compound seam: óp-getun came out [ɔbɡə-] and avék-gelaygt [avɛɡɡə-], losing
# the prefix consonant to the following ge-. Devoicing stays unrestricted; its
# spec examples (zugt, libt) are plosives.
#
# AUDIT 2026-08-07: /f/ left the target set. There is no gold evidence for it --
# the one gold primary with a voiceless obstruent before a voiced one is
# vˈiljamsburɡ, where the s does NOT voice -- and it was actively wrong at the
# separable-prefix seam, turning arˈoʊfɡajn into arˈoʊvɡajn and arˈupɡəkimən
# into ˈaravɡəkimən. The engine already asserts the same for /p/ (אפגעטון ->
# ɔpɡɛtin, "the p does not voice before ɡ"), and §10.1 licenses only a move to
# the VOICED COUNTERPART, which for the אראפ- family is b, never v.
_VOICING_TARGETS = frozenset("sʃʦʧ")

_WORD_SPLIT = re.compile(r"(\s+)")


def _postlexical_word(word: str) -> str:
    core = word.rstrip(".,!?;:\"()'")
    trail = word[len(core):]
    chars = list(core)

    # v3 §10.1 VOICING-ward assimilation only, right to left so a chain settles
    # in one pass (the rightmost obstruent is the one that never changes).
    # v3 §10.2 turns devoicing OFF everywhere, so the devoicing-ward branch that
    # used to sit here (zuɡt -> *zukt, ʃraːbt -> *ʃraːpt) is gone.
    for i in range(len(chars) - 2, -1, -1):
        cur, nxt = chars[i], chars[i + 1]
        if cur not in _OBSTRUENTS or nxt not in _OBSTRUENTS:
            continue
        if nxt in _VOICING_TRIGGERS and cur in _VOICING_TARGETS:
            chars[i] = _VOICE.get(cur, cur)

    # Degemination. Assimilation regularly produces a doubled consonant at a
    # morpheme seam (ge-red-t -> ɡərɛdt -> ɡərɛtt), and a repeated phone is not
    # a legal TTS phoneme sequence -- Yiddish has no geminates. §9's own
    # past-participle shape (ge- + stem + devoiced final) is the common case.
    chars = [c for i, c in enumerate(chars)
             if i == 0 or c != chars[i - 1] or c not in _OBSTRUENTS]

    # v3 §10.2: NO word-final devoicing. iz, zuɡt, hub, ɔjb, vejɡ, kind, ruv, jid
    # all stay voiced; the lexicalized devoiced forms (jˈankəf, ʃkˈɔjəx) are
    # lexicon entries, not the output of a rule.

    return "".join(chars) + trail


def postlexical(ipa: str) -> str:
    """v3 §10: voicing-ward assimilation + degemination. Devoicing is OFF."""
    return "".join(
        tok if (not tok or tok.isspace()) else _postlexical_word(tok)
        for tok in _WORD_SPLIT.split(ipa)
    )


def reduce_unstressed(ipa: str) -> str:
    """Reduce unstressed /ɛ/ to /ə/ within each word.

    Central Yiddish reduces unstressed vowels heavily -- the grammatical prefixes
    (ge-, be-, der-, tse-) and the neutral suffixes (-er, -en, -es) are all schwa
    in running speech: gemakht is /ɡəmaxt/, not /ɡɛmaxt/; beser is /bɛsər/.
    Without this every unstressed e surfaced as a full ɛ, which is the single
    most audible artifact separating synthetic from natural Yiddish.

    The stress marker makes this decidable: the vowel immediately after ˈ is the
    stressed one, everything else in the word is not. Words carrying no mark are
    monosyllables, whose only vowel is stressed by definition, so they are left
    alone -- reducing them would flatten mentsh to mənʧ.
    """
    words = []
    for word in ipa.split(" "):
        if STRESS not in word:
            words.append(word)
            continue
        keep = word.index(STRESS) + 1
        words.append(
            "".join(
                "ə" if (ch == "ɛ" and i != keep) else ch
                for i, ch in enumerate(word)
            )
        )
    return " ".join(words)


def normalize_ipa_affricates(ipa: str) -> str:
    for pattern, lig in _AFFRICATE_DECOMPOSE:
        ipa = pattern.sub(lig, ipa)
    return ipa


def normalize_ipa_spacing(ipa: str) -> str:
    ipa = re.sub(r"\s+", " ", ipa).strip()
    for punct in [",", ".", "!", "?", ";", ":", "'"]:
        ipa = ipa.replace(f" {punct}", punct)
    return ipa


def g2p_fingerprint() -> str:
    """Short hash of the rules that determine Yiddish phoneme output.

    Inference regenerates IPA from source text with the LIVE code, while a
    checkpoint is frozen on whatever phonemes it trained on. Nothing errors when
    those diverge -- the audio just degrades, which cost a full listening
    evaluation once. Stamp this into a checkpoint or a demo manifest so the
    mismatch is visible instead of silent.
    """
    import json as _json

    parts = [
        _json.dumps(_STRESS_OVERRIDES, sort_keys=True, ensure_ascii=False),
        _json.dumps(_LOSHN_KOYDESH, sort_keys=True, ensure_ascii=False),
        _json.dumps(_WORD_LATIN, sort_keys=True, ensure_ascii=False),
        repr(_UNSTRESSED_PREFIXES), repr(_SEPARABLE_PREFIXES),
        repr(_TONIC_SUFFIXES), repr(_PRETONIC_SUFFIXES), repr(_NEUTRAL_SUFFIXES),
        repr(_CIRCUMFIX_ENDINGS), repr(_LATIN_TO_IPA), repr(_COMPOUND_SEAMS),
        "schwa_reduction=1", "postlexical=1",
        # The v3 routing layer decides ~64% of running tokens, so it belongs in
        # the drift stamp exactly as much as the rules do.
        _json.dumps(
            {k: v["ipa_primary"] for k, v in GOLD_LEXICON.items()},
            sort_keys=True, ensure_ascii=False,
        ),
        _json.dumps(_ABBREVIATIONS, sort_keys=True, ensure_ascii=False),
        _json.dumps(_MULTIWORD, sort_keys=True, ensure_ascii=False),
        repr(sorted(_MULTIWORD_LEGACY)), repr(sorted(_CLITIC_IPA.items())),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _rule_path_ipa(text: str, stress: bool = True, lk_penult: bool = False,
                   stem_subs: bool = True) -> str:
    """The Germanic/LK RULE PATH: orthography -> Latin -> IPA, no gold lookup.

    This is the engine as it existed before the v3 lexicon layer, and it is what
    routing falls back to when no table knows the token (§3.6).
    ``stem_subs=False`` is for fully pointed input -- see hebrew_to_latin().
    """
    text = _preprocess_hebrew(strip_tags(text))
    latin = hebrew_to_latin(text, stem_subs=stem_subs)
    if stress:
        latin = add_stress(latin, penult=lk_penult)
    ipa = latin_to_ipa(latin)
    ipa = postlexical(ipa)
    if stress:
        ipa = reduce_unstressed(ipa)
    ipa = normalize_ipa_affricates(ipa)
    return normalize_ipa_spacing(ipa)


# =====================================================================
# STAGE 0: LEXICON ROUTING  (spec v3 §2 normalization, §3 routing order,
# §8 abbreviations & multiword, §9 homographs, §12 per-token output)
#
# Everything above this point is the rule path. Routing sits in FRONT of it:
# the gold lexicon (authority #1, native-verified) is consulted first for whole
# tokens and overrides every rule and every legacy dict below it.
# =====================================================================


def _load_gold_lexicon() -> dict:
    """Import data/gold_lexicon.py by path; an absent/broken file is not fatal.

    The module is generated (scripts/build_gold_lexicon.py) and committed. The
    engine must still import and run without it -- a checkout that has not
    regenerated it degrades to the rule path rather than failing to import.
    """
    path = Path(__file__).resolve().parent / "data" / "gold_lexicon.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_yiddish_gold_lexicon", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return dict(getattr(module, "GOLD_LEXICON", {}) or {})
    except FileNotFoundError:
        return {}  # not generated in this checkout: degradation is deliberate
    except Exception as exc:  # a table that EXISTS but will not load is a bug
        raise RuntimeError(
            f"{path} exists but could not be loaded ({exc!r}). Returning an "
            "empty table here would silently drop every verdict it holds; "
            "regenerate it with its builder in scripts/."
        ) from exc


GOLD_LEXICON: dict[str, dict] = _load_gold_lexicon()


def _load_audio_endorsed() -> dict:
    """data/audio_endorsed_lk.py, keyed by lexicon_key; absent file degrades."""
    path = Path(__file__).resolve().parent / "data" / "audio_endorsed_lk.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_yiddish_audio_lk", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, "AUDIO_ENDORSED_LK", {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}  # not generated in this checkout: degradation is deliberate
    except Exception as exc:  # a table that EXISTS but will not load is a bug
        raise RuntimeError(
            f"{path} exists but could not be loaded ({exc!r}). Returning an "
            "empty table here would silently drop every verdict it holds; "
            "regenerate it with its builder in scripts/."
        ) from exc


_AUDIO_ENDORSED: dict[str, dict] = _load_audio_endorsed()


def _load_homograph_lk() -> dict:
    """data/homograph_lk.py, keyed by lexicon_key; absent file degrades."""
    path = Path(__file__).resolve().parent / "data" / "homograph_lk.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_yiddish_homograph_lk", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, "HOMOGRAPH_LK", {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}  # not generated in this checkout: degradation is deliberate
    except Exception as exc:  # a table that EXISTS but will not load is a bug
        raise RuntimeError(
            f"{path} exists but could not be loaded ({exc!r}). Returning an "
            "empty table here would silently drop every verdict it holds; "
            "regenerate it with its builder in scripts/."
        ) from exc


_HOMOGRAPH_LK: dict[str, dict] = _load_homograph_lk()


def _load_audio_pe() -> dict:
    """data/audio_pe_lk.py, keyed by lexicon_key; absent file degrades.

    Audio-confirmed /p/ readings for words the §4 pe-default would read with
    /f/ (scripts/build_audio_pe_lexicon.py). Consulted after every gold and
    legacy lexicon — audio never outranks a native or published verdict — and
    before the rule path."""
    path = Path(__file__).resolve().parent / "data" / "audio_pe_lk.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_yiddish_audio_pe_lk", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, "AUDIO_PE_LK", {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}  # not generated in this checkout: degradation is deliberate
    except Exception as exc:  # a table that EXISTS but will not load is a bug
        raise RuntimeError(
            f"{path} exists but could not be loaded ({exc!r}). Returning an "
            "empty table here would silently drop every verdict it holds; "
            "regenerate it with its builder in scripts/."
        ) from exc


_AUDIO_PE: dict[str, dict] = _load_audio_pe()


def _load_audio_vowel() -> dict:
    """data/audio_vowel_lk.py, keyed by lexicon_key; absent file degrades.

    Audio-confirmed vowel corrections for alef-default words
    (scripts/build_audio_vowel_lexicon.py): the engine's own stressed reading
    with clean-target vowel slots substituted. Same authority slot as the
    audio-pe table."""
    path = Path(__file__).resolve().parent / "data" / "audio_vowel_lk.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_yiddish_audio_vowel_lk", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, "AUDIO_VOWEL_LK", {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}  # not generated in this checkout: degradation is deliberate
    except Exception as exc:  # a table that EXISTS but will not load is a bug
        raise RuntimeError(
            f"{path} exists but could not be loaded ({exc!r}). Returning an "
            "empty table here would silently drop every verdict it holds; "
            "regenerate it with its builder in scripts/."
        ) from exc


_AUDIO_VOWEL: dict[str, dict] = _load_audio_vowel()


def _load_sefaria_pointed() -> dict:
    """data/sefaria_pointed_lk.py, keyed by lexicon_key; absent file degrades."""
    path = Path(__file__).resolve().parent / "data" / "sefaria_pointed_lk.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_yiddish_sefaria_lk", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, "SEFARIA_POINTED_LK", {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}  # not generated in this checkout: degradation is deliberate
    except Exception as exc:  # a table that EXISTS but will not load is a bug
        raise RuntimeError(
            f"{path} exists but could not be loaded ({exc!r}). Returning an "
            "empty table here would silently drop every verdict it holds; "
            "regenerate it with its builder in scripts/."
        ) from exc


_SEFARIA_POINTED: dict[str, dict] = _load_sefaria_pointed()


def _load_model_pointed() -> dict:
    """data/model_pointed_lk.py (phonikud-yi v3 guesses), keyed by lexicon_key."""
    path = Path(__file__).resolve().parent / "data" / "model_pointed_lk.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_yiddish_model_lk", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, "MODEL_POINTED_LK", {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}  # not generated in this checkout: degradation is deliberate
    except Exception as exc:  # a table that EXISTS but will not load is a bug
        raise RuntimeError(
            f"{path} exists but could not be loaded ({exc!r}). Returning an "
            "empty table here would silently drop every verdict it holds; "
            "regenerate it with its builder in scripts/."
        ) from exc


_MODEL_POINTED: dict[str, dict] = _load_model_pointed()


# --- §8 abbreviations --------------------------------------------------------
# A token with a gershayim inside it is an abbreviation by definition (§2.6) and
# never takes the rule path: ש-ל-י-ט-א read as letters is nonsense. Values are
# (primary, [alternates]); they agree with the gold rows for the same spellings.
_ABBREVIATIONS: dict[str, tuple[str, list[str]]] = {
    "ר'": ("rɛb", []),
    "ה'": ("haʃˈɛm", []),
    'שליט"א': ("ʃlˈitə", []),
    'זצ"ל': ("zaʦˈal", []),
    'ז"ל': ("zal", []),
    'זי"ע': ("zxisˈɔj jˈuɡin ulˈajni", []),
    'יו"ט': ("jˈɔntəf", ["jˈɔntif"]),
    'ב"ה': ("bˈurəx haʃˈɛm", []),
}
_ABBREVIATIONS = {lexicon_key(k): v for k, v in _ABBREVIATIONS.items()}

# --- §7.5a acronyms that are pronounced as WORDS -----------------------------
# The letter-name fallback below is right for an acronym the reader does not
# recognize (a gematria year, an unfamiliar rosh-teyves).  It is wrong -- and
# loudly wrong, because these are the most frequent gershayim tokens in the
# corpus -- for the established acronyms that are never spelled out in speech:
# nobody says "rajʃ ʃin jid" for רש"י, they say rˈaʃi.  Spelling those out
# converts a known-unknown into a confidently-shaped wrong answer.
#
# Scope is deliberately narrow: only acronyms whose word reading is settled
# usage.  Everything else (dates כ"ה, years תשפ"ה, ע"ה, רמ"א, הקב"ה, ...) keeps
# the letter-name fallback, which is honest about not knowing.
#
# The readings are editorial, not audio-verified, so they route to the table
# (layer A) but carry LOW confidence: the segmental shape is settled, the
# stress placement is not, and LOW keeps every one of them in the verification
# queue.
#
# Corroboration, gathered before the table was written:
#   * PhoneticXeus (data/audio_lexicon/hebrew_verify.jsonl) heard רש"י as
#     [r ʃ ə] (2 clips) and אר"י as [a r i] (2/3 clips) -- word readings both,
#     neither spelled out.
#   * data/canonical_pointing.tsv points 15 of these 17 independently, and
#     `reconcile` accepts the pointing against the reading below for every one
#     of them (חַזַ"ל, רַמְבַּ"ם, רַמְבַּ"ן, תַּרְיַ"ג, לַ"ג, מַהֲרַ"ם, מַהֲרַ"ל,
#     שַׁ"ס, הַשַּׁ"ס, תַּנַ"ךְ, חַבַּ"ד round-trip through the rule path as well).
#   * The two it disagrees with are kept anyway and are the only judgement
#     calls in the table: של"ה (pointed שְׁלַ"ה, i.e. *ʃla*; the Shloh is called
#     ʃlu) and בעש"ט (pointed בַּעַשְׁ"ט letter by letter; the Baal Shem Tov is
#     the Besht).  האר"י was left OUT for the same reason with no such
#     counterweight -- הָאֲרִ"י wants hu-, and nothing here settles it.
_ACRONYM_WORDS: dict[str, tuple[str, list[str]]] = {
    'רש"י': ("rˈaʃi", []),
    'חז"ל': ("xazˈal", []),
    'רמב"ם': ("rˈambam", []),
    'רמב"ן': ("rˈamban", []),
    'חיד"א': ("xˈidə", []),
    'של"ה': ("ʃlu", []),
    'אר"י': ("ˈari", []),
    'תרי"ג': ("tarjˈaɡ", []),
    'ל"ג': ("laɡ", []),
    'מהר"ם': ("mahˈaram", []),
    'מהר"ל': ("mahˈaral", []),
    'מהרש"א': ("mahˈarʃə", []),
    'בעש"ט': ("bɛʃt", []),
    'ש"ס': ("ʃas", []),
    'הש"ס': ("haʃˈas", []),
    'תנ"ך': ("tˈanax", []),
    'חב"ד': ("xabˈad", []),
}
_ACRONYM_WORDS = {lexicon_key(k): v for k, v in _ACRONYM_WORDS.items()}

# --- §7.5 letter-name fallback for unknown abbreviations ---------------------
# An abbreviation that is not in the table above used to be quarantined: the
# rule path over the gershayim-less string invents a word that nobody says
# (תשפ"ה is not a word). Spec v2 §7.5 says to "treat unknowns by letter-name
# fallback" -- a reader who does not know the acronym spells it out, and that
# is also exactly right for the very common gematria years (תש..).
#
# Values are the Hasidic letter names of §5, written in the §1 closed
# inventory. Final forms never reach here (lexicon_key folds them), but they
# are listed for callers that pass a raw surface string.
_LETTER_NAMES: dict[str, str] = {
    "א": "ˈaləf",
    "ב": "bajs",
    "ג": "ɡiml",
    "ד": "dˈuləd",
    "ה": "haj",
    "ו": "vuv",
    "ז": "zˈajən",
    "ח": "xɛs",
    "ט": "tɛs",
    "י": "jid",
    "כ": "xuf",
    "ך": "xuf",
    "ל": "lˈaməd",
    "מ": "mɛm",
    "ם": "mɛm",
    "נ": "nin",
    "ן": "nin",
    "ס": "sˈaməx",
    "ע": "ˈajən",
    "פ": "paj",
    "ף": "paj",
    "צ": "ʦˈadik",
    "ץ": "ʦˈadik",
    "ק": "kif",
    "ר": "rajʃ",
    "ש": "ʃin",
    "ת": "tuf",
}


def letter_name_ipa(key: str) -> str:
    """§7.5: spell ``key`` out as letter names, or "" if it is not spellable.

    ``key`` is a lexicon key (points stripped, finals folded); the gershayim
    and any geresh are ignored, every remaining character must be a Hebrew
    letter with a name in §5. Digits, Latin letters and the YIVO digraph
    ligatures return "" so the caller can keep its old fallback.
    """
    letters = [c for c in key if c not in '"\'']
    if not letters or any(c not in _LETTER_NAMES for c in letters):
        return ""
    return " ".join(_LETTER_NAMES[c] for c in letters)


# --- §7.5b tokens that SPELL OUT a single letter's name ----------------------
# מ"ם, כ"ף, יו"ד are not acronyms at all: they are the Hebrew names of the
# letters mem, khof and yud, written with a gershayim the way a name is.  The
# letter-name fallback doubles them (מ"ם -> "mɛm mɛm") or expands them
# character by character (יו"ד -> "jid vuv dˈuləd"), which is wrong twice over.
# Keys are the letter-name spellings under `lexicon_key` (finals folded,
# gershayim removed); values are the same §5 names `_LETTER_NAMES` uses.
#
# Spellings that double as a common abbreviation are deliberately absent:
# פ"א / פ"ה (also a folio reference), ת"ו (תיבנה ותיכונן after a city name).
_LETTER_NAME_WORDS: dict[str, str] = {
    "אלף": "ˈaləf",
    "בית": "bajs",
    "גימל": "ɡiml",
    "גמל": "ɡiml",
    "דלת": "dˈuləd",
    "הא": "haj",
    "וו": "vuv",     # ו"ו
    "ווו": "vuv",    # וו"ו — same name, ו spelled with the Yiddish digraph
    "זין": "zˈajən",
    "חית": "xɛs",
    "טית": "tɛs",
    "יוד": "jid",
    "כף": "xuf",
    "למד": "lˈaməd",
    "מם": "mɛm",
    "נון": "nin",
    "סמך": "sˈaməx",
    "עין": "ˈajən",
    "צדי": "ʦˈadik",
    "צדיק": "ʦˈadik",
    "קוף": "kif",
    "ריש": "rajʃ",
    "שין": "ʃin",
    "תיו": "tuf",
}
_LETTER_NAME_WORDS = {lexicon_key(k): v for k, v in _LETTER_NAME_WORDS.items()}


def letter_name_word_ipa(key: str) -> str:
    """§7.5b: the §5 name of the letter ``key`` spells out, or "".

    ``key`` is a lexicon key; the gershayim is what marks it as a name rather
    than the ordinary word with the same letters, so callers must only reach
    here for a token that carries one.
    """
    return _LETTER_NAME_WORDS.get(key.replace('"', "").replace("'", ""), "")

# --- §8 multiword ------------------------------------------------------------
# Spec §8: בית־מדרש / בית מדרש -> bis-mˈɛdrəʃ, while בית alone is bajs (the gold
# row for בית carries "bis- in bˈis-mədrəʃ" as its bracketed note, kept here as
# the alternate). Reduced forms fire only inside the compound, which is exactly
# what a multiword entry expresses.
_MULTIWORD: dict[str, tuple[str, list[str]]] = {
    "בית מדרש": ("bis-mˈɛdrəʃ", ["bˈis-mədrəʃ"]),
    "בית-מדרש": ("bis-mˈɛdrəʃ", ["bˈis-mədrəʃ"]),
    # אַ פּאָר "a pair / a few": after the article, פאר is the noun pur, not the
    # preposition far. The fused spelling אפאר is gold (apˈur | ˈapɔr,
    # chezky-verified "apooor"); the spaced bigram is the same lexeme, so it
    # inherits that verdict. פאר alone stays gold far.
    "א פאר": ("a pˈur", ["a pˈɔr"]),
}

# --- corpus-mined lexicalized MWEs (scripts/mine_lk_mwe.py) ------------------
# Every key below occurs >= 15x in data/yiddish_tts_dataset.tsv (count in the
# comment) and is a fixed collocation, not a quoted verse: the reading is the
# MERGED register (spec v2 §5/§7 -- shuruk -> i, final kometz-hey -> ə), which
# is what embedded LK uses. Verse fragments that the miner also surfaces (מה
# נשתנה, נחמו נחמו, וזאת הברכה ...) are deliberately NOT here: they are quoted
# Whole-Hebrew and belong to read_pointed_wh, not to a lexicalized-Yiddish
# table.
#
# CONFIDENCE. These are a frequency miner's candidates plus an author's reading,
# not a native verdict and not audio: by the standing authority order (gold >
# audio > books > guesses) they rank BELOW the Sefaria and model tables, which
# ship LOW and stay queued. So they ship LOW too, reason 'mwe-mined', and stay
# in the verification queue until a native or an aligner signs off. _MULTIWORD's
# two hand-verified entries (בית מדרש, from the gold's own bracketed note) keep
# HIGH.
#
# The spaced spelling and the makef/hyphen spelling are ONE entry: the aliasing
# loop below registers כ-form keys for every space key, because the space/makef
# choice is an orthographic accident and must not change the phones.
_MULTIWORD_MINED: dict[str, tuple[str, list[str]]] = {
    "זכרונו לברכה": ("zixrˈɔjni livrˈuxə", []),      # 847  zichroyni livruche
    "יום טוב": ("jˈɔntəv", ["jɔjm tɔjv"]),            # 569  = יום-טובֿ (LK table)
    "ארץ ישראל": ("ˈɛrəʦ jisrˈuəl", []),              # 563  both parts gold
    "ראש השנה": ("rˈuʃəʃunə", ["rɔjʃ haʃˈunə"]),      # 415  = ראש-השנה (LK table)
    "כלל ישראל": ("klal jisrˈuəl", []),               # 417  both parts gold
    "יום כיפור": ("jˈunkipər", []),                   # 391  = יום-כּיפּור (LK table)
    "שבת קודש": ("ʃˈabəs kˈɔjdəʃ", []),               # 353  koydesh, not *kidsh
    "ראש חודש": ("rɔjʃ xˈɔjdəʃ", []),                 # 297  both parts gold
    "רבונו של עולם": ("ribˈɔjnə ʃɛl ˈɔjləm", []),     # 245  riboyne shel oylem
    "ריבונו של עולם": ("ribˈɔjnə ʃɛl ˈɔjləm", []),    # 32   (yud-full spelling)
    "ברוך השם": ("bˈurəx haʃˈɛm", []),                # 218  = the ב"ה expansion
    "תשעה באב": ("tˈiʃə buv", []),                    # 177  = תּישעה-באָב (LK table)
    "לשון הרע": ("lˈuʃn hˈurə", []),                  # 153  loshn hore
    "בעל הבית": ("bˈaləbus", ["baləbˈus"]),           # 132  = בעל-הבית, spec §6.2
    "בית דין": ("bˈɛs din", ["bajs din"]),            # 126  LK בעסדין; בית alone bajs
    "ערב שבת": ("ˈɛrɛv ʃˈabəs", []),                  # 123  erev shabes
    "ראש ישיבה": ("rɔjʃ jəʃˈivə", []),                # 112  both parts gold
    "עולם הבא": ("ˈɔjləm habˈu", []),                 # 109  oylem habu
    "אהבת ישראל": ("ˈahavas jisrˈuəl", []),           # 80   both parts gold
    "ימים טובים": ("jˈumim tˈɔjvim", []),             # 79   yumim toyvim
    "קידוש השם": ("kˈidəʃ haʃˈɛm", []),               # 79   both parts gold
    "חול המועד": ("xɔl hamˈɔjəd", []),                # 75   chol hamoyed
    "שבע ברכות": ("ʃˈɛvə brˈuxəs", []),               # 60   sheve bruches
    "עולם הזה": ("ˈɔjləm hˈazɛ", []),                 # 58   oylem haze
    "מחצית השקל": ("maxˈaʦis haʃˈɛkɛl", []),          # 50   machtsis hashekel
    "מלך מלכי המלכים": ("mˈɛlɛx malxˈaj hamlˈuxim", []),  # 40
    "בעזרת השם": ("bəˈɛzras haʃˈɛm", []),             # 35   b'ezras hashem
    "בר מצוה": ("bar mˈiʦvə", []),                    # 33   bar mitsve
    "אשת חיל": ("ˈajʃɛs xˈajil", []),                 # 29   eyshes chayil
    "בעלי בתים": ("baləbˈatim", []),                  # 29   = בעלי-בתים, spec §6.2
    "מסירת נפש": ("məsˈiras nˈejfiʃ", []),            # 24   נפש is gold nˈejfiʃ
    "אם ירצה השם": ("im jˈirʦə haʃˈɛm", []),          # 22   im yirtze hashem
    "מלוה מלכה": ("məlˈavə mˈalkə", []),              # 19   melave malke
    "פסח שני": ("pˈajsəx ʃˈajni", []),                # 19   peysach sheyni
    "שלום בית": ("ʃˈuləm bˈajis", []),                # 18   sholem bayis
    "גמילות חסדים": ("ɡəmˈilus xasˈudim", []),        # 16   חסדים audio xasudim
    "זאת אומרת": ("zɔjs ɔjmˈɛrɛs", []),               # 16   discourse marker
    "יום טובים": ("jˈuntɔjvim", []),                  # 15   = יום-טובֿים (LK table)
}
_MULTIWORD.update(_MULTIWORD_MINED)
_MULTIWORD = {lexicon_key(k): v for k, v in _MULTIWORD.items()}
# Keys that carry the mined provenance, i.e. LOW confidence at emission.
_MULTIWORD_MINED_KEYS: set[str] = {lexicon_key(k) for k in _MULTIWORD_MINED}
# One lexeme, two spellings: יום־טוב is יום טוב. Registered here rather than in
# the table so a new entry cannot forget it; a hyphen key written out by hand
# (בית-מדרש) already exists and is left alone.
for _mw_key in list(_MULTIWORD):
    _mw_hyphen = _mw_key.replace(" ", "-")
    if _mw_hyphen != _mw_key and _mw_hyphen not in _MULTIWORD:
        _MULTIWORD[_mw_hyphen] = _MULTIWORD[_mw_key]
        if _mw_key in _MULTIWORD_MINED_KEYS:
            _MULTIWORD_MINED_KEYS.add(_mw_hyphen)


def _multiword_confidence(key: str) -> tuple[str, str]:
    """(confidence, reason) for a multiword hit -- see _MULTIWORD_MINED."""
    return ("LOW", "mwe-mined") if key in _MULTIWORD_MINED_KEYS else ("HIGH", "")


# Space-separated loshn-koydesh entries (בית המדרש, בית דין, בית עולם ...) were
# matched across token boundaries by _LK_PATTERN when the engine phonemized whole
# strings. Per-token routing would lose them, so they are registered as multiword
# keys whose value is computed by the rule path over the joined string.
_MULTIWORD_LEGACY = {lexicon_key(k) for k in _LOSHN_KOYDESH if " " in k}
_MAX_MULTIWORD = max(
    [len(k.split()) for k in list(_MULTIWORD) + list(_MULTIWORD_LEGACY)] or [1]
)

# --- §2.5 clitics ------------------------------------------------------------
_CLITIC_IPA = {"ס": "s", "מ": "m", "כ": "x"}

# --- §3 LK detector ----------------------------------------------------------
_LK_DETECT = re.compile(r"[חת]|שׂ|כּ")
# The vowel letters of §3's shape heuristic: א ע י ו and the digraphs יי / וי,
# including their precomposed ligatures (ײ ױ), which are the same nuclei.
_VOWEL_LETTERS = frozenset("אעיוײױ")

# --- §4 ambiguous graphemes --------------------------------------------------
# Every runtime application of a DEFAULT on א (a, when the word is not in a
# lexicon that pins u or ɔ) is logged for triage -- §4's own instruction. פ is
# logged the same way; those two account for most naked-rule errors.
A_DEFAULT_LOG: Counter = Counter()
P_DEFAULT_LOG: Counter = Counter()


def reset_default_logs() -> None:
    A_DEFAULT_LOG.clear()
    P_DEFAULT_LOG.clear()


def _lk_detector(bare: str) -> bool:
    """§3: does this token look like unlexiconed loshn-koydesh?

    The marker clauses (ח / ת / שׂ / כּ, suffix ־ות) are near-perfect: Germanic
    Yiddish writes /x/ with כ and /t/ with ט, so those letters ARE the diagnosis.

    The shape clause is where §3's phrasing -- "fewer than 1 vowel-letter per 3
    consonants" -- has to be calibrated, because it decides quarantine (§6.3) and
    a false positive deletes an ordinary word from the training set. Measured
    against the gold's own layer column (authority #1: 96 L rows, 355 G rows):

        cons >= 3 and cons > 3*vowels   (spec-literal)  45 TP / 10 FP
        no vowel letter at all                          48 TP /  0 FP

    The literal reading fires on מענטשן, העלפן, קענסט, האלטן, דארפן, שטארק,
    טרעפן, ברענגט, פונקט -- and in the corpus on פלעגט, שטיקל, דאָרטן, זינגט,
    גאנצן -- because v3's own syllabic finals put three or four consonants around
    a single written vowel. The zero-vowel reading is strictly better on the gold
    on BOTH counts, so it is what runs.
    """
    if _LK_DETECT.search(bare):
        return True
    letters = _strip_points(bare)
    if letters.endswith("ות"):
        return True
    consonants = sum(1 for c in letters if _HEBREW_CHAR.match(c) and c not in _VOWEL_LETTERS)
    vowels = sum(1 for c in letters if c in _VOWEL_LETTERS)
    return vowels == 0 and consonants >= 2


def _lk_nikud(core: str) -> bool:
    """§6.2 side-channel: did the token arrive with its own pointing?

    v3 §6.2 wants the pointing fetched from a pointed source -- a Tashma /
    Sefaria index -- which is not reachable from this engine. What IS available
    is §2.1's side-channel: the token's own nikud, which the letter table
    already reads through the §5 vowel rows. When it is there, the word is not
    an OOV-LK fallback; when it is not, §6.3 applies.
    """
    return any(_vowel_point(marks) for _, marks in _split_units(core))


def _lk_evidence(core: str) -> bool:
    """Whether §6 has ANY lexical footing for an LK-detected token.

    §6.1 merged-LK lexicon -- as a whole word (_LK_PATTERN) or as a _STEM_SUBS
    stem inside an inflected form (חסידישע, שבתדיק) -- or the §6.2 pointing.
    Without one of these, §6.3 is explicit: log OOV-LK, emit nothing.
    """
    if _LK_PATTERN.search(core) or _has_stem_sub(core):
        return True
    return _lk_nikud(core)


def _has_ambiguous_alef(core: str) -> bool:
    """An unpointed א that the rule path must resolve by default (a / ɔ / u)."""
    for ch, marks in _split_units(core):
        if ch == "א" and not _vowel_point(marks):
            return True
    return False


def _has_ambiguous_pe(core: str) -> bool:
    """An unpointed פ/ף, where f vs p is lexical (§4)."""
    for ch, marks in _split_units(core):
        if ch in ("פ", "ף") and DAGESH not in marks and RAFE not in marks:
            return True
    return False


def _pe_point_contradicts(core: str, ipa: str) -> bool:
    """Whether an explicit פּ/פֿ point rules out a stripped-key lexicon reading.

    The lexicon is keyed with nikud stripped (§2.1), so pointed פּאָר lands on
    the gold entry for פאר (far). פ is the one letter whose written point flips
    a phoneme the key can't see: a dagesh promises /p/ and a rafe /f/, and a
    reading with none of the promised phone anywhere cannot be what the writer
    pointed. The rule path reads the pointed form itself, so falling through is
    always safe."""
    for ch, marks in _split_units(core):
        if ch in ("פ", "ף"):
            if DAGESH in marks and "p" not in ipa:
                return True
            if RAFE in marks and "f" not in ipa:
                return True
    return False


def _known_word(core: str) -> bool:
    """Whether some lexicon -- gold or legacy -- pins this whole token."""
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))
    return key in GOLD_LEXICON or bare in _LK_BARE or bare in _WORD_LATIN


def _entry_result(word: str, primary: str, variants: list[str], layer: str,
                  route: str, confidence: str, reason: str = "") -> dict:
    return {
        "word": word,
        "ipa_primary": primary,
        "variants": [v for v in variants if v != primary],
        "layer": layer,
        "route": route,
        "confidence": confidence,
        "reason": reason,
    }


# =====================================================================
# WORD-FINAL DEVOICING VARIANTS
#
# Audio evidence (PhoneticXeus, episode 100313, 40 chunks) shows heavy surface
# devoicing of word-final voiced obstruents -- final /z/ heard as [s] 79:0,
# final /d/ as [t] 24:8 -- while the native reviewer keeps the underlying
# voiced form as the citation reading. v3 therefore leaves every PRIMARY
# voiced and ships the devoiced reading as an extra VARIANT, so forced
# alignment can vote for whichever the speaker actually produced.
#
# The map is the plain voiced->voiceless obstruent pairing. Sonorants
# (m n ŋ l r j) and the already-voiceless set are untouched.
# =====================================================================
FINAL_DEVOICE_MAP = {
    "b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s", "ʒ": "ʃ", "ʤ": "ʧ",
}


def devoiced_final(ipa: str) -> str:
    """The word-final-devoiced reading of ``ipa``, or "" if nothing devoices.

    Multiword ("bˈurəx haʃˈɛm") and hyphenated ("bis-mˈɛdrəʃ") primaries have a
    word boundary at every space and hyphen, so EACH part's final obstruent is
    devoiced -- the audio shows the effect at every such boundary, not only at
    the end of the record.
    """
    parts = re.split(r"([ -])", ipa)
    out: list[str] = []
    changed = False
    for part in parts:
        if part and part[-1] in FINAL_DEVOICE_MAP:
            out.append(part[:-1] + FINAL_DEVOICE_MAP[part[-1]])
            changed = True
        else:
            out.append(part)
    return "".join(out) if changed else ""


def _with_auto_variants(rec: dict) -> dict:
    """Append the auto-generated devoiced-final variant to a §12 record.

    Appended LAST, after any lexicon/gold variants, and only when it is not
    already among them: the gold rows that already list the devoiced reading
    (זאגט zuɡt|zukt, טאג tuɡ|tuk, ביז biz|bis) keep their hand-verified order,
    and the auto pass adds nothing. ipa_primary, layer, route, confidence and
    reason are never touched -- this is additive only. The names of the
    auto-generated forms are recorded in ``auto_variants`` for provenance.
    """
    rec["variants"] = list(rec["variants"])
    rec["auto_variants"] = []
    primary = rec["ipa_primary"]
    if not primary or rec["route"] == "fallback":
        return rec
    auto = devoiced_final(primary)
    if auto and auto != primary and auto not in rec["variants"]:
        rec["variants"].append(auto)
        rec["auto_variants"] = [auto]
    return rec


def _multiword_ipa(key: str) -> str:
    if key in _MULTIWORD:
        return _MULTIWORD[key][0]
    return _rule_path_ipa(key, stress=True)


def _guess_layer(core: str) -> str:
    """Layer for a token no lexicon covers: L if the LK detector fires, else G."""
    return "L" if _lk_detector(_strip_points(core)) else "G"


def _route_token(core: str) -> dict:
    """Route a token, then enforce §1 on the answer before it can be emitted.

    The shape check sits OUTSIDE the routing order on purpose: a clitic split, a
    hyphen join or a lexicon entry can produce an unspeakable string just as a
    bare rule application can, and §1 says such a string must never reach corpus
    output no matter which path built it. Offenders are turned into
    route='fallback' / confidence='LOW' records whose ipa_primary is a flagged
    approximation for the quarantine file.
    """
    result = _route_token_inner(core)
    primary = result["ipa_primary"]
    # A vowelless Hebrew proclitic is a defect no routing step owns: it can
    # come out of the rule path, the stem substitution or a table entry alike.
    # Repair it here, where every path has already had its say.
    if primary:
        repaired = _repair_lk_proclitic(core, primary)
        if repaired is not None:
            result, primary = repaired, repaired["ipa_primary"]
    if result["route"] == "fallback" or not primary:
        return _audio_endorsed_or(core, result)
    flags = []
    if violates_vowel_ratio(primary):
        flags.append("vowel-ratio")
    if ipa_phone_violations(primary):
        flags.append("bad-phone")
    if not flags:
        return result
    reason = ",".join(filter(None, [result["reason"], *flags]))
    return _audio_endorsed_or(core, _entry_result(
        core, primary, result["variants"], result["layer"],
        "fallback", "LOW", reason))


def _audio_endorsed_or(core: str, fallback_result: dict) -> dict:
    """Rescue a would-be-quarantined token with its audio-endorsed reading.

    data/audio_endorsed_lk.py holds readings from the corpus's UNVERIFIED
    pointed tier that PhoneticXeus confirmed against episode audio (mean
    agreement >= 0.70, >= 2 clips). They are emitted at LOW confidence with a
    distinct reason so they stay visible in the verification queue — audio
    endorsement is evidence, not native judgement, and the entries are replaced
    the moment a Chezky verdict lands (the main lexicons route before this).
    """
    entry = _AUDIO_ENDORSED.get(lexicon_key(core))
    if entry is None:
        return _homograph_or(core, fallback_result)
    return _entry_result(core, entry["ipa"], [], "L",
                         "rule", "LOW", "pointed-audio-endorsed")


def _homograph_or(core: str, fallback_result: dict) -> dict:
    """Rescue #1.5: a word the verified editions point more than one way.

    data/homograph_lk.py holds the 'homograph-conflict' types the Sefaria
    rescue refused because no vocalization reached its 80% dominance bar. Two
    kinds live here and each carries its own reason:

      'homograph-collapsed' — EVERY attested pointing READS the same once
        phonemic_fold() drops the editions' cosmetic disagreements (te'amim,
        dagesh lene, holam male). The conflict was in the print, not the mouth,
        so there is nothing for audio to decide and the reading is free. A
        thinly-attested rival reading does not count as collapsed: a handful of
        types qualify, and a word with a live second reading is never here.
      'audio-homograph' — a real two-way split, decided against episode audio
        (>= 3 decided occurrences, winner >= 75% of them). The losing readings
        ride along as variants so alignment can still pick them up.

    It sits BETWEEN the audio-endorsed table and the Sefaria one: an explicit
    audio endorsement of a single-reading word is stronger evidence than a
    verdict picked out of a candidate set, and both are stronger than a book
    pointing that could not even name one reading for this spelling. LOW
    confidence with a distinct reason either way — these stay in the
    verification queue until a native verdict replaces them.
    """
    entry = _HOMOGRAPH_LK.get(lexicon_key(core))
    if entry is None:
        return _sefaria_pointed_or(core, fallback_result)
    return _entry_result(core, entry["ipa"], list(entry.get("variants") or []),
                         "L", "rule", "LOW", entry["reason"])


def _sefaria_pointed_or(core: str, fallback_result: dict) -> dict:
    """Rescue #2: the reading of this word in a VERIFIED pointed edition.

    data/sefaria_pointed_lk.py holds words whose unpointed form has exactly one
    vocalization (or a dominant one, >= 80%) across Sefaria's MAM Tanakh and
    Torat Emet Mishnah/Siddur.

    REGISTER (scripts/register_policy.py): the pointing is read as EMBEDDED
    loshn-koydesh by default — read_pointed_merged(), shuruk -> [i], final
    komets-hey -> [ə] — because that is what the word is doing in a Yiddish
    sentence. read_pointed_wh() is used only where the evidence says the word is
    being QUOTED (audio, or >= 70% of its corpus tokens inside a run of LK
    words). Whichever register loses ships as a VARIANT on the record, so an
    aligner can still vote for the other reading; the table stores it and this
    function passes it through.

    It is consulted only after _AUDIO_ENDORSED misses:
    hearing the word in an episode outranks finding it in a book, because the
    book says how the posuk is chanted and the audio says how this community
    says the word in a Yiddish sentence. Hence LOW confidence and a distinct
    reason — these stay in the verification queue like the audio ones.

    SCOPE, precisely: this fires wherever _route_token() would otherwise hand
    back a fallback, which is NOT the same as "only tokens whose final route is
    'fallback'". The §2.5 clitic split re-routes the stripped core through
    _route_token(), so a rescued core can end up inside a composed token that
    reports route='rule' / reason='clitic' (5 hapax types corpus-wide, e.g.
    כ'אחד xaxd -> xˈɛxud). That is intended — the alternative is emitting the
    Germanic letter table's garbage for the same core — and the LOW confidence
    propagates, so the composed token still lands in the verification queue.
    Sweeping for this requires clearing _ROUTE_CACHE between runs; it will
    otherwise serve pre-rescue answers and hide the difference.
    """
    entry = _SEFARIA_POINTED.get(lexicon_key(core))
    if entry is not None:
        return _entry_result(core, entry["ipa"],
                             list(entry.get("variants") or []), "L",
                             "rule", "LOW", "sefaria-pointed")
    # LAST link — the no-drop policy (2026-08-08): phonikud-yi v3's contextual
    # guess (data/model_pointed_lk.py, 97% held-out accuracy on evidence-backed
    # Hebrew). A guess is better than silence, and it is outranked by every
    # other source above, stays LOW, and stays in the verification queue. Same
    # register policy as the Sefaria table above, on the same evidence.
    entry = _MODEL_POINTED.get(lexicon_key(core))
    if entry is not None:
        return _entry_result(core, entry["ipa"],
                             list(entry.get("variants") or []), "L",
                             "rule", "LOW", "model-pointed-guess")
    return fallback_result


# =====================================================================
# RESCUE #4: LK ROOT + GERMANIC SUFFIX
#
# Every lexicon in the chain above is keyed on the WHOLE token, so an LK root
# that took a Yiddish ending misses all of them at once: רבנוס (רבנו + the
# possessive ס) is not in the gold, not in Sefaria, not in the model table, and
# not even LK-DETECTED any more — the suffix hands the shape heuristic a token
# it reads as Germanic — so it left the rule path as 'rbnis'.
#
# The closed suffix list below is the productive Germanic morphology that
# attaches to LK bases in this corpus: the possessive/plural ס, the plural ן,
# the adjectivizer דיג/דיגע/דיגן, the agent ניק/ניקעס, the feminines טע/שע and
# the diminutive plural לעך. Strip one, route the ROOT through the same chain
# (gold > merged-LK list > audio > homograph > Sefaria > model), and concatenate
# root-IPA + suffix-IPA. The root's stress mark rides along untouched, which is
# the whole point: the suffix is unstressed in every one of these.
#
# It is the LAST link, after the full-form model guess, because a whole-token
# entry is always the better evidence — and _STEM_SUBS (חסידישע, אמתדיג) is an
# earlier and finer mechanism that substitutes a POINTED base inside the rule
# path, so a core it covers is left alone here.
#
# FOOTPRINT, measured over the WHOLE corpus (all 23,666 rows, 92,651 token
# types, _ROUTE_CACHE cleared and _stem_suffix_rescue disabled for the
# 'before' side): 75 types / 274 tokens change, every one of them from a
# consonant string to a word (רבין rbin -> rˈɛbən 66 tokens, רביס rbis ->
# rˈɛbəs 48, גמראס ɡmras -> ɡəmˈurəs, פשטלעך fʃtlɛx -> pʃatləx). A few-thousand
# -row sample is NOT a substitute for this measurement: the sample this rescue
# was first sized on (3,000 rows -> 19 types / 39 tokens) happened to contain
# almost none of the pointed input, which is where the mechanism was doing its
# damage — 236 of its 270 model-guess tokens were pointed GERMANIC words being
# rebuilt out of Hebrew readings before _model_guess_root_is_lk() stopped them.
# =====================================================================
_GERMANIC_SUFFIX_IPA: list[tuple[str, str]] = [
    ("ניקעס", "nikəs"),
    ("דיגע", "diɡə"), ("דיגן", "diɡn"),
    ("דיג", "diɡ"), ("ניק", "nik"), ("לעך", "ləx"),
    ("טע", "tə"), ("שע", "ʃə"),
    ("ס", "s"), ("ן", "n"),
]

# Shortest root the stemmer will accept. Two letters is one letter too few:
# הרשע (ha-ROshe) split as הר 'har' + שע, because a two-letter LK root is
# almost always also the tail of a longer unrelated word, and the tables are
# dense at that length. Every real split measured on the corpus -- פשט+לעך,
# כעס+ן, כבוד+ן, תורה+דיג, צדיק+ס -- has three or more.
_MIN_STEM_ROOT = 3

# Monomorphemic LK words that merely END in a suffix string. They are the
# residue of the corpus-wide sweep: with the length and evidence guards in
# place these were the only splits that came out wrong, and both are the same
# accident -- a root ending in י followed by a genuine root ן, where the
# lookalike split (רבי + ן -> rˈɛbən) is correct 66 times over.
#   בנין  binyen 'building', not בני + ן
#   כדין  kədin 'as required', not כדי + ן
#   שמחס  the defective plural of שמחה (simkhes), not the adjective שמח
#         sumˈajxa 'joyful' + ס -- the seam is clean, so only the lexeme
#         itself says the split is wrong.
# Whole-token entries in any lexicon are the permanent fix; this is the
# stop-gap until one lands. Keys are lexicon_key form.
_STEM_NO_SPLIT: frozenset[str] = frozenset({"בנינ", "כדינ", "שמחס"})


def _lk_table_reading(core: str) -> tuple[str, str] | None:
    """(ipa, reason) for a whole token from any LK source, or None.

    The same lookups the rescue chain runs, in the same order of authority, but
    reusable on a substring: gold (only its L rows -- a Germanic gold word is
    not an LK root), the merged-LK list read through the rule path, then the
    three evidence tables and the model guess.
    """
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))
    entry = GOLD_LEXICON.get(key)
    if entry is not None and entry["layer"] == "L":
        return entry["ipa_primary"], "gold"
    if bare in _LK_BARE:
        return _rule_path_ipa(core, stress=True), "lk-lexicon"
    entry = _AUDIO_ENDORSED.get(key)
    if entry is not None:
        return entry["ipa"], "pointed-audio-endorsed"
    entry = _HOMOGRAPH_LK.get(key)
    if entry is not None:
        return entry["ipa"], entry["reason"]
    entry = _SEFARIA_POINTED.get(key)
    if entry is not None:
        return entry["ipa"], "sefaria-pointed"
    entry = _MODEL_POINTED.get(key)
    if entry is not None:
        return entry["ipa"], "model-pointed-guess"
    return None


def _full_form_resolves(core: str) -> bool:
    """Whether ANY whole-token mechanism already answers for this core.

    The stemmer must never outrank one of these: a full-form entry was written
    for the inflected spelling on purpose, and _STEM_SUBS is the older, more
    precise route into the rule path (אמתדיג -> ˈɛməzdiɡ) that would otherwise
    be shadowed by a gold reading of the bare root.
    """
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))
    if key in GOLD_LEXICON or key in _MULTIWORD or key in _MULTIWORD_LEGACY:
        return True
    if bare in _LK_BARE or bare in _WORD_LATIN:
        return True
    if (key in _AUDIO_ENDORSED or key in _HOMOGRAPH_LK
            or key in _SEFARIA_POINTED or key in _MODEL_POINTED):
        return True
    return _has_stem_sub(core)


def _model_guess_root_is_lk(root: str) -> bool:
    """Is a model-table root really LK -- judged by a test this input can answer?

    _lk_detector's decisive clause is a SHAPE test: no vowel LETTER means the
    vowels are unwritten, i.e. Hebrew. On POINTED input that premise is gone --
    the vowels are written as points, so an ordinary Germanic word hands the
    heuristic a bare skeleton and it fires on every one of them:
    _lk_detector('שנסט'), ('שלכט'), ('שטר'), ('קלפ'), ('פלג') are all True while
    the unpointed spellings שענסט / שלעכט / שטער / קלאפ / פלעג are all correctly
    False. That is how the guard here came to rebuild pointed GERMANIC words out
    of Hebrew readings -- שֶׁנְסְטֶע ʃˈɛnstə -> *ʃnustə, פְלֶגְן flɛɡn -> *pˈɛləɡn --
    which is exactly what the guard exists to prevent (the docstring's own
    negative example, וורטלעך, passes only because it is unpointed).

    So when the root carries its own pointing, only the MARKER clauses count:
    ח / ת / שׂ / כּ and the ־ות suffix, letters Germanic Yiddish does not use to
    spell those sounds. Unpointed roots keep the full detector, shape clause
    included.
    """
    bare = _strip_points(root)
    if not _lk_nikud(root):
        return _lk_detector(bare)
    return bool(_LK_DETECT.search(normalize_surface(root)) or bare.endswith("ות"))


def _phone_list(ipa: str) -> list[str]:
    """``ipa`` split into §1 phone symbols; separators and marks dropped."""
    out: list[str] = []
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS or ipa[i] in PHONE_MARKS:
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i) and sym not in PHONE_MARKS:
                out.append(sym)
                i += len(sym)
                break
        else:
            out.append(ipa[i])
            i += 1
    return out


def _join_stem(root_ipa: str, suffix_ipa: str) -> str | None:
    """Glue a root reading to a suffix reading, or None if the seam is unspeakable.

    Bare concatenation is not a phonology. Two things go wrong at the seam and
    both reached corpus output at route='rule' (QA gate (a) passes them -- they
    are legal phones in a legal ratio):

      GEMINATE   the root's last phone and the suffix's first are the same
                 consonant, or the root reading already doubles one:
                 כוסס *kɔjss, חַזֶרְנֶן *xˈazərnn, תאוועס *taˈavvɛs. Yiddish has no
                 geminates, so such a join is evidence the split (or the root
                 reading behind it) is wrong -- decline it and let the rule path
                 answer.
      HIATUS     the syllabic plural ן on a vowel-final root needs the linking
                 vowel Yiddish actually pronounces: שעה 'ʃu' + ן is shoen ʃuən,
                 not *ʃun. A root already ending in ə takes no second one
                 (תורה + ן -> tɔjrən), and the sibilant ס takes none at all --
                 it attaches straight to the vowel (תעשׂה + ס -> taˈasɛs).
    """
    root_phones = _phone_list(root_ipa)
    suffix_phones = _phone_list(suffix_ipa)
    if not root_phones or not suffix_phones:
        return None
    link = ""
    if (suffix_ipa == "n" and root_phones[-1] in PHONE_VOWELS
            and root_phones[-1] != "ə"):
        link = "ə"
    joined = root_ipa + link + suffix_ipa
    phones = _phone_list(joined)
    if any(a == b and a in PHONE_CONSONANTS for a, b in zip(phones, phones[1:])):
        return None
    return joined


def _stem_suffix_rescue(core: str) -> dict | None:
    """Rescue #4: LK root + Germanic suffix, or None when nothing applies.

    Fires only when (a) no whole-token mechanism answers for the full form,
    (b) the core ends in a listed suffix leaving a root of >= _MIN_STEM_ROOT
    letters, (c) that root is LK -- either an LK-only table holds it, or the §3
    detector fires on it in a form the input can actually answer
    (_model_guess_root_is_lk) -- and (d) the seam it produces is speakable and
    costs no syllable (_join_stem, plus the nucleus-count floor below).
    """
    if _full_form_resolves(core) or lexicon_key(core) in _STEM_NO_SPLIT:
        return None
    units = _split_units(normalize_surface(core))
    letters = "".join(base for base, _ in units)
    for suffix, suffix_ipa in _GERMANIC_SUFFIX_IPA:
        if not letters.endswith(suffix):
            continue
        root_units = units[:-len(suffix)]
        root = unicodedata.normalize(
            "NFC", "".join(base + marks for base, marks in root_units))
        root_bare = _strip_points(root)
        if sum(1 for c in root_bare if _HEBREW_CHAR.match(c)) < _MIN_STEM_ROOT:
            continue
        reading = _lk_table_reading(root)
        if reading is None:
            continue
        root_ipa, reason = reading
        if not root_ipa:
            continue
        # A hit in gold-L / the merged-LK list / audio / homograph / Sefaria IS
        # the LK evidence. The model table is not: it guesses for every
        # quarantined type, so a defectively spelled GERMANIC stem can be
        # sitting in it (וורט 'vvɛrt', וופּ 'vuf') and would have the stemmer
        # rebuild ordinary Yiddish words out of Hebrew readings. On that last
        # source the root has to be LK-DETECTED as well -- and on POINTED input
        # that has to be the marker test, not the shape test (see
        # _model_guess_root_is_lk).
        if reason == "model-pointed-guess" and not _model_guess_root_is_lk(root):
            continue
        joined = _join_stem(root_ipa, suffix_ipa)
        if joined is None:
            continue
        # A rescue may not COST a syllable. The root reading is for the root as
        # a free word, and some of them swallow the vowel that the inflected
        # spelling still writes: חַתֶנֶעס xˈasənəs -> *xˈasnəs, מַחְלוֹקֶס
        # mˈaxlɔjkəs -> *mˈaxlɔjks. The rule path reads every letter of the full
        # form, so its nucleus count is the floor; below it the split is losing
        # information rather than adding it.
        if (vowel_consonant_counts(joined)[0]
                < vowel_consonant_counts(_rule_path_ipa(core, stress=False))[0]):
            continue
        return _entry_result(core, joined, [], "L",
                             "rule", "LOW", reason + "+suffix")
    return None


_PREFIX_RESCUE_TABLE: list[tuple[str, str, str]] = [
    # separable + ge (past participles)
    ("אויסגע", "separable", "ˈoʊzɡə"),
    ("אויפגע", "separable", "ˈoʊfɡə"),
    ("איינגע", "separable", "ˈaːnɡə"),
    # אפ־ is [up], not [ɔp]: gold אפשאצן ˈupʃaʦn (Chezky), אראפ arˈup below,
    # and the corpus audio (ˈupɡəmaxt 4/4, ˈupɡəhaltn 4/4) all agree.
    ("אפגע", "separable", "ˈupɡə"),
    ("אנגע", "separable", "ˈunɡə"),
    ("אונטערגע", "separable", "ˈintərɡə"),
    ("איבערגע", "separable", "ˈibərɡə"),
    # ˈ sits immediately before the stressed VOWEL (§1), never on the onset.
    ("דורכגע", "separable", "dˈirxɡə"),
    ("מיטגע", "separable", "mˈitɡə"),
    ("צוגע", "separable", "ʦˈiɡə"),
    ("ארויסגע", "separable", "arˈoʊzɡə"),
    ("ארויפגע", "separable", "arˈoʊfɡə"),
    ("אראפגע", "separable", "arˈupɡə"),
    ("אריינגע", "separable", "arˈaːnɡə"),
    ("ארונטערגע", "separable", "arˈintərɡə"),
    ("אריבערגע", "separable", "arˈibərɡə"),
    ("אדורכגע", "separable", "adˈirxɡə"),
    ("אוועקגע", "separable", "avˈɛkɡə"),
    ("אהיימגע", "separable", "ahˈajmɡə"),
    ("צוריקגע", "separable", "ʦirˈikɡə"),
    # directional
    ("ארויס", "separable", "arˈoʊs"),
    ("ארויפ", "separable", "arˈoʊf"),
    ("אראפ", "separable", "arˈup"),
    ("אריין", "separable", "arˈaːn"),
    ("ארונטער", "separable", "arˈintər"),
    ("אריבער", "separable", "arˈibər"),
    ("אדורך", "separable", "adˈirx"),
    ("אוועק", "separable", "avˈɛk"),
    ("אהיים", "separable", "ahˈajm"),
    ("אנידער", "separable", "anˈidər"),
    ("ארום", "separable", "arˈim"),
    ("אהין", "separable", "ahˈin"),
    ("אהער", "separable", "ahˈɛr"),
    ("צוריק", "separable", "ʦirˈik"),
    # separable
    ("אונטער", "separable", "ˈintər"),
    ("איבער", "separable", "ˈibər"),
    ("דורכ", "separable", "dˈirx"),
    ("פארביי", "separable", "farbˈaj"),
    ("צוזאמען", "separable", "ʦizˈamən"),
    ("אנטקעגן", "separable", "antkˈejɡn"),
    ("אויס", "separable", "ˈoʊs"),
    ("אויפ", "separable", "ˈoʊf"),
    ("איינ", "separable", "ˈaːn"),
    ("אפ", "separable", "ˈup"),
    ("אנ", "separable", "ˈun"),
    ("מיט", "separable", "mˈit"),
    ("צו", "separable", "ʦˈi"),
    # unstressed
    ("גע", "unstressed", "ɡə"),
    ("בא", "unstressed", "ba"),
    ("בע", "unstressed", "bə"),
    ("פאר", "unstressed", "far"),
    ("דער", "unstressed", "dər"),
    ("צע", "unstressed", "ʦə"),
    ("מיס", "unstressed", "mis"),
]
_PREFIX_RESCUE_TABLE.sort(key=lambda item: len(item[0]), reverse=True)

# Words that LOOK like prefix+stem but are single lexemes; the coincidental
# tail resolves in a lexicon, so only the lexeme itself can say the split is
# wrong (same idea as _STEM_NO_SPLIT for suffixes). Keys are lexicon_key form.
#   פארעם  'form' (fˈɔrəm), not far+עם -> *farˈejm
#   איינטאג 'one day': the NUMBER איין (ajn), not the verbal prefix aːn —
#           the corpus audio agrees (ˈajntuɡ)
_PREFIX_NO_SPLIT: frozenset[str] = frozenset({"פארעמ", "אײנטאג", "איינטאג"})


# Stems that must never anchor a VERBAL composition (separable prefixes and
# the participle גע־) even though a lexicon resolves them: closed-class
# function words and contraction homographs are never verb stems —
#   אנ+דער   is not the article (אנדער 'other' is ˈandər, gold אנדערע agrees)
#   אנ+כי    is not LK ki (אנכי is the biblical 'I')
#   גע+פארן  is not the contraction far+n (the participle vowel is u: the
#            audio pool has אראפגעפארן arˈupɡəfurn 4/4)
# The same stems ARE legitimate after unstressed דער־/פאר־, which build the
# pronominal-adverb class: דער+פון dərfˈin, פאר+דעם fardˈejm, דער+פאר dərfˈar
# (the gold anchors נאכדעם nuxdˈejm and דערנאך dərnˈux fix the pattern).
# Bare (point-stripped) forms.
_PREFIX_BAD_STEMS: frozenset[str] = frozenset({
    "דער", "דעם", "די", "דאס", "דו", "ער", "זי", "עס", "איר", "מיר",
    "זיי", "אים", "אונז", "כי", "עם", "אין", "פון", "פאר", "פארן",
    "צו", "ביי", "נאך", "און", "אבער", "נאר", "אויך", "איז", "האט",
    "וואס", "ווי", "ווער", "וועט",
})


def _ensure_stress(ipa: str) -> str:
    """Mark the first vowel if the reading carries no stress mark.

    A stem that was a monosyllable on its own (§1: unmarked) becomes a
    polysyllable once a prefix is attached and must take a mark. The scan is by
    phone TOKEN, not by character: ej/oʊ begin with 'e'/'o', which are not
    phones, so a character scan silently leaves bejtn -> *ɡəbejtn unmarked.
    """
    if STRESS in ipa:
        return ipa
    i = 0
    while i < len(ipa):
        if any(ipa.startswith(v, i) for v in PHONE_VOWELS):
            return ipa[:i] + STRESS + ipa[i:]
        i += 1
    return ipa


def _get_stem_reading(stem: str) -> tuple[str, str, str] | tuple[None, None, None]:
    """Look up a candidate stem in gold/legacy tables.

    Audio tables (_AUDIO_PE/_AUDIO_VOWEL) are deliberately NOT stem sources:
    they are MED acoustic verdicts about one word, and composition would
    amplify them onto words the audio never voted on (גע+טראפן must not
    inherit the noun trapn 'drops' onto the participle of treffen). Stems
    must be anchored in a native or curated reading."""
    key = lexicon_key(stem)
    bare = _strip_points(stem)
    if key in GOLD_LEXICON:
        entry = GOLD_LEXICON[key]
        return entry["ipa_primary"], entry.get("layer", "G"), "gold"
    if bare in _WORD_LATIN:
        ipa = hebrew_to_ipa(bare, stress=True)
        return ipa, "G", "word-latin"
    if bare in _LK_BARE:
        ipa = hebrew_to_ipa(bare, stress=True)
        return ipa, "L", "lk"
    return None, None, None


def _prefix_stem_rescue(core: str) -> dict | None:
    """Rescue #5: Known prefix + gold/lexicon stem.

    Fires only when the whole form has missed all lexicons, and the word begins
    with a recognized verbal/morphological prefix whose stem is anchored in authority #1
    (gold lexicon) or legacy/audio tables. Preserves stem vowel classes (untershraybn ->
    ˈintərʃraːbn, opzogn -> ˈupzuɡn) and prefix stress (aynkoyfn -> ˈaːnkɔjfn).
    """
    if _full_form_resolves(core) or lexicon_key(core) in _PREFIX_NO_SPLIT:
        return None
    bare = _strip_points(normalize_surface(core))
    if len(bare) < 4:
        return None
    for prefix, ptype, prefix_ipa in _PREFIX_RESCUE_TABLE:
        if bare.startswith(prefix) and len(bare) >= len(prefix) + 2:
            stem = bare[len(prefix):]
            # verbal contexts (separable prefixes, participle גע־) never take
            # a function-word stem; unstressed דער־/פאר־ etc. may (pronominal
            # adverbs: דערפון, פארדעם) — see _PREFIX_BAD_STEMS.
            if (ptype == "separable" or prefix == "גע") \
                    and stem in _PREFIX_BAD_STEMS:
                continue
            stem_ipa, layer, src = _get_stem_reading(stem)
            if stem_ipa is None:
                continue
            if ptype == "unstressed":
                stem_ipa = _ensure_stress(stem_ipa)
                raw = prefix_ipa + stem_ipa
            else:
                stem_nostress = stem_ipa.replace(STRESS, "")
                raw = prefix_ipa + stem_nostress
            joined = reduce_unstressed(postlexical(raw))
            if ipa_phone_violations(joined):
                continue
            return _entry_result(core, joined, [], layer, "rule", "MED",
                                 f"prefix-rescue:{src}+{prefix}")
    return None


# Hebrew proclitics: one letter, written joined, carrying their own vowel that
# the unpointed spelling does not show. Without them the rule path reads the
# letter as a bare consonant and emits an unpronounceable onset cluster —
# השבת -> *hʃˈabəs, בשבת -> *bʃˈabəs. Values are the ordinary Ashkenazi
# readings; the article הַ is [ha], the rest are shva-na [ə] except מִ.
_LK_PROCLITICS: dict[str, str] = {
    "ה": "ha",   # definite article
    "ב": "bə",   # in / with
    "ל": "lə",   # to / for      (לחיים ləxˈajim, the pointed sources agree)
    "כ": "kə",   # like / as
    "ד": "də",   # Aramaic relative
    "ש": "ʃə",   # relative
    "מ": "mi",   # from
    "ו": "və",   # and
}


def _lk_stem_reading(stem: str) -> tuple[str, str] | None:
    """A loshn-koydesh reading for a would-be stem, from a REAL source.

    Deliberately narrower than _lk_table_reading: the model-guess table is
    excluded. Composing a prefix onto a model guess is inference stacked on
    inference, and the result would look no different from evidence.
    """
    key = lexicon_key(stem)
    bare = _strip_points(normalize_surface(stem))
    entry = GOLD_LEXICON.get(key)
    if entry is not None and entry["layer"] == "L":
        return entry["ipa_primary"], "gold"
    if bare in _LK_BARE:
        return _rule_path_ipa(stem, stress=True), "lk-lexicon"
    entry = _AUDIO_ENDORSED.get(key)
    if entry is not None:
        return entry["ipa"], "audio-endorsed"
    entry = _SEFARIA_POINTED.get(key)
    if entry is not None:
        return entry["ipa"], "sefaria"
    if any(pat.fullmatch(bare) for pat, _ in _STEM_SUB_RE):
        # a §6.2 base in its own right (חסיד -> כאָסיד): the substitution table
        # is an audio-matched reading of exactly this root, not a guess
        return _rule_path_ipa(stem, stress=True), "stem-sub"
    return None


# The consonant each proclitic contributes, for recognising the defect.
_PROCLITIC_ONSET: dict[str, str] = {
    "ה": "h", "ב": "b", "ל": "l", "כ": "k",
    "ד": "d", "ש": "ʃ", "מ": "m", "ו": "v",
}


def _repair_lk_proclitic(core: str, ipa: str) -> dict | None:
    """Repair a proclitic that was read as a bare consonant.

    An unpointed ה/ב/ל/כ/ד/ש/מ/ו in front of a loshn-koydesh root carries a
    vowel the spelling does not write. Whatever path produced the reading —
    usually the §6.2 stem substitution, which fixes the ROOT's vowels and
    leaves the proclitic bare — the result is an onset cluster no speaker can
    say: השבת -> *hʃˈabəs, בשבת -> *bʃˈabəs, החסיד -> *hxˈusid.

    This is deliberately a REPAIR, not a routing step. It fires only on a
    reading that is already defective (proclitic consonant + another consonant,
    no vowel between), so a word that already reads well — including one with a
    published pointing of the whole form, like המלך hamˈɛlɛx — is never
    recomposed and no authority is inverted.

    The stem must be attested on its own. That is what separates a proclitic
    from a lookalike first letter: שלחן (shulkhn) and בדחן (badkhn) are single
    roots whose tails לחן/דחן are attested nowhere, so they stay untouched —
    the same trap the Germanic prefix rescue hit with אנדער.

    LOW confidence: the segmentation is an inference about an unpointed word,
    so the repaired reading stays in the verification queue.
    """
    bare = _strip_points(normalize_surface(core))
    if len(bare) < 4 or bare[0] not in _LK_PROCLITICS:
        return None
    naked = ipa.replace(STRESS, "")
    onset = _PROCLITIC_ONSET[bare[0]]
    if not naked.startswith(onset):
        return None
    rest = naked[len(onset):]
    if not rest or any(rest.startswith(v) for v in PHONE_VOWELS):
        return None  # the proclitic already has its vowel: nothing to repair
    if not _lk_detector(bare):
        return None  # a Germanic word is never carrying a Hebrew proclitic
    stem = bare[1:]
    if len(stem) < 3:
        return None  # two letters is too little to be sure of a root
    got = _lk_stem_reading(stem)
    if got is None:
        return None
    stem_ipa, src = got

    # A pointing of the WHOLE word always wins over this composition. It is
    # direct evidence about this token and it knows two things the rule cannot:
    # that the article contracts into the proclitic (בַּתּוֹרָה is ba-, not
    # bə-), and which morphological form the root takes — gold בית is the
    # CONSTRUCT bajs (bays-medresh), while הבית needs the absolute habˈajis.
    # Composing gold onto a proclitic would have shipped *habˈajs.
    #
    # Where such a pointing disagrees with gold about the root (sefaria reads
    # הַשַּׁבָּת as haʃˈabus where Chezky says ʃˈabəs), that is the standing
    # pointing-vs-gold conflict class: it belongs in the native queue, not in a
    # silent flip decided here.
    whole = _lk_table_reading(core)
    if whole is not None:
        whole_ipa, whole_src = whole
        naked_whole = whole_ipa.replace(STRESS, "")
        if (not naked_whole.startswith(onset)
                or any(naked_whole[len(onset):].startswith(v)
                       for v in PHONE_VOWELS)):
            if whole_ipa == ipa or ipa_phone_violations(whole_ipa):
                return None
            return _entry_result(core, whole_ipa, [], "L", "rule", "LOW",
                                 f"lk-proclitic-pointed:{whole_src}")
        return None  # the pointing has the same defect: nothing better to say

    joined = reduce_unstressed(postlexical(
        _LK_PROCLITICS[bare[0]] + _ensure_stress(stem_ipa)))
    if ipa_phone_violations(joined) or joined == ipa:
        return None
    return _entry_result(core, joined, [], "L", "rule", "LOW",
                         f"lk-proclitic:{src}+{bare[0]}")


def _route_token_inner(core: str) -> dict:
    """Route one punctuation-free token through §3's strict order."""
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))

    # 1. abbreviation table (§3.1). A mid-word gershayim never reaches the rules.
    if key in _ABBREVIATIONS:
        primary, alts = _ABBREVIATIONS[key]
        return _entry_result(core, primary, [primary, *alts], "A", "lexicon", "HIGH")
    if '"' in key:
        # §7.5a: an acronym that is read as a WORD (רש"י -> rˈaʃi) must never
        # be spelled out. Editorial reading -> table route at LOW confidence,
        # so it stays in the verification queue.
        if key in _ACRONYM_WORDS:
            primary, alts = _ACRONYM_WORDS[key]
            return _entry_result(core, primary, [primary, *alts], "A",
                                 "lexicon", "LOW", "acronym-word")
        # §7.5b: the token IS a letter's name (מ"ם -> mɛm), not an acronym.
        named = letter_name_word_ipa(key)
        if named:
            return _entry_result(core, named, [], "A", "rule", "LOW",
                                 "letter-name-word")
        # §7.5: an unknown acronym is read out letter by letter (תשפ"ה ->
        # tuf ʃin paj haj). Only spellings with an unnamed character (digits,
        # Latin) still fall through to the quarantined rule approximation.
        spelled = letter_name_ipa(key)
        if spelled:
            return _entry_result(core, spelled, [], "A", "rule", "LOW",
                                 "letter-names")
        approx = _rule_path_ipa(key.replace('"', ""), stress=True)
        return _entry_result(core, approx, [], "A", "fallback", "LOW",
                             "unknown-abbreviation")

    # §2.2 second pass: an apostrophe at either edge is a quote mark, not part
    # of the word (זיין' , 'ישראל'). It is stripped only now, because the same
    # character is load-bearing inside the abbreviations handled just above.
    unquoted = core.strip("'")
    if unquoted and unquoted != core:
        return _route_token(unquoted)

    # 2. multiword table (§3.2) -- only reachable when a caller hands us the
    #    joined string; hebrew_to_ipa matches it over the token stream instead.
    if key in _MULTIWORD or key in _MULTIWORD_LEGACY:
        primary = _multiword_ipa(key)
        alts = _MULTIWORD.get(key, ("", []))[1]
        conf, why = _multiword_confidence(key)
        return _entry_result(core, primary, [primary, *alts], "L", "lexicon",
                             conf, why)

    # 3. gold lexicon (§3.3) -- authority #1, overrides every rule and every
    #    legacy dict. Skipped only for the pointed Whole-Hebrew readings that the
    #    engine deliberately distinguishes from their merged spellings (עוֹלָם),
    #    and when the writer's own פּ/פֿ point contradicts the reading the
    #    point-stripped key looked up (פּאָר must not read gold far).
    entry = GOLD_LEXICON.get(key)
    if entry is not None and not (bare in _WH_WHEN_POINTED and _vowel_point(core)) \
            and not _pe_point_contradicts(core, entry["ipa_primary"]):
        return _entry_result(core, entry["ipa_primary"], entry["variants"],
                             entry["layer"], "lexicon", "HIGH")

    # 4./5. legacy whole-token lexicons: the merged-LK list (§6.1), the
    #       high-frequency word list and the loan list (§7). Still lexicon hits.
    # (§2.1: the key is point-stripped. The old "and not _vowel_point(core)"
    #  guard dropped a pointed token out of its own lexicon entry.)
    if bare in _LK_BARE or bare in _WORD_LATIN:
        primary = _rule_path_ipa(core, stress=True)
        layer = "L" if bare in _LK_BARE else _guess_layer(core)
        return _entry_result(core, primary, [], layer, "lexicon", "HIGH")

    # 5.5 audio-confirmed pe flips: words whose §4 f-default the corpus audio
    #     contradicts unanimously (scripts/xeus_pe_sweep.py). Below every
    #     gold/legacy table, above the rule path; MED because the evidence is
    #     acoustic, not native. An explicit rafe still wins via the rule path
    #     upstream (the token then never carries reason pe-default).
    entry = _AUDIO_PE.get(key)
    if entry is not None and _strip_points(core) == core:
        return _entry_result(core, entry["ipa"], [], _guess_layer(core),
                             "lexicon", "MED", "audio-pe")

    # 5.6 audio-confirmed vowel corrections (alef-default words heard with a
    #     clean-target vowel across clips). Same tier and guards as 5.5.
    entry = _AUDIO_VOWEL.get(key)
    if entry is not None and _strip_points(core) == core:
        return _entry_result(core, entry["ipa"], [], _guess_layer(core),
                             "lexicon", "MED", "audio-vowel")

    # §2.3 hyphen / makef: process the parts separately, unless the whole string
    # was matched above by the multiword or LK table.
    if "-" in core and any(p for p in core.split("-")):
        parts = [p for p in core.split("-") if p]
        routed = [_route_token(p) for p in parts]
        primary = "-".join(r["ipa_primary"] for r in routed)
        route = "lexicon" if all(r["route"] == "lexicon" for r in routed) else (
            "fallback" if any(r["route"] == "fallback" for r in routed) else "rule")
        conf = ("LOW" if any(r["confidence"] == "LOW" for r in routed)
                else "MED" if any(r["confidence"] == "MED" for r in routed) else "HIGH")
        return _entry_result(core, primary, [], routed[0]["layer"], route, conf,
                             "hyphen-split")

    # §2.5 clitics: ס' מ' כ' detach, then the remainder is processed as a word.
    # (ר' and ה' were taken by the abbreviation table above.) The apostrophe-less
    # variant fires only when the remainder is itself a known word: סאיז -> s+iz.
    clitic = ""
    rest = ""
    if len(core) > 2 and core[0] in _CLITIC_IPA and core[1] == "'":
        clitic, rest = core[0], core[2:]
    elif len(core) > 2 and core[0] in _CLITIC_IPA and core[1] == "א" and _known_word(core[1:]):
        clitic, rest = core[0], core[1:]
    if clitic and rest:
        tail = _route_token(rest)
        primary = _CLITIC_IPA[clitic] + tail["ipa_primary"]
        conf = "MED" if tail["confidence"] == "HIGH" else tail["confidence"]
        return _entry_result(core, primary, [], tail["layer"], "rule", conf, "clitic")

    # 5.5 stemmer (rescue #4). Every whole-token table has missed by now, and
    #     the clitic / hyphen splits above have had their turn, so this is where
    #     an LK root wearing a Yiddish suffix can finally be seen. It runs ahead
    #     of the rule path rather than inside the fallback chain because the
    #     suffix can hide the root from the §3 LK detector entirely (רבנוס reads
    #     as Germanic and would otherwise leave here as a confident 'rbnis',
    #     never reaching a rescue at all).
    stemmed = _stem_suffix_rescue(core)
    if stemmed is not None:
        return stemmed

    # 5.6 prefix-stem rescue (rescue #5). Prefixed verbs/words where the stem
    #     is anchored in the gold lexicon or legacy/audio tables. Preserves
    #     stem vowel classes (untershraybn -> ˈintərʃraːbn, opzogn -> ˈupzuɡn)
    #     and separable prefix stress (aynkoyfn -> ˈaːnkɔjfn).
    prefixed = _prefix_stem_rescue(core)
    if prefixed is not None:
        return prefixed

    # 6. rule path (§3.6). Confidence is MED unless an ambiguous grapheme had to
    #    be defaulted or the LK detector fired on a word no lexicon knows -- both
    #    are LOW by §12 and both are logged for the next verification batch.
    is_lk = _lk_detector(bare)
    # §6.2: a pointed LK token is read through the §5 nikud table and takes the
    # §11.5 penult default rather than the Germanic §11.7 initial one -- unless
    # a _STEM_SUBS base is what is being read. Those are LK bases that have
    # TAKEN YIDDISH MORPHOLOGY (אמתדיג, שבתדיגע, חסידישע); the substitution
    # rewrites them into their Yiddish spelling and Yiddish stems take §11.7
    # initial stress. Without this the pointed and unpointed spellings of one
    # word came out differently stressed (ˈɛməzdiɡ vs *əmˈɛzdiɡ).
    primary = _rule_path_ipa(core, stress=True,
                             lk_penult=(is_lk and _lk_nikud(core)
                                        and not _has_stem_sub(core)))
    reasons = []
    if _has_ambiguous_alef(core):
        reasons.append("alef-default")
    if _has_ambiguous_pe(core):
        reasons.append("pe-default")
    if is_lk:
        reasons.append("lk-fallback")
        if not _lk_evidence(core):
            # §6.3, LITERALLY: "No pointing found -> log OOV-LK, emit nothing to
            # the training set for that token (a flagged schwa-filled
            # approximation may go to a quarantine file)." The engine used to
            # append the reason and then hand the Germanic letter table's answer
            # back as route='rule', so 13,308 unlexiconed LK types (91,937
            # tokens, 5.0% of the corpus) were emitted as well-shaped garbage --
            # hkdiʃ, xlilh, mʦrim, mxliks, iʃxr -- which neither QA gate (a) nor
            # (b) can see. route='fallback' is what the quarantine file and
            # hebrew_to_ipa both key on.
            return _entry_result(core, primary, [], "L", "fallback", "LOW",
                                 ",".join(reasons))
    # §1/§6.3 shape enforcement happens in _route_token, which wraps this.
    conf = "LOW" if reasons else "MED"
    return _entry_result(core, primary, [], _guess_layer(core), "rule", conf,
                         ",".join(reasons))


_ROUTE_CACHE: dict[str, dict] = {}


def g2p_token(word: str, context: str | None = None) -> dict:
    """Phonemize ONE token and report how the answer was reached (§12).

    Returns ``{word, ipa_primary, variants, layer, route, confidence, reason}``:

      route       'lexicon' (gold / abbreviation / multiword / legacy list),
                  'rule' (Germanic or LK rule path), or 'fallback' (the output
                  is not fit for corpus emission -- quarantine it)
      confidence  HIGH = lexicon, MED = unambiguous rule, LOW = a defaulted
                  ambiguous א/פ, an LK fallback, or a §1 violation

    ``context`` is accepted for the §9 homograph disambiguators (adverb slot,
    plural-noun context, ...). None are wired yet: the primary is emitted and
    every alternate reading is returned in ``variants`` for the forced-alignment
    vote, which is what §9 asks for today.
    """
    _, core, _ = split_affixes(normalize_surface(word))
    if not core:
        return _entry_result(word, "", [], "X", "fallback", "LOW", "empty")
    cached = _ROUTE_CACHE.get(core)
    if cached is None:
        cached = _route_token(core)
        _ROUTE_CACHE[core] = cached
    result = _with_auto_variants(dict(cached, word=word))
    if "alef-default" in result["reason"]:
        A_DEFAULT_LOG[core] += 1
    if "pe-default" in result["reason"]:
        P_DEFAULT_LOG[core] += 1
    return result


def _multiword_match(tokens: list[str], i: int) -> tuple[int, str, str] | None:
    """Longest multiword-table match starting at ``tokens[i]``, if any.

    Punctuation BETWEEN the members blocks the match. The key is built from the
    cores alone, so without this guard "ער איז א בעל. הבית איז גוט" fuses across
    the full stop into one בעל-הבית record and the stop is deleted with it (the
    record keeps only the first member's lead and the last member's trail) --
    two sentences silently merged into one word. A fixed collocation is a
    contiguous span; if the writer put a comma or a period in the middle, it is
    not that collocation.
    """
    for n in range(min(_MAX_MULTIWORD, len(tokens) - i), 1, -1):
        split = [split_affixes(t) for t in tokens[i:i + n]]
        cores = [core for _, core, _ in split]
        if not all(cores):
            continue
        if any(trail for _, _, trail in split[:-1]) or any(
                lead for lead, _, _ in split[1:]):
            continue
        key = lexicon_key(" ".join(cores))
        if key in _MULTIWORD or key in _MULTIWORD_LEGACY:
            return n, key, _multiword_ipa(key)
    return None


def g2p_tokens(text: str) -> list[dict]:
    """Route a whole string and return one §12 record per token, in order.

    Multiword entries (§8) are matched over the token stream and come back as a
    single record whose ``word`` is the joined spelling. Each record carries the
    surrounding punctuation in ``lead``/``trail`` so a caller can rebuild the
    line; hebrew_to_ipa does exactly that.
    """
    tokens = normalize_surface(strip_tags(text)).split()
    records: list[dict] = []
    i = 0
    while i < len(tokens):
        match = _multiword_match(tokens, i)
        if match is not None:
            count, key, ipa = match
            cores = [split_affixes(t)[1] for t in tokens[i:i + count]]
            conf, why = _multiword_confidence(key)
            rec = _with_auto_variants(
                _entry_result(" ".join(cores), ipa, [], "L", "lexicon",
                              conf, why))
            rec["lead"] = split_affixes(tokens[i])[0]
            rec["trail"] = split_affixes(tokens[i + count - 1])[2]
            records.append(rec)
            i += count
            continue
        lead, core, trail = split_affixes(tokens[i])
        if core:
            rec = g2p_token(core)
        else:
            rec = _entry_result(tokens[i], "", [], "X", "fallback", "LOW", "punctuation")
            lead, trail = tokens[i], ""
        rec["lead"], rec["trail"] = lead, trail
        records.append(rec)
        i += 1
    return records


def hebrew_to_ipa(text: str, stress: bool = True, quarantine: bool = True) -> str:
    """Hebrew-script Yiddish -> IPA, emitting the lexicon primary per token.

    ``stress=True`` (the production path) runs §3 routing: gold lexicon first,
    rules only where no table knows the word.

    ``quarantine=True`` (the default) enforces §1 and §6.3 on the STRING, not
    only on the per-token record: a token whose route is 'fallback' -- a
    vowel-less LK consonant string (mlx, ʃm, mdrʃ, lbrxh), an unlexiconed
    unpointed LK word, an out-of-inventory token such as a phone number or a URL
    -- contributes nothing but its surrounding punctuation. The router always
    knew; this function used to build its string from ``ipa_primary`` alone and
    threw the verdict away, so every consumer that is not the corpus runner got
    the forbidden strings. Pass ``quarantine=False`` for triage tooling that
    wants to SEE the flagged approximation.

    ``stress=False`` reproduces the pre-prosody rule-path output unchanged. The
    gold IPA is stress-bearing and post-reduction (ʃˈabəs), so it cannot be
    projected back onto an unstressed, unreduced string without inventing a
    reading; the flag therefore selects the rule path wholesale rather than a
    half-lexiconed hybrid. Every gold-reproduction gate is defined on
    ``stress=True``, which is also what g2p_token and the corpus runner use.
    """
    if not stress:
        return _rule_path_ipa(text, stress=False)
    out = [
        # §2.2: quotes around a word are stripped, not phonemized -- they carry
        # no prosodic content, unlike . , ! ? which downstream TTS reads as
        # phrase breaks. Mid-word gershayim never reach here (abbreviation table).
        r["lead"].replace('"', "") + ("" if quarantine and r["route"] == "fallback"
                     else r["ipa_primary"]) + r["trail"].replace('"', "")
        for r in g2p_tokens(text)
    ]
    return normalize_ipa_spacing(" ".join(out))


# Letters that essentially only occur in Hebrew/Aramaic-origin words: Germanic
# Yiddish writes /kh/ with כ and /t/ with ט, so ח / ת / שׂ surviving Stage 1 mark
# a word the Loshn-Koydesh lexicon does not know. Such words get Germanic
# initial stress by default, which is wrong for most of them (penultimate is the
# LK norm), so they are worth surfacing rather than silently mis-stressing.
_LK_MARKER = re.compile(r"[\u05d7\u05ea]|\u05e9\u05c2")


def find_oov_loshn_koydesh(text: str) -> list[str]:
    """Words that look Hebrew/Aramaic but are missing from the LK lexicon.

    These fall through to the Germanic initial-stress default. Use this to grow
    _LOSHN_KOYDESH / _STRESS_OVERRIDES from real corpus data instead of guessing.
    """
    after_lexicon = _LK_PATTERN.sub(_lk_replace, strip_tags(text))
    out = []
    for word in after_lexicon.split():
        bare = _strip_points(word.strip(".,!?;:\"'()-"))
        if bare and _LK_MARKER.search(bare) and bare not in _LK_BARE:
            out.append(bare)
    return out


def validate_ipa_vocab(ipa: str, char_to_id: dict[str, int]) -> tuple[str, list[str]]:
    missing = sorted({ch for ch in ipa if ch not in char_to_id and not ch.isspace()})
    return ipa, missing

# =====================================================================
# STAGE 6: WHOLE-HEBREW (WH) READING REGISTER  — spec v2 §7.1
#
# OPT-IN ONLY. Nothing above this line calls into it: hebrew_to_ipa,
# g2p_token, _route_token and every gate keep the MERGED register exactly as
# it was. The entry point is read_pointed_wh(), which the Sefaria rescue
# builder calls for VERIFIED pointed quotations (pesukim, citations) where the
# book pointing is trustworthy and the word is being *read as Hebrew*, not
# absorbed into Yiddish.
#
# WHAT DIFFERS FROM THE MERGED §5 TABLE (_POINT_TO_LATIN):
#   (a) shuruk / kubuts read [u]. The merged register routes them through the
#       Latin label "u", which _LATIN_TO_IPA maps to [i] (the near-exceptionless
#       Yiddish u->i shift: קִדּוּשׁ kidesh). In a quoted posuk that shift does
#       not apply — וּבֵרַכְתִּי is uvajraxti, not *ivajraxti.
#   (b) shva-na is pronounced [ə]. The merged register reads every sheva as
#       silent and lets the Latin layer re-insert a schwa only where a syllable
#       demands one, which drops the vowel of לַחְמְךָ (*laxmxu).
#   (c) a word-final komets-hey reads [u] (Toyru, chochmu, Torah-style), not the
#       merged [ə] of the Yiddish feminine ending (bruxə, ʃirə).
#   (d) everything else is ordinary Ashkenazi: komets [u], pasekh [a],
#       tsere [aj], segol [ɛ], cholam [ɔj], chirik [i].
#
# The reader is deliberately SELF-CONTAINED (its own point table, its own
# consonant table, its own stress pass) rather than a flag threaded through
# _word_to_latin / latin_to_ipa. The merged path is 500-gold-locked; a flag
# would have to be proven not to leak, whereas a separate function cannot.
# =====================================================================

# Ashkenazi values of the points when reading Hebrew as Hebrew.
_WH_POINT_TO_IPA: dict[str, str] = {
    HIRIQ: "i",
    TSERE: "aj",
    "ֶ": "ɛ",   # segol
    PATAH: "a",
    QAMATS: "u",     # komets gadol AND katan -> [u] (Shabus, Toyru)
    HOLAM: "ɔj",     # cholam
    "ֺ": "ɔj",  # holam haser for vav
    QUBUTS: "u",     # kubuts -> [u]; NO Yiddish u->i shift in this register
    "ׇ": "u",   # qamats qatan
    "ֱ": "ɛ",   # hataf segol
    "ֲ": "a",   # hataf patah
    "ֳ": "u",   # hataf qamats
}

# The points that historically spell a LONG vowel. A sheva after one of these
# is a shva-na (rule 3 below): שׁוֹמְרִים -> ʃɔjmərim. Komets KATAN (U+05C7) is
# excluded — it is short by definition, so חׇכְמָה is xuxmu and not *xuxəmu.
_WH_LONG_POINTS = frozenset({QAMATS, TSERE, HOLAM, "ֺ", QUBUTS})

# Consonants, straight to IPA. ב/כ/פ/ת take their value from the DAGESH, which
# is the correct rule for a fully pointed book text (bare = fricative) — except
# word-initially, where dagesh lene is grammatically obligatory and the letter
# is a plosive whether or not the edition printed the dot (see
# _wh_consonant); ג and ד
# have no Ashkenazi fricative reflex and stay [ɡ]/[d]. א and ע are silent
# carriers; ה is [h] only at a syllable onset (see _wh_word).
_WH_CONSONANT: dict[str, str] = {
    "ב": "v", "ג": "ɡ", "ד": "d", "ה": "h", "ו": "v", "ז": "z", "ח": "x",
    "ט": "t", "י": "j", "כ": "x", "ך": "x", "ל": "l", "מ": "m", "ם": "m",
    "נ": "n", "ן": "n", "ס": "s", "פ": "f", "ף": "f", "צ": "ʦ", "ץ": "ʦ",
    "ק": "k", "ר": "r", "ש": "ʃ", "ת": "s", "א": "", "ע": "",
}
_WH_DAGESH_HARD: dict[str, str] = {"ב": "b", "כ": "k", "ך": "k", "פ": "p",
                                   "ף": "p", "ת": "t"}


def _wh_consonant(ch: str, marks: str, initial: bool = False) -> str:
    """IPA of one Hebrew consonant in the Whole-Hebrew register.

    ``initial`` = this is the word's first letter. A begadkefat letter there
    takes dagesh LENE and is a plosive when the word is read from a pause,
    which is exactly how a quoted word is read; the dot is grammatically
    predictable and the editions print it inconsistently (Torat Emet has
    כְנֶסֶת, פֵאוֹת, כְתוּבָה with no dagesh at all, and no dotted variant exists
    to fall back on). Reading the bare letter literally gave xənˈɛsɛs /
    fˈajɔjs. An explicit RAFE still forces the fricative.
    """
    if ch == "ש":
        return "s" if SIN_DOT in marks else "ʃ"
    if ch in _WH_DAGESH_HARD and RAFE not in marks and (
            DAGESH in marks or initial):
        return _WH_DAGESH_HARD[ch]
    return _WH_CONSONANT.get(ch, "")


def _wh_vowel_point(marks: str) -> str:
    """Like _vowel_point, but also sees the vav-only holam haser (U+05BA)."""
    for mark in marks:
        if mark in _WH_POINT_TO_IPA:
            return mark
    return ""


def _wh_sheva_is_na(units: list[tuple[str, str]], i: int, emitted_vowel: bool,
                    prev_point: str, prev_was_nach: bool) -> bool:
    """Whether the sheva at unit ``i`` is a shva-NA (pronounced [ə]).

    The standard schoolroom heuristic, kept deliberately small:
      1. sheva on the FIRST consonant of the word is na      (בְּרֵאשִׁית -> bə-)
      2. the SECOND of two adjacent shevas is na             (לַחְמְךָ -> -mə-)
      3. a sheva after a LONG vowel is na                    (שׁוֹמְרִים -> -mə-)
      4. a word-final sheva is always nach (silent), even under rules 2-3
    Everything else — the ordinary sheva closing a short syllable — is nach.
    Not modelled: dagesh chazak (gemination is not realised in this register
    anyway) and the sheva under a doubled letter; both would need syllable
    weight the reader does not track.
    """
    if i == len(units) - 1:
        return False                      # 4: word-final
    if not emitted_vowel:
        return True                       # 1: first consonant of the word
    if prev_was_nach:
        return True                       # 2: second of two adjacent shevas
    return prev_point in _WH_LONG_POINTS  # 3: after a long vowel


def _wh_word(word: str) -> list[tuple[str, bool]]:
    """Read one pointed Hebrew word into (segment, is_vowel) pairs."""
    units = [(b, m) for b, m in _split_units(word) if b.strip()]
    out: list[tuple[str, bool]] = []
    n = len(units)
    i = 0
    prev_point = ""          # the point whose vowel was last realised
    prev_was_nach = False    # the previous consonant carried a silent sheva
    emitted_vowel = False

    def nxt(k: int) -> str:
        return units[k][0] if 0 <= k < n else ""

    while i < n:
        ch, marks = units[i]
        point = _wh_vowel_point(marks)

        # --- vav: shuruk / cholam male are VOWELS, everything else is /v/ ----
        if ch == "ו" and DAGESH in marks and not point:
            out.append(("u", True))       # shuruk — [u], not the Yiddish [i]
            prev_point, prev_was_nach, emitted_vowel = QUBUTS, False, True
            i += 1
            continue
        if ch == "ו" and point in (HOLAM, "ֺ") and not prev_was_nach:
            out.append(("ɔj", True))      # cholam male
            prev_point, prev_was_nach, emitted_vowel = HOLAM, False, True
            i += 1
            continue
        # bare vav / bare yud spelling out the preceding point: matres, silent.
        if ch == "ו" and not marks and prev_point in (HOLAM, "ֺ", QUBUTS):
            i += 1
            continue
        if ch == "י" and not marks and prev_point in (HIRIQ, TSERE):
            i += 1
            continue
        # word-final ה with no point of its own is silent — and after a komets
        # that is exactly the (c) case: תּוֹרָה -> tɔjru, not *tɔjrə.
        if ch == "ה" and i == n - 1 and not point:
            i += 1
            continue

        # Pasekh genuvah (furtive patah): word-final guttural (ח, ע, ה with mappiq)
        # carrying a patah after a non-a vowel emits the [a] BEFORE the consonant.
        # e.g. רוּחַ -> rˈuax; מַשְׁגִּיחַ -> maʃɡˈiax; תַּפּוּחַ -> tapˈuax; כֹּחַ -> kˈɔjax;
        # מַפְתֵּחַ -> maftˈajax; שָׁמוֹעַ -> ʃumˈɔja.
        if (
            i == n - 1
            and ch in ("ח", "ע", "ה")
            and point in (PATAH, "ֲ")
            and emitted_vowel
            and prev_point not in (PATAH, QAMATS, "ׇ", "ֲ")
        ):
            out.append(("a", True))
            cons = _wh_consonant(ch, marks, initial=False)
            if cons:
                out.append((cons, False))
            prev_point, prev_was_nach, emitted_vowel = PATAH, False, True
            i += 1
            continue

        cons = _wh_consonant(ch, marks, initial=(i == 0))
        if cons:
            out.append((cons, False))

        if SHEVA in marks:
            na = _wh_sheva_is_na(units, i, emitted_vowel, prev_point,
                                 prev_was_nach)
            if na:
                out.append(("ə", True))
                emitted_vowel = True
            prev_point, prev_was_nach = "", not na
            i += 1
            continue
        if point:
            out.append((_WH_POINT_TO_IPA[point], True))
            prev_point = point
            prev_was_nach = False
            emitted_vowel = True
            i += 1
            continue

        prev_point, prev_was_nach = "", False
        i += 1

    return out


def _wh_stress(segments: list[tuple[str, bool]]) -> str:
    """Join segments, marking WH default stress: penult, ˈ before the vowel.

    Mil'el retraction is the Ashkenazi reading default, so the mark goes on the
    second-to-last syllable. A shva-na syllable is never a stress bearer in
    Hebrew, so schwas are skipped when counting candidates — לַחְמְךָ comes out
    lˈaxməxu (la-xmə-xu, retracted past the schwa) rather than laxmˈəxu, and
    שׁוֹמְרִים ʃˈɔjmərim. Monosyllables carry no mark, matching the engine-wide
    convention in add_stress().
    """
    vowels = [k for k, (_, is_v) in enumerate(segments) if is_v]
    if len(vowels) < 2:
        return "".join(s for s, _ in segments)
    full = [k for k in vowels if segments[k][0] != "ə"]
    if len(full) >= 2:
        at = full[-2]
    elif full:
        at = full[0]
    else:
        at = vowels[-2]
    return "".join(
        (STRESS + s) if k == at else s for k, (s, _) in enumerate(segments)
    )


def read_pointed_wh(pointed: str) -> str:
    """Read VERIFIED pointed Hebrew in the Whole-Hebrew register -> IPA.

    Opt-in sibling of hebrew_to_ipa() for quoted, book-pointed loshn-koydesh.
    Nothing in the live pipeline routes through it; the Sefaria rescue builder
    calls it directly. Output stays inside the closed v3 phone inventory.

    >>> read_pointed_wh("וּבֵרַכְתִּי")
    'uvajrˈaxti'
    >>> read_pointed_wh("וַיֹּאמֶר")
    'vajˈɔjmɛr'
    """
    out = []
    for word in re.split(r"[\s־]+", strip_tags(pointed)):
        segments = _wh_word(word) if word else []
        if segments:
            out.append(_wh_stress(segments))
    return " ".join(out)


_BEGADKEFAT = frozenset("בכךפףת")


def _explicit_begadkefat(word: str) -> str:
    """Write the begadkefat stop/fricative choice into the marks, explicitly.

    The merged reader is the engine's ordinary Hebrew-script reader, and that
    reader has to cope with text where a bare ב is far more often Germanic /b/
    (האָבן, אָבער) than Hebrew /v/. Its guards for the Hebrew case are therefore
    conservative and they miss the two environments a book pointing produces
    most: a ב with a sheva (אַבְרָהָם -> *abrˈuhum, Abrohom for Avrohom) and a ב
    whose vowel is spelled by a following mater (אָבוֹת -> *ˈubɔjs).

    read_pointed_merged() does not have that ambiguity — its input is pointed
    Hebrew by contract — so the choice is made HERE, where the context is known,
    and written into the string as a dagesh or a rafe that the shared reader
    then simply obeys. The rule is the Hebrew one, identical to _wh_consonant():
    a begadkefat letter is a plosive word-initially (dagesh lene is obligatory
    from a pause, and editions print it inconsistently) and a fricative
    elsewhere. Letters that already carry an explicit dagesh or rafe are left
    exactly as the edition set them.

    Marking rather than special-casing keeps the register difference where it
    belongs: read_pointed_merged() still runs the gold-locked merged VOWEL path
    and adds no consonant logic of its own.
    """
    units = _split_units(word)
    out: list[str] = []
    first = True
    for ch, marks in units:
        if _HEBREW_CHAR.match(ch) and ch in _BEGADKEFAT and not (
                DAGESH in marks or RAFE in marks):
            marks += DAGESH if first else RAFE
        if _HEBREW_CHAR.match(ch):
            first = False
        out.append(ch + marks)
    return "".join(out)


def read_pointed_merged(pointed: str) -> str:
    """Read pointed Hebrew in the MERGED register (embedded loshn-koydesh) -> IPA.

    The register sibling of read_pointed_wh(). Same input — a pointed Hebrew
    word — read as a word that has been ABSORBED INTO YIDDISH rather than
    quoted as Hebrew, which is what an LK word in a running Yiddish sentence
    almost always is (spec v2 §5/§7, native-verified against the gold CSV):

      shuruk / kubuts   [i], not [u]   — the u->i shift, 'near-exceptionless':
                                         חִדּוּשׁ xˈidiʃ, שִׁדּוּךְ ʃˈidix, תְּרוּמָה trˈimə
      final komets-hey  [ə], not [u]   — the Yiddish feminine ending:
                                         תּוֹרָה tɔjrə, בְּרָכָה brˈuxə
      sheva             [ə] only where the syllable is actually pronounced
                                         with one — יְשִׁיבָה jəʃˈivə but בְּרָכָה
                                         brˈuxə, which no straight shva-na rule
                                         gets right
      stress            §11.5 LK penult, as in the WH reader

    DELEGATES, deliberately. This is not a second implementation of the merged
    table: it is _rule_path_ipa(lk_penult=True), i.e. the very path that
    _route_token_inner() takes for a pointed LK token (§6.2) and that the 500
    gold rows lock down. A hand-written twin of _POINT_TO_LATIN here would be
    free to drift away from the register the gold defines — and the sheva rule
    above is emergent from the Latin layer's syllable repair, not statable as a
    point table at all. read_pointed_wh() is self-contained for the opposite
    reason: it has to differ from the locked path, so it cannot reuse it.

    Note the asymmetry with hebrew_to_ipa(): this function reads the POINTING
    it is handed and consults no lexicon, so a builder gets the evidence's own
    reading rather than an answer some table already had. Callers that want the
    lexicon's verdict should call g2p_token()/hebrew_to_ipa() instead.

    >>> read_pointed_merged("תּוֹרָה")
    'tˈɔjrə'
    >>> read_pointed_merged("חִדּוּשׁ")
    'xˈidiʃ'
    >>> read_pointed_merged("וּבֵרַכְתִּי")
    'ivajrˈaxti'
    """
    out = []
    for word in re.split(r"[\s־]+", strip_tags(pointed)):
        if not word:
            continue
        ipa = _rule_path_ipa(_explicit_begadkefat(word),
                             stress=True, lk_penult=True, stem_subs=False)
        if ipa:
            out.append(ipa)
    return normalize_ipa_spacing(" ".join(out))
