"""
Hybrid Yiddish G2P: Hebrew-script Yiddish -> IPA for TTS.

ARCHITECTURE: Three-Stage (Orthography -> Latin Base -> Central Phonology)
DIALECT: Poylish/Galitzyaner/Modern Hasidic

Handles BOTH spelling systems:
  - Pointed YIVO orthography (אַ אָ פּ בֿ ייִ ...)
  - Unpointed Hasidic spelling (א = vowel a/o, י = vowel i, no diacritics)
"""

from __future__ import annotations
import re
import unicodedata

_HEBREW_CHAR = re.compile(r"[\u0590-\u05FF]")


def _strip_points(text: str) -> str:
    """Remove Hebrew diacritics (nikud) so pointed lexicon keys also match unpointed text."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


# =====================================================================
# STAGE 1: ORTHOGRAPHY & LEXICON (Loshn-Koydesh -> phonetic respelling)
# =====================================================================
_LOSHN_KOYDESH = {
    # === Religion, Holidays, and Time ===
    "שבת": "שאָבעס",
    "שבתים": "שאָבאָסים",
    "יום-טובֿ": "יאָנטעוו",
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
    "תּורה": "טויערע",
    "ספֿר": "סייפער",
    "ספֿרים": "ספֿאָרים",
    "מעשׂה": "מײַסע",
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
}

_LK_ALL: dict[str, str] = dict(_LOSHN_KOYDESH)
for _k, _v in _LOSHN_KOYDESH.items():
    _bare = _strip_points(_k)
    if _bare != _k and _bare not in _LK_ALL:
        _LK_ALL[_bare] = _v

_LK_PATTERN = re.compile(
    rf"(?<![\w\u0590-\u05FF])({'|'.join(re.escape(k) for k in sorted(_LK_ALL, key=len, reverse=True))})(?![\w\u0590-\u05FF])"
)

# =====================================================================
# STAGE 1.5: HIGH-FREQUENCY WORD LEXICON (unpointed spelling -> Latin base)
# =====================================================================
_WORD_LATIN: dict[str, str] = {
    # --- mid-word א = 'o' ---
    "וואס": "vos",
    "האט": "hot",
    "דאס": "dos",
    "דא": "do",
    "נאר": "nor",
    "נאך": "nokh",
    "נאכדעם": "nokhdem",
    "נאכער": "nokher",
    "נאכאמאל": "nokhamol",
    "זאל": "zol",
    "זאלן": "zoln",
    "זאלסט": "zolst",
    "זאג": "zog",
    "זאגן": "zogn",
    "זאגט": "zogt",
    "זאגסט": "zogst",
    "געזאגט": "gezogt",
    "אנזאגן": "onzogn",
    "יא": "yo",
    "יאר": "yor",
    "יארן": "yorn",
    "האב": "hob",
    "האבן": "hobn",
    "האסט": "host",
    "וואלט": "volt",
    "וואלטן": "voltn",
    "מאל": "mol",
    "אמאל": "amol",
    "קיינמאל": "keynmol",
    "געווארן": "gevorn",
    "פארוואס": "farvos",
    "אבער": "ober",
    "אדער": "oder",
    "טאג": "tog",
    "פרייטאג": "fraytog",
    "דארט": "dort",
    "דארטן": "dortn",
    "דאך": "dokh",
    "וואך": "vokh",
    "גארנישט": "gornisht",
    "געוואלט": "gevolt",
    "לאמיר": "lomir",
    "אראפ": "arop",
    "פארט": "fort",
    "אפ": "op",
    "אפט": "oft",
    "מארגן": "morgn",
    "קאפ": "kop",
    "גאט": "got",
    "נאז": "noz",
    "גראב": "grob",
    "אווענט": "ovnt",
    "אוודאי": "avade",

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
    "בעצם": "beetsem",
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
    "כל": "kol",
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
    "לכל": "lekol",
    "אוו": "ov",
    "וויבאלד": "vibald",

    # --- Hasidic pronunciation of common words ---
    "אויף": "af",
    "אויפן": "afn",
}


# Sub-word substitutions for Loshn-Koydesh bases that take Yiddish morphology
_STEM_SUBS: list[tuple[str, str]] = [
    ("שבת", "שאָבעס"),
    ("אמת", "עמעס"),
    ("חן", "כיין"),
    ("פסק", "פּאַסק"),
    ("פטר", "פּאַטער"),
    ("הרג", "האַרג"),
]

# =====================================================================
# STAGE 2: BASE TRANSLITERATION (context-aware)
# =====================================================================
_BASE_MAP: list[tuple[str, str]] = [
    ("דזש", "dzh"),
    ("דז", "dz"),
    ("זש", "zh"),
    ("טש", "tsh"),
    ("ייִ", "yi"),
    ("וווּ", "vu"),
    ("וואו", "vu"),
    ("ווּ", "vu"),
    ("וו", "v"),
    ("ױ", "oy"),
    ("וי", "oy"),
    ("ײַ", "ay"),
    ("ײ", "ey"),
    ("יי", "ey"),
    ("יִ", "i"),
    ("אַ", "a"),
    ("אָ", "o"),
    ("וּ", "u"),
    ("בֿ", "v"),
    ("כּ", "k"),
    ("פּ", "p"),
    ("פֿ", "f"),
    ("שׂ", "s"),
    ("תּ", "t"),
    ("װ", "v"),
]

_SINGLE_MAP: dict[str, str] = {
    "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ז": "z", "ח": "kh", "ט": "t",
    "כ": "kh", "ך": "kh", "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n",
    "ס": "s", "ע": "e", "פ": "f", "ף": "f", "צ": "ts", "ץ": "ts", "ק": "k",
    "ר": "r", "ש": "sh", "ת": "s",
}

_LATIN_VOWELS = set("aeiou")
_NUCLEUS_START = set("ויױײ")


def _word_to_latin(word: str) -> str:
    """Transliterate one Hebrew-script word to Latin base with context rules."""
    out: list[str] = []
    i = 0
    n = len(word)

    def emitted_any() -> bool:
        return any(t for t in out)

    def last_char() -> str:
        for t in reversed(out):
            if t:
                return t[-1]
        return ""

    while i < n:
        ch = word[i]

        if not _HEBREW_CHAR.match(ch):
            out.append(" " if ch == "-" else ch)
            i += 1
            continue

        matched = False
        for src, dst in _BASE_MAP:
            if word.startswith(src, i):
                out.append(dst)
                i += len(src)
                matched = True
                break
        if matched:
            continue

        if ch == "א":
            nxt = word[i + 1] if i + 1 < n else ""
            if not emitted_any() and nxt in _NUCLEUS_START:
                i += 1
            else:
                out.append("a")
                i += 1
            continue

        if ch == "י":
            nxt = word[i + 1] if i + 1 < n else ""
            prev = last_char()
            if not emitted_any() and nxt in "אעו":
                out.append("y")
            elif prev in _LATIN_VOWELS:
                out.append("y")
            elif nxt == "ו" and (i + 2 >= n or word[i + 2] != "ו"):
                out.append("y")
            else:
                out.append("i")
            i += 1
            continue

        if ch == "ו":
            out.append("u")
            i += 1
            continue

        out.append(_SINGLE_MAP.get(ch, ""))
        i += 1

    return "".join(out)


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

    # 1. Contractions
    text = re.sub(r"ס'איז", "סיז", text)
    text = re.sub(r"כ'", "איך ", text)
    text = re.sub(r"מ'", "מע ", text)
    text = re.sub(r"ס'", "עס ", text)

    # Drop remaining intra-word apostrophes
    text = re.sub(r"(?<=[\u0590-\u05FF])'(?=[\u0590-\u05FF])", "", text)
    
    # 2. Hasidic Silent 'ה' Patch (strips 'ה' before terminal ן, סט, ט)
    text = re.sub(r"(?<=[אעיווי])ה(?=ן\b|סט\b|ט\b)", "", text)

    # 3. Loshn-Koydesh lexical swap
    text = _LK_PATTERN.sub(lambda m: _LK_ALL[m.group(1)], text)

    return text


def hebrew_to_latin(text: str) -> str:
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
            if core in _WORD_LATIN:
                latin = _WORD_LATIN[core]
            else:
                for stem, repl in _STEM_SUBS:
                    if stem in core:
                        core = core.replace(stem, repl)
                latin = _word_to_latin(core)
            latin_parts.append(lead + latin + trail)
        if latin_parts:
            out_tokens.append(" ".join(latin_parts))
    return re.sub(r"\s+", " ", " ".join(out_tokens)).strip()


# =====================================================================
# STAGE 3: CENTRAL YIDDISH PHONOLOGY
# =====================================================================
_LATIN_TO_IPA: list[tuple[str, str]] = [
    ("tsh", "ʧ"), ("dzh", "ʤ"), ("dz", "ʣ"), ("ts", "ʦ"),
    ("kh", "x"), ("sh", "ʃ"), ("zh", "ʒ"),
    ("ey", "aɪ"), ("ay", "aː"), ("oy", "ɔɪ"),
    ("a", "a"), ("o", "u"), ("e", "ɛ"), ("u", "i"), ("i", "i"),
    ("b", "b"), ("v", "v"), ("d", "d"), ("h", "h"), ("z", "z"),
    ("t", "t"), ("l", "l"), ("m", "m"), ("n", "n"), ("s", "s"),
    ("f", "f"), ("p", "p"), ("k", "k"), ("g", "ɡ"), ("r", "r"),
    ("y", "j"), ("-", " "),
]

_APOSTROPHE = re.compile(r"'")

_AFFRICATE_DECOMPOSE = [
    (re.compile(r"t\u0361s|t͡s", re.I), "ʦ"),
    (re.compile(r"t\u0361ʃ|t͡ʃ|tʃ", re.I), "ʧ"),
    (re.compile(r"d\u0361ʒ|d͡ʒ|dʒ", re.I), "ʤ"),
    (re.compile(r"d\u0361z|d͡z|dz", re.I), "ʣ"),
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
            out.append("ʔ")
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

    # Syllabic n/m smoothing
    ipa = re.sub(r"([bvdɡktpsfzʒʃxrlmʦʧʤʣ])([nm])\b", r"\1ə\2", ipa)
    
    # Word-final epsilon to schwa reduction (protects single-syllable words)
    ipa = re.sub(r"(\S{2,})ɛ\b", r"\1ə", ipa)
    
    return ipa


def normalize_ipa_affricates(ipa: str) -> str:
    for pattern, lig in _AFFRICATE_DECOMPOSE:
        ipa = pattern.sub(lig, ipa)
    return ipa


def normalize_ipa_spacing(ipa: str) -> str:
    ipa = re.sub(r"\s+", " ", ipa).strip()
    for punct in [",", ".", "!", "?", ";", ":", "'"]:
        ipa = ipa.replace(f" {punct}", punct)
    return ipa


def hebrew_to_ipa(text: str) -> str:
    text = _preprocess_hebrew(strip_tags(text))
    latin = hebrew_to_latin(text)
    ipa = latin_to_ipa(latin)
    ipa = normalize_ipa_affricates(ipa)
    return normalize_ipa_spacing(ipa)


def validate_ipa_vocab(ipa: str, char_to_id: dict[str, int]) -> tuple[str, list[str]]:
    missing = sorted({ch for ch in ipa if ch not in char_to_id and not ch.isspace()})
    return ipa, missing