"""
Hybrid Yiddish G2P: Hebrew-script Yiddish -> IPA for TTS.

ARCHITECTURE: Three-Stage (Orthography -> Latin Base -> Central Phonology)
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
import re
import unicodedata

_HEBREW_CHAR = re.compile(r"[\u0590-\u05FF]")


def _strip_points(text: str) -> str:
    """Remove Hebrew diacritics (nikud) so pointed lexicon keys also match unpointed text."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


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
_POINT_TO_LATIN: dict[str, str] = {
    SHEVA: "",
    "\u05b1": "e",   # hataf segol
    "\u05b2": "a",   # hataf patah
    "\u05b3": "o",   # hataf qamats
    HIRIQ: "i",
    TSERE: "ey",
    "\u05b6": "e",   # segol
    PATAH: "a",
    QAMATS: "o",
    HOLAM: "oy",
    QUBUTS: "u",
    "\u05c7": "o",   # qamats qatan
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

# Keyed on the unpointed form so a word matches whether it arrives unpointed,
# YIVO-pointed or fully pointed by the nikud model.
_LK_BARE: dict[str, str] = {_strip_points(_k): _v for _k, _v in _LOSHN_KOYDESH.items()}

_LK_PATTERN = re.compile(
    r"(?<![\w\u0590-\u05FF])("
    + "|".join(_tolerant(k) for k in sorted(_LK_BARE, key=len, reverse=True))
    + r")(?![\w\u0590-\u05FF])"
)


def _lk_replace(match: re.Match) -> str:
    return _LK_BARE.get(_strip_points(match.group(1)), match.group(1))

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
    ("וואו", "vu"),
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
            out.append(latin)
            prev_point = point_used if latin else ""
            prev_consonant_point = ""
            i += size
            continue

        latin_c = _consonant(ch, marks)
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
        out.append(latin_c)
        # A vowel point on a consonant is realised unless the next letter already
        # spells that vowel independently, which would double it up.
        point = _vowel_point(marks)
        prev_point = prev_consonant_point = ""
        if point and not _spells_own_vowel(units, i + 1):
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
        return ("ey", 1, "")
    if ch == "ױ":
        return ("oy", 1, "")

    if ch == "י":
        # Word-initial pointed yud before א/ע is consonantal /j/ carrying that
        # vowel, and the vowel letter just restates it (יֶעדְן -> yedn, not eedn).
        if point and not emitted and nxt in "אע" and not _vowel_point(nxt_marks):
            return ("y" + _POINT_TO_LATIN[point], 2, point)
        if nxt == "י":
            # tsvey yudn: a hiriq on either yud is ייִ /yi/, a pasekh marks /ay/
            if HIRIQ in marks or HIRIQ in nxt_marks:
                return ("yi", 2, HIRIQ)
            if PATAH in marks or PATAH in nxt_marks or prev_consonant_point == PATAH:
                return ("ay", 2, PATAH)
            return ("ey", 2, "")
        if point:
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
        if point:
            return (_POINT_TO_LATIN[point], 1, point)
        if prev_consonant_point:
            return ("", 1, "")  # the consonant's point already spelled this vowel
        if ch == "א":
            if not emitted and nxt in _NUCLEUS_START:
                return ("", 1, "")  # silent alef before a nucleus
            return ("a", 1, "")
        return ("e", 1, "")

    # Word-final bare ה after a vowel is silent (חתונה -> khasene).
    if ch == "ה" and not marks and i == n - 1 and prev_latin in _LATIN_VOWELS:
        return ("", 1, "")

    return None


def letter_at(units: list[tuple[str, str]], idx: int) -> str:
    return units[idx][0] if 0 <= idx < len(units) else ""


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

    # 1. Contractions (diacritic-tolerant: pointed input carries marks on ס/כ/מ)
    text = re.sub(_tolerant("ס") + r"'" + _tolerant("איז"), "סיז", text)
    text = re.sub(_tolerant("כ") + r"'", "איך ", text)
    text = re.sub(_tolerant("מ") + r"'", "מע ", text)
    text = re.sub(_tolerant("ס") + r"'", "עס ", text)

    # Drop remaining intra-word apostrophes
    text = re.sub(r"(?<=[\u0590-\u05FF])'(?=[\u0590-\u05FF])", "", text)

    # 2. Hasidic Silent 'ה' Patch (strips 'ה' before terminal ן, סט, ט)
    text = re.sub(
        rf"([אעיו]{_MARKS_CLASS})ה{_MARKS_CLASS}"
        rf"(?=ן{_MARKS_CLASS}\b|ס{_MARKS_CLASS}ט{_MARKS_CLASS}\b|ט{_MARKS_CLASS}\b)",
        r"\1",
        text,
    )

    # 3. Loshn-Koydesh lexical swap
    text = _LK_PATTERN.sub(_lk_replace, text)

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
            # _WORD_LATIN only guesses vowels that unpointed spelling leaves open,
            # so it applies to the bare form only while the word itself carries no
            # vowel point. Otherwise the diacritics are the better evidence
            # (unpointed דאך is /dokh/, but pointed דאַך is /dakh/).
            bare = _strip_points(core)
            if bare in _WORD_LATIN and not _vowel_point(core):
                latin = _WORD_LATIN[bare]
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

# Latin vowel nuclei, longest first so digraphs win.
_NUCLEI = ("ey", "ay", "oy", "a", "e", "i", "o", "u")

# Inseparable prefixes that never take stress: it falls on the following stem
# syllable. Matched only when at least one nucleus follows, so the bare words
# (der, far, tsu ...) are not mis-analysed as prefixed forms.
_UNSTRESSED_PREFIXES = ("ge", "be", "der", "far", "tsu", "tse", "tser", "ant", "ent", "ba", "dis")

# Function words that carry no lexical stress. Marking these would dilute the
# meaning of the marker; espeak leaves the equivalent clitics bare too.
_CLITICS = frozenset({
    "a", "an", "di", "der", "dos", "dem", "den", "de",
    "in", "im", "un", "az", "tsu", "mit", "fun", "far", "bay", "ba",
    "oyf", "iber", "unter", "es", "zi", "er", "ix", "mir", "dir",
    "zix", "ze", "do", "vi", "ober", "nor", "oder", "ven",
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
    "ndik", "shaft", "kayt", "heyt", "lekh", "dik", "ung", "es", "en", "er",
    "l", "s",
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
    "mishpokhe": 1,    # mish-PO-khe
    "tsedoke":   1,    # tse-DO-ke
    "meshuge":   1,    # me-SHU-ge
    "mekhutn":   1,    # me-KHU-tn
    "rebetsn":   0,    # RE-be-tsn
    "balebos":   2,    # ba-le-BOS
    "balebuste": 2,    # ba-le-BUS-te
    "yeshive":   1,    # ye-SHI-ve
    "khevre":    0,    # KHEV-re
    "shabes":    0,    # SHA-bes
    "yontev":    0,    # YON-tev
    "khasene":   0,    # KHA-se-ne
    "mazltov":   0,    # MAZL-tov
    "seykhl":    0,
    "khoydesh":  0,
    "kholem":    0,
    "afile":     1,    # a-FI-le
    "efsher":    0,
    "asakh":     1,    # a-SAKH
    "bishas":    1,
    "beemes":    1,
    "stam":      0,

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
    "aroys":     1,
    "arayn":     1,
    "anider":    1,
    "amol":      1,    # a-MOL
    "aza":       1,    # a-ZA
    "arop":      1,    # a-ROP
    "aroyf":     1,
    "ahin":      1,
    "aher":      1,
    "atsind":    1,

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


# Consonants that make a following word-final n/m syllabic, mirroring the schwa
# insertion in latin_to_ipa (arbetn -> arbetən). Without this, gutn/hobn/zogn
# count as one syllable and the monosyllable rule wrongly leaves them unmarked.
_SYLLABIC_TRIGGER = set("bvdgktpsfzhlmnrxjw")


_SYLLABIC_NASAL = re.compile(r"[bvdgktpsfzhlmrxjw][nm](?![aeiou])")


def _syllable_count(word: str) -> int:
    """Phonetic syllable count: vowel nuclei plus every syllabic n/m.

    Mirrors the schwa insertion in latin_to_ipa, including mid-word cases such
    as arbet|n|dik, so the monosyllable rule and the phonology agree.
    """
    return len(_nuclei_spans(word)) + len(_SYLLABIC_NASAL.findall(word))


# --- Separable (directional) prefixes: these CARRY the stress -----------------
# arop-geyn, avek-forn, unter-shraybn. Only when material follows: bare "arum"
# is a-RUM, but "arumgeyn" is ARUM-geyn.
_SEPARABLE_PREFIXES = (
    "aroys", "arayn", "arum", "arop", "aroyf", "avek", "anider", "tsurik",
    "unter", "iber", "arunter", "aruf", "mit", "oys", "on", "uf", "oyf", "ayn",
    "tsuzamen", "farbay", "adurkh", "antkegn",
)

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
    for pre in _SEPARABLE_PREFIXES:
        if word.startswith(pre) and len(word) > len(pre):
            tail = word[len(pre) :]
            if _nuclei_spans(tail) and _nuclei_spans(pre):
                return 0

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


def add_stress(latin: str) -> str:
    """Insert a primary-stress marker before the stressed syllable of each word."""
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

    # Syllabic n/m smoothing. The old form anchored on \b, so it only fired at
    # word end: arbetn -> arbetən but arbetndik stayed arbetndik, leaving an
    # unpronounceable tnd cluster. A syllabic nasal arises wherever a consonant
    # precedes it and no vowel follows, including before -dik / -shaft.
    ipa = re.sub(
        r"([bvdɡktpsfzʒʃxrlmʦʧʤʣ])([nm])(?![aeiouɑɐəɛɪɵɔʊʉæœøyɨɤʌɜɒ])",
        r"\1ə\2",
        ipa,
    )
    
    # Word-final epsilon to schwa reduction (protects single-syllable words)
    ipa = re.sub(r"(\S{2,})ɛ\b", r"\1ə", ipa)
    
    return ipa


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


def hebrew_to_ipa(text: str, stress: bool = True) -> str:
    """Hebrew-script Yiddish -> IPA. ``stress=False`` reproduces pre-prosody output."""
    text = _preprocess_hebrew(strip_tags(text))
    latin = hebrew_to_latin(text)
    if stress:
        latin = add_stress(latin)
    ipa = latin_to_ipa(latin)
    if stress:
        ipa = reduce_unstressed(ipa)
    ipa = normalize_ipa_affricates(ipa)
    return normalize_ipa_spacing(ipa)


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