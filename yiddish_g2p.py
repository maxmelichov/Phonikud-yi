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
    "פ": ("p", "f", "f"),
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

        out.append(_consonant(ch, marks))
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
        return ("ay", 1, PATAH) if PATAH in marks else ("ey", 1, "")
    if ch == "ױ":
        return ("oy", 1, "")

    if ch == "י":
        if nxt == "י":
            # tsvey yudn: a hiriq on either yud is ייִ /yi/, a pasekh marks /ay/
            if HIRIQ in marks or HIRIQ in nxt_marks:
                return ("yi", 2, HIRIQ)
            if PATAH in marks or PATAH in nxt_marks:
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