"""GENERATED — audio-confirmed vowel corrections for alef-default words.

Source: PhoneticXeus positional tags (scripts/xeus_pe_sweep.py /
xeus_lk_sweep.py) folded by scripts/build_audio_vowel_lexicon.py. Each entry
is the engine's own stressed rule-path reading with >= 3-vote,
>= 2/3-majority vowel slots substituted — and only toward CLEAN_TARGETS
(['u']), phones the recognizer never produces spuriously, so its known
biases cannot inject a vowel. Stress marks are untouched by construction.

Consulted after every gold/legacy lexicon and after data/audio_pe_lk.py,
before the rule path; emitted at MED confidence, reason 'audio-vowel'.
Regenerate: .venv/bin/python scripts/build_audio_vowel_lexicon.py
Never hand-edit.
"""

AUDIO_VOWEL_LK = {
    'אבוא': {"ipa": 'ˈubia', "slots": '0:a->u(3/3)'},  # אבוא
    'אדמ': {"ipa": 'udm', "slots": '0:a->u(34/40)'},  # אדם
    'אוגעשוואכט': {"ipa": 'ˈuɡəʃvaxt', "slots": '0:i->u(2/3)'},  # אוגעשוואכט
    'אודמ': {"ipa": 'udm', "slots": '0:i->u(2/3)'},  # אודם
    'אודער': {"ipa": 'ˈudər', "slots": '0:i->u(12/14)'},  # אודער
    'אומגעגרייט': {"ipa": 'ˈumɡəɡrajt', "slots": '0:i->u(7/7)'},  # אומגעגרייט
    'אומגעזעצט': {"ipa": 'ˈumɡəzəʦt', "slots": '0:i->u(5/6)'},  # אומגעזעצט
    'אומגעמאכט': {"ipa": 'ˈumɡəmaxt', "slots": '0:i->u(3/3)'},  # אומגעמאכט
    'אומגעפילט': {"ipa": 'ˈumɡəfilt', "slots": '0:i->u(3/3)'},  # אומגעפילט
    'אומגעפער': {"ipa": 'ˈumɡəfər', "slots": '0:i->u(4/6)'},  # אומגעפער
    'אומקומענ': {"ipa": 'ˈumkimən', "slots": '0:i->u(2/3)'},  # אומקומען
    'אומקוקנ': {"ipa": 'ˈumkikn', "slots": '0:i->u(2/3)'},  # אומקוקן
    'אונדזערע': {"ipa": 'ˈundzərə', "slots": '0:i->u(2/3)'},  # אונדזערע
    'אורה': {"ipa": 'ˈurə', "slots": '0:i->u(2/3)'},  # אורה
    'אורנ': {"ipa": 'urn', "slots": '0:i->u(2/3)'},  # אורן
    'איבערגעזאגט': {"ipa": 'ˈibərɡəzuɡt', "slots": '7:a->u(2/3)'},  # איבערגעזאגט
    'איינטאג': {"ipa": 'ˈajntuɡ', "slots": '3:a->u(2/3)'},  # איינטאג
    'איכה': {"ipa": 'ˈixu', "slots": '2:ə->u(6/8)'},  # איכה
    'אלנו': {"ipa": 'ˈalnu', "slots": '3:i->u(4/5)'},  # אלנו
    'אלעזר': {"ipa": 'ˈaluzr', "slots": '2:ə->u(5/7)'},  # אלעזר
    'אמאאל': {"ipa": 'ˈamaul', "slots": '3:a->u(2/3)'},  # אמאאל
    'אמאליג': {"ipa": 'ˈamaluɡ', "slots": '4:i->u(2/3)'},  # אמאליג
    'אמאליגע': {"ipa": 'ˈamuliɡə', "slots": '2:a->u(4/5)'},  # אמאליגע
    'אנגעגרייט': {"ipa": 'ˈunɡəɡrajt', "slots": '0:a->u(3/4)'},  # אנגעגרייט
    'אנגעהאפט': {"ipa": 'ˈunɡəhaft', "slots": '0:a->u(3/3)'},  # אנגעהאפט
    'אנגעטאנ': {"ipa": 'ˈunɡətan', "slots": '0:a->u(3/4)'},  # אנגעטאן
    'אנגעטראגנ': {"ipa": 'ˈanɡətruɡn', "slots": '6:a->u(2/3)'},  # אנגעטראגן
    'אנגעכאפט': {"ipa": 'ˈunɡəxaft', "slots": '0:a->u(2/3)'},  # אנגעכאפט
    'אנגעמאכט': {"ipa": 'ˈunɡəmaxt', "slots": '0:a->u(3/3)'},  # אנגעמאכט
    'אנגענומענ': {"ipa": 'ˈunɡənimən', "slots": '0:a->u(3/3)'},  # אנגענומען
    'אנגעפילט': {"ipa": 'ˈunɡəfilt', "slots": '0:a->u(5/5)'},  # אנגעפילט
    'אנגעקומענ': {"ipa": 'ˈunɡəkimən', "slots": '0:a->u(16/21)'},  # אנגעקומען
    'אנגערירט': {"ipa": 'ˈunɡərirt', "slots": '0:a->u(3/4)'},  # אנגערירט
    'אנגעשלאפנ': {"ipa": 'ˈanɡəʃlufn', "slots": '6:a->u(3/3)'},  # אנגעשלאפן
    'אנונג': {"ipa": 'ˈanunɡ', "slots": '2:i->u(4/5)'},  # אנונג
    'אנטשולדיג': {"ipa": 'ˈanʧuldiɡ', "slots": '3:i->u(2/3)'},  # אנטשולדיג
    'אנטשולדיקט': {"ipa": 'ˈanʧuldikt', "slots": '3:i->u(3/3)'},  # אנטשולדיקט
    'אנע': {"ipa": 'ˈunə', "slots": '0:a->u(5/6)'},  # אנע
    'אנצוג': {"ipa": 'ˈanʦuɡ', "slots": '3:i->u(2/3)'},  # אנצוג
    'אנקומענ': {"ipa": 'ˈunkimən', "slots": '0:a->u(7/8)'},  # אנקומען
    'אפגעבורט': {"ipa": 'ˈɔpɡəburt', "slots": '5:i->u(3/3)'},  # אפגעבורט
    'אפגעהאלטנ': {"ipa": 'ˈupɡəhaltn', "slots": '0:ɔ->u(2/3)'},  # אפגעהאלטן
    'אפגעזאגט': {"ipa": 'ˈɔpɡəzuɡt', "slots": '5:a->u(5/7)'},  # אפגעזאגט
    'אפגעמאכט': {"ipa": 'ˈupɡəmaxt', "slots": '0:ɔ->u(2/3)'},  # אפגעמאכט
    'אפגעקויפט': {"ipa": 'ˈupɡəkɔjft', "slots": '0:ɔ->u(4/4)'},  # אפגעקויפט
    'אפגעשניטנ': {"ipa": 'ˈupɡəʃnitn', "slots": '0:ɔ->u(2/3)'},  # אפגעשניטן
    'אראפגעפארנ': {"ipa": 'arˈupɡəfurn', "slots": '7:a->u(4/4)'},  # אראפגעפארן
    'ארוב': {"ipa": 'ˈarub', "slots": '2:i->u(4/4)'},  # ארוב
    'ארויסגעזאגט': {"ipa": 'arˈoʊzɡəzuɡt', "slots": '7:a->u(3/3)'},  # ארויסגעזאגט
    'ארויסגעפארנ': {"ipa": 'arˈoʊzɡəfurn', "slots": '7:a->u(3/3)'},  # ארויסגעפארן
    'ארויפשרייבנ': {"ipa": 'arˈufʃrajbn', "slots": '2:oʊ->u(4/4)'},  # ארויפשרייבן
    'ארור': {"ipa": 'urˈir', "slots": '0:a->u(5/7)'},  # ארור
    'באצאלנ': {"ipa": 'baʦˈuln', "slots": '3:a->u(3/4)'},  # באצאלן
    'בארואיגט': {"ipa": 'barˈuaːɡt', "slots": '3:i->u(3/3)'},  # בארואיגט
    'בואו': {"ipa": 'bˈui', "slots": '1:i->u(2/3)'},  # בואו
    'בלאזט': {"ipa": 'bluzt', "slots": '2:a->u(4/4)'},  # בלאזט
    'בלאזנ': {"ipa": 'bluzn', "slots": '2:a->u(4/5)'},  # בלאזן
    'בלאזער': {"ipa": 'blˈuzər', "slots": '2:a->u(3/3)'},  # בלאזער
    'בלאזשאווער': {"ipa": 'blˈuʒavər', "slots": '2:a->u(3/3)'},  # בלאזשאווער
    'בלאזשעווער': {"ipa": 'blˈuʒəvər', "slots": '2:a->u(3/3)'},  # בלאזשעװער
    'גאולה': {"ipa": 'ɡˈulə', "slots": '1:i->u(7/10)'},  # גאולה
    'גאזאגט': {"ipa": 'ɡˈazuɡt', "slots": '3:a->u(3/3)'},  # גאזאגט
    'גאלמעסער': {"ipa": 'ɡˈulməsər', "slots": '1:a->u(4/4)'},  # גאלמעסער
    'גזאגט': {"ipa": 'ɡzuɡt', "slots": '2:a->u(3/4)'},  # גזאגט
    'געטראגנ': {"ipa": 'ɡətrˈuɡn', "slots": '4:a->u(4/5)'},  # געטראגן
    'געשלאגנ': {"ipa": 'ɡəʃlˈuɡn', "slots": '4:a->u(3/4)'},  # געשלאגן
    'געשלאפנ': {"ipa": 'ɡəʃlˈufn', "slots": '4:a->u(9/10)'},  # געשלאפן
    'דאג': {"ipa": 'duɡ', "slots": '1:a->u(2/3)'},  # דאג
    'דאסנ': {"ipa": 'dusn', "slots": '1:a->u(3/3)'},  # דאסן
    'האדמה': {"ipa": 'hˈadmu', "slots": '4:ə->u(3/3)'},  # האדמה
    'האלאו': {"ipa": 'hˈalu', "slots": '3:i->u(3/3)'},  # האלאו
    'הארונ': {"ipa": 'hˈurin', "slots": '1:a->u(4/4)'},  # הארון
    'ואנכי': {"ipa": 'ˈiunxi', "slots": '1:a->u(4/5)'},  # ואנכי
    'וארא': {"ipa": 'ˈiaru', "slots": '3:a->u(2/3)'},  # וארא
    'ואשה': {"ipa": 'ˈiaʃu', "slots": '3:ə->u(2/3)'},  # ואשה
    'והארצ': {"ipa": 'ˈihurʦ', "slots": '2:a->u(3/3)'},  # והארץ
    'וואגנ': {"ipa": 'vuɡn', "slots": '1:a->u(12/18)'},  # וואגן
    'וואוס': {"ipa": 'vus', "slots": '1:i->u(2/3)'},  # וואוס
    'וואוסט': {"ipa": 'vust', "slots": '1:i->u(3/4)'},  # וואוסט
    'וואורקער': {"ipa": 'vˈurkər', "slots": '1:i->u(5/6)'},  # וואורקער
    'טאגס': {"ipa": 'tuɡs', "slots": '1:a->u(4/5)'},  # טאגס
    'טאווער': {"ipa": 'tˈavur', "slots": '3:ə->u(2/3)'},  # טאווער
    'טאר': {"ipa": 'tur', "slots": '1:a->u(4/6)'},  # טאר
    'טראג': {"ipa": 'truɡ', "slots": '2:a->u(4/5)'},  # טראג
    'טראגט': {"ipa": 'truɡt', "slots": '2:a->u(2/3)'},  # טראגט
    'טראגנ': {"ipa": 'truɡn', "slots": '2:a->u(2/3)'},  # טראגן
    'טשאולנט': {"ipa": 'ʧulnt', "slots": '1:i->u(3/4)'},  # טשאולנט
    'יארצייט': {"ipa": 'jˈurʦajt', "slots": '1:a->u(23/29)'},  # יארצייט
    'יארצייטנ': {"ipa": 'jˈurʦajtn', "slots": '1:a->u(4/5)'},  # יארצייטן
    'יבואו': {"ipa": 'ˈubii', "slots": '0:i->u(2/3)'},  # יבואו
    'יקרא': {"ipa": 'ˈikru', "slots": '3:a->u(4/5)'},  # יקרא
    'ישמעאל': {"ipa": 'ˈiʃməul', "slots": '4:a->u(3/4)'},  # ישמעאל
    'כהוא': {"ipa": 'xhˈiu', "slots": '3:a->u(2/3)'},  # כהוא
    'לאדמ': {"ipa": 'ludm', "slots": '1:a->u(2/3)'},  # לאדם
    'לארצ': {"ipa": 'lurʦ', "slots": '1:a->u(3/3)'},  # לארץ
    'להגאל': {"ipa": 'lhɡul', "slots": '3:a->u(2/3)'},  # להגאל
    'ליכא': {"ipa": 'lˈuxa', "slots": '1:i->u(2/3)'},  # ליכא
    'מאנטיג': {"ipa": 'mˈuntiɡ', "slots": '1:a->u(2/3)'},  # מאנטיג
    'מאנטיק': {"ipa": 'mˈuntik', "slots": '1:a->u(3/3)'},  # מאנטיק
    'מוזיקאליש': {"ipa": 'mˈuzikaliʃ', "slots": '1:i->u(6/6)'},  # מוזיקאליש
    'מוזיקאלישע': {"ipa": 'mˈuzikaliʃə', "slots": '1:i->u(4/6)'},  # מוזיקאלישע
    'מוזיקאנט': {"ipa": 'muzikˈant', "slots": '1:i->u(3/3)'},  # מוזיקאנט
    "מורא'דיקע": {"ipa": 'mˈirudikə', "slots": '3:a->u(2/3)'},  # מורא'דיקע
    'מיהאלו': {"ipa": 'mˈihalu', "slots": '5:i->u(3/3)'},  # מיהאלו
    'מרכא': {"ipa": 'mrxu', "slots": '3:a->u(2/3)'},  # מרכא
    'נאטור': {"ipa": 'natˈur', "slots": '3:i->u(3/3)'},  # נאטור
    'נאכגעזאגט': {"ipa": 'nˈuxɡəzuɡt', "slots": '1:a->u(3/4);6:a->u(3/4)'},  # נאכגעזאגט
    'נאס': {"ipa": 'nus', "slots": '1:a->u(4/4)'},  # נאס
    'סוגיא': {"ipa": 'sˈuɡia', "slots": '1:i->u(10/11)'},  # סוגיא
    'סטודיא': {"ipa": 'stˈudia', "slots": '2:i->u(3/3)'},  # סטודיא
    'סרואל': {"ipa": 'srˈual', "slots": '2:i->u(3/3)'},  # סרואל
    'פאבליק': {"ipa": 'fˈublik', "slots": '1:a->u(2/3)'},  # פאבליק
    'פאפער': {"ipa": 'fˈufər', "slots": '1:a->u(2/3)'},  # פאפער
    'פאריגע': {"ipa": 'furˈiɡə', "slots": '1:a->u(6/8)'},  # פאריגע
    'צוגעזאגט': {"ipa": 'ʦˈiɡəzuɡt', "slots": '5:a->u(3/3)'},  # צוגעזאגט
    'צוויאי': {"ipa": 'ʦvˈiu', "slots": '3:aː->u(2/3)'},  # צוויאי
    'צונויפגעזאמלט': {"ipa": 'ʦˈinufɡəzamlt', "slots": '3:ɔj->u(3/3)'},  # צונויפגעזאמלט
    'קאטנ': {"ipa": 'kutn', "slots": '1:a->u(3/3)'},  # קאטן
    'קאמפאזיטאר': {"ipa": 'kˈamfazutar', "slots": '6:i->u(2/3)'},  # קאמפאזיטאר
    'קארטנ': {"ipa": 'kurtn', "slots": '1:a->u(4/5)'},  # קארטן
    'קיימא': {"ipa": 'kˈuma', "slots": '1:aj->u(2/3)'},  # קיימא
    'קיימאל': {"ipa": 'kˈajmul', "slots": '3:a->u(2/3)'},  # קיימאל
    'קלאגנ': {"ipa": 'kluɡn', "slots": '2:a->u(2/3)'},  # קלאגן
    'ראו': {"ipa": 'ru', "slots": '1:i->u(2/3)'},  # ראו
    'ראשי': {"ipa": 'rˈuʃi', "slots": '1:a->u(23/27)'},  # ראשי
    'רואיג': {"ipa": 'rˈuaːɡ', "slots": '1:i->u(6/8)'},  # רואיג
    'רואיגקייט': {"ipa": 'rˈuaːɡkaːt', "slots": '1:i->u(3/3)'},  # רואיגקייט
    'רואיק': {"ipa": 'rˈuaːk', "slots": '1:i->u(3/3)'},  # רואיק
    'שוא': {"ipa": 'ʃˈiu', "slots": '2:a->u(2/3)'},  # שוא
    'שואל': {"ipa": 'ʃˈual', "slots": '1:i->u(2/3)'},  # שואל
    'שולאכצ': {"ipa": 'ʃˈulaxʦ', "slots": '1:i->u(8/8)'},  # שולאכץ
    'שטאט': {"ipa": 'ʃtut', "slots": '2:a->u(29/40)'},  # שטאט
    'שטראפ': {"ipa": 'ʃtruf', "slots": '3:a->u(8/9)'},  # שטראף
    'שלאפט': {"ipa": 'ʃluft', "slots": '2:a->u(6/9)'},  # שלאפט
    'שמואליק': {"ipa": 'ʃmˈiulik', "slots": '3:a->u(3/3)'},  # שמואליק
    'שפיטאל': {"ipa": 'ʃpˈitul', "slots": '4:a->u(9/9)'},  # שפיטאל
}
