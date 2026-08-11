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
    'אדמ': {"ipa": 'udm', "slots": '0:a->u(21/27)'},  # אדם
    'אודער': {"ipa": 'ˈudər', "slots": '0:i->u(9/11)'},  # אודער
    'אומגעגרייט': {"ipa": 'ˈumɡəɡrajt', "slots": '0:i->u(4/4)'},  # אומגעגרייט
    'אומגעזעצט': {"ipa": 'ˈumɡəzəʦt', "slots": '0:i->u(5/6)'},  # אומגעזעצט
    'אומגעפער': {"ipa": 'ˈumɡəfər', "slots": '0:i->u(4/6)'},  # אומגעפער
    'אומעט': {"ipa": 'ˈumət', "slots": '0:i->u(2/3)'},  # אומעט
    'אומקומענ': {"ipa": 'ˈumkimən', "slots": '0:i->u(2/3)'},  # אומקומען
    'אונדזערע': {"ipa": 'ˈundzərə', "slots": '0:i->u(2/3)'},  # אונדזערע
    'אורנ': {"ipa": 'urn', "slots": '0:i->u(2/3)'},  # אורן
    'אזוינס': {"ipa": 'ˈazuns', "slots": '2:ɔj->u(3/4)'},  # אזוינס
    'איבערגעזאגט': {"ipa": 'ˈibərɡəzuɡt', "slots": '7:a->u(2/3)'},  # איבערגעזאגט
    'איכה': {"ipa": 'ˈixu', "slots": '2:ə->u(6/8)'},  # איכה
    'אלעזר': {"ipa": 'ˈaluzr', "slots": '2:ə->u(5/7)'},  # אלעזר
    'אמאאל': {"ipa": 'ˈamaul', "slots": '3:a->u(2/3)'},  # אמאאל
    'אמאליג': {"ipa": 'ˈamaluɡ', "slots": '4:i->u(2/3)'},  # אמאליג
    'אמאליגע': {"ipa": 'ˈamuliɡə', "slots": '2:a->u(4/5)'},  # אמאליגע
    'אנגעגרייט': {"ipa": 'ˈunɡəɡrajt', "slots": '0:a->u(3/3)'},  # אנגעגרייט
    'אנגעטראגנ': {"ipa": 'ˈanɡətruɡn', "slots": '6:a->u(2/3)'},  # אנגעטראגן
    'אנגעמאכט': {"ipa": 'ˈunɡəmaxt', "slots": '0:a->u(3/3)'},  # אנגעמאכט
    'אנגענומענ': {"ipa": 'ˈunɡənimən', "slots": '0:a->u(3/3)'},  # אנגענומען
    'אנגעפילט': {"ipa": 'ˈunɡəfilt', "slots": '0:a->u(5/5)'},  # אנגעפילט
    'אנגעקומענ': {"ipa": 'ˈunɡəkimən', "slots": '0:a->u(16/21)'},  # אנגעקומען
    'אנגעשלאפנ': {"ipa": 'ˈanɡəʃlufn', "slots": '6:a->u(3/3)'},  # אנגעשלאפן
    'אנגעשריבנ': {"ipa": 'ˈunɡəʃribn', "slots": '0:a->u(2/3)'},  # אנגעשריבן
    'אנגרייטנ': {"ipa": 'ˈunɡrajtn', "slots": '0:a->u(2/3)'},  # אנגרייטן
    'אנהויב': {"ipa": 'ˈunhɔjb', "slots": '0:a->u(4/6)'},  # אנהויב
    'אנונג': {"ipa": 'ˈanunɡ', "slots": '2:i->u(3/4)'},  # אנונג
    'אנטשולדיגט': {"ipa": 'ˈanʧuldiɡt', "slots": '3:i->u(2/3)'},  # אנטשולדיגט
    'אנע': {"ipa": 'ˈunə', "slots": '0:a->u(5/6)'},  # אנע
    'אנקומענ': {"ipa": 'ˈunkimən', "slots": '0:a->u(3/3)'},  # אנקומען
    'אפגעבורט': {"ipa": 'ˈɔpɡəburt', "slots": '5:i->u(3/3)'},  # אפגעבורט
    'אפגעהאלטנ': {"ipa": 'ˈupɡəhaltn', "slots": '0:ɔ->u(2/3)'},  # אפגעהאלטן
    'אפגעקויפט': {"ipa": 'ˈupɡəkɔjft', "slots": '0:ɔ->u(4/4)'},  # אפגעקויפט
    'אפגעשניטנ': {"ipa": 'ˈupɡəʃnitn', "slots": '0:ɔ->u(2/3)'},  # אפגעשניטן
    'אראפגעפארנ': {"ipa": 'arˈupɡəfurn', "slots": '7:a->u(4/4)'},  # אראפגעפארן
    'ארויסגעפארנ': {"ipa": 'arˈoʊzɡəfurn', "slots": '7:a->u(3/3)'},  # ארויסגעפארן
    'ארויפשרייבנ': {"ipa": 'arˈufʃrajbn', "slots": '2:oʊ->u(3/3)'},  # ארויפשרייבן
    'אשה': {"ipa": 'ˈaʃu', "slots": '2:ə->u(2/3)'},  # אשה
    'באצאלנ': {"ipa": 'baʦˈuln', "slots": '3:a->u(3/4)'},  # באצאלן
    'בארצ': {"ipa": 'burʦ', "slots": '1:a->u(2/3)'},  # בארץ
    'בואו': {"ipa": 'bˈiu', "slots": '2:i->u(3/4)'},  # בואו
    'בלאזט': {"ipa": 'bluzt', "slots": '2:a->u(4/4)'},  # בלאזט
    'בלאזנ': {"ipa": 'bluzn', "slots": '2:a->u(4/5)'},  # בלאזן
    'גאולה': {"ipa": 'ɡˈulə', "slots": '1:i->u(7/10)'},  # גאולה
    'געטראגנ': {"ipa": 'ɡətrˈuɡn', "slots": '4:a->u(4/5)'},  # געטראגן
    'געצאלט': {"ipa": 'ɡəʦˈult', "slots": '3:a->u(3/4)'},  # געצאלט
    'געשלאפנ': {"ipa": 'ɡəʃlˈufn', "slots": '4:a->u(9/10)'},  # געשלאפן
    'דאסנ': {"ipa": 'dusn', "slots": '1:a->u(3/3)'},  # דאסן
    'דורכגעגאנגענ': {"ipa": 'dˈurxɡəɡanɡən', "slots": '1:i->u(2/3)'},  # דורכגעגאנגען
    'האלאו': {"ipa": 'hˈalu', "slots": '3:i->u(3/3)'},  # האלאו
    'הארונ': {"ipa": 'hˈurin', "slots": '1:a->u(3/3)'},  # הארון
    'וואגנ': {"ipa": 'vuɡn', "slots": '1:a->u(12/18)'},  # וואגן
    'וואוס': {"ipa": 'vus', "slots": '1:i->u(2/3)'},  # וואוס
    'וואורקער': {"ipa": 'vˈurkər', "slots": '1:i->u(3/3)'},  # וואורקער
    'טאגס': {"ipa": 'tuɡs', "slots": '1:a->u(3/3)'},  # טאגס
    'טאווער': {"ipa": 'tˈavur', "slots": '3:ə->u(2/3)'},  # טאווער
    'טאר': {"ipa": 'tur', "slots": '1:a->u(4/5)'},  # טאר
    'טראג': {"ipa": 'truɡ', "slots": '2:a->u(4/5)'},  # טראג
    'טראגט': {"ipa": 'truɡt', "slots": '2:a->u(2/3)'},  # טראגט
    'טראגנ': {"ipa": 'truɡn', "slots": '2:a->u(2/3)'},  # טראגן
    'יארצייט': {"ipa": 'jˈurʦajt', "slots": '1:a->u(23/29)'},  # יארצייט
    'יארצייטנ': {"ipa": 'jˈurʦajtn', "slots": '1:a->u(4/5)'},  # יארצייטן
    'יקרא': {"ipa": 'ˈikru', "slots": '3:a->u(4/5)'},  # יקרא
    'לארצ': {"ipa": 'lurʦ', "slots": '1:a->u(3/3)'},  # לארץ
    'מאנטיג': {"ipa": 'mˈuntiɡ', "slots": '1:a->u(2/3)'},  # מאנטיג
    'מאנטיק': {"ipa": 'mˈuntik', "slots": '1:a->u(3/3)'},  # מאנטיק
    'מוזיקאליש': {"ipa": 'mˈuzikaliʃ', "slots": '1:i->u(4/4)'},  # מוזיקאליש
    'מוזיקאלישע': {"ipa": 'mˈuzikaliʃə', "slots": '1:i->u(3/3)'},  # מוזיקאלישע
    "מורא'דיקע": {"ipa": 'mˈirudikə', "slots": '3:a->u(2/3)'},  # מורא'דיקע
    'מוראדיגער': {"ipa": 'mˈuradiɡər', "slots": '1:i->u(3/4)'},  # מוראדיגער
    'מיהאלו': {"ipa": 'mˈihalu', "slots": '5:i->u(3/3)'},  # מיהאלו
    'נאכגעזאגט': {"ipa": 'nˈuxɡəzuɡt', "slots": '1:a->u(3/4);6:a->u(3/4)'},  # נאכגעזאגט
    'נאנט': {"ipa": 'nunt', "slots": '1:a->u(2/3)'},  # נאנט
    'נאנטער': {"ipa": 'nˈuntər', "slots": '1:a->u(3/4)'},  # נאנטער
    'נאס': {"ipa": 'nus', "slots": '1:a->u(4/4)'},  # נאס
    'סוגיא': {"ipa": 'sˈuɡia', "slots": '1:i->u(5/5)'},  # סוגיא
    'פאבליק': {"ipa": 'fˈublik', "slots": '1:a->u(2/3)'},  # פאבליק
    'פאריגע': {"ipa": 'furˈiɡə', "slots": '1:a->u(6/8)'},  # פאריגע
    'צוויאי': {"ipa": 'ʦvˈiu', "slots": '3:aː->u(2/3)'},  # צוויאי
    'קאמפאזיטאר': {"ipa": 'kˈamfazutar', "slots": '6:i->u(2/3)'},  # קאמפאזיטאר
    'קארטנ': {"ipa": 'kurtn', "slots": '1:a->u(4/5)'},  # קארטן
    'קלאגנ': {"ipa": 'kluɡn', "slots": '2:a->u(2/3)'},  # קלאגן
    'ראו': {"ipa": 'ru', "slots": '1:i->u(2/3)'},  # ראו
    'ראשי': {"ipa": 'rˈuʃi', "slots": '1:a->u(23/27)'},  # ראשי
    'רבא': {"ipa": 'rbu', "slots": '2:a->u(4/6)'},  # רבא
    'רואיג': {"ipa": 'rˈuaːɡ', "slots": '1:i->u(4/6)'},  # רואיג
    'רואיגקייט': {"ipa": 'rˈuaːɡkaːt', "slots": '1:i->u(3/3)'},  # רואיגקייט
    'שאפראנ': {"ipa": 'ʃˈafrun', "slots": '4:a->u(2/3)'},  # שאפראן
    'שולאכצ': {"ipa": 'ʃˈulaxʦ', "slots": '1:i->u(8/8)'},  # שולאכץ
    'שטאט': {"ipa": 'ʃtut', "slots": '2:a->u(18/26)'},  # שטאט
    'שלאגנ': {"ipa": 'ʃluɡn', "slots": '2:a->u(3/4)'},  # שלאגן
    'שלאפט': {"ipa": 'ʃluft', "slots": '2:a->u(6/9)'},  # שלאפט
    'שמואליק': {"ipa": 'ʃmˈiulik', "slots": '3:a->u(3/3)'},  # שמואליק
    'שפיטאל': {"ipa": 'ʃpˈitul', "slots": '4:a->u(9/9)'},  # שפיטאל
}
