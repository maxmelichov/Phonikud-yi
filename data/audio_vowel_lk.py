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
    'אומגעזעצט': {"ipa": 'ˈumɡəzəʦt', "slots": '0:i->u(5/6)'},  # אומגעזעצט
    'אומגעפער': {"ipa": 'ˈumɡəfər', "slots": '0:i->u(4/6)'},  # אומגעפער
    'אלעזר': {"ipa": 'ˈaluzr', "slots": '2:ə->u(5/7)'},  # אלעזר
    'אנגענומענ': {"ipa": 'ˈunɡənimən', "slots": '0:a->u(3/3)'},  # אנגענומען
    'אנגעפילט': {"ipa": 'ˈunɡəfilt', "slots": '0:a->u(5/5)'},  # אנגעפילט
    'אנגעקומענ': {"ipa": 'ˈunɡəkimən', "slots": '0:a->u(16/21)'},  # אנגעקומען
    'אנגעשלאפנ': {"ipa": 'ˈanɡəʃlufn', "slots": '6:a->u(3/3)'},  # אנגעשלאפן
    'אנונג': {"ipa": 'ˈanunɡ', "slots": '2:i->u(3/4)'},  # אנונג
    'אנע': {"ipa": 'ˈunə', "slots": '0:a->u(5/6)'},  # אנע
    'אפגעבורט': {"ipa": 'ˈɔpɡəburt', "slots": '5:i->u(3/3)'},  # אפגעבורט
    'אפגעהאלטנ': {"ipa": 'ˈupɡəhaltn', "slots": '0:ɔ->u(2/3)'},  # אפגעהאלטן
    'אפגעשניטנ': {"ipa": 'ˈupɡəʃnitn', "slots": '0:ɔ->u(2/3)'},  # אפגעשניטן
    'אראפגעפארנ': {"ipa": 'arˈupɡəfurn', "slots": '7:a->u(4/4)'},  # אראפגעפארן
    'ארויסגעפארנ': {"ipa": 'arˈoʊzɡəfurn', "slots": '7:a->u(3/3)'},  # ארויסגעפארן
    'באצאלנ': {"ipa": 'baʦˈuln', "slots": '3:a->u(3/4)'},  # באצאלן
    'בלאזט': {"ipa": 'bluzt', "slots": '2:a->u(4/4)'},  # בלאזט
    'בלאזנ': {"ipa": 'bluzn', "slots": '2:a->u(4/5)'},  # בלאזן
    'גאולה': {"ipa": 'ɡˈulə', "slots": '1:i->u(7/10)'},  # גאולה
    'געטראגנ': {"ipa": 'ɡətrˈuɡn', "slots": '4:a->u(4/5)'},  # געטראגן
    'געשלאפנ': {"ipa": 'ɡəʃlˈufn', "slots": '4:a->u(9/10)'},  # געשלאפן
    'וואגנ': {"ipa": 'vuɡn', "slots": '1:a->u(12/18)'},  # וואגן
    'וואוס': {"ipa": 'vus', "slots": '1:i->u(2/3)'},  # וואוס
    'טאווער': {"ipa": 'tˈavur', "slots": '3:ə->u(2/3)'},  # טאווער
    'טאר': {"ipa": 'tur', "slots": '1:a->u(4/5)'},  # טאר
    'טראג': {"ipa": 'truɡ', "slots": '2:a->u(4/5)'},  # טראג
    'טראגט': {"ipa": 'truɡt', "slots": '2:a->u(2/3)'},  # טראגט
    'טראגנ': {"ipa": 'truɡn', "slots": '2:a->u(2/3)'},  # טראגן
    'יארצייט': {"ipa": 'jˈurʦajt', "slots": '1:a->u(23/29)'},  # יארצייט
    'יארצייטנ': {"ipa": 'jˈurʦajtn', "slots": '1:a->u(4/5)'},  # יארצייטן
    'יקרא': {"ipa": 'ˈikru', "slots": '3:a->u(4/5)'},  # יקרא
    'מאנטיג': {"ipa": 'mˈuntiɡ', "slots": '1:a->u(2/3)'},  # מאנטיג
    'נאכגעזאגט': {"ipa": 'nˈuxɡəzuɡt', "slots": '1:a->u(3/4);6:a->u(3/4)'},  # נאכגעזאגט
    'נאס': {"ipa": 'nus', "slots": '1:a->u(4/4)'},  # נאס
    'פאריגע': {"ipa": 'furˈiɡə', "slots": '1:a->u(6/8)'},  # פאריגע
    'צוויאי': {"ipa": 'ʦvˈiu', "slots": '3:aː->u(2/3)'},  # צוויאי
    'קארטנ': {"ipa": 'kurtn', "slots": '1:a->u(4/5)'},  # קארטן
    'קלאגנ': {"ipa": 'kluɡn', "slots": '2:a->u(2/3)'},  # קלאגן
    'ראשי': {"ipa": 'rˈuʃi', "slots": '1:a->u(23/27)'},  # ראשי
    'שולאכצ': {"ipa": 'ʃˈulaxʦ', "slots": '1:i->u(8/8)'},  # שולאכץ
    'שלאפט': {"ipa": 'ʃluft', "slots": '2:a->u(6/9)'},  # שלאפט
    'שמואליק': {"ipa": 'ʃmˈiulik', "slots": '3:a->u(3/3)'},  # שמואליק
    'שפיטאל': {"ipa": 'ʃpˈitul', "slots": '4:a->u(9/9)'},  # שפיטאל
}
