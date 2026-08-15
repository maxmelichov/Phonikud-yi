#!/usr/bin/env python
"""Test suite for yiddish_g2p.py — run: .venv/bin/python scripts/test_g2p.py

Covers the three input modes (unpointed Hasidic, YIVO, fully pointed) plus the
known-bug cases, marked XFAIL so a fix flips them to PASS visibly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yiddish_g2p import g2p_token, hebrew_to_ipa  # noqa: E402

# LK root + Germanic suffix (rescue #4). Every positive is a real corpus type
# that used to leave the rule path as a consonant string; the expected reading
# is root-IPA + suffix-IPA with the root's own stress kept. (word, ipa, reason).
STEM_CASES: list[tuple[str, str, str, str]] = [
    ("פשטלעך", "pʃatləx", "gold+suffix", "פשט gold + לעך; was fʃtlɛx"),
    ("רבין", "rˈɛbən", "gold+suffix", "רבי + ן, 66 corpus tokens; was rbin"),
    ("גמראס", "ɡəmˈurəs", "gold+suffix", "גמרא + ס; was ɡmras"),
    ("כבודן", "kˈuvədn", "lk-lexicon+suffix", "merged-LK root + ן; was xbidn"),
    ("צדיקס", "ʦˈadiks", "lk-lexicon+suffix", "merged-LK root + ס; was ʦdiks"),
    ("תעשׂהס", "taˈasɛs", "sefaria-pointed+suffix", "book-pointed root + ס"),
    ("הפקרדיגע", "hˈɛfkajrdiɡə", "pointed-audio-endorsed+suffix",
     "audio-endorsed root + דיגע; was the quarantined hfkrdˈiɡə"),
    ("שעהן", "ʃuən", "gold+suffix",
     "שעה 'ʃu' + ן takes the linking vowel: shoen, not *ʃun"),
    # --- negatives: the stemmer must keep its hands off ------------------
    # POINTED GERMANIC. _lk_detector's shape clause reads 'no vowel letter' as
    # 'Hebrew', which is meaningless once the vowels are written as points:
    # שנסט / שלכט / שטר / קלפ / פלג all pass it while שענסט / שלעכט / שטער /
    # קלאפ / פלעג correctly do not. The model-guess guard therefore has to use
    # the marker clauses on pointed roots, or the stemmer rebuilds ordinary
    # Yiddish words out of Hebrew readings.
    ("שֶׁנְסְטֶע", "ʃˈɛnstə", "", "pointed Germanic; was *ʃnustə"),
    ("שְׁלֶכְטֶע", "ʃlˈɛxtə", "", "pointed Germanic; was *ʃˈɛlxutə"),
    ("שְׁטֶרְן", "ʃtɛrn", "lk-fallback", "pointed Germanic; was *ʃtarn"),
    ("קְלַפְּן", "klapn", "lk-fallback", "pointed Germanic; was *kəlfn"),
    ("פְלֶגְן", "flɛɡn", "pe-default,lk-fallback", "pointed Germanic; was *pˈɛləɡn"),
    # SEAM. Root-IPA + suffix-IPA is not a phonology: a geminate at the join or
    # a lost nucleus means the split (or the root reading) is wrong.
    ("כוסס", "xis", "", "geminate *kɔjss declined"),
    ("חַזֶרְנֶן", "xˈazərnən", "lk-fallback", "geminate *xˈazərnn declined"),
    ("תאוועס", "sˈavəs", "alef-default,lk-fallback", "geminate *taˈavvɛs declined"),
    ("מַחְלוֹקֶס", "mˈaxlɔjkəs", "lk-fallback", "syllable loss *mˈaxlɔjks declined"),
    ("חַתֶנֶעס", "xˈasənəs", "lk-fallback", "syllable loss *xˈasnəs declined"),
    ("שמחס", "ʃmxs", "lk-fallback",
     "the defective plural of שמחה, not שמח + ס (_STEM_NO_SPLIT)"),
    # _STEM_SUBS owns these whether or not the input is pointed: one lexeme,
    # one path, one reading.
    ("אֱמֶתְדִיג", "ˈɛməzdiɡ", "lk-fallback", "== אמתדיג; was *ˈɛməsdiɡ"),
    ("שַׁבָּתְדִיגֶע", "ʃˈabəzdiɡə", "lk-fallback", "== שבתדיגע; was *ʃˈabəsdiɡə"),
    ("אמתדיג", "ˈɛməzdiɡ", "alef-default,lk-fallback",
     "_STEM_SUBS owns this one: the pointed base wins over root+suffix"),
    ("חסידישע", "xˈusidiʃə", "lk-fallback", "_STEM_SUBS again (חסיד -> כאָסיד)"),
    ("מענטשן", "mɛnʧn", "", "Germanic word, whole-token lexicon entry"),
    ("גוטע", "ɡˈitə", "", "Germanic -טע is not an LK root + suffix"),
    ("וורטלעך", "vrtlɛx", "",
     "Germanic וורט sits in the model-guess table; a guess alone is not LK evidence"),
    ("הרשע", "hrʃə", "", "ha-ROshe: הר + שע is a two-letter root, below the floor"),
    ("בנין", "bnin", "", "binyen is one morpheme, not בני + ן (_STEM_NO_SPLIT)"),
    ("כדין", "xdin", "", "kədin is one morpheme, not כדי + ן (_STEM_NO_SPLIT)"),
    # Documented gap, not a target: the root רבנו is in NO table (it is never
    # quarantined, so the model builder never saw it), so the stemmer has
    # nothing to resolve and the possessive stays wrong. A whole-token or root
    # entry for רבנו is what fixes it -- 717 corpus tokens ride on it.
    ("רבנוס", "rbnis", "", "XFAIL-ish: unresolvable until רבנו is in a lexicon"),
]

CASES: list[tuple[str, str, str]] = [
    # (input, expected IPA, note)
    # --- unpointed Hasidic (regression: pre-rewrite behavior) ---
    ("וואס", "vus", "mid-word alef = o (Central u)"),
    ("היינט", "haːnt", "ay flattens to long aː (haant), distinct from ey->aj"),  # v3
    ("איך וויל גיין", "ix vil ɡajn", "basic Germanic"),  # v3: aɪ -> aj
    ("שבת", "ʃabɛs", "LK lexicon swap"),
    ("אויף", "oʊf", "v3 §9: oʊf standalone; afn is the fused/reduced form"),  # v3
    # --- pointed loshn-koydesh (the new capability) ---
    ("כְּלַל", "klal", "sheva+patah resolved"),
    ("בֵּית", "bajs", "tsere + matres yud"),  # v3: aɪ -> aj
    ("בְּרָכָה", "bruxə", "feminine -e ending"),
    ("שִׂמְחָה", "simxə", "sin dot"),
    ("פְּשַׁט", "pʃat", "dagesh pe = p"),
    ("מִצְוָה", "miʦvə", "vav as consonant /v/"),
    ("כִּי", "ki", "dagesh kaf = k"),
    ("יוֹם", "jɔjm", "holam = oy"),  # v3: ɔɪ -> ɔj
    # --- pointed Yiddish (redundant Hasidic pointing) ---
    ("דֶער", "dɛr", "consonant point + restating ayin stays single"),
    ("זָאגְט", "zuɡt", "komets alef digraph; v3 §10.2 devoicing is OFF, so zuɡt"),  # v3
    ("ווֶען", "vɛn", "pointed vov-vov"),
    # --- the פאר homograph (far vs pur) ---
    ("פּאָר", "pur", "explicit dagesh-pe overrides the stripped-key gold far"),
    ("פֿאַר", "far", "explicit rafe agrees with gold; lexicon hit unchanged"),
    # --- known bugs (XFAIL until fixed) ---
    ("וַוייל", "vaːl", "pasekh on vav makes yy=ay -> aː"),
    ("צְבִי", "ʦvi", "bare bet in pointed LK is /v/ (was XFAIL, now fixed)"),
]


# Stress marking (hebrew_to_ipa(..., stress=True)).
STRESS_CASES: list[tuple[str, str, str]] = [
    ("שבת", "ʃˈabəs", "LK penultimate"),
    ("משפּחה", "miʃpˈuxə", "LK penultimate, 3 syllables"),
    ("חתונה", "xˈasənə", "LK penultimate"),
    ("געקומען", "ɡəkˈimən", "unstressed prefix ge-"),
    ("פארשטיין", "farʃtˈajn", "unstressed prefix far-"),  # v3: aɪ -> aj
    ("ארבעטן", "ˈarbətn", "v3 §1: syllabic -n takes no epenthetic ə; ע is a written vowel so the word IS marked"),  # v3
    ("אונטערגיין", "ˈintərɡajn", "separable prefix IS stressed"),  # v3: aɪ -> aj
    # --- prefix-stem rescue (R26) stress + no-split regressions -----------
    ("געבעטן", "ɡəbˈejtn", "rescue must MARK an unmarked ej/oʊ stem (phone-token scan, not char scan)"),
    ("דעראויף", "dərˈoʊf", "rescue must mark an unmarked oʊ stem"),
    ("פארעם", "farˈɛm", "_PREFIX_NO_SPLIT: 'form' is one lexeme, never far+עם -> *farˈejm"),
    ("אנדער", "ˈandər", "_PREFIX_BAD_STEMS: verbal אנ־ never takes the article דער as stem (gold אנדערע agrees)"),
    ("געפארן", "ɡəfˈarn", "גע־ never takes the contraction פארן as stem; stays LOW-queued (audio hints u)"),
    ("פארדעם", "fardˈejm", "pronominal adverb: unstressed פאר־ MAY fuse the article (gold נאכדעם anchors dejm)"),
    ("דערצו", "dərʦˈi", "pronominal adverb: unstressed דער־ + gold צו"),
    # --- R27 LK proclitic repair ------------------------------------------
    ("בשבת", "bəʃˈabəs", "proclitic ב + gold שבת; the bare read was *bʃˈabəs"),
    ("החסיד", "haxˈusid", "proclitic ה + the §6.2 stem-sub base חסיד"),
    ("הבית", "habˈajis", "a whole-word pointing beats composition: gold בית is the construct bajs"),
    ("שלחן", "ʃlxajn", "one root (shulkhn): ש is not a proclitic here, tail לחן unattested"),
    ("בדחן", "bdxajn", "one root (badkhn), not ב + דחן"),
    ("גיין", "ɡajn", "monosyllable stays unmarked"),  # v3: aɪ -> aj
    # --- _STRESS_OVERRIDE entries confirmed against audio (scripts/stress_eval.py) ---
    ("אזוי", "azˈɔj", "override: unstressed initial a-, not *Azoy"),  # v3: ɔɪ -> ɔj
    ("חנוכה", "xˈanikə", "override: LK with initial, not penultimate, stress"),
    ("אינטערעסאנט", "intərəsˈant", "override: loanword keeps donor stress"),
    # --- v3 core-phonology regressions ------------------------------------
    ("מאכן", "maxn", "v3: syllabic -n, one written nucleus -> monosyllable, NO mark"),
    ("זאגן", "zuɡn", "v3 §1/§11.2: zuɡn, no ə, no stress mark"),
    ("פארן", "farn", "v3: farn is a monosyllable by the written-nucleus count"),
    ("מענטשן", "mɛnʧn", "v3 §11.2: one written nucleus (ע) -> monosyllable, no mark; gold agrees"),
    ("איז", "iz", "v3 §10.2: no final devoicing"),
    ("אויב", "ɔjb", "v3 §10.2: ɔjb, not ɔjp"),
    ("וועג", "vejɡ", "v3 §5/§10.2: ej away from r, and no final devoicing"),
    ("מער", "mɛr", "v3 §5: the ־ער default is ɛr; ejr never occurs before r"),
    ("ווער", "vɛr", "v3 §5: ɛr default"),
    ("שווער", "ʃvir", "v3 §5: the closed ir-list survives"),
    ("לערנען", "lˈirnən", "v3 §5 ir-list spelling is lirnen, not leernen"),
    ("מסביר", "mazbˈir", "v3 §10.1: voicing-ward assimilation stays ON"),
    ("צוויי", "ʦvaj", "v3 §10.1: /v/ is not a voicing trigger"),
    ("זעהן", "zejn", "v3 §5: ה silent after a vowel before a consonant"),
    ("ייד", "jid", "v3 §4: word-initial יי = ji"),
    ("וואו", "vi", "v3 §5: א silent before a vowel-ו"),
    ("אונז", "inz", "v3 §5 + §10.2"),
    ("אייביג", "ˈajbiɡ", "v3 §5: suffix ־יג = iɡ, no devoicing"),
    ("ארויס", "arˈoʊs", "v3 §11.4: directional a(r)- stresses the second nucleus"),
    ("אראפ", "arˈup", "v3 §11.4"),
    ("אהיים", "ahˈajm", "v3 §11.4"),
    ("אוועק", "avˈɛk", "v3 §11.4"),
    ("אריין", "arˈaːn", "v3 §11.4"),
    # --- the פאר homograph (far vs pur): MWE fires only on the §3 routing path
    ("א פאר יאר", "a pˈur jur", "MWE: article + pair-noun (gold אפאר apˈur)"),
    ("פאר דער וואך", "far dɛr vɔx", "bare פאר without the article stays gold far"),
    # --- audio-pe table (data/audio_pe_lk.py, xeus_pe_sweep votes) ---
    ("פעקל", "pɛkl", "audio-pe flip: p=23/f=0 across clips"),
    ("פֿעקל", "fɛkl", "written rafe outranks the audio-pe table"),
    ("כאפן", "xapn", "audio-pe: khapn 'to catch', p=12/f=1"),
    ("פסוקים", "psˈikim", "audio-pe: LK plural, p=11/f=0"),
    ("עפל", "ɛpl", "audio-pe: epl 'apple', p=4/f=0"),
    ("דאפלט", "daplt", "audio-pe: doplt 'double', p=12/f=0"),
    ("ראפשיצער", "rˈapʃiʦər", "audio-pe: Ropshitser (Hasidic name), p=12/f=0"),
    ("קאפיטל", "kˈapitl", "audio-pe: kapitl 'chapter', p=5/f=0"),
    ("געקלאפט", "ɡəklˈapt", "audio-pe: geklapt, root קלאפ, p=3/f=0"),
    ("אפטייקער", "ˈaptajkər", "audio-pe: apteyker 'pharmacist', p=3/f=0"),
    ("א פאר מינוט", "a pˈur minˈit", "MWE + audio loop: a pur minit"),
    # --- audio-vowel table (data/audio_vowel_lk.py): komets vowels the
    #     a-default missed, heard as u across clips; stress marks unmoved ---
    ("יארצייט", "jˈurʦajt", "audio-vowel: yurtsayt, u=23/29 clips"),
    ("שפיטאל", "ʃpˈitul", "audio-vowel: shpitul, u=9/9"),
    ("געשלאפן", "ɡəʃlˈufn", "audio-vowel: geshlufn, u=9/10"),
    ("אנגעקומען", "ˈunɡəkimən", "audio-vowel: ungekumen, u=16/21"),
    ("אראפגעפארן", "arˈupɡəfurn", "audio-vowel: arupgefurn, both u slots"),
]


def main() -> int:
    passed = failed = xfailed = fixed = 0
    for text, want, note in CASES + [(t, w, "stress: " + n) for t, w, n in STRESS_CASES]:
        got = hebrew_to_ipa(text, stress=note.startswith("stress:"))
        xfail = note.startswith("XFAIL")
        ok = got == want
        if ok and not xfail:
            passed += 1
            status = "PASS "
        elif ok and xfail:
            fixed += 1
            status = "FIXED"
        elif xfail:
            xfailed += 1
            status = "xfail"
        else:
            failed += 1
            status = "FAIL "
        print(f"{status}  {text!r:30s} -> {got!r:18s} (want {want!r})  {note}")
    for text, want, want_reason, note in STEM_CASES:
        rec = g2p_token(text)
        got, got_reason = rec["ipa_primary"], rec["reason"]
        ok = got == want and got_reason == want_reason
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"{'PASS ' if ok else 'FAIL '}  {text!r:30s} -> {got!r:18s} "
              f"[{got_reason}] (want {want!r} [{want_reason}])  stem: {note}")
    print(f"\n{passed} passed, {failed} FAILED, {xfailed} known-bugs, {fixed} newly fixed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
