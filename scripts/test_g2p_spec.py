#!/usr/bin/env python
"""SPEC regression suite for yiddish_g2p.py — the phonemization guide as tests.

Encodes the "Contemporary Hasidic Yiddish — Phonemization Guide (G2P Spec)"
(Unterland/Satmar koine) as executable expectations. Unlike scripts/test_g2p.py,
which pins the engine's CURRENT behaviour, this file pins the SPEC's target
behaviour, so a large number of FAILs here is expected and informative: each one
is a place the engine and the spec disagree.

Run:  python3 scripts/test_g2p_spec.py      (or .venv/bin/python)

TARGET PHONE MAP (spec romanization -> IPA), fixed for this whole file:
    a  [a]    aa [aː]   e  [ɛ]    ey [ej]   ay [aj]   i  [i]
    o  [ɔ]    oy [ɔj]   ou [oʊ]   u  [u]    @  [ə]
    kh [x]  sh [ʃ]  zh [ʒ]  ts [ʦ]  tsh [ʧ]  dzh [ʤ]  dz [ʣ]  ng [ŋ]
    g [ɡ]   y [j]   one l, one r (no dark-l / r-variant diacritics)
Stress mark ˈ sits immediately BEFORE the stressed vowel, not before the onset.

NOTE ON THE ENGINE'S LATIN LAYER: the engine's internal romanization uses
different labels from the spec ("ey" = spec ay, "ay" = spec aa, "o" = spec u).
That is an internal detail; only the IPA output is asserted here.

SPEC v3 (2026-08-06) — this file has been migrated. The changes, all marked
"# v3" on the affected line:
  * notation aɪ -> aj and ɔɪ -> ɔj (v3 §1); ʣ left the inventory, dz is d + z
  * v3 §10.2 turns DEVOICING OFF everywhere: no final devoicing (kind, ɔjb, hub,
    iz, vejɡ, briv) and no devoicing-ward assimilation (zuɡt, davkə, rivkə).
    Only voicing-ward assimilation survives (§10.1: ʃabɛzdik, mazbˈir).
  * v3 §1 syllabic finals -n -l -m get NO epenthetic ə: zuɡn, not zuɡən.
  * v3 §11.2 counts WRITTEN vowel nuclei, so maxn / zuɡn / farn are
    monosyllables and carry no stress mark; ˈarbətn is marked (ע is written).
  * v3 §5 settles the ־ער system at the ɛr default (mɛr, vɛr, bɛrɡ, klɛrt) with
    a closed ir-list (ʃvir, virn, virt, hirn, lirnen, ɡəhˈirt). Several cases
    that were recorded here as spec/lexicon CONFLICTS are therefore now plain
    passes.
  * v3 §11.4 stresses the second nucleus of every directional a(r)- word,
    bare or compounded.

Segmental cases run with stress=False so unstressed /ɛ/ is not reduced to /ə/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yiddish_g2p import g2p_token, g2p_tokens, hebrew_to_ipa  # noqa: E402

# (hebrew input, expected IPA, note).  Notes carry the spec romanization and the
# Weinreich diaphoneme class.  A note containing "XFAIL" marks a known/accepted
# divergence rather than a defect to fix blindly.
CASES: list[tuple[str, str, str]] = [
    # =================================================================
    # §11 MINIMAL-PAIR TEST SET (the spec's own regression list, 1-15)
    # =================================================================
    # 1. man / maan
    ("מאן", "man", "§11.1 man 'husband' (11)"),
    ("מיין", "maːn", "§11.1 maan 'my' (34)"),
    # 2. hant / haant
    ("האנט", "hant", "§11.2 hant 'hand' (11)"),
    ("היינט", "haːnt", "§11.2 haant 'today' (34), also §8"),
    # 3. vays(know) / vaas(white) — IDENTICAL spelling ווייס, lexical only (§3).
    #    One reading is chosen here: the verb, which is far the more frequent in
    #    running text.  The adjective reading would be [vaːs] (34) and cannot be
    #    distinguished by any rule.
    ("ווייס", "vajs", "§11.3 vays 'I know' (24); the 'white' reading is vaas [vaːs] (34), same spelling"),
    # 4. oykh / boukh — the 42/44 vs 54 split, the top transcription-error source
    ("אויך", "oʊx", "§11.4 oykh 'also'; v3 §4 puts oukh on the oʊ-list"),  # v3
    ("בויך", "boʊx", "§11.4 boukh 'belly' (54)"),
    # 5. broyt / hout
    ("ברויט", "brɔjt", "§11.5 broyt 'bread' (42/44)"),
    ("הויט", "hoʊt", "§11.5 hout 'skin' (54)"),
    # 6. ayn-(one) / aan-(in-)
    ("איין", "ajn", "§11.6 ayn 'one' (24)"),
    ("איינקוקן", "aːnkikn", "XFAIL §11.6 aankikn: aan- 'in-' prefix (34) — same unpointed spelling as ayn- (24), lexical only"),  # v3
    # 7. mern(carrots, 21) / meyrn(increase, 25) — identical spelling מערן
    ("מערן", "mɛrn", "§11.7 mern 'carrots' (21); v3 §1 no epenthetic ə in the syllabic -n"),  # v3
    # 8. spec: no ber/beyr pair — test berl (21) vs leyrn (25) instead
    ("בערל", "bɛrl", "§11.8 berl (21) — diminutive never *beyrl (§9)"),
    ("לערנען", "lirnɛn", "v3 §5: לערנען is on the closed ir-list — lirnen, not leyrnen"),  # v3
    # 9. in(and) / in(in) — true homophone, two spellings
    ("און", "in", "§11.9 in 'and' (51) — homophone of אין"),
    ("אין", "in", "§11.9 in 'in' (51)"),
    # 10. zin(sun) / zin(son) — 51/52 merger, one spelling
    ("זון", "zin", "§11.10 zin — 'sun' (51) and 'son' (52) merge"),
    # 11. shtayn(stand) / shtayn(stone) — homophones here, unlike StY
    ("שטיין", "ʃtajn", "§11.11 shtayn 'stand' AND 'stone' (22/24)"),
    # 12. Got(o) / gut->git(i)
    ("גאט", "ɡɔt", "§11.12 Got (41) — short o, never lengthened"),
    ("גוט", "ɡit", "§11.12 git 'good' (52) — long u -> i"),
    # 13. nishmusu/nishmusoy needs pointed WH input (§5 suffix ־יו); the
    #     unpointed spelling נשמתו/נשמתה is not resolvable — deliberately omitted.
    # 14. melech(WH) / maylech(Yiddish) — same spelling מלך (§7.3)
    ("מֶלֶךְ", "mɛlɛx", "§11.14 melech (WH, segol=e, 21) — pointed, quoted register"),
    ("מלך", "majlɛx", "§11.14 maylech (merged Yiddish) — LK lexicon entry; the rule path alone emits the vowel-less mlx, which v3 §1 forbids"),  # v3
    # 15. Toyru(formal) / Toyre(casual)
    ("תורה", "tɔjrə", "§11.15 Toyre (casual, final kometz-hey -> @); formal WH is Toyru [tɔɪru]"),

    # =================================================================
    # §2 CHAIN-SHIFT CLASSES (unpointed Hasidic spelling)
    # =================================================================
    # --- class 11: a (short a) ---
    ("גאס", "ɡas", "11 gas"),
    ("נאכט", "naxt", "11 nakht"),
    ("הארץ", "harʦ", "11 harts"),
    ("גלאז", "ɡlaz", "v3 §10.2: NO final devoicing — ɡlaz stays voiced"),  # v3
    # --- class 12/13: u (MHG ā / lengthened a) ---
    ("וואס", "vus", "12/13 vus, also §8"),
    ("יאר", "jur", "12/13 yur"),
    ("דאס", "dus", "12/13 dus, also §8"),
    ("שלאפן", "ʃlufn", "12/13 shlufn; v3 §1 syllabic -n takes no ə"),  # v3
    ("זאגן", "zuɡn", "12/13 zugn — v3 §1's own example of the bare syllabic -n"),  # v3
    ("נאך", "nux", "12/13 nukh"),
    # --- class 21: e [ɛ] (closed syll., geminates, r+C) ---
    ("בעט", "bɛt", "21 bet"),
    ("מענטש", "mɛnʧ", "21 mentsh"),
    ("געלט", "ɡɛlt", "21 gelt"),
    ("טעלער", "tɛlɛr", "21 teler"),
    ("בערג", "bɛrɡ", "v3 §5: r+cluster words take the ɛr default; the barg lexicon entry is retired"),  # v3
    # --- class 25: ey [ej] (e lengthened in open syllable) ---
    ("מער", "mɛr", "v3 §5: the ־ער default is ɛr — ejr never occurs before r"),  # v3
    ("געווען", "ɡɛvejn", "25 geveyn 'been', also §8 (unstressed ge- keeps ɛ at stress=False)"),
    ("זען", "zejn", "25 zeyn 'see'"),
    ("לעבן", "lejbn", "25 leybn; v3 §1 syllabic -n"),  # v3
    ("שווער", "ʃvir", "v3 §5 ir-list — no longer a spec/lexicon conflict"),  # v3
    ("הערן", "hirn", "v3 §5 ir-list — no longer a spec/lexicon conflict"),  # v3
    # --- class 22/24: ay [aɪ] (MHG ei / ē) ---
    ("היים", "hajm", "22/24 haym"),
    ("איינס", "ajns", "22/24 ayns"),
    ("צוויי", "ʦvaj", "22/24 tsvay"),
    ("גיין", "ɡajn", "22/24 gayn"),
    ("קיין", "kajn", "22/24 kayn 'no'"),
    ("זיידע", "zajdə", "22/24 zayde (final unstressed -e = @)"),
    # --- class 31/32: i ---
    ("טיש", "tiʃ", "31/32 tish"),
    ("נישט", "niʃt", "31/32 nisht"),
    ("זיבן", "zibn", "31/32 zibn; v3 §1 syllabic -n"),  # v3
    ("זיצן", "ziʦn", "31/32 zitsn; v3 §1 syllabic -n"),  # v3
    ("בריוו", "briv", "v3 §10.2: no final devoicing — briv"),  # v3
    # --- class 34: aa [aː] (MHG ī, monophthongized) ---
    ("זיין", "zaːn", "34 zaan 'be' (contrast zayn 'his', also 34 here)"),
    ("שרייבן", "ʃraːbn", "34 shraabn; v3 §1 syllabic -n"),  # v3
    ("ווייט", "vaːt", "34 vaat"),
    ("צייט", "ʦaːt", "34 tsaat"),
    ("דריי", "draː", "34 draa 'three' (§8 numbers)"),
    # --- class 41: o [ɔ] (short o, never lengthened) ---
    ("דארט", "dɔrt", "41 dort"),
    ("וואך", "vɔx", "41 vokh"),
    ("קאפ", "kɔp", "41 kop"),
    ("מארגן", "mɔrɡn", "41 morgn; v3 §1 syllabic -n"),  # v3
    ("האט", "hɔt", "41 hot"),
    # --- class 42/44: oy [ɔɪ] (MHG ō / ou) ---
    ("גרויס", "ɡrɔjs", "42/44 groys"),
    ("רויט", "rɔjt", "42/44 royt"),
    ("שוין", "ʃɔjn", "42/44 shoyn"),
    ("אזוי", "azɔj", "42/44 azoy"),
    # --- class 51: i (short u) ---
    ("פון", "fin", "51 fin 'from', also §8"),
    # CASE FIXED: -en is [ɛn] at stress=False by this suite's own convention
    # (cf. לערנען -> lejrnɛn below); the [ə] reduction is a stress=True rule.
    ("קומען", "kimɛn", "51 kimen"),
    ("פרום", "frim", "51 frim"),
    ("זומער", "zimɛr", "51 zimer"),
    ("הונדערט", "hindɛrt", "51 hindert"),
    # --- class 52: i (long u) ---
    ("שול", "ʃil", "52 shil"),
    ("בוך", "bix", "52 bikh — contrast בויך boukh [boʊx] (54)"),
    ("ברידער", "bridɛr", "52 brider"),
    ("שטוב", "ʃtib", "52 shtib — v3 §10.2, no final devoicing"),  # v3
    # --- class 54: ou [oʊ] (MHG ū) ---
    ("הויז", "hoʊz", "54 hous — v3 §10.2, no final devoicing"),  # v3
    ("מויל", "moʊl", "54 moul"),
    ("טויזנט", "toʊznt", "54 touznt (§8 numbers)"),
    ("ארויס", "aroʊs", "54 arous"),

    # =================================================================
    # §4 POSTLEXICAL RULES
    # =================================================================
    # 4.1 final devoicing  b d ɡ v z ʒ -> p t k f s ʃ
    ("טאג", "tuɡ", "v3 §10.2: tuɡ, not tuk"),  # v3
    ("וועג", "vejɡ", "v3 §5 ej away from r + §10.2 no devoicing"),  # v3
    ("קינד", "kind", "v3 §10.2: kind is one of the spec's own named examples"),  # v3
    # 4.2 regressive voicing assimilation — the SECOND consonant wins
    ("זאגט", "zuɡt", "v3 §10.2: no devoicing-ward assimilation — zuɡt (gold primary)"),  # v3
    ("ליבט", "libt", "v3 §10.2: libt, not lipt"),  # v3
    ("שבתדיק", "ʃabɛzdik", "v3 §10.1: voicing-ward assimilation stays ON — s voiced before d"),  # v3
    ("דווקא", "davkə", "v3 §10.2: no devoicing-ward assimilation — davkə"),  # v3
    ("ריווקע", "rivkə", "v3 §10.2: Rivke stays rivkə"),  # v3
    ("חשוון", "xɛʒvn", "§8 frozen month name Chezhvn (lexical ʒ); v3 §1 syllabic -n"),  # v3
    ("יעקב", "jankɛv", "v3 §10.2 + gold primary jˈankəv; jankəf is the lexicalized variant"),  # v3
    # =================================================================
    # §5 LOSHN-KOYDESH NIKUD TABLE (pointed input)
    # =================================================================
    # CASE FIXED: §4.1 final devoicing applies to names too (the spec's own
    # example is Yankev -> Yankef), so the lexical Duvid surfaces as [duvit].
    ("דָּוִד", "duvid", "§5 kometz -> u + chirik -> i; v3 §10.2 keeps the final d"),  # v3
    ("בָּרוּךְ", "burix", "§5 kometz -> u, shuruk -> i (51/52); Burich"),
    ("בַּת", "bas", "§5 pasekh -> a (11); bas"),
    ("אֱמֶת", "ɛmɛs", "§5 segol / chatef-segol -> e (21); emes"),
    ("פֵּסַח", "pajsɛx", "§5 tsere -> ay (22/24); Paysech"),
    ("תּוֹרָה", "tɔjrə", "§5 cholam -> oy (42/44), casual final kometz-hey -> @ (§7.2)"),
    ("יוֹם", "jɔjm", "§5 cholam -> oy (42/44)"),
    ("שִׂמְחָה", "simxə", "§5 chirik -> i, sin dot, shva na deleted, final -e = @"),
    ("כְּלַל", "klal", "§5 shva na deleted + pasekh -> a (11)"),
    ("כִּי", "ki", "§5 dagesh kaf = k, chirik -> i"),
    ("רוּחַ", "riəx", "XFAIL §5 pasekh genuvah inserts @: riech; not implemented"),  # v3
    ("עולם", "ɔjlɛm", "§7.3 merged 'der oylem' (cholam oy); the WH reading is oylum [ɔɪlum]"),
    ("משה", "mɔjʃə", "§8 names — Moyshe (frozen form)"),

    # =================================================================
    # §8 HIGH-FREQUENCY IRREGULARS AND FUNCTION WORDS (hard-coded)
    # =================================================================
    ("אויף", "oʊf", "v3 §9 homograph: oʊf standalone, afn fused/reduced (gold primary oʊf)"),  # v3
    ("אויפן", "afn", "v3 §9: the fused token אויפן is afn"),  # v3
    ("צו", "ʦi", "§8 tsi 'to/too' (52)"),
    ("דו", "di", "§8 di 'you' (52) — merges toward the article"),
    ("וואו", "vi", "§8 vi 'where' (52) — merges with vi 'how'"),
    ("זיי", "zaj", "§8 zay 'they' (22/24)"),
    ("ניין", "naːn", "§8 naan 'no' (34) — vs ayn 'one' (24)"),
    ("אונדז", "indz", "§8 'us'; v3 §1 has no ʣ, so dz is two phones, and nothing devoices"),  # v3
    ("עץ", "ɛʦ", "§8 ets — 2pl nominative, hallmark Unterland form"),
    ("ער", "ɛr", "v3 §5/gold: ɛr is the default and the gold primary — no longer a divergence"),  # v3
    ("ווער", "vɛr", "v3 §5/gold: vɛr"),  # v3
    ("פיר", "fir", "§8 numbers — fir"),
    ("זעקס", "zɛks", "§8 numbers — zeks (21)"),
    ("אכט", "axt", "§8 numbers — akht (11)"),

    # =================================================================
    # AUDIT REGRESSIONS — one case per confirmed finding, 2026-08-06.
    # =================================================================
    # --- word-initial א/ע + double-vav lost its vowel entirely -----------
    ("אוועק", "avɛk", "audit: א before consonantal וו is a vowel — was 'vɛk' (§6.1 avék-)"),
    ("אוונט", "uvnt", "audit: 12/13 uvnt, unpointed Hasidic spelling — was 'vnt', no vowel at all"),
    ("אוונטן", "uvntn", "audit: uvntn; v3 §1 syllabic -n"),  # v3
    ("אוועקגיין", "avɛkɡajn", "audit: avek- reachable in compounds; §4.2 does not voice the k (see §4.2 block)"),
    ("עוף", "if", "audit: word-initial ע before a vav-nucleus is silent (§5) — was 'ɛif'"),

    # --- class 34 [aː] gaps: words listed in the spec's own §2 column ----
    ("ווייב", "vaːb", "34 vaab; v3 §10.2 no final devoicing"),  # v3
    ("בלייבן", "blaːbn", "34 blaabn; v3 §1 syllabic -n"),  # v3
    ("גלייבן", "ɡlaːbn", "34, sister of blaabn"),  # v3
    ("הייזער", "haːzɛr", "audit: 34 (YIVO pasekh-tsvey-yudn הײַזער)"),
    ("ווייבער", "vaːbɛr", "audit: 34, plural of vaab"),
    # -kaat / -haat (§9) is the same class and is productive, not lexical
    ("אידישקייט", "jidiʃkaːt", "§9 -kaat = class 34; v3 initial-yud gives jidiʃ- (gold jˈidiʃə)"),  # v3
    ("פרומקייט", "frimkaːt", "§9 frimkaat"),
    ("געזונטהייט", "ɡɛzinthaːt", "§9 gezinthaat — the -haat sister suffix"),
    # aráan- keeps class 34 inside compounds, not only as a bare word
    ("אריינקומען", "araːnkimɛn", "§6.1 aráankimen — was [araɪnkimən], [aː] only on the bare word"),
    ("אריינגיין", "araːnɡajn", "§6.1 aráangayn"),

    # --- ambiguous א: האב/האבן are class 41, like their paradigm mate האט ---
    ("האב", "hɔb", "class 41; v3 §10.2 no final devoicing (gold hub | hɔb)"),  # v3
    ("האבן", "hɔbn", "41 hobn; v3 §1 syllabic -n"),  # v3
    # --- class 25 across a paradigm, not per surface form ---------------
    ("קלערט", "klɛrt", "v3 §5: the class-25-before-r paradigm is retired — ɛr default"),  # v3
    ("קלערסט", "klɛrst", "v3 §5 ditto"),  # v3
    ("שטערנס", "ʃtɛrns", "v3 §5 ditto"),  # v3
    # --- §5 pointed Whole-Hebrew ----------------------------------------
    ("אֲבֵלִים", "avajlim", "§5 avaylim — bare bet with its own vowel point is /v/; was [abaɪlim]"),
    ("יִשְׂרָאֵל", "jisruɛl", "§5 chirik row Yisruel — initial yud is consonantal; was [israɪl]"),
    # probe: עולם removed from _WH_WHEN_POINTED — e2e probe showed 3/5 pointed
    # firings were merged-Yiddish contexts (עולם הבא, "the crowd"), so pointed
    # עוֹלָם now reads merged too. See data/verification/e2e_pointing.md.
    ("עוֹלָם", "ɔjlɛm", "pointed follows the merged lexicon (WH hatch kept only for מלך)"),
    # probe: pointed א before tsvey-yudn contributes no vowel of its own — the
    # ajaj doubled-diphthong fix (leave_one_out.md + e2e_pointing.md).
    ("אֵיין", "ajn", "over-pointed alef before ײ digraph: single aj, not *ajajn"),
    ("אֵייבֶּערְשְׁטֶער", "ajbɛrʃtɛr", "ajaj fix: same reading as bare אייבערשטער"),
    ("קִדּוּשׁ", "kidɛʃ", "§5 shuruk row kidesh (frozen; not derivable from the nikud table)"),
    ("חֻמָּשׁ", "ximɛʃ", "§5 kubuts row chimesh"),
    ("עֵדוּת", "ajdɛs", "§5 tsere row aydes"),  # v3
    # --- §4.2 mesivta, one of the spec's three named examples -----------
    ("מתיבתא", "mɛsiftə", "§4.2 mesivta -> mesifte — was [msipsa]"),
    ("מסיבתא", "mɛsiftə", "§4.2 variant spelling"),

    # --- §4.2 must not create geminates, and must not voice plosives ----
    ("גערעדט", "ɡɛrɛdt", "v3 §10.2: no devoicing, so dt survives (gold primary ɡərˈɛt is the lexical variant)"),  # v3
    ("רעדט", "rɛdt", "v3 §10.2 + gold primary rɛdt"),  # v3
    ("אפגעטון", "ɔpɡɛtin", "§6.1 óp- prefix; the p does not voice before ɡ — was [avɡɛtin]"),
    ("אפגעשטעלט", "ɔpɡɛʃtɛlt", "§6.1 óp-"),

    # --- REGRESSION GUARD: postlexical must not feed the affricate fuser -
    # §4.2 devoices d before ʃ, giving a legitimate t+ʃ cluster. Fusing that
    # into ʧ deletes a phoneme; only tie-bar forms may be fused.
    ("קודש", "kidʃ", "v3 §10.2: no devoicing, so d+ʃ; still never fused to ʤ"),  # v3
    ("חסידשע", "xusidʃə", "v3 §10.2: d+ʃ stays two phones"),  # v3
    ("דשמיא", "dʃmia", "v3 §10.2: d+ʃ stays two phones"),  # v3
    # --- §6.2 merged LK forms listed verbatim in the spec ---------------
    ("חסיד", "xusid", "§6.2 CHUsid; v3 §10.2 no final devoicing"),  # v3
    ("ספר", "sejfɛr", "§6.2 SEYfer — class 25 [ej], was [saɪfɛr]"),
    ("תענית", "tunɛs", "§7.3 merged tunes — was [sɛnis]"),
    ("שידוך", "ʃidɛx", "§6.2 SHIdech — post-tonic ו reduces, was [ʃidix]"),
    ("קידוש", "kidɛʃ", "§6.2 KIdesh — was [kidiʃ]"),
    ("בוחר", "buxɛr", "§6.2 BUcher — was [bixr], no epenthetic vowel"),
    ("בחורים", "baxirɛm", "§6.2 baCHIrem — was [pxirim]"),

    # --- §7.5 abbreviations: ʔ is not in the phone set ------------------
    ("ה'", "haʃɛm", "§7.5 ה' -> Hashem — was [hʔ]"),
    ("ר׳", "rɛb", "§8 ר' -> rɛb (v3 abbreviation table); no final devoicing"),  # v3
    ("זיין'", "zaːn", "audit: a word-edge geresh is dropped, never phonemized as ʔ"),

    # =================================================================
    # AUDIT REGRESSIONS — three-auditor round, 2026-08-07
    # =================================================================
    # --- §5 digraph table: שפ -> ʃp, "after ש always p" (§4) -------------
    ("שפילן", "ʃpiln", "§5 digraph שפ -> ʃp — was ʃfiln"),
    ("שפראך", "ʃprax", "§5 שפ -> ʃp — was ʃfrax"),
    ("שפיץ", "ʃpiʦ", "§5 שפ -> ʃp — was ʃfiʦ"),
    ("שפעט", "ʃpɛt", "§5 שפ -> ʃp — was ʃfɛt"),
    ("אשפיז", "aʃpiz", "§5 שפ -> ʃp mid-word — was ˈaʃfiz"),
    ("שפֿילט", "ʃfilt", "an EXPLICIT rafe still wins over the שפ digraph"),
    # --- §5 suffix spelling ־ליך -> ləx ----------------------------------
    ("ערליך", "ɛrlɛx", "§5 ־ליך -> lekh (ə only after the stress stage) — was ˈɛrlix"),
    ("הערליכע", "hɛrlɛxə", "§5 ־ליך inflected — was hˈɛrlixə; gold has hˈɛrləx"),
    ("ליכט", "lixt", "NOT the suffix: ליכט keeps lix"),
    # --- word-final ה is never [h]: the LK feminine ending is ə ----------
    ("מדינה", "mdinə", "audit: unpointed LK feminine -ה -> ə, was mdinh"),
    # --- the ה/digraph collision: two phonemes, not one merged one -------
    ("מזוזה", "mzizə", "audit: ז+ה no longer fuses to ʒ (was mziʒ)"),
    ("עצה", "ɛʦə", "audit: צ+ה no longer fuses to ʧ (was ɛʧ)"),
]


# =====================================================================
# §6 STRESS  (hebrew_to_ipa(..., stress=True); unstressed ɛ reduces to ə)
# =====================================================================
STRESS_CASES: list[tuple[str, str, str]] = [
    ("שבת", "ʃˈabəs", "§6.2 SHAbes — merged LK retracts to the penult, rest reduces to @"),
    ("תורה", "tˈɔjrə", "§6.2 TOYre"),
    ("חתונה", "xˈasənə", "§6.2 CHAsene"),
    ("ישיבה", "jəʃˈivə", "§6.2 YEshive"),
    ("אונטערגיין", "ˈintərɡajn", "§6.1 ún(n)ter- separable prefix IS stressed"),
    # CASE FIXED: §4.2 regressive voicing applies at the prefix seam too
    # (same juncture type as shabes+dik -> shabezdik), so s -> z before ɡ.
    # CASE FIXED (2): the stress sits on the prefix's SECOND nucleus. §6.1
    # writes aróus-, and the bare word already came out arˈoʊs, so initial
    # stress here contradicted the engine's own reading of the same morpheme.
    ("ארויסגיין", "arˈoʊzɡajn", "§6.1 aróusgayn (54) + §4.2 s->z before ɡ"),
    ("געקומען", "ɡəkˈimən", "§6.1 ge- is an unstressed prefix; ge- reduces to [ɡə] (§9 participle)"),
    ("געזאגט", "ɡəzˈuɡt", "§11.3 ge- unstressed; v3 §10.2 keeps ɡt (gold zuɡt)"),  # v3
    ("אזוי", "azˈɔj", "§6.1 a-ZOY — unstressed initial a-"),
    ("מלך", "mˈajləx", "§6.2 MAYlech — LK lexicon (the rule path alone gives the vowel-less mlx)"),  # v3
    ("געווען", "ɡəvˈejn", "§6.1 geveyn (25) — ge- unstressed"),
    ("נעבעך", "nˈɛbəx", "§10 Slavic component: fixed penult stress, NEbech"),

    # --- audit regressions, stress side ---------------------------------
    ("אוועק", "avˈɛk", "§6.1 avék (bare adverb)"),
    ("אוועקגעלייגט", "avˈɛkɡəlajɡt", "v3 §11.4: directional a(r)- stresses the SECOND nucleus, in compounds too"),  # v3
    ("אריינקומען", "arˈaːnkimən", "§6.1 aráankimen — prefix stress on its SECOND nucleus"),
    ("צוריקגעקומען", "ʦirˈikɡəkimən", "§6.1 tsurík-gekimen — ditto"),
    ("בעל-הבית", "bˈaləbus", "§6.2 BAlebus — listed with SHAbes/TOYre/CHAsene; was ba-le-BUS"),
    ("בעלי-בתים", "baləbˈatim", "§6.2 plural stress shift baleBAtim"),
    ("חסיד", "xˈusid", "§6.2 CHUsid; v3 §10.2"),  # v3
    ("גייען", "ɡˈajən", "§6.1 root-initial: ge- is not a prefix here, it is g|ey|en"),
    ("לייענט", "lˈajənt", "§6.1 root-initial: -ent is not the loan suffix here"),
    ("זייען", "zˈajən", "§6.1 sister form that was already right — guard against a fix that breaks it"),
    ("פרעזידענט", "prəzidˈɛnt", "the real -ent tonic suffix still applies"),

    # --- audit round 2026-08-07 -----------------------------------------
    # §11.4 directional a(r)-, the two families that were still initial-stressed
    ("אראפגעקומען", "arˈupɡəkimən", "§11.4 + §4: אראפ- is arˈup-, and its פ is /p/ (was ˈaravɡəkimən)"),
    ("אראפנעמען", "arˈupnəmən", "§11.4 אראפ- (was ˈarafnəmən)"),
    ("אראפגעפאלן", "arˈupɡəfaln", "§11.4 אראפ- (was ˈaravɡəfaln)"),
    ("ארויפגיין", "arˈoʊfɡajn", "§11.4 ארויפ- ; §10.1 does not voice the f before ɡ (was ˈaroʊvɡajn)"),
    ("ארויפלייגן", "arˈoʊflajɡn", "§11.4 ארויפ-"),
    ("ארויפגעלייגט", "arˈoʊfɡəlajɡt", "§11.4 ארויפ- with ge-"),
    ("ארונטער", "arˈintər", "§11.4 applies to the BARE directional adverb too"),
    ("אדורך", "adˈirx", "§11.4 bare adverb"),
    # §5 ־ליך suffix, stressed side
    ("ערליך", "ˈɛrləx", "§5 ־ליך -> ləx"),
    ("הערליכע", "hˈɛrləxə", "§5 ־ליך; gold הערליך = hˈɛrləx"),
    ("שרעקליכע", "ʃrˈɛkləxə", "§5 ־ליך"),
    ("נעמליך", "nˈɛmləx", "§5 ־ליך"),
    # §2.1 the lookup key is point-stripped: pointing must not disable a lexicon
    ("מאָרגן", "mɔrɡn", "§2.1 pointed spelling keeps the §4 class-41 pinning (was murɡn)"),
    ("גאָט", "ɡɔt", "§2.1 (was ɡut)"),
    ("אָפֿט", "ɔft", "§2.1 (was uft)"),
    ("דאָלאַר", "dˈɔlar", "§2.1 (was dˈular)"),
    # §1 shape gate must not eat ordinary Germanic four-consonant runs
    ("אייבערשטן", "ˈajbərʃtn", "§1: r-ʃ-t-n is an ordinary Yiddish run, not a quarantine"),
    ("דארפסטו", "dˈarfsti", "§1: r-f-s-t"),
    # pˈinktləx: the f was the §4 default when this case was authored; the
    # xeus pe-sweep heard p 9-0 across clips (data/audio_pe_lk.py), and audio
    # outranks an author guess. The §1 cluster the case exists for is unchanged.
    ("פינקטלעך", "pˈinktləx", "§1: n-k-t-l"),
    # §6.1 merged-LK entries added from the OOV-LK log
    ("שם", "ʃɛm", "§6.3: was the forbidden vowel-less ʃm"),
    ("מדרש", "mˈɛdrəʃ", "§6.3: was mdrʃ"),
    ("שלמה", "ʃlˈɔjmə", "§6.3: was ʃlmh"),
    ("לברכה", "livrˈuxə", "§6.3: was lbrxh"),
    ("חלילה", "xalˈilə", "§6.3: was xlilh"),
    ("עבודה", "avˈɔjdə", "audit: was the illegal word-final /h/ ˈɛbidh"),
    ("אמונה", "əmˈinə", "audit: was ˈaminh"),
    ("שירה", "ʃˈirə", "audit: was ʃirh"),
    # sefaria — the reading of a verified pointing now reaches the emitted
    # string for words that used to be quarantined outright. Which REGISTER the
    # pointing is read in is decided per type by scripts/register_policy.py:
    # merged-register by default (the word is embedded in a Yiddish sentence),
    # Whole-Hebrew where the word is quoted or where the merged reader misreads
    # the pointing. Both readings ship; the loser is a variant.
    ("כזית", "kazˈajis",
     "sefaria: כַּזַּיִת — WH kept: merged kˈazis loses the yud's own chirik"),
    ("כהן", "kˈɔjhajn", "sefaria: כֹּהֵן — dagesh kaf [k], tsere [aj]"),
    # merged-register: shuruk takes the Yiddish u->i shift (was WH zəxˈus) and
    # the shva-na goes, as in gold brˈuxə. The audio agrees — zkhis, not zkhus.
    ("זכות", "zxis", "sefaria: זְכוּת — merged-register, audio-arbitrated"),
    # merged-register: shva-na dropped, b'sheym. WH bəʃˈajm is now the variant.
    ("בשם", "bʃajm", "sefaria: בְּשֵׁם — merged-register, audio-arbitrated"),
    # homograph rescue (data/homograph_lk.py), between the audio and sefaria
    # tables. 'homograph-collapsed': EVERY attested pointing READS the same, so
    # no audio verdict was needed to emit the word.
    ("עגלות", "aɡˈulɔjs", "collapsed: עֲגָלוֹת — pointings differ, reading does not"),
    # 'audio-homograph': rivals that really do sound different, decided per
    # occurrence against episode audio.
    ("בשלח", "bəʃˈalax", "audio: בְּשַׁלַּח the parsha — NOT biʃlˈɔjxa 'in sending'"),
    # no-drop policy (2026-08-08): words the evidence chain cannot settle are no
    # longer withheld — phonikud-yi v3 guesses them in context at LOW confidence
    # (reason 'model-pointed-guess'). The guess is never the raw consonant
    # skeleton: חתם is xˈasam (not xˈɔjsum, not xsm), דבר is dˈuvur (not dbr).
    ("דער הייליגער חתם סופר", "dɛr hˈajliɡər xˈasam sifr",
     "no-drop: חתם guessed xˈasam by phonikud-v3"),
    ("השלך על ה'", "hˈaʃlajx al haʃˈɛm",
     "no-drop: השלך guessed hˈaʃlajx in context"),
    ("דער מלך האט געזאגט אז דבר איז", "dɛr mˈajləx hut ɡəzˈuɡt az dˈuvur iz",
     "no-drop: דבר guessed dˈuvur, never the skeleton dbr"),
    ("רופט אן 845-554-0338", "rift un",
     "§1(b): an out-of-inventory token is withheld, digits never reach output"),

    # --- §7.5 letter-name fallback for UNKNOWN abbreviations ------------
    # An acronym the table does not know is spelled out, the way a reader
    # spells out one he does not recognize; it is no longer quarantined and
    # never becomes an invented word (תשפ"ה was 'sʃpə').
    ("תשפ\"ה", "tuf ʃin paj haj",
     "§7.5 unknown gematria year -> letter names, not a fake word"),
    ("כ\"ה", "xuf haj",
     "§7.5 an unrecognized date is spelled out, not turned into a word"),
    # --- §7.5a acronyms pronounced as WORDS -----------------------------
    # The fallback must NOT fire on the established acronyms: they are never
    # spelled out in speech, and the corpus's single most frequent gershayim
    # token (רש"י, 1157 occurrences) is one of them.
    ("רש\"י", "rˈaʃi", "§7.5a word-pronounced acronym, not 'rajʃ ʃin jid'"),
    ("חז\"ל", "xazˈal", "§7.5a word-pronounced acronym"),
    ("רמב\"ם", "rˈambam", "§7.5a final ם folds in the key, reading is a word"),
    ("של\"ה", "ʃlu", "§7.5a word-pronounced acronym"),
    ("תרי\"ג", "tarjˈaɡ", "§7.5a word-pronounced acronym"),
    # --- §7.5b tokens that spell a single LETTER'S NAME -----------------
    ("מ\"ם", "mɛm", "§7.5b the letter mem's name, not 'mɛm mɛm'"),
    ("כ\"ף", "xuf", "§7.5b the letter khof's name, not 'xuf paj'"),
    ("יו\"ד", "jid", "§7.5b the letter yud's name, not 'jid vuv dˈuləd'"),
    # The known entries keep their real readings: the fallback fires only on a
    # miss, never over the abbreviation table.
    ("שליט\"א", "ʃlˈitə", "§8 a KNOWN abbreviation is untouched by the §7.5 fallback"),

    # --- §8 lexicalized multiword LK (scripts/mine_lk_mwe.py) ------------
    # The phrase, not the word, is the unit: per-word routing gives a reading
    # nobody says (baːl habˈajis, ʃˈabəs kidʃ). Merged register throughout.
    ("בעל הבית", "bˈaləbus",
     "§8 spaced spelling reads like the hyphenated בעל-הבית (§6.2 BAlebus)"),
    ("בעלי בתים", "baləbˈatim", "§8 spaced plural = בעלי-בתים, baleBAtim"),
    ("שבת קודש", "ʃˈabəs kˈɔjdəʃ",
     "§8 shabes KOYdesh — קודש alone rules out to the vowel-poor kidʃ"),
    ("זכרונו לברכה", "zixrˈɔjni livrˈuxə",
     "§8 the corpus's most frequent LK phrase (847x); was zxrˈini"),
    ("ריבונו של עולם", "ribˈɔjnə ʃɛl ˈɔjləm",
     "§8 three-token match; §5 merged final kometz-hey -> ə"),
    ("יום טוב", "jˈɔntəv", "§8 spaced spelling of the LK compound יום-טובֿ"),
    ("ראש השנה", "rˈuʃəʃunə", "§8 spaced spelling of the LK compound ראש-השנה"),
    ("אם ירצה השם", "im jˈirʦə haʃˈɛm",
     "§8 phrase-initial אם is im, not the alef-default am"),
    ("בית דין", "bˈɛs din", "§8 LK בעסדין; בית on its own stays bajs"),
    ("ראש חודש", "rɔjʃ xˈɔjdəʃ",
     "§8 an entry may confirm the per-word reading — it pins it as HIGH"),
]


# =====================================================================
# §12 ROUTE / CONFIDENCE HONESTY  (g2p_token, not the phone string)
#
# §3.4 routes an LK-detected token to the LK path and §6.3 says a word with no
# lexicon and no pointing must not be emitted, so 'rule' is the one answer that
# record may not carry. The mirror-image defect matters just as much: the §3
# shape heuristic used to fire on ordinary Germanic words (פלעגט, שטיקל, זינגט,
# דאָרטן, גאנצן), tagging correct output LOW and stuffing the OOV-LK
# verification batch with 19k types, most of which needed no verification.
# =====================================================================
ROUTE_CASES: list[tuple[str, str, str, str]] = [
    # (word, expected route, expected confidence, note)
    # audio-endorsed rescue (data/audio_endorsed_lk.py): previously quarantined
    # LK whose unverified pointing PhoneticXeus confirmed against episode audio.
    ("צדקה", "rule", "LOW", "pointed-audio-endorsed rescue -> ʦdˈukə emitted at LOW"),
    ("בחינה", "rule", "LOW", "pointed-audio-endorsed rescue -> bxˈinə"),
    # sefaria — rescue #2 (data/sefaria_pointed_lk.py), consulted only after the
    # audio table misses: a single agreed pointing in the verified editions,
    # read in the register that type is used in. Still LOW, 'sefaria-pointed' —
    # the register decision does not make a book pointing a native verdict.
    ("כזית", "rule", "LOW", "sefaria: כַּזַּיִת, 92.6% of 81 sources -> kazˈajis"),
    ("כהן", "rule", "LOW", "sefaria: כֹּהֵן -> kˈɔjhajn (was quarantined as khn)"),
    ("זכות", "rule", "LOW", "sefaria: זְכוּת -> zxis, merged-register (was zəxˈus)"),
    ("בשם", "rule", "LOW", "sefaria: בְּשֵׁם -> bʃajm, merged-register"),
    # homograph rescue #1.5 (data/homograph_lk.py), consulted between the audio
    # and sefaria tables. Collapsed types are free — one reading, several
    # printings — but still LOW: the collapse is an inference about editions.
    ("עגלות", "rule", "LOW", "homograph-collapsed: עֲגָלוֹת -> aɡˈulɔjs (was withheld)"),
    ("בשלח", "rule", "LOW", "audio-homograph: בְּשַׁלַּח -> bəʃˈalax, 86% of 7 clips"),
    # A type whose rivals were merely thin, or whose audio never separated them,
    # stays quarantined — the collapsed branch must not launder either into a
    # free single-reading rescue.
    ("חתם", "rule", "LOW", "no-drop: model-pointed-guess xˈasam"),
    ("הלבבות", "rule", "LOW", "no-drop: model-pointed-guess halvˈuvɔjs"),
    ("דבר", "rule", "LOW", "no-drop: model-pointed-guess dˈuvur"),
    ("מקבל", "rule", "LOW", "sefaria: מְקַבֵּל -> məkˈabajl (was §6.3-withheld mkbl)"),
    ("תהילים", "rule", "LOW", "no-drop: model-pointed-guess təhˈilim"),
    ("חידושים", "rule", "LOW", "no-drop: model-pointed-guess xidˈuʃim"),
    ("מחלוקת", "rule", "LOW", "no-drop: model-pointed-guess maxalˈɔjkɛs (was quarantined mxliks)"),
    ("שם", "lexicon", "HIGH", "§6.1 now covers it, so it is emitted"),
    ("חסידישע", "rule", "LOW", "§6.1 _STEM_SUBS is lexical evidence -> still emitted"),
    ("פלעגט", "rule", "LOW", "§3 must not call this LK (LOW here is only the פ default)"),
    ("שטיקל", "rule", "MED", "§3 shape heuristic must not fire on a Germanic word"),
    ("זינגט", "rule", "MED", "ditto — was conf=LOW, reason=lk-fallback"),
    ("דאָרטן", "lexicon", "HIGH", "ditto, and §2.1 keeps the pointed form on the lexicon"),
    # §7.5: the letter-name fallback is a rule answer, not a quarantined one,
    # but it stays LOW — spelling out is a guess about how the reader copes.
    ("תשפ\"ה", "rule", "LOW", "§7.5 unknown abbreviation -> letter-names, un-quarantined"),
    # §7.5a: the reading comes from a table (route 'lexicon') but it is
    # editorial, not audio-verified — the segmental shape is settled usage, the
    # stress is a judgement — so it stays LOW and in the verification queue.
    ("רש\"י", "lexicon", "LOW", "§7.5a word-pronounced acronym: tabled but LOW"),
    ("מ\"ם", "rule", "LOW", "§7.5b letter-name token, derived like the fallback"),
    ("שליט\"א", "lexicon", "HIGH", "§8 known abbreviation still routes to the table"),
]


# §8 multiword records: the phrase must come back as ONE lexicon record over the
# token stream (g2p_tokens), i.e. the multiword match has to fire BEFORE
# per-word routing. Confidence is the entry's provenance, not the mechanism's:
# the corpus-mined entries are unverified readings and ship LOW ('mwe-mined'),
# so they stay in the verification queue like every other unverified rescue;
# only the hand-verified בית מדרש is HIGH.
MWE_RECORD_CASES: list[tuple[str, int, str, str]] = [
    ("דער בעל הבית", 2, "בעל הבית", "LOW"),
    ("אויף שבת קודש", 2, "שבת קודש", "LOW"),
    ("א גוטן יום טוב", 3, "יום טוב", "LOW"),
    ("דער ריבונו של עולם", 2, "ריבונו של עולם", "LOW"),
    ("אין בית מדרש", 2, "בית מדרש", "HIGH"),
]

# §2.3: the space/makef choice is orthography, not phonology. Every multiword
# entry must read the same spaced, hyphenated and with a makef.
MWE_SPELLING_CASES: list[str] = [
    "יום טוב", "בית דין", "לשון הרע", "עולם הבא", "זכרונו לברכה", "עולם הזה",
    "ימים טובים", "שבע ברכות", "בית מדרש", "ראש השנה", "בעל הבית",
]

# Punctuation between the members blocks the fusion: a fixed collocation is a
# contiguous span, and fusing across a full stop deletes it and merges two
# sentences into one word.
MWE_PUNCT_CASES: list[tuple[str, str]] = [
    ("ער איז א בעל. הבית איז גוט", "ɛr iz a baːl. habˈajis iz ɡit"),
    ("דער יום, טוב איז גוט", "dɛr jɔjm, tɔjv iz ɡit"),
]


def main() -> int:
    passed = failed = xfailed = fixed = 0
    all_cases = CASES + [(t, w, "stress: " + n) for t, w, n in STRESS_CASES]
    for text, want, note in all_cases:
        stress = note.startswith("stress:")
        got = hebrew_to_ipa(text, stress=stress)
        xfail = "XFAIL" in note
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
        print(f"{status}  {text!r:22s} -> {got!r:16s} (want {want!r:16s})  {note}")
    for word, route, conf, note in ROUTE_CASES:
        rec = g2p_token(word)
        ok = rec["route"] == route and rec["confidence"] == conf
        passed += ok
        failed += not ok
        print(f"{'PASS ' if ok else 'FAIL '}  {word!r:22s} -> "
              f"{rec['route']}/{rec['confidence']:4s} "
              f"(want {route}/{conf})  {note}")
    for text, n_records, phrase, conf in MWE_RECORD_CASES:
        recs = g2p_tokens(text)
        hit = [r for r in recs if r["word"] == phrase]
        ok = (len(recs) == n_records and len(hit) == 1
              and hit[0]["route"] == "lexicon" and hit[0]["confidence"] == conf)
        passed += ok
        failed += not ok
        got = hit[0] if hit else {"ipa_primary": "-", "route": "-",
                                  "confidence": "-"}
        print(f"{'PASS ' if ok else 'FAIL '}  {text!r:22s} -> "
              f"{len(recs)} records, {phrase!r} = {got['ipa_primary']!r} "
              f"{got['route']}/{got['confidence']}  §8 multiword fires before "
              f"per-word routing")
    for phrase in MWE_SPELLING_CASES:
        spaced = hebrew_to_ipa(phrase)
        hyphen = hebrew_to_ipa(phrase.replace(" ", "-"))
        makef = hebrew_to_ipa(phrase.replace(" ", "\u05be"))
        ok = spaced == hyphen == makef
        passed += ok
        failed += not ok
        print(f"{'PASS ' if ok else 'FAIL '}  {phrase!r:22s} -> "
              f"{spaced!r} / {hyphen!r} / {makef!r}  §2.3 space == hyphen == makef")
    for text, want in MWE_PUNCT_CASES:
        got = hebrew_to_ipa(text)
        ok = got == want
        passed += ok
        failed += not ok
        print(f"{'PASS ' if ok else 'FAIL '}  {text!r:22s} -> {got!r} "
              f"(want {want!r})  §8 punctuation blocks multiword fusion")
    total = (len(all_cases) + len(ROUTE_CASES) + len(MWE_RECORD_CASES)
             + len(MWE_SPELLING_CASES) + len(MWE_PUNCT_CASES))
    print(
        f"\n{total} spec cases: {passed} passed, {failed} FAILED, "
        f"{xfailed} known divergences, {fixed} newly fixed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
