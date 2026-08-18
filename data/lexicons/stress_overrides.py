# Harvested from audio (Gemini Flash judgments, conf>=0.8, >=2 agreeing
# occurrences in the 70% harvest split). bare Hebrew word -> nucleus index.
_STRESS_OVERRIDE: dict[str, int] = {
    "אזא": 1,  # ˈaza -> stress 'a-za'[1]
    "אזוי": 1,  # ˈazɔɪ -> stress 'a-zɔɪ'[1]
    "אזעלכע": 1,  # ˈazɛlxə -> stress 'a-zɛl-xɛ'[1]
    "אינטערעסאנט": 3,  # ˈintɛrɛsant -> stress 'in-tɛ-rɛ-sant'[3]
    "אינטערעסאנטע": 3,  # ˈintɛrɛsantə -> stress 'in-tɛ-rɛ-san-tɛ'[3]
    "אמאל": 1,  # ˈamul -> stress 'a-mul'[1]
    "אראפ": 1,  # ˈarup -> stress 'a-rup'[1]
    "אראפגעקומען": 1,  # ˈarafɡɛkimɛn -> stress 'a-raf-ɡɛ-ki-mɛn'[1]
    "ארויס": 1,  # ˈarɔɪs -> stress 'a-rɔɪs'[1]
    "אריין": 1,  # ˈaraːn -> stress 'a-raːn'[1]
    "טויערע": 0,  # tɔɪˈɛrə -> stress 'tu-jɛ-rɛ'[0]
    "ישראל": 1,  # ˈiʃral -> stress 'i-ʃral'[1]
    "כאניקע": 0,  # xanˈikə -> stress 'xa-ni-kɛ'[0]
    "כביכול": 1,  # xbˈixil -> stress 'xbi-xil'[1]
    "פראבלעם": 1,  # prˈublɛm -> stress 'pru-blɛm'[1]
    "צוריק": 1,  # ʦˈirik -> stress 'ʦi-rik'[1]
    "קימאט": 1,  # kˈimat -> stress 'ki-mat'[1]
}
