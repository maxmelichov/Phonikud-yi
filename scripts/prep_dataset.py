"""Segment the yiddish24 audio on the Yiddish Labs phrase timestamps and emit
training manifests for Parakeet (NeMo ASR), Qwen3-TTS, and a generic TTS
recipe (LJSpeech-style metadata).

Usage:  python scripts/prep_dataset.py <yiddish24 folder> [out dir]

Layout produced here:
  clips/16k/<clip_id>.wav     16 kHz mono PCM16   (ASR)
  clips/24k/<clip_id>.wav     24 kHz mono PCM16   (TTS)
  clips.tsv                   master table, one row per clip, all flags
  parakeet_{train,dev,test}.jsonl     audio_filepath / duration / text
  qwen3tts_{train,dev,test}.jsonl     audio / text(IPA) / orth_text / speaker / duration
  tts_{train,dev,test}.csv            clip_id|orth_text|ipa   (LJSpeech-style, pipe separated)

No numpy: each source file is decoded once by ffmpeg to raw PCM and sliced by
byte offset with the stdlib wave module.
"""
import os, re, sys, csv, json, glob, wave, random, subprocess, hashlib
from collections import Counter, defaultdict

# python scripts/prep_dataset.py <yiddish24 folder with *.mp3 + *.yl.json> [out dir]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "..", "yiddish24")
HERE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "data", "dataset_out")
LEX = os.path.join(REPO, "data", "lexicon_out")
sys.path.insert(0, REPO)
import yiddish_g2p as g

# ---------------- knobs ----------------
TTS_MIN, TTS_MAX = 1.0, 15.0      # seconds
ASR_MIN, ASR_MAX = 0.5, 20.0
TARGET = 10.0                     # greedy merge target length
TTS_MAX_LOW_SHARE = 0.30          # share of tokens read at LOW confidence
END_PAD = 0.05                    # seconds kept after the next phrase starts
TEST_CATS = {"237", "236", "276"}        # held-out speakers (shiurim) -> test
DEV_CATS = {"229", "167"}                # held-out speaker -> dev
BULLETIN_SPLIT = (0.90, 0.05)     # bulletin_57 is one studio voice: split by item
random.seed(20260903)

# ---------------- text ----------------
TS = re.compile(r"<\|(\d+):(\d+(?:\.\d+)?)\|>")
MARK = re.compile(r"⟦#\d+⟧")
HEB = re.compile(r"[א-ת]")
TOK = re.compile(r"[א-ת]+(?:['\"״׳][א-ת]+)*|[^\sא-ת]+")

def clean(t):
    t = t.replace("״", '"').replace("׳", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t

def phrases(text, duration):
    """[(start, end, text)] from the timestamped transcript."""
    text = MARK.sub("\n", text)
    out = []
    pos = [(m.start(), m.end(), int(m.group(1)) * 60 + float(m.group(2))) for m in TS.finditer(text)]
    for i, (s, e, t0) in enumerate(pos):
        nxt = pos[i + 1][0] if i + 1 < len(pos) else len(text)
        seg = text[e:nxt]
        para_break = "\n" in seg.rstrip("\n ") or seg.endswith("\n")
        t1 = pos[i + 1][2] if i + 1 < len(pos) else duration
        txt = clean(seg)
        if txt and HEB.search(txt):
            out.append([t0, min(t1 + END_PAD, duration), txt, "\n\n" in seg])
    return out

def merge(phr):
    """Greedy merge of consecutive phrases up to TARGET seconds; never across a paragraph break."""
    clips, cur = [], None
    for t0, t1, txt, brk in phr:
        if cur and (t1 - cur[0] <= TARGET) and not cur[3]:
            cur[1] = t1; cur[2] += " " + txt; cur[3] = brk
        else:
            if cur: clips.append(cur)
            cur = [t0, t1, txt, brk]
    if cur: clips.append(cur)
    return clips

# ---------------- IPA ----------------
lex = {}
with open(os.path.join(LEX, "lexicon_merged.tsv"), encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        lex[r["word"]] = (r["ipa"], r["confidence"])
_cache = {}
def word_ipa(w):
    if w in lex: return lex[w]
    if w not in _cache:
        try:
            r = g.g2p_token(w); _cache[w] = (r["ipa_primary"], r["confidence"])
        except Exception:
            _cache[w] = ("", "LOW")
    return _cache[w]

def to_ipa(text):
    out, n, low = [], 0, 0
    for tok in TOK.findall(text):
        if HEB.search(tok):
            ipa, conf = word_ipa(tok); n += 1; low += conf == "LOW"
            out.append(ipa or tok)
        elif tok in ",.;:!?": out.append(tok)
    return " ".join(out).replace(" ,", ",").replace(" .", "."), (low / n if n else 1.0)

# ---------------- audio ----------------
def decode(path, sr):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                       capture_output=True, check=True)
    return p.stdout

def write_wav(path, pcm, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)

def slice_pcm(pcm, sr, t0, t1):
    a = int(t0 * sr) * 2; b = int(t1 * sr) * 2
    return pcm[a:b]

# ---------------- main ----------------
def main():
    for d in ("clips/16k", "clips/24k"): os.makedirs(os.path.join(HERE, d), exist_ok=True)
    rows = []
    files = sorted(glob.glob(os.path.join(AUDIO, "*", "*.yl.json")))
    print(len(files), "transcribed files")
    for jf in files:
        mp3 = jf.replace(".yl.json", ".mp3")
        if not os.path.exists(mp3): continue
        d = json.load(open(jf, encoding="utf-8")); d = d.get("data", d)
        folder = os.path.basename(os.path.dirname(jf))
        base = os.path.basename(mp3)[:-4]
        cat, item = base.split("_")[0], base.split("_")[1]
        duration = float(d.get("duration_seconds") or 0)
        clips = merge(phrases(d["text"], duration))
        if not clips: continue
        pcm16 = decode(mp3, 16000); pcm24 = decode(mp3, 24000)
        real_dur = len(pcm16) / 32000
        for k, (t0, t1, txt, _) in enumerate(clips):
            t1 = min(t1, real_dur)
            if t1 - t0 <= 0.2: continue
            cid = f"{cat}_{item}_{k:04d}"
            write_wav(os.path.join(HERE, "clips/16k", cid + ".wav"), slice_pcm(pcm16, 16000, t0, t1), 16000)
            write_wav(os.path.join(HERE, "clips/24k", cid + ".wav"), slice_pcm(pcm24, 24000, t0, t1), 24000)
            ipa, low_share = to_ipa(txt)
            rows.append(dict(clip_id=cid, folder=folder, cat=cat, item=item, source=base, start=round(t0, 2),
                             end=round(t1, 2), duration=round(t1 - t0, 2), text=txt, ipa=ipa,
                             low_share=round(low_share, 3), has_digits=bool(re.search(r"\d", txt)),
                             has_latin=bool(re.search(r"[A-Za-z]", txt)), n_words=sum(1 for t in TOK.findall(txt) if HEB.search(t))))
        print(f"{base[:45]:45s} {len(clips):5d} clips")

    # splits
    bulletin_items = sorted({r["item"] for r in rows if r["cat"] == "57"})
    random.shuffle(bulletin_items)
    n = len(bulletin_items); ntr = int(n * BULLETIN_SPLIT[0]); ndv = int(n * BULLETIN_SPLIT[1])
    b_split = {it: ("train" if i < ntr else "dev" if i < ntr + ndv else "test") for i, it in enumerate(bulletin_items)}
    for r in rows:
        if r["cat"] == "57": r["split"] = b_split[r["item"]]
        elif r["cat"] in TEST_CATS: r["split"] = "test"
        elif r["cat"] in DEV_CATS: r["split"] = "dev"
        else: r["split"] = "train"
        ok_txt = not r["has_digits"] and not r["has_latin"]
        r["asr_ok"] = ok_txt and ASR_MIN <= r["duration"] <= ASR_MAX
        r["tts_ok"] = ok_txt and TTS_MIN <= r["duration"] <= TTS_MAX and r["low_share"] <= TTS_MAX_LOW_SHARE

    cols = ["clip_id", "split", "folder", "cat", "item", "source", "start", "end", "duration", "n_words",
            "text", "ipa", "low_share", "has_digits", "has_latin", "asr_ok", "tts_ok"]
    with open(os.path.join(HERE, "clips.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    for split in ("train", "dev", "test"):
        rs = [r for r in rows if r["split"] == split]
        with open(os.path.join(HERE, f"parakeet_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                if r["asr_ok"]:
                    f.write(json.dumps({"audio_filepath": f"clips/16k/{r['clip_id']}.wav", "duration": r["duration"],
                                        "text": r["text"]}, ensure_ascii=False) + "\n")
        with open(os.path.join(HERE, f"qwen3tts_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                if r["tts_ok"]:
                    f.write(json.dumps({"audio": f"clips/24k/{r['clip_id']}.wav", "text": r["ipa"], "orth_text": r["text"],
                                        "speaker": f"cat{r['cat']}", "duration": r["duration"]}, ensure_ascii=False) + "\n")
        with open(os.path.join(HERE, f"tts_{split}.csv"), "w", encoding="utf-8") as f:
            for r in rs:
                if r["tts_ok"]:
                    f.write(f"{r['clip_id']}|{r['text']}|{r['ipa']}\n")

    tot = Counter(); dur = Counter()
    for r in rows:
        for k in ("asr_ok", "tts_ok"):
            if r[k]: tot[(k, r["split"])] += 1; dur[(k, r["split"])] += r["duration"]
    print(f"\n{len(rows)} clips, {sum(r['duration'] for r in rows)/3600:.1f} h total")
    for k in ("asr_ok", "tts_ok"):
        for s in ("train", "dev", "test"):
            print(f"  {k} {s:5s} {tot[(k,s)]:6d} clips {dur[(k,s)]/3600:5.2f} h")
    print("dropped (digits/latin):", sum(1 for r in rows if r["has_digits"] or r["has_latin"]),
          " over-length for TTS:", sum(1 for r in rows if r["duration"] > TTS_MAX),
          " low_share too high:", sum(1 for r in rows if r["low_share"] > TTS_MAX_LOW_SHARE))

if __name__ == "__main__":
    main()
