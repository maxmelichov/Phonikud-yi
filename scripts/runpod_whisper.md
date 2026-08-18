# RunPod + faster-whisper large-v3 on Hasidic Yiddish — experiment log

**Date:** 2026-07-28 · **Verdict: NOT viable as a bulk transcription layer.** ~6-7% exact
word overlap with the Gemini reference. Do not scale to the 264-episode corpus.

---

## 1. What was run

| | |
|---|---|
| Pod ID | `e5drdw0okcdq7d` (terminated) |
| GPU | 1x NVIDIA RTX A5000 24 GB, **Secure** cloud |
| Image | `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204` (CUDA 12.8, Py 3.12) |
| Disk | 30 GB container, 0 GB volume |
| Price | **$0.27/hr** |
| Wall clock | ~6.5 min from create to terminate |
| **Cost** | **~$0.03** (account balance 116.9327 -> 116.9039) |
| Model | `faster-whisper` 1.2.1 / ctranslate2 4.8.1, `large-v3`, float16 |
| Input | `data/chunks/161701/chunk_00000..00009.mp3` (10 x 30 s) |
| Output | `data/scratch/runpod_whisper_sample.jsonl` |

Throughput measured: **RTF 0.09-0.24** on the A5000 (300 s of audio in 28-71 s),
i.e. roughly **4-10x realtime**, plus ~5 s model load.

## 2. Reproducing / scaling the mechanics

Community cloud had **no A5000/A4000/4090/3090/A6000 instances available** at the time
(`create pod: There are no instances currently available` / `This machine does not have
the resources to deploy your pod`). Secure cloud A5000 at $0.27/hr succeeded on the first
try. When scripting this, iterate over a candidate GPU list and fall back COMMUNITY ->
SECURE rather than retrying one type.

```python
# create — REST v1, Bearer auth
import os, requests
H = {"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]}
body = {
    "name": "yi-whisper",
    "imageName": "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204",
    "gpuTypeIds": ["NVIDIA RTX A5000"],   # list = RunPod picks any available
    "gpuCount": 1,
    "cloudType": "SECURE",                # try COMMUNITY first, it is ~40% cheaper
    "containerDiskInGb": 30, "volumeInGb": 0,
    "ports": ["22/tcp"],
    "env": {"PUBLIC_KEY": open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()},
    "supportPublicIp": True,
}
pod = requests.post("https://rest.runpod.io/v1/pods", headers=H, json=body).json()
```

Then poll `GET https://rest.runpod.io/v1/pods/<id>` until `publicIp` **and**
`portMappings["22"]` are both populated (~2.5 min here; `desiredStatus` is `RUNNING`
long before SSH is actually reachable — do not trust it). Balance/price checks and the
GPU catalogue are easier over GraphQL:

```
POST https://api.runpod.io/graphql?api_key=<KEY>
{myself{clientBalance currentSpendPerHr pods{id desiredStatus}}}
{gpuTypes{id lowestPrice(input:{gpuCount:1}){uninterruptablePrice stockStatus}}}
```

On the pod, `pip install faster-whisper` is sufficient — the RunPod PyTorch image already
ships the cuDNN 9 / cuBLAS libs ctranslate2 needs, no `LD_LIBRARY_PATH` fiddling required.
`scp -P <port> root@<ip>:...` works normally on secure-cloud pods with a public IP.

**Terminate with `DELETE https://rest.runpod.io/v1/pods/<id>` (expect 204), then confirm
`myself.pods` is empty.** Stopping is not enough — a stopped pod still bills for disk.

Cheapest GPUs seen (community, $/hr): 3070 0.13, **A5000 0.16**, A4000 0.17, 3080Ti 0.18,
A4500 0.19, 3090 0.22, 4090 0.34. large-v3 float16 needs ~5 GB VRAM, so the 16 GB A4000
is ample — VRAM is not the constraint, availability is.

### If you did scale it (you should not — see §3)

264 episodes x ~30 min = ~132 h of audio. At RTF ~0.1 on one A5000 that is ~13 GPU-hours,
~$2-4 on community cloud with batching (`BatchedInferencePipeline`, batch_size 8-16, would
cut it further). Cost is genuinely trivial — **the blocker is quality, not price.**

## 3. Quality vs the Gemini reference (`data/annotations/161701.jsonl`, chunks 0-9)

Normalization before scoring: NFC, strip all U+0591-U+05C7 (niqqud **and** the rafe/dagesh
that the Gemini orthography uses in פֿלעגט / אַז), drop non-Hebrew-letter chars.

| config | word recall | precision | F1 | mean char-sim |
|---|---|---|---|---|
| A: `language="yi"`, beam=5, no VAD | 5.7% | 8.7% | 6.9% | 7.4% |
| B: A + VAD + `temperature=0` + `condition_on_previous_text=False` | 6.6% | 7.1% | 6.8% | 2.6% |

Fuzzy scoring (a ref word counts as hit if any hypothesis word is >=0.8 character-similar)
lifts recall only to **28%** — that is the phonetic-gist ceiling, not usable text.

Config B is the one saved as `text_whisper`; A is kept as `text_whisper_novad`.
B is worth using anyway: it eliminated two hard failure modes that A hit in 10 chunks —
chunk 8 came back **completely empty**, and chunk 4/7 collapsed into hallucinated
**English and Korean** mid-sentence ("*particular small shop in Goules ... when the price
is low*"). B still loops (chunk 7 = `אין` repeated ~30x).

`language_probability` was **1.00 for all 10 chunks** — Whisper is fully confident it is
producing Yiddish while producing near-garbage. Confidence is useless as a quality filter here.

### Side by side

```
chunk 0
  GEMINI : בשם השם נעשה ונצליח האלטן מיר אין פרשת דברים שבת חזון תשפ"ו לפרט קטן
  WHISPER: בי שיים אשם לאה סבן עצליי חלט מדו פרשז דה וורם, שבס חזון טופשי אין פאי ווב לפרטקוטן

chunk 2
  GEMINI : דער קאָזשניצער מגיד פֿלעגט זיך פֿירן אַז יעדער ערבֿ יום כיפּור, האָט ער געבענטשט אַלע זיינע קינדער
  WHISPER: קרמנטס ומגט פליגט שח פירן, אס ידע אירו ויום קיפר, עד דער גבנשט אלה דערה קינדר

chunk 6
  GEMINI : נו, שוין באַלד צוויי טויזנט יאָר וואָס מ'זענען שוין אָפּגעטריבן פון דאָרטן, און ס'ווערט ליידער פאַרגעסן
  WHISPER: שם באו 2000 יוב איזן שטיימא פן דארט איזן וייט לידי פגאסן
```

### Qualitative read

It is **not** random garbage — it is a **phonetic mis-transcription in the wrong
orthography**. Script is correct (Hebrew), prosody/segmentation is roughly right, and the
loud stressed content words survive: `חזון`, `שכינה`, `אייבערשטע`, `קינדער`, `צער`,
`פארגעסן`, `קילאמעטער`, `2000 יאר`. Everything between them dissolves.

Three compounding failures:

1. **Orthography.** Whisper writes Yiddish the way Israeli Hebrew is spelled — vowel
   letters dropped (`פליגט` for `פֿלעגט`, `גבנשט` for `געבענטשט`, `קינדר` for `קינדער`,
   `דא` for `דער`). Almost every word is 1-2 letters off, which is exactly why exact
   overlap is 6% while fuzzy overlap is 28%.
2. **Loshn-koydesh / Hasidic register.** The fixed religious phrases that open nearly every
   episode are mangled: `בשם השם נעשה ונצליח` -> `בי שיים אשם לאה סבן עצליי`,
   `משנכנס אב ממעטין בשמחה` -> `מי שנכנס עוב ממהטן בסמכה`. Proper nouns are lost outright
   (`דער קאָזשניצער מגיד` -> `קרמנטס ומגט`). These are the highest-value tokens in the corpus.
3. **Dialect.** large-v3's Yiddish training data is overwhelmingly YIVO/standard and
   heavily weighted toward text, not Hasidic Yiddish speech. Word boundaries in fast
   connected Hasidic speech are consistently misplaced.

## 4. Recommendation

**Do not scale to 264 episodes.** Cost is not the problem ($2-4 of GPU); output at ~7%
agreement would need more correction effort than transcribing from scratch, and it would
poison the g2p/annotation pipeline with plausible-looking wrong Hebrew text. Keep Gemini
as the transcription layer.

Whisper on RunPod is still worth keeping for narrower jobs where phonetic-level output is
enough and orthography does not matter:

- **Word/VAD-level timestamps and speaker turns** (`word_timestamps=True`) to align or
  re-chunk the existing Gemini transcripts — timing was solid even where the text was not.
- **Silence / music / non-speech detection** to skip dead chunks before paying for Gemini.
- **Cheap triage**: flag episodes where the transcript collapses (empty output, repetition
  loops, non-Hebrew script) as bad audio.

The only path to usable open ASR here is **fine-tuning** `large-v3` (or `ivrit-ai`'s
Hebrew-adapted checkpoints, which already carry the right script conventions) on the
existing Gemini `text_yi` annotations as pseudo-labels — the corpus for that is exactly
what `data/annotations/` is accumulating. Revisit once a few dozen episodes are annotated;
at ~$0.20/hr for an A5000 the fine-tune itself is affordable.
