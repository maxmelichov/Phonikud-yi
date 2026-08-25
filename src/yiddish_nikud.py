"""
Yiddish diacritization (nikud) via the phonikud-yi **v5** ONNX model.

WHY THIS EXISTS: the Yiddish training targets were phonemized from *diacritized*
text, not from raw ASR text -- reproducing them requires nikud first. Diacritics
are what let yiddish_g2p resolve /a/ vs /o/ for א, /p/ vs /f/ for פ, /ey/ vs
/ay/ for יי, and the vowels of Loshn-Koydesh words that unpointed Hasidic
spelling omits.

WHAT CHANGED FROM THE PREVIOUS VERSION OF THIS FILE (read this):

  The old module pointed at the `onnx_yiddish` export and was validated by how
  faithfully it reproduced the corpus's stored `nikud` column (99/100). That is
  the wrong target: the stored column IS the defect. It was written by an
  audio-annotation prompt with no pointing conventions, so it disagrees with
  itself -- האט appears pointed 18 different ways, האבן 31, and פאר is פֿאַר
  ("far") even in אַ פּאָר יאָר ("a pur yor", a few years). A voice trained on
  it learns three dialects at once, which is what the released voice does.

  This module points at v5, which was finetuned on labels where every disputed
  reading was settled by a fixed authority chain: native verdicts (503 gold
  words) > corpus audio (PhoneticXeus votes over 900 episode chunks) > published
  pointing > model guesses. It therefore *deliberately disagrees* with the
  stored nikud column on exactly the words that column got wrong. Do not
  "fix" it back by scoring it against that column.

Self-contained: the char vocab, the three head class orderings and the special
token ids are read from the model.onnx custom metadata, so no sidecar file can
drift out of sync with the weights. Needs only onnxruntime + numpy.

Three details are load-bearing and are asserted at load time:
  1. Token ids come from the model metadata (built from the tokenizer's true id
     space), NOT vocab.txt -- vocab.txt is off by one against the real ids, so
     reading it shifts every Hebrew letter and the output is garbage.
  2. Head class orderings come from the metadata (yi_head_labels.json in this
     directory is a copy for reference; the metadata wins).
  3. Rafe is only emitted on בכפגדת. The rafe head fires on other letters too,
     and honouring it there produces spurious ֿ marks.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata as ud

_HERE = os.path.dirname(os.path.abspath(__file__))

# Where the v5 export lives. In this repo it is under models/; in a shipped
# bundle it sits next to this file. $PHONIKUD_YI_MODEL overrides both.
_CANDIDATES = (
    os.environ.get("PHONIKUD_YI_MODEL", ""),
    os.path.join(_HERE, "onnx_yiddish_v6"),
    os.path.join(_HERE, "onnx_yiddish_v5"),
    os.path.join(os.path.dirname(_HERE), "models", "phonikud_yi_v6", "v6.onnx"),
    os.path.join(os.path.dirname(_HERE), "models", "phonikud_yi_v5", "v5.onnx"),
)


def _find_model_dir() -> str:
    for cand in _CANDIDATES:
        if cand and os.path.exists(os.path.join(cand, "model.onnx")):
            return cand
    raise RuntimeError(
        "no phonikud-yi v5 export found (looked for model.onnx in: "
        + ", ".join(c for c in _CANDIDATES if c)
        + "). Set $PHONIKUD_YI_MODEL to the export directory."
    )


MODEL_DIR = next(
    (c for c in _CANDIDATES if c and os.path.exists(os.path.join(c, "model.onnx"))),
    os.path.join(_HERE, "onnx_yiddish_v5"),  # reported by _find_model_dir on use
)

# Rafe is a Yiddish/Hebrew fricative marker; it is only meaningful on these.
_RAFE_LETTERS = frozenset("בכפגדת")
_HEBREW_LETTERS = frozenset(chr(c) for c in range(0x05D0, 0x05EB))
_WS = re.compile(r"(\s+)")


def _strip_marks(text: str) -> str:
    """Drop existing diacritics so the model sees the consonantal skeleton."""
    return "".join(
        c for c in ud.normalize("NFD", text) if ud.category(c) != "Mn"
    )


class YiddishNikud:
    """Adds nikud to raw Hebrew-script Yiddish text.

    >>> YiddishNikud().add("מיט א פאר יאר צוריק")
    'מִיט אַ פּאָר יאָר צוּרִיק'
    """

    def __init__(
        self,
        model_dir: str | None = None,
        intra_op_threads: int = 0,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        model_dir = model_dir or _find_model_dir()
        self.model_dir = model_dir
        opts = ort.SessionOptions()
        if intra_op_threads:
            # Left at 0 (= all cores) by default. Set to 1 when running many
            # worker processes, or they oversubscribe and throughput collapses.
            opts.intra_op_num_threads = intra_op_threads
        # CPU by default, deliberately. Auto-selecting CUDA fails in the normal
        # case: the dataset is rebuilt on the same box that is training, both
        # GPUs sit near capacity, and ORT dies allocating its buffer. This model
        # is small and CPU is fast enough (~4200 chars/sec). Pass providers=[...]
        # explicitly if the GPUs are known to be idle -- and verify GPU and CPU
        # agree first, since diacritics decide the phonemes.
        self._session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.providers = self._session.get_providers()

        meta = self._session.get_modelmeta().custom_metadata_map
        missing = {"vocab", "nikud_classes", "shin_classes", "rafe_classes",
                   "cls_id", "sep_id", "pad_id", "unk_id"} - set(meta)
        if missing:
            raise RuntimeError(
                f"{model_dir}/model.onnx is missing runtime metadata {sorted(missing)}. "
                "Re-embed it with scripts/embed_onnx_metadata.py; do NOT fall back "
                "to vocab.txt (its ids are off by one)."
            )
        itos = json.loads(meta["vocab"])
        self._vocab = {c: i for i, c in enumerate(itos)}
        self._nikud = json.loads(meta["nikud_classes"])
        self._shin = json.loads(meta["shin_classes"])
        self._rafe = json.loads(meta["rafe_classes"])
        self._cls = int(meta["cls_id"])
        self._sep = int(meta["sep_id"])
        self._pad = int(meta["pad_id"])
        self._unk = int(meta["unk_id"])
        self.max_len = int(meta.get("max_len", 512))
        # room for [CLS]/[SEP] plus slack, so no row is silently truncated
        self._chunk = self.max_len - 8

        heads = {o.name: o.shape[-1] for o in self._session.get_outputs()}
        want = sorted((len(self._nikud), len(self._shin), len(self._rafe)))
        got = sorted(v for v in heads.values() if isinstance(v, int))
        if got != want:
            raise RuntimeError(f"head sizes {got} != class lists {want}")

    # -----------------------------------------------------------------
    def _encode(self, text: str) -> tuple[str, list[int]]:
        base = _strip_marks(ud.normalize("NFKC", text))
        ids = [self._cls] + [self._vocab.get(c, self._unk) for c in base] + [self._sep]
        return base, ids

    def _decode(self, base: str, nikud, shin, rafe) -> str:
        out: list[str] = []
        for i, ch in enumerate(base, start=1):  # +1 skips [CLS]
            out.append(ch)
            if ch not in _HEBREW_LETTERS:
                continue
            if ch == "ש":
                out.append(self._shin[shin[i]])
            out.append(self._nikud[nikud[i]])
            if ch in _RAFE_LETTERS:
                out.append(self._rafe[rafe[i]])
        return ud.normalize("NFC", "".join(out))

    def _split(self, text: str) -> list[str]:
        """Whitespace-run split into <= _chunk pieces; rejoining is exact."""
        if len(text) <= self._chunk:
            return [text]
        out: list[str] = []
        cur = ""
        for piece in _WS.split(text):
            if cur and len(cur) + len(piece) > self._chunk:
                out.append(cur)
                cur = piece
            else:
                cur += piece
        if cur:
            out.append(cur)
        return out

    def add(self, text: str) -> str:
        """Diacritize a single string (any length)."""
        return self.add_batch([text])[0]

    def add_batch(self, texts: list[str]) -> list[str]:
        """Diacritize a batch. Pads to the longest row; keep batches size-similar.

        Rows longer than the model's window are split on whitespace, pointed,
        and rejoined -- separators pass through untouched, so the result is
        character-exact against the input. Every output is verified to strip
        back to its input letters; a row that fails raises rather than silently
        shipping a mutated spelling (models rewrite ligatures given the chance,
        and a mutated target poisons TTS training).
        """
        import numpy as np

        if not texts:
            return []
        # flatten (row -> segments) so one session.run covers the whole batch
        segs: list[str] = []
        owner: list[int] = []
        bases: list[str] = []
        for r, t in enumerate(texts):
            for s in self._split(_strip_marks(ud.normalize("NFKC", t))):
                segs.append(s)
                owner.append(r)
        encoded = [self._encode(s) for s in segs]
        width = max(len(ids) for _, ids in encoded)
        input_ids = np.full((len(encoded), width), self._pad, dtype=np.int64)
        attention = np.zeros((len(encoded), width), dtype=np.int64)
        for row, (_, ids) in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            attention[row, : len(ids)] = 1

        nikud, shin, rafe = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention}
        )
        nikud, shin, rafe = nikud.argmax(-1), shin.argmax(-1), rafe.argmax(-1)

        parts: list[list[str]] = [[] for _ in texts]
        for r, (base, _) in enumerate(encoded):
            parts[owner[r]].append(self._decode(base, nikud[r], shin[r], rafe[r]))
        out = ["".join(p) for p in parts]
        for src, pointed in zip(texts, out):
            want = _strip_marks(ud.normalize("NFKC", src))
            if _strip_marks(pointed) != want:
                raise ValueError(
                    "pointing changed the letters (model rewrote a spelling); "
                    f"input={src!r} output={pointed!r}"
                )
        return out


_INSTANCE: YiddishNikud | None = None


def add_nikud(text: str) -> str:
    """Diacritize using a lazily-created process-wide model instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = YiddishNikud()
    return _INSTANCE.add(text)


def add_nikud_batch(texts: list[str]) -> list[str]:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = YiddishNikud()
    return _INSTANCE.add_batch(texts)
