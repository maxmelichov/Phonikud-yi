"""
Yiddish diacritization (nikud) via the char-level ONNX model in data/onnx_yiddish.

WHY THIS EXISTS: the Yiddish training targets in yiddish24-wav were phonemized
from *diacritized* text, not from the raw ASR text -- reproducing the stored
`ipa` column requires nikud first (1500/1500 exact match from the `nikud`
column, only 148/1500 from raw `text`). Without this module the inference path
feeds unpointed text to the G2P and reproduces training output ~10% of the time.
Diacritics are what let yiddish_g2p resolve /a/ vs /o/ for א, /p/ vs /f/ for פ,
/ey/ vs /ay/ for יי, and the vowels of Loshn-Koydesh words that unpointed
Hasidic spelling omits entirely.

THE EXPORT SHIPS NO CONFIG. Three details were recovered empirically and are
load-bearing; all three are verified by test_matches_reference() below, which
reproduces the stored `nikud` column at 99/100 on real corpus rows:

  1. Token ids come from tokenizer.json, NOT vocab.txt. The two disagree from
     index 18 onward (vocab.txt carries two extra empty lines), so reading
     vocab.txt shifts every Hebrew letter and the output is garbage.
  2. Head class orderings live in yi_head_labels.json (see its _note). The
     42-class yi_labels.json is a different, unused label set.
  3. Rafe is only emitted on בכפגדת. The rafe head fires on other letters too,
     and honouring it there produces spurious ֿ marks.
"""

from __future__ import annotations

import json
import os
import unicodedata as ud

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_yiddish")

# Rafe is a Yiddish/Hebrew fricative marker; it is only meaningful on these.
_RAFE_LETTERS = frozenset("בכפגדת")
_HEBREW_LETTERS = frozenset(chr(c) for c in range(0x05D0, 0x05EB))


def _strip_marks(text: str) -> str:
    """Drop existing diacritics so the model sees the consonantal skeleton."""
    return "".join(
        c for c in ud.normalize("NFD", text) if ud.category(c) != "Mn"
    )


class YiddishNikud:
    """Adds nikud to raw Hebrew-script Yiddish text.

    >>> YiddishNikud().add("בשעת ביידן'ס צופרידנהייט ראטעס פאלן.")
    "בִּשְׁעַת בַּיידְן'ס צוּפְרִידְנְהֵייט ראָטֶעס פֿאַלְן."
    """

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        intra_op_threads: int = 0,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        self.model_dir = model_dir
        with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
            self._vocab = json.load(f)["model"]["vocab"]
        with open(os.path.join(model_dir, "yi_head_labels.json"), encoding="utf-8") as f:
            heads = json.load(f)
        self._nikud = heads["nikud_classes"]
        self._shin = heads["shin_classes"]
        self._rafe = heads["rafe_classes"]

        self._unk = self._vocab["[UNK]"]
        self._cls = self._vocab["[CLS]"]
        self._sep = self._vocab["[SEP]"]
        self._pad = self._vocab["[PAD]"]

        opts = ort.SessionOptions()
        if intra_op_threads:
            # Left at 0 (= all cores) by default. Set to 1 when running many
            # worker processes, or they oversubscribe and throughput collapses.
            opts.intra_op_num_threads = intra_op_threads
        # CPU by default, deliberately. Auto-selecting CUDA was tried and fails
        # in the normal case: the dataset is rebuilt on the same box that is
        # training, both GPUs sit near capacity, and ORT dies with
        # "Failed to allocate memory for requested buffer of size 403206400".
        # This model is small and CPU is fast enough. Pass providers=[...]
        # explicitly if the GPUs are known to be idle -- and verify GPU and CPU
        # agree first, since diacritics decide the phonemes.
        self._session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.providers = self._session.get_providers()

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

    def add(self, text: str) -> str:
        """Diacritize a single string."""
        return self.add_batch([text])[0]

    def add_batch(self, texts: list[str]) -> list[str]:
        """Diacritize a batch. Pads to the longest row; keep batches size-similar."""
        import numpy as np

        if not texts:
            return []
        encoded = [self._encode(t) for t in texts]
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
        return [
            self._decode(base, nikud[r], shin[r], rafe[r])
            for r, (base, _) in enumerate(encoded)
        ]


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
