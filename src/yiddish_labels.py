"""Front door for the Yiddish label stack: text -> nikud -> IPA.

IMPORT THIS, NOT ``yiddish_g2p`` DIRECTLY -- in this repo and on any box the
bundle is shipped to.

WHY. ``yiddish_g2p`` loads its knowledge from seven generated tables in
``data/``. Every loader swallows a missing file and returns ``{}``: degradation
is deliberate, so the engine keeps running while a table is being regenerated.
The cost is that an incomplete deployment emits plausible-looking IPA with ZERO
native verdicts and zero audio corrections, and says nothing about it:

    with tables:     פעקל -> pɛkl    יארצייט -> jˈurʦajt   האף -> huf
    without tables:  פעקל -> fɛkl    יארצייט -> jˈarʦajt   האף -> haf

Both columns look like reasonable Yiddish; only the left one is right. This
module asserts at import that every table loaded and that readings only the
tables can produce are actually produced, so a bad deployment fails here
instead of surfacing months later in a listening test.

Usage:
    from yiddish_labels import text_to_ipa, text_to_nikud
    text_to_nikud("מיט א פאר יאר צוריק")   # 'מִיט אַ פּאָר יאָר צוּרִיק'
    text_to_ipa("מיט א פאר יאר צוריק")     # 'mit a pˈur jur ʦirˈik'

Layout-independent: it finds the engine and its tables whether it is sitting in
this repo's ``src/`` or unpacked next to them in a shipped bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find_engine() -> Path:
    """The directory holding yiddish_g2p.py AND its data/ tables.

    Both must live in the same directory -- the engine resolves ``data/``
    relative to its own file, so a yiddish_g2p.py found without its tables is
    the silent-degradation case this module exists to prevent.
    """
    for cand in (_HERE, _HERE.parent):
        if (cand / "yiddish_g2p.py").exists() and (cand / "data").is_dir():
            return cand
    raise RuntimeError(
        "cannot locate yiddish_g2p.py next to a data/ directory "
        f"(looked in {_HERE} and {_HERE.parent})"
    )


ENGINE_DIR = _find_engine()
# ENGINE_DIR must be importable (yiddish_g2p lives there), but THIS directory
# has to win every name collision: the repo root also carries an older
# yiddish_nikud.py aimed at a superseded export, and importing that one would
# regenerate the very labels this stack exists to replace. Order is forced
# rather than guarded, because a caller may already have put either path on
# sys.path before importing us.
for _p in (str(ENGINE_DIR), str(_HERE)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)  # _HERE inserted last -> ends up first

import yiddish_g2p as _g2p  # noqa: E402

# Table sizes at the time this guard was written. A table that fails to load
# reads 0; a legitimately regenerated table grows. The check is ">=" so growth
# is fine, and a shortfall names the table.
_EXPECTED = {
    "GOLD_LEXICON": 502,
    "_AUDIO_PE": 77,
    "_AUDIO_VOWEL": 42,
    "_AUDIO_ENDORSED": 100,
    "_HOMOGRAPH_LK": 200,
    "_SEFARIA_POINTED": 3000,
    "_MODEL_POINTED": 7000,
}

# Readings no rule path can invent -- each comes from exactly one table.
_CANARIES = {
    "פעקל": "pɛkl",             # audio-pe (PhoneticXeus voted p 23-0)
    "יארצייט": "jˈurʦajt",       # audio-vowel (komets u, 23/29 clips)
    "האף": "huf",               # gold (Chezky 2026-08-10, "ho not ha")
    "שבת": "ʃˈabəs",            # gold / merged-LK
    "א פאר יאר": "a pˈur jur",   # multiword: 'a few years', not 'far'
}


def verify(strict: bool = True) -> dict:
    """Check every table loaded and every canary reads correctly.

    Returns the report; raises RuntimeError listing the problems when strict.
    """
    problems: list[str] = []
    sizes: dict[str, int] = {}
    for name, want in _EXPECTED.items():
        got = len(getattr(_g2p, name, {}) or {})
        sizes[name] = got
        if got < want:
            problems.append(
                f"{name}: {got} entries, expected >= {want}"
                + (" -- table missing from data/" if got == 0 else "")
            )
    readings: dict[str, str] = {}
    for word, want in _CANARIES.items():
        got = _g2p.hebrew_to_ipa(word, stress=True)
        readings[word] = got
        if got != want:
            problems.append(f"{word}: reads {got!r}, expected {want!r}")
    report = {"engine_dir": str(ENGINE_DIR), "sizes": sizes,
              "canaries": readings, "problems": problems}
    if problems and strict:
        raise RuntimeError(
            "Yiddish label stack is not correctly deployed:\n  "
            + "\n  ".join(problems)
            + f"\n\nEngine loaded from {ENGINE_DIR}. The data/*.py tables must "
              "sit beside yiddish_g2p.py; without them the engine silently "
              "drops every native verdict and audio correction."
        )
    return report


verify()  # fail at import, not at inference


def text_to_ipa(text: str) -> str:
    """Phonemes for Hebrew-script Yiddish, via the full authority chain."""
    return _g2p.hebrew_to_ipa(text, stress=True)


def token_detail(word: str) -> dict:
    """Per-token record: ipa_primary, route, confidence, reason.

    LOW confidence marks a defaulted ambiguous grapheme or a rescued reading --
    the human-verification queue, not noise.
    """
    return _g2p.g2p_token(word)


def text_to_nikud(text: str) -> str:
    """Diacritized text via phonikud-yi v5 (lazy-loads the ONNX model)."""
    from yiddish_nikud import add_nikud
    return add_nikud(text)


def text_to_nikud_batch(texts: list[str]) -> list[str]:
    """Batch diacritization -- use this when labelling a dataset."""
    from yiddish_nikud import add_nikud_batch
    return add_nikud_batch(texts)


if __name__ == "__main__":
    import json
    print(json.dumps(verify(strict=False), ensure_ascii=False, indent=1))
