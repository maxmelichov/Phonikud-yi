#!/usr/bin/env python
"""Smoke test: rule engine on sample sentences + one tiny gateway call.

  .venv/bin/python scripts/smoke_test.py
  .venv/bin/python scripts/smoke_test.py --no-gateway
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonikud_yi.gateway import MODEL_FLASH, MODEL_PRO, Gateway, text_message  # noqa: E402
from phonikud_yi.segment import have_ffmpeg  # noqa: E402
from yiddish_g2p import hebrew_to_ipa, hebrew_to_latin  # noqa: E402

SAMPLES = [
    "שבת איז געווען א שיינער טאג אין בארא פארק",
    "דער רבי האט געזאגט אז מען דארף לערנען תורה יעדן טאג",
    "מסתמא וועט ער קומען צו דער חתונה מיט זיין משפחה",
    "איך האב אים געזען אויף דער גאס פארגאנגענע וואך",
    "עס איז א מצווה צו העלפן א צווייטן איד",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gateway", action="store_true")
    ap.add_argument("--audio", action="store_true", help="also test a base64 audio call")
    args = ap.parse_args()

    print("== ffmpeg ==")
    print("  available" if have_ffmpeg() else "  MISSING -> brew install ffmpeg")

    print("\n== rule engine (yiddish_g2p) ==")
    for s in SAMPLES:
        print(f"  yi : {s}")
        print(f"  lat: {hebrew_to_latin(s)}")
        print(f"  ipa: {hebrew_to_ipa(s)}\n")

    if args.no_gateway:
        return 0

    print("== AI gateway ==")
    try:
        gw = Gateway()
        models = gw.list_models()
        for name, slug in (("flash", MODEL_FLASH), ("pro", MODEL_PRO)):
            status = "present" if slug in models else "NOT IN /models"
            print(f"  {name}: {slug} ({status})")
        reply = gw.chat(
            [text_message("user", "Reply with exactly: PHONIKUD-YI-OK")],
            model=MODEL_FLASH,
            max_tokens=2048,  # gemini-3.x are reasoning models: leave headroom
        )
        print(f"  chat({MODEL_FLASH}) -> {reply.strip()!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        return 1

    if args.audio and have_ffmpeg():
        import subprocess
        import tempfile

        from phonikud_yi.gateway import audio_message

        with tempfile.TemporaryDirectory() as td:
            tone = Path(td) / "tone.mp3"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "sine=frequency=440:duration=3", "-ac", "1", "-ar", "16000", str(tone)],
                check=True,
            )
            try:
                r = gw.chat(
                    [audio_message(tone, "Describe this audio in 5 words, then say AUDIO-OK.")],
                    model=MODEL_FLASH,
                    max_tokens=2048,
                )
                print(f"  audio -> {r.strip()[:100]!r}")
            except Exception as exc:  # noqa: BLE001
                print(f"  audio FAILED: {exc}")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
