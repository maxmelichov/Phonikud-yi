"""Split an mp3 into fixed-length chunks with the ffmpeg CLI.

Requires ffmpeg on PATH (`brew install ffmpeg` on macOS).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegMissing(RuntimeError):
    pass


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def require_ffmpeg() -> None:
    if not have_ffmpeg():
        raise FFmpegMissing(
            "ffmpeg/ffprobe not found on PATH. Install with: brew install ffmpeg"
        )


def duration_s(path: str | Path) -> float:
    require_ffmpeg()
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_entries", "format=duration", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


@dataclass
class Chunk:
    idx: int
    start_s: float
    end_s: float
    path: Path


def chunk_mp3(
    src: str | Path,
    out_dir: str | Path,
    chunk_s: float = 30.0,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> list[Chunk]:
    """Segment `src` into ~chunk_s mono mp3 files under out_dir.

    Uses ffmpeg's segment muxer (single decode pass). Returns chunk metadata.
    """
    require_ffmpeg()
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = out_dir / "chunk_%05d.mp3"
    existing = sorted(out_dir.glob("chunk_*.mp3"))
    if existing and not overwrite:
        files = existing
    else:
        for f in existing:
            f.unlink()
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(src),
                "-vn", "-ac", "1", "-ar", str(sample_rate), "-b:a", "64k",
                "-f", "segment", "-segment_time", str(chunk_s),
                "-reset_timestamps", "1",
                str(pattern),
            ],
            check=True, capture_output=True,
        )
        files = sorted(out_dir.glob("chunk_*.mp3"))

    total = duration_s(src)
    chunks: list[Chunk] = []
    for i, f in enumerate(files):
        start = i * chunk_s
        chunks.append(Chunk(i, start, min(start + chunk_s, total), f))
    return chunks
