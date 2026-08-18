#!/usr/bin/env python3
"""Scrape the Yiddish24 category 162 ("שבת פעקל") episode list and audio.

Usage:
    python scraper/scrape_yiddish24.py                 # scrape metadata + download first 20 mp3s
    python scraper/scrape_yiddish24.py --all           # scrape metadata + download every mp3
    python scraper/scrape_yiddish24.py --limit 50      # download the first N episodes
    python scraper/scrape_yiddish24.py --no-download   # metadata only

Outputs:
    data/corpus/episodes.jsonl        one JSON object per episode
    data/audio/<id>.mp3        downloaded audio
    data/corpus/audio_manifest.jsonl  one JSON object per downloaded file
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

CAT_ID = 162
TOTAL_PAGES = 27
PAGE_LIMIT = 10
BASE_URL = "https://www.yiddish24.com/cat/%d" % CAT_ID
AJAX_URL = "https://www.yiddish24.com/ajax/cat_pagination.php"
USER_AGENT = "Mozilla/5.0"
DELAY = 1.0

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
EPISODES_PATH = DATA_DIR / "corpus" / "episodes.jsonl"
MANIFEST_PATH = DATA_DIR / "corpus" / "audio_manifest.jsonl"

# <h1 id="song-title161701">TITLE</h1> ... <span class="date">DATE</span>
TITLE_RE = re.compile(
    r'<h1\s+id="song-title(\d+)"\s*>(.*?)</h1>\s*<span\s+class="date"\s*>(.*?)</span>',
    re.S,
)
# <div class="... item-data-161701 audio" data-id="161701" data-song-url="https://...mp3"
SONG_RE = re.compile(
    r'data-id="(\d+)"\s+data-song-url="(https?://[^"]+?\.mp3)"',
    re.I,
)
BLOCK_RE = re.compile(r"item-data-(\d+)\b")
DURATION_RE = re.compile(r'<div class="song-duration[^"]*"\s*><i>([^<]*)</i>', re.S)
TAG_RE = re.compile(r"<[^>]+>")


def parse_durations(markup: str) -> dict[str, str]:
    """Map song id -> duration, scoped to each episode's own markup block."""
    marks = [(m.start(), m.group(1)) for m in BLOCK_RE.finditer(markup)]
    out: dict[str, str] = {}
    for i, (start, song_id) in enumerate(marks):
        if song_id in out:
            continue
        end = marks[i + 1][0] if i + 1 < len(marks) else len(markup)
        found = DURATION_RE.search(markup, start, end)
        if found:
            out[song_id] = clean(found.group(1))
    return out


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", text))).strip()


def parse_episodes(markup: str, page: int) -> list[dict]:
    """Extract episode records from a category page or AJAX HTML fragment."""
    titles: dict[str, tuple[str, str]] = {}
    for song_id, title, date in TITLE_RE.findall(markup):
        # The page renders each block twice (a commented-out copy plus the live
        # one); first occurrence wins, they are identical.
        titles.setdefault(song_id, (clean(title), clean(date)))

    durations = parse_durations(markup)

    episodes: list[dict] = []
    seen: set[str] = set()
    for song_id, mp3_url in SONG_RE.findall(markup):
        if song_id in seen:
            continue
        seen.add(song_id)
        title, date = titles.get(song_id, ("", ""))
        episodes.append(
            {
                "id": song_id,
                "title": title,
                "date": date,
                "duration": durations.get(song_id, ""),
                "mp3_url": mp3_url,
                "page": page,
            }
        )
    return episodes


def fetch_page(session: requests.Session, page: int) -> str:
    if page == 1:
        resp = session.get(BASE_URL, timeout=60)
        resp.raise_for_status()
        return resp.text

    resp = session.post(
        AJAX_URL,
        data={
            "page_no": page,
            "data_id": CAT_ID,
            "total_pages": TOTAL_PAGES,
            "page_limit": PAGE_LIMIT,
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    result = payload.get("result")
    if not result or result is True:
        raise ValueError("page %d returned no result payload: %r" % (page, payload))
    return result


def scrape(session: requests.Session) -> tuple[list[dict], list[tuple[int, str]]]:
    episodes: list[dict] = []
    seen_urls: set[str] = set()
    failures: list[tuple[int, str]] = []

    for page in range(1, TOTAL_PAGES + 1):
        try:
            markup = fetch_page(session, page)
            found = parse_episodes(markup, page)
        except Exception as exc:  # network, JSON, or parse failure
            failures.append((page, "%s: %s" % (type(exc).__name__, exc)))
            print("page %2d FAILED: %s" % (page, exc), file=sys.stderr)
        else:
            new = [e for e in found if e["mp3_url"] not in seen_urls]
            seen_urls.update(e["mp3_url"] for e in new)
            episodes.extend(new)
            print(
                "page %2d: %d episodes (%d new, running total %d)"
                % (page, len(found), len(new), len(episodes))
            )
        if page < TOTAL_PAGES:
            time.sleep(DELAY)

    return episodes, failures


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def download(session: requests.Session, episodes: list[dict]) -> list[dict]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for idx, ep in enumerate(episodes, 1):
        dest = AUDIO_DIR / ("%s.mp3" % ep["id"])
        if dest.exists() and dest.stat().st_size > 0:
            print("[%d/%d] skip %s (exists)" % (idx, len(episodes), dest.name))
            manifest.append(
                {
                    "id": ep["id"],
                    "path": str(dest.relative_to(ROOT)),
                    "bytes": dest.stat().st_size,
                    "mp3_url": ep["mp3_url"],
                }
            )
            continue

        tmp = dest.with_suffix(".mp3.part")
        try:
            with session.get(ep["mp3_url"], stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            os.replace(tmp, dest)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            print(
                "[%d/%d] FAILED %s: %s" % (idx, len(episodes), ep["id"], exc),
                file=sys.stderr,
            )
            continue

        size = dest.stat().st_size
        print("[%d/%d] %s -> %.1f MB" % (idx, len(episodes), dest.name, size / 1e6))
        manifest.append(
            {
                "id": ep["id"],
                "path": str(dest.relative_to(ROOT)),
                "bytes": size,
                "mp3_url": ep["mp3_url"],
            }
        )
        time.sleep(DELAY)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="download every episode, not just the first 20"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="how many episodes to download (default 20)"
    )
    parser.add_argument(
        "--no-download", action="store_true", help="scrape metadata only"
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": BASE_URL,
        }
    )

    episodes, failures = scrape(session)
    write_jsonl(EPISODES_PATH, episodes)
    print("\nwrote %d episodes to %s" % (len(episodes), EPISODES_PATH))
    if failures:
        print("failed pages: %s" % ", ".join(str(p) for p, _ in failures))

    if args.no_download:
        return 1 if failures else 0

    targets = episodes if args.all else episodes[: args.limit]
    print("\ndownloading %d/%d episodes..." % (len(targets), len(episodes)))
    manifest = download(session, targets)
    write_jsonl(MANIFEST_PATH, manifest)
    total = sum(row["bytes"] for row in manifest)
    print(
        "\nwrote %d rows to %s (%.1f MB total)"
        % (len(manifest), MANIFEST_PATH, total / 1e6)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
