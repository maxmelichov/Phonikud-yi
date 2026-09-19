#!/usr/bin/env python3
"""Run ingest_printed_index with a PyMuPDF extractor (no poppler pdftotext needed).

Emulates `pdftotext -layout` on each column crop from raw glyphs: characters
are clustered into lines by y, read right-to-left, split into cells at a wide
gap (the hebrew | phonetic column seam) and into words at space glyphs. Raw
glyphs are used instead of PyMuPDF "words" because the zero-width '*' the font
uses for yud-with-hiriq is dropped by the word extractor. A superscript
homograph numeral glued to the hebrew cell is emitted the way pdftotext did:
<hebrew>  <digit> then <phonetic> on its own record.

    python scripts/ingest_printed_index_fitz.py <index.pdf> [--records cache.json]
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_printed_index as ing

CELL_GAP = 9.0
Y_TOL = 4.0
_DIGIT_TAIL = re.compile(r"^(.*[א-תיִ-ﭏ]) ?(\d)$")
_MARK = re.compile(r"[ְ-ׇ]")
_SWAP = {"(": ")", ")": "("}

def _page_chars(page):
    """Glyph units: a base char plus the combining marks that follow it."""
    units = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                for c in s["chars"]:
                    x0, y0, x1, y1 = c["bbox"]
                    ch = c["c"]
                    if (_MARK.match(ch) or ch == "*") and units:
                        u = units[-1]; units[-1] = (u[0], u[1], u[2], u[3], u[4] + ch)
                    else:
                        units.append((x0, y0, x1, y1, _SWAP.get(ch, ch)))
    return units

def extract_records_fitz(pdf: Path) -> list[list[str]]:
    doc = fitz.open(str(pdf))
    records: list[list[str]] = []
    for page in doc:
        chars = list(_page_chars(page))
        for x, width in ing.COLUMN_CROPS:
            blk = [c for c in chars if x <= c[0] < x + width and c[4] != "\n"]
            blk.sort(key=lambda c: c[1])
            lines: list[list] = []
            for c in blk:
                if lines and abs(c[1] - lines[-1][0][1]) <= Y_TOL:
                    lines[-1].append(c)
                else:
                    lines.append([c])
            for ln in lines:
                ln.sort(key=lambda c: -c[0])
                cells: list[str] = [""]
                prev = None
                for c in ln:
                    ch = ing._BIDI.sub("", c[4])
                    if not ch:
                        continue
                    if prev is not None and (prev[0] - c[2]) > CELL_GAP:
                        cells.append("")
                    elif ch == " ":
                        if cells[-1] and not cells[-1].endswith(" "):
                            cells[-1] += " "
                        prev = c
                        continue
                    cells[-1] += ch
                    prev = c
                cells = [re.sub(r"\s+", " ", c).strip() for c in cells]
                cells = [c for c in cells if c]
                cells.reverse()   # pdftotext order: hebrew then phonetic
                if not cells:
                    continue
                m = _DIGIT_TAIL.match(cells[0])
                if m and len(cells) == 2:
                    records.append([m.group(1), m.group(2)])
                    records.append([cells[1]])
                    continue
                records.append(cells)
    return records

if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else ing.PDF
    ing.extract_records = lambda: extract_records_fitz(pdf)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    ing.main()
