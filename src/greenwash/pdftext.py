"""PDF / text extraction and sentence segmentation."""
from __future__ import annotations

import io
import re

from pypdf import PdfReader

_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|incl|no|fig|plc|ltd|inc|co|mr|mrs|ms|dr|st|jan|feb|mar|apr|jun|"
                     r"jul|aug|sep|sept|oct|nov|dec|[A-Z])\.$", re.I)
_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[\"'“(]?[A-Z0-9])")


def _split(text: str) -> list[str]:
    parts, buf = [], ""
    for p in _SPLIT_RX.split(text):
        buf = f"{buf} {p}" if buf else p
        if not _ABBREV.search(buf):
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


def pdf_pages(data: bytes, max_pages: int = 400) -> list[str]:
    """Extract text per page from PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a broken page shouldn't kill the report
            pages.append("")
    return pages


def clean(text: str) -> str:
    text = text.replace("­", "").replace("‐", "-").replace("‑", "-")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate line breaks
    text = re.sub(r"[•▪●◦■□►]", ". ", text)          # bullets become sentence breaks
    text = re.sub(r"\n{2,}", ". ", text)                  # paragraph breaks
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\.\s*\.(\s*\.)*", ".", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def sentences(text: str, min_words: int = 6, max_words: int = 60) -> list[str]:
    """Split cleaned text into candidate sentences, dropping fragments, tables and boilerplate."""
    out = []
    for s in _split(clean(text)):
        s = s.strip(" .;:-–")
        n = len(s.split())
        if n < min_words or n > max_words:
            continue
        letters = sum(c.isalpha() for c in s)
        if letters < 0.6 * len(s.replace(" ", "")):      # mostly numbers: a table row
            continue
        caps = sum(w[:1].isupper() for w in s.split())
        if n >= 12 and caps > 0.6 * n:                    # title-case run: headings, infographics, SDG lists
            continue
        out.append(s + ".")
    return out


def document_sentences(pages: list[str]) -> list[tuple[int, str]]:
    """[(page_number, sentence)], 1-based pages."""
    return [(i + 1, s) for i, p in enumerate(pages) for s in sentences(p)]
