"""Book ingestion: parse EPUB/PDF/TXT into a single text + chapter map, then chunk."""
from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .db import conn_ctx


@dataclass
class RawBook:
    title: str
    author: str | None
    text: str
    # list of (char_offset, chapter_title) sorted by char_offset
    chapter_marks: list[tuple[int, str]]


def _parse_txt(data: bytes, filename: str) -> RawBook:
    text = data.decode("utf-8", errors="replace")
    return RawBook(title=Path(filename).stem, author=None, text=text, chapter_marks=[(0, "Body")])


def _parse_pdf(data: bytes, filename: str) -> RawBook:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    chapter_marks: list[tuple[int, str]] = []
    cursor = 0
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        chapter_marks.append((cursor, f"Page {i + 1}"))
        parts.append(page_text)
        cursor += len(page_text) + 2  # account for separator
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else Path(filename).stem) or Path(filename).stem
    author = reader.metadata.author if reader.metadata else None
    return RawBook(title=title, author=author, text="\n\n".join(parts), chapter_marks=chapter_marks)


def _parse_epub(data: bytes, filename: str) -> RawBook:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    # ebooklib reads from file path; write to temp
    tmp = settings.data_dir / f"_tmp_{uuid.uuid4().hex}.epub"
    tmp.write_bytes(data)
    try:
        book = epub.read_epub(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    title_meta = book.get_metadata("DC", "title")
    title = title_meta[0][0] if title_meta else Path(filename).stem
    author_meta = book.get_metadata("DC", "creator")
    author = author_meta[0][0] if author_meta else None

    parts: list[str] = []
    chapter_marks: list[tuple[int, str]] = []
    cursor = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        # Try to find a chapter title from h1/h2/h3
        heading_tag = soup.find(["h1", "h2", "h3", "title"])
        chapter_title = (heading_tag.get_text().strip() if heading_tag else item.get_name()) or item.get_name()
        text = soup.get_text("\n").strip()
        if not text:
            continue
        chapter_marks.append((cursor, chapter_title))
        parts.append(text)
        cursor += len(text) + 2
    return RawBook(title=title, author=author, text="\n\n".join(parts), chapter_marks=chapter_marks)


def parse_upload(data: bytes, filename: str, content_type: str | None) -> RawBook:
    name = filename.lower()
    if name.endswith(".epub") or content_type == "application/epub+zip":
        return _parse_epub(data, filename)
    if name.endswith(".pdf") or content_type == "application/pdf":
        return _parse_pdf(data, filename)
    return _parse_txt(data, filename)


def _chunk_text(text: str, target: int, overlap: int) -> list[tuple[int, int, str]]:
    """Split into overlapping chunks at paragraph boundaries when possible.

    Returns list of (char_start, char_end, chunk_text).
    """
    chunks: list[tuple[int, int, str]] = []
    if not text:
        return chunks
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + target, n)
        if end < n:
            # try to back off to a paragraph or sentence boundary
            window = text[pos:end]
            for sep in ("\n\n", "\n", ". ", " "):
                idx = window.rfind(sep)
                if idx > target * 0.5:
                    end = pos + idx + len(sep)
                    break
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append((pos, end, chunk))
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)
    return chunks


def _chapter_for(offset: int, marks: list[tuple[int, str]]) -> str | None:
    current = None
    for off, title in marks:
        if off <= offset:
            current = title
        else:
            break
    return current


def ingest_book(data: bytes, filename: str, content_type: str | None = None) -> str:
    """Parse, chunk, and store a book. Returns book_id."""
    raw = parse_upload(data, filename, content_type)
    book_id = uuid.uuid4().hex[:12]
    chunks = _chunk_text(raw.text, settings.chunk_target_chars, settings.chunk_overlap_chars)

    # persist source for reference
    src_path = settings.data_dir / f"{book_id}_{re.sub(r'[^A-Za-z0-9._-]', '_', filename)}"
    src_path.write_bytes(data)

    with conn_ctx() as conn:
        conn.execute(
            "INSERT INTO books (id, title, author, source_path, char_count) VALUES (?, ?, ?, ?, ?)",
            (book_id, raw.title, raw.author, str(src_path), len(raw.text)),
        )
        rows = []
        for i, (cs, ce, ctext) in enumerate(chunks):
            rows.append(
                (
                    f"{book_id}_{i:05d}",
                    book_id,
                    i,
                    _chapter_for(cs, raw.chapter_marks),
                    cs,
                    ce,
                    ctext,
                )
            )
        conn.executemany(
            "INSERT INTO chunks (id, book_id, idx, chapter, char_start, char_end, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return book_id
