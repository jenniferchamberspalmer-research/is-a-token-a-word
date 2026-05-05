"""BM25 retrieval over a book's chunks, with a small in-process cache."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import NamedTuple

from rank_bm25 import BM25Okapi

from .db import conn_ctx


class Chunk(NamedTuple):
    id: str
    idx: int
    chapter: str | None
    char_start: int
    char_end: int
    text: str


_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s)]


class _BookIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self._bm25 or not self.chunks:
            return []
        q = _tokenize(query)
        scores = list(self._bm25.get_scores(q))
        # BM25 IDF can flatten to 0 on tiny corpora; fall back to term-frequency overlap
        if max(scores, default=0) <= 0:
            qset = set(q)
            scores = [sum(1 for t in _tokenize(c.text) if t in qset) for c in self.chunks]
        ranked = sorted(zip(self.chunks, scores), key=lambda t: t[1], reverse=True)
        return [(c, s) for c, s in ranked[:k] if s > 0]


@lru_cache(maxsize=32)
def _index_for(book_id: str) -> _BookIndex:
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT id, idx, chapter, char_start, char_end, text FROM chunks WHERE book_id = ? ORDER BY idx",
            (book_id,),
        ).fetchall()
    chunks = [Chunk(r["id"], r["idx"], r["chapter"], r["char_start"], r["char_end"], r["text"]) for r in rows]
    return _BookIndex(chunks)


def invalidate(book_id: str) -> None:
    _index_for.cache_clear()


def retrieve(book_id: str, query: str, k: int = 6) -> list[Chunk]:
    return [c for c, _ in _index_for(book_id).search(query, k)]


def get_all_chunks(book_id: str) -> list[Chunk]:
    return _index_for(book_id).chunks


def get_chunk(book_id: str, chunk_id: str) -> Chunk | None:
    for c in _index_for(book_id).chunks:
        if c.id == chunk_id:
            return c
    return None
