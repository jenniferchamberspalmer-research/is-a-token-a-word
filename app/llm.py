"""Claude wrapper. Uses prompt caching on book context for cost-efficient repeated queries."""
from __future__ import annotations

from typing import Iterable

import anthropic

from .config import settings
from .retrieval import Chunk, get_all_chunks, retrieve


SYSTEM_BASE = (
    "You are a study companion for a single book. Answer the reader's questions strictly from "
    "the provided book excerpts. If the answer isn't in the excerpts, say so plainly. "
    "When you make a claim, attach a citation in the form [#<idx>] where <idx> is the chunk index. "
    "Keep answers tight and grounded in the text."
)


def _client() -> anthropic.Anthropic:
    if not settings.anthropic_api_key:
        # SDK will read ANTHROPIC_API_KEY from env if our setting is empty.
        return anthropic.Anthropic()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _format_excerpts(chunks: Iterable[Chunk]) -> str:
    parts = []
    for c in chunks:
        header = f"[#{c.idx}]"
        if c.chapter:
            header += f" ({c.chapter})"
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _book_context_block(book_id: str, max_chars: int = 80_000) -> str:
    """Render a stable book-context prefix for prompt caching.

    Truncates if the book is too long for the cache prefix; the per-turn retrieval
    layer carries the load when this is hit.
    """
    chunks = get_all_chunks(book_id)
    pieces: list[str] = []
    used = 0
    for c in chunks:
        h = f"[#{c.idx}]"
        if c.chapter:
            h += f" ({c.chapter})"
        block = f"{h}\n{c.text}"
        if used + len(block) > max_chars:
            break
        pieces.append(block)
        used += len(block) + 4
    return "\n\n---\n\n".join(pieces)


def chat(
    book_id: str,
    user_message: str,
    history: list[dict] | None = None,
    k: int = 6,
    position_chunk_id: str | None = None,
) -> tuple[str, list[Chunk]]:
    """Answer a user question. Returns (assistant_text, cited_chunks)."""
    history = history or []
    cited = retrieve(book_id, user_message, k=k)

    # If the user has a position, lightly bias context with surrounding chunks.
    extra: list[Chunk] = []
    if position_chunk_id:
        all_chunks = get_all_chunks(book_id)
        for i, c in enumerate(all_chunks):
            if c.id == position_chunk_id:
                extra = all_chunks[max(0, i - 1) : i + 2]
                break
    seen_ids = set()
    merged: list[Chunk] = []
    for c in list(extra) + list(cited):
        if c.id not in seen_ids:
            merged.append(c)
            seen_ids.add(c.id)

    excerpts = _format_excerpts(merged)
    book_prefix = _book_context_block(book_id)

    system = [
        {"type": "text", "text": SYSTEM_BASE},
        {
            "type": "text",
            "text": f"FULL BOOK (for cached recall):\n\n{book_prefix}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    user_turn = (
        f"Relevant excerpts for this question:\n\n{excerpts}\n\n"
        f"Question: {user_message}"
    )
    messages = list(history) + [{"role": "user", "content": user_turn}]

    resp = _client().messages.create(
        model=settings.chat_model,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, merged


def overview(book_id: str, style: str = "brief") -> str:
    book_prefix = _book_context_block(book_id)
    if style == "outline":
        prompt = "Produce a structured outline of this book: top-level sections with 2-4 bullets each."
    elif style == "detailed":
        prompt = "Write a detailed multi-paragraph overview: setup, key arguments or arc, and stakes."
    else:
        prompt = "Write a tight 5-8 sentence overview of this book."

    system = [
        {"type": "text", "text": "You are summarizing a book the reader is working through."},
        {
            "type": "text",
            "text": f"BOOK:\n\n{book_prefix}",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    resp = _client().messages.create(
        model=settings.overview_model,
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def info_graphic(book_id: str, topic: str, fmt: str = "mermaid") -> str:
    book_prefix = _book_context_block(book_id)
    if fmt == "mermaid":
        instr = (
            "Output ONLY a Mermaid diagram inside a ```mermaid fenced code block. "
            "Choose the diagram type (flowchart, mindmap, timeline, classDiagram) that best fits the topic."
        )
    else:
        instr = (
            "Output ONLY a self-contained SVG (no script) that visualizes the topic. "
            "Use simple shapes, text labels, and a viewBox."
        )

    system = [
        {"type": "text", "text": "You produce information graphics that illuminate themes and structures in a book."},
        {
            "type": "text",
            "text": f"BOOK:\n\n{book_prefix}",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    resp = _client().messages.create(
        model=settings.overview_model,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": f"Topic: {topic}\n\n{instr}"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
