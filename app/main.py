"""FastAPI application: bring-your-own-book multimodal study companion."""
from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from . import audio, llm
from .db import conn_ctx, init_db
from .ingest import ingest_book
from .retrieval import get_all_chunks, get_chunk, invalidate
from .schemas import (
    AnnotationCreate,
    AnnotationOut,
    BookOut,
    ChatRequest,
    ChatResponse,
    ChunkRef,
    Citation,
    GraphicRequest,
    NotebookEntry,
    OverviewRequest,
    TTSRequest,
)

app = FastAPI(title="Book Companion API", version="0.1.0")
init_db()


# ---------- books ----------

@app.post("/books", response_model=BookOut, status_code=201)
async def create_book(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    book_id = ingest_book(data, file.filename or "book.txt", file.content_type)
    invalidate(book_id)
    return _load_book(book_id)


@app.get("/books", response_model=list[BookOut])
def list_books():
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT b.id, b.title, b.author, b.char_count, "
            "(SELECT COUNT(*) FROM chunks c WHERE c.book_id = b.id) AS chunk_count "
            "FROM books b ORDER BY b.created_at DESC"
        ).fetchall()
    return [BookOut(**dict(r)) for r in rows]


@app.get("/books/{book_id}", response_model=BookOut)
def get_book(book_id: str):
    return _load_book(book_id)


@app.delete("/books/{book_id}", status_code=204)
def delete_book(book_id: str):
    with conn_ctx() as conn:
        cur = conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "book not found")
    invalidate(book_id)
    return Response(status_code=204)


@app.get("/books/{book_id}/chunks", response_model=list[ChunkRef])
def list_chunks(book_id: str, limit: int = 100, offset: int = 0):
    chunks = get_all_chunks(book_id)[offset : offset + limit]
    return [
        ChunkRef(
            chunk_id=c.id,
            idx=c.idx,
            chapter=c.chapter,
            char_start=c.char_start,
            char_end=c.char_end,
            snippet=c.text[:240],
        )
        for c in chunks
    ]


def _load_book(book_id: str) -> BookOut:
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT b.id, b.title, b.author, b.char_count, "
            "(SELECT COUNT(*) FROM chunks c WHERE c.book_id = b.id) AS chunk_count "
            "FROM books b WHERE b.id = ?",
            (book_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "book not found")
    return BookOut(**dict(row))


# ---------- chat ----------

@app.post("/books/{book_id}/chat", response_model=ChatResponse)
def chat(book_id: str, req: ChatRequest):
    _load_book(book_id)
    session_id = req.session_id or _new_session(book_id)
    history = _history(session_id)

    text, cited = llm.chat(
        book_id=book_id,
        user_message=req.message,
        history=history,
        k=req.k,
        position_chunk_id=req.position_chunk_id,
    )

    citations = [
        Citation(chunk_id=c.id, idx=c.idx, snippet=c.text[:240]) for c in cited
    ]
    _append_message(session_id, "user", req.message, [])
    _append_message(session_id, "assistant", text, citations)
    return ChatResponse(session_id=session_id, message=text, citations=citations)


def _new_session(book_id: str) -> str:
    sid = uuid.uuid4().hex[:12]
    with conn_ctx() as conn:
        conn.execute("INSERT INTO sessions (id, book_id) VALUES (?, ?)", (sid, book_id))
    return sid


def _history(session_id: str) -> list[dict]:
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _append_message(session_id: str, role: str, content: str, citations: list[Citation]) -> None:
    with conn_ctx() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, citations) VALUES (?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex[:12],
                session_id,
                role,
                content,
                json.dumps([c.model_dump() for c in citations]) if citations else None,
            ),
        )


# ---------- annotations ----------

@app.post("/books/{book_id}/annotations", response_model=AnnotationOut, status_code=201)
def create_annotation(book_id: str, ann: AnnotationCreate):
    _load_book(book_id)
    anchor: dict = {}
    if ann.chunk_id:
        anchor["chunk_id"] = ann.chunk_id
        if ann.char_start is not None:
            anchor["char_start"] = ann.char_start
        if ann.char_end is not None:
            anchor["char_end"] = ann.char_end
    if ann.timestamp_s is not None:
        anchor["timestamp_s"] = ann.timestamp_s

    aid = uuid.uuid4().hex[:12]
    with conn_ctx() as conn:
        conn.execute(
            "INSERT INTO annotations (id, book_id, kind, anchor, body) VALUES (?, ?, ?, ?, ?)",
            (aid, book_id, ann.kind, json.dumps(anchor), ann.body),
        )
        row = conn.execute("SELECT * FROM annotations WHERE id = ?", (aid,)).fetchone()
    return _row_to_annotation(row)


@app.get("/books/{book_id}/annotations", response_model=list[AnnotationOut])
def list_annotations(book_id: str):
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT * FROM annotations WHERE book_id = ? ORDER BY created_at",
            (book_id,),
        ).fetchall()
    return [_row_to_annotation(r) for r in rows]


@app.delete("/books/{book_id}/annotations/{annotation_id}", status_code=204)
def delete_annotation(book_id: str, annotation_id: str):
    with conn_ctx() as conn:
        cur = conn.execute(
            "DELETE FROM annotations WHERE id = ? AND book_id = ?",
            (annotation_id, book_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "annotation not found")
    return Response(status_code=204)


def _row_to_annotation(row) -> AnnotationOut:
    return AnnotationOut(
        id=row["id"],
        book_id=row["book_id"],
        kind=row["kind"],
        anchor=json.loads(row["anchor"]),
        body=row["body"],
        created_at=row["created_at"],
    )


# ---------- overview / graphic / notebook ----------

@app.post("/books/{book_id}/overview", response_model=NotebookEntry)
def overview(book_id: str, req: OverviewRequest):
    _load_book(book_id)
    body = llm.overview(book_id, style=req.style)
    return _save_notebook(book_id, "overview", title=f"Overview ({req.style})", body=body, meta={"style": req.style})


@app.post("/books/{book_id}/graphic", response_model=NotebookEntry)
def graphic(book_id: str, req: GraphicRequest):
    _load_book(book_id)
    body = llm.info_graphic(book_id, topic=req.topic, fmt=req.format)
    return _save_notebook(book_id, "graphic", title=req.topic, body=body, meta={"format": req.format})


@app.get("/books/{book_id}/notebook", response_model=list[NotebookEntry])
def notebook(book_id: str, kind: Optional[str] = None):
    with conn_ctx() as conn:
        if kind:
            rows = conn.execute(
                "SELECT * FROM notebook_entries WHERE book_id = ? AND kind = ? ORDER BY created_at DESC",
                (book_id, kind),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notebook_entries WHERE book_id = ? ORDER BY created_at DESC",
                (book_id,),
            ).fetchall()
    return [_row_to_notebook(r) for r in rows]


@app.post("/books/{book_id}/notebook", response_model=NotebookEntry, status_code=201)
def add_note(book_id: str, entry: NotebookEntry):
    _load_book(book_id)
    return _save_notebook(book_id, "note", title=entry.title, body=entry.body, meta=entry.meta or {})


def _save_notebook(book_id: str, kind: str, title: str | None, body: str, meta: dict) -> NotebookEntry:
    nid = uuid.uuid4().hex[:12]
    with conn_ctx() as conn:
        conn.execute(
            "INSERT INTO notebook_entries (id, book_id, kind, title, body, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (nid, book_id, kind, title, body, json.dumps(meta) if meta else None),
        )
        row = conn.execute("SELECT * FROM notebook_entries WHERE id = ?", (nid,)).fetchone()
    return _row_to_notebook(row)


def _row_to_notebook(row) -> NotebookEntry:
    return NotebookEntry(
        id=row["id"],
        book_id=row["book_id"],
        kind=row["kind"],
        title=row["title"],
        body=row["body"],
        meta=json.loads(row["meta"]) if row["meta"] else None,
        created_at=row["created_at"],
    )


# ---------- audio ----------

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    data = await file.read()
    text = audio.get_stt().transcribe(data, file.content_type)
    return JSONResponse({"text": text})


@app.post("/tts")
def tts(req: TTSRequest):
    data, ctype = audio.get_tts().synthesize(req.text, req.voice)
    return Response(content=data, media_type=ctype)


@app.post("/books/{book_id}/voice-chat")
async def voice_chat(
    book_id: str,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    k: int = Form(default=6),
    position_chunk_id: Optional[str] = Form(default=None),
):
    """Listen → transcribe → answer → speak. Returns audio with answer in headers."""
    _load_book(book_id)
    audio_bytes = await file.read()
    transcript = audio.get_stt().transcribe(audio_bytes, file.content_type)

    sid = session_id or _new_session(book_id)
    history = _history(sid)
    text, cited = llm.chat(
        book_id=book_id,
        user_message=transcript,
        history=history,
        k=k,
        position_chunk_id=position_chunk_id,
    )
    citations = [Citation(chunk_id=c.id, idx=c.idx, snippet=c.text[:240]) for c in cited]
    _append_message(sid, "user", transcript, [])
    _append_message(sid, "assistant", text, citations)

    audio_out, ctype = audio.get_tts().synthesize(text, voice=None)
    headers = {
        "X-Session-Id": sid,
        "X-Transcript": _safe_header(transcript),
        "X-Answer": _safe_header(text),
        "X-Citations": json.dumps([c.model_dump() for c in citations])[:4096],
    }
    return Response(content=audio_out, media_type=ctype, headers=headers)


def _safe_header(s: str) -> str:
    # HTTP headers must be latin-1 safe and reasonably short
    return s.encode("ascii", errors="replace").decode("ascii")[:1024]


@app.get("/healthz")
def health():
    return {"ok": True}
