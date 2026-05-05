# Book Companion API

A bring-your-own-book RAG service. Upload an EPUB, PDF, or plain-text book and
get a multimodal study companion: chat with citations, voice-in / voice-out,
annotations anchored to text or audio position, on-demand overviews, and
generated information graphics. State persists per-book in SQLite so the
notebook becomes a repository for the reader's thinking with the book.

> Built for non-copyrighted material (e.g. Project Gutenberg). Don't upload
> books you don't have the rights to.

## Architecture at a glance

```
upload (epub/pdf/txt)
    │
    ▼
ingest.py  ──► chunks (1.8K chars, 200 overlap) ──► SQLite
    │
    ▼
retrieval.py  (BM25 + TF fallback for tiny corpora)
    │
    ▼
llm.py  ──►  Anthropic Claude (claude-opus-4-7)
              + prompt caching on full-book context
              + per-turn retrieved excerpts
    │
    ▼
main.py (FastAPI)
    ├── /books, /books/{id}/chunks
    ├── /books/{id}/chat               text chat with citations
    ├── /books/{id}/voice-chat         audio in → transcript → answer → audio out
    ├── /books/{id}/annotations        text + voice notes, anchored to chunk or timestamp
    ├── /books/{id}/overview           brief / detailed / outline
    ├── /books/{id}/graphic            mermaid / svg info graphics
    ├── /books/{id}/notebook           the reader's repository for thinking
    └── /stt, /tts                     audio adapters
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs for the interactive API.

## Quick walkthrough

```bash
# Ingest a public-domain text
curl -F "file=@samples/sample.txt" http://localhost:8000/books
# → {"id":"abc123","title":"sample","char_count":1996,"chunk_count":2}

# Chat
curl -X POST http://localhost:8000/books/abc123/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the relationship between tokens and words?"}'

# Annotate (text or voice; anchor to chunk + char range, or audio timestamp)
curl -X POST http://localhost:8000/books/abc123/annotations \
  -H 'Content-Type: application/json' \
  -d '{"kind":"text","body":"this is the thesis","chunk_id":"abc123_00001"}'

# Generate an overview
curl -X POST http://localhost:8000/books/abc123/overview \
  -H 'Content-Type: application/json' -d '{"style":"outline"}'

# Generate a Mermaid info graphic
curl -X POST http://localhost:8000/books/abc123/graphic \
  -H 'Content-Type: application/json' \
  -d '{"topic":"how subword tokenization changes what the model sees","format":"mermaid"}'

# Voice chat (audio in, audio out)
curl -F "file=@question.wav" http://localhost:8000/books/abc123/voice-chat -o answer.wav
# Headers carry: X-Transcript, X-Answer, X-Citations, X-Session-Id
```

## Audio (STT/TTS)

The `audio` module ships with a `stub` provider so the multimodal endpoints
return valid responses without external services. To plug in real audio:

1. Add a class implementing `STTProvider` or `TTSProvider` in `app/audio.py`
   (e.g. wrapping OpenAI Whisper, ElevenLabs, AssemblyAI, Azure Speech).
2. Register it in `get_stt()` / `get_tts()` behind a provider name.
3. Set `BOOK_STT_PROVIDER=...` / `BOOK_TTS_PROVIDER=...` in your environment.

## Configuration

All settings are environment-variable driven (prefix `BOOK_`):

| Var | Default | Purpose |
| --- | --- | --- |
| `BOOK_DATA_DIR` | `data` | Where uploaded source files live |
| `BOOK_DB_PATH` | `data/books.db` | SQLite database path |
| `BOOK_CHAT_MODEL` | `claude-opus-4-7` | Model used for chat |
| `BOOK_OVERVIEW_MODEL` | `claude-opus-4-7` | Model for overviews/graphics |
| `BOOK_CHUNK_TARGET_CHARS` | `1800` | Target chunk size |
| `BOOK_CHUNK_OVERLAP_CHARS` | `200` | Sliding overlap |
| `BOOK_RETRIEVAL_K` | `6` | Default top-k for retrieval |
| `BOOK_STT_PROVIDER` / `BOOK_TTS_PROVIDER` | `stub` | Audio adapters |
| `ANTHROPIC_API_KEY` | _(unset)_ | Standard Anthropic SDK env var |

## Why prompt caching matters here

For any book longer than a few thousand tokens, every chat turn re-sends the
full text as context. `llm.py` puts the rendered book on a `cache_control:
ephemeral` block so subsequent turns within the cache window cost ~10% of the
first turn's input price. Empirically, this is the difference between an
unusable demo and one you can actually have a conversation with.

## Tests

```bash
python -m unittest tests.test_smoke -v
```

The smoke test ingests, retrieves, annotates, and exercises the audio stubs
without calling Anthropic. LLM-backed endpoints (`/chat`, `/overview`,
`/graphic`) need a real API key.

## What this is and isn't

It is a working scaffold for a book-aware multimodal study companion — Notebook
LM-style features plugged into the listen/pause/speak/annotate loop of an
audiobook.

It is not:
- A distribution channel — you supply the book, you bear the licensing.
- A replacement for embeddings at scale — BM25 is fine for a single book; if
  you grow to thousands of books, swap `retrieval.py` for a vector store.
- A production audio stack — the stub providers exist to keep the API surface
  exercisable; pick a real STT/TTS before shipping.
