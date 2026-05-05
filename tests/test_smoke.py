"""Smoke tests that exercise ingestion, retrieval, and non-LLM endpoints.

These do not call the Anthropic API; the LLM-backed endpoints are exercised
manually with a real key, or wired up with mocks if the team adds them later.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Use a temp DB so the test does not touch real data
TEST_DIR = Path(tempfile.mkdtemp(prefix="bookapp_test_"))
os.environ["BOOK_DATA_DIR"] = str(TEST_DIR)
os.environ["BOOK_DB_PATH"] = str(TEST_DIR / "books.db")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.retrieval import retrieve  # noqa: E402


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        sample = Path(__file__).resolve().parents[1] / "samples" / "sample.txt"
        cls.sample_bytes = sample.read_bytes()

    def test_full_flow(self):
        # ingest
        resp = self.client.post(
            "/books",
            files={"file": ("sample.txt", self.sample_bytes, "text/plain")},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        book = resp.json()
        self.assertGreater(book["chunk_count"], 0)
        book_id = book["id"]

        # list
        resp = self.client.get("/books")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(any(b["id"] == book_id for b in resp.json()))

        # chunks endpoint
        resp = self.client.get(f"/books/{book_id}/chunks")
        self.assertEqual(resp.status_code, 200)
        chunks = resp.json()
        self.assertGreater(len(chunks), 0)

        # retrieval finds the right chunk for a query
        hits = retrieve(book_id, "byte-pair encoding", k=3)
        self.assertTrue(any("byte" in h.text.lower() for h in hits))

        # annotation create + list
        resp = self.client.post(
            f"/books/{book_id}/annotations",
            json={
                "kind": "text",
                "body": "Worth re-reading - the word/token gap is the thesis.",
                "chunk_id": chunks[0]["chunk_id"],
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        ann_id = resp.json()["id"]

        resp = self.client.get(f"/books/{book_id}/annotations")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(any(a["id"] == ann_id for a in resp.json()))

        # tts returns audio bytes
        resp = self.client.post("/tts", json={"text": "hello"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/wav")
        self.assertGreater(len(resp.content), 100)

        # stt returns a string
        resp = self.client.post(
            "/stt",
            files={"file": ("clip.wav", b"\x00\x01\x02\x03", "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text", resp.json())

        # delete
        resp = self.client.delete(f"/books/{book_id}")
        self.assertEqual(resp.status_code, 204)


if __name__ == "__main__":
    unittest.main()
