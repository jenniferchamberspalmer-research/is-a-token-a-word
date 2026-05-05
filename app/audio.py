"""STT/TTS adapter interface.

The default 'stub' provider returns predictable placeholder responses so the
multi-modal API surface is exercisable end-to-end without external services.
Plug in real providers (OpenAI Whisper, ElevenLabs, AssemblyAI, etc.) by
implementing the two functions below and selecting the provider via settings.
"""
from __future__ import annotations

import io
import wave
from typing import Protocol

from .config import settings


class STTProvider(Protocol):
    def transcribe(self, audio: bytes, content_type: str | None) -> str: ...


class TTSProvider(Protocol):
    def synthesize(self, text: str, voice: str | None) -> tuple[bytes, str]:
        """Return (audio_bytes, content_type)."""


class _StubSTT:
    def transcribe(self, audio: bytes, content_type: str | None) -> str:
        return f"[stub-stt: received {len(audio)} bytes of {content_type or 'audio'}]"


class _StubTTS:
    """Generates a brief silent WAV so the response is a valid audio file."""

    def synthesize(self, text: str, voice: str | None) -> tuple[bytes, str]:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            # ~0.25s of silence; real provider would synthesize text/voice
            w.writeframes(b"\x00\x00" * 4000)
        return buf.getvalue(), "audio/wav"


def get_stt() -> STTProvider:
    if settings.stt_provider == "stub":
        return _StubSTT()
    raise NotImplementedError(f"STT provider '{settings.stt_provider}' not configured")


def get_tts() -> TTSProvider:
    if settings.tts_provider == "stub":
        return _StubTTS()
    raise NotImplementedError(f"TTS provider '{settings.tts_provider}' not configured")
