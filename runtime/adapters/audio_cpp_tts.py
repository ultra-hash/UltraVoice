"""
UltraVoice adapter for the local audio.cpp OpenAI-compatible TTS server.

Current local topology:
    llama.cpp  -> 127.0.0.1:8080
    Whisper    -> 127.0.0.1:8081
    audio.cpp  -> 127.0.0.1:8082

Current audio.cpp model:
    kokoro-tts

The installed audio.cpp model is currently reported as mode="offline".
Therefore each send_text() generates one WAV response. This adapter strips
the WAV container and exposes raw mono int16 PCM to the existing
live_agent.py playback code.

Install this file as:
    runtime/adapters/audio_cpp_tts.py
"""

from __future__ import annotations

import asyncio
import io
import json
import wave
from typing import Any, Optional

import httpx

try:
    from runtime.config import settings
except Exception:
    settings = None


class AudioCppTTSError(RuntimeError):
    pass


class _EndMarker:
    pass


class _Session:
    def __init__(self, owner: "AudioCppTTS") -> None:
        self._owner = owner

    async def __aenter__(self) -> "_Session":
        self._owner._ensure_worker()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # The live agent explicitly calls end() before leaving the context.
        # Do not enqueue another end marker here because that could terminate
        # a subsequent session unexpectedly.
        return None

    async def send_text(self, text: str) -> None:
        await self._owner._send_text(text)

    async def end(self) -> None:
        await self._owner._end()

    async def receive(self) -> bytes | str:
        return await self._owner._receive()


class AudioCppTTS:
    """Small compatibility layer for live_agent.py."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082",
        model: str = "kokoro-tts",
        sample_rate: int = 24000,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        instruction: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sample_rate = sample_rate
        self.voice = voice
        self.speed = speed
        self.instruction = instruction
        self.timeout = timeout

        self._items: asyncio.Queue[Any] = asyncio.Queue()
        self._requests: asyncio.Queue[Any] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._started = False

    @classmethod
    def from_config(cls) -> "AudioCppTTS":
        cfg: dict[str, Any] = {}

        try:
            tts = getattr(settings, "tts", None)
            if tts is not None:
                if hasattr(tts, "model_dump"):
                    cfg = tts.model_dump()
                elif hasattr(tts, "dict"):
                    cfg = tts.dict()
                elif isinstance(tts, dict):
                    cfg = dict(tts)
                else:
                    for name in (
                        "base_url", "model", "audio_cpp_base_url",
                        "audio_cpp_model", "sample_rate",
                        "audio_cpp_sample_rate", "voice",
                        "voice_id", "speed", "instruction",
                    ):
                        if hasattr(tts, name):
                            cfg[name] = getattr(tts, name)
        except Exception:
            cfg = {}

        return cls(
            base_url=(
                cfg.get("audio_cpp_base_url")
                or "http://127.0.0.1:8082"
            ),
            model=(
                cfg.get("audio_cpp_model")
                or "kokoro-tts"
            ),
            sample_rate=int(
                cfg.get("audio_cpp_sample_rate")
                or 24000
            ),
            voice=cfg.get("voice") or cfg.get("voice_id"),
            speed=cfg.get("speed"),
            instruction=cfg.get("instruction"),
        )

    def persistent_session(self) -> _Session:
        self._ensure_worker()
        return _Session(self)

    def _ensure_worker(self) -> None:
        # Each conversation turn ends the current HTTP worker.  A later
        # persistent_session() must therefore start a fresh worker instead of
        # putting requests onto the completed task's queue.
        if (
            self._worker_task is None
            or self._worker_task.done()
            or self._worker_task.cancelled()
        ):
            self._worker_task = asyncio.create_task(self._worker())

    async def _send_text(self, text: str) -> None:
        self._ensure_worker()
        await self._requests.put(("text", text))

    async def _end(self) -> None:
        self._ensure_worker()
        await self._requests.put(("end", None))

    async def _receive(self) -> bytes | str:
        return await self._items.get()

    async def _worker(self) -> None:
        url = f"{self.base_url}/v1/audio/speech"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                while True:
                    kind, value = await self._requests.get()

                    if kind == "end":
                        await self._items.put(json.dumps({"type": "done"}))
                        return

                    text = str(value).strip()
                    if not text:
                        continue

                    payload: dict[str, Any] = {
                        "model": self.model,
                        "input": text,
                        "response_format": "wav",
                    }

                    if self.voice:
                        payload["voice"] = self.voice

                    if self.speed is not None:
                        payload["speed"] = self.speed

                    # Kokoro in audio.cpp does not accept the OpenAI-style
                    # "instructions" request option. Keep the configured
                    # instruction available on the adapter for compatibility,
                    # but do not send it to the current Kokoro endpoint.
                    try:
                        response = await client.post(url, json=payload)
                    except httpx.RequestError as exc:
                        raise AudioCppTTSError(
                            f"Cannot connect to audio.cpp at {url}: {exc}"
                        ) from exc

                    if response.status_code >= 400:
                        raise AudioCppTTSError(
                            f"audio.cpp HTTP {response.status_code}: "
                            f"{response.text[:2000]}"
                        )

                    pcm = self._wav_to_pcm(response.content)
                    if pcm:
                        await self._items.put(pcm)

        except Exception as exc:
            await self._items.put(
                json.dumps({"type": "error", "message": str(exc)})
            )

    def _wav_to_pcm(self, data: bytes) -> bytes:
        # Raw PCM fallback.
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return data

        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())
        except wave.Error as exc:
            raise AudioCppTTSError(f"Invalid WAV from audio.cpp: {exc}") from exc

        if channels != 1 or sample_width != 2:
            raise AudioCppTTSError(
                f"Unsupported audio.cpp WAV format: "
                f"{channels} channel(s), {sample_width * 8}-bit"
            )

        if rate != self.sample_rate:
            raise AudioCppTTSError(
                f"audio.cpp returned {rate} Hz, but live_agent expects "
                f"{self.sample_rate} Hz"
            )

        return pcm
