"""
Breeze TTS adapter — text-to-speech generation.

Supports two modes:
  1. HTTP streaming  (POST /v1/audio/speech) — simple, one-shot
  2. WebSocket       — streaming text, instruction changes, cancellation

Breeze-TTS-2 generates speech in four stages:
  1. T5Gemma2 text encoder → conditioning sequence
  2. Qwen3 backbone → codebook 0 at 12.5 fps
  3. Depth decoder → codebooks 1-15
  4. Vocoder → 24 kHz waveform (s16le PCM)

Three generation modes (auto-detected from reference):
  - Voice design   (no reference, driven by instruction)
  - Voice clone    (reference audio provided)
  - Voice direction (reference + delivery instruction)
"""

from __future__ import annotations

import asyncio
import io
import json
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import httpx
import websockets


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class BreezeVoice(str, Enum):
    """Available quantisation presets — maps to --voice CLI flag."""

    HARBOUR = "harbour"
    YOUNG_WOMAN = "young_woman"
    # Add more as you register saved voices via POST /v1/voices


@dataclass(frozen=True)
class SynthesisResult:
    """Single synthesis output."""

    pcm: bytes                                    # raw s16le PCM
    sample_rate: int = 24000
    duration_secs: float = 0.0
    text: str = ""
    voice_id: str = ""
    frames: int = 0                               # 1 frame = 80 ms = 1920 samples


@dataclass
class VoiceInfo:
    """Metadata about a registered/saved voice."""

    id: str
    seconds: float
    frames: int
    encode_ms: float = 0.0
    saved: bool = False
    ref_text: str = ""


# ---------------------------------------------------------------------------
# HTTP Adapter (one-shot speech generation)
# ---------------------------------------------------------------------------


class BreezeHTTPAdapter:
    """Talks to breeze-server via HTTP POST /v1/audio/speech.

    Streams PCM chunks as they are generated.
    """

    def __init__(
        self,
        base_url: str,
        instruction: str = "Speak clearly and naturally.",
        cfg_scale: float = 1.0,
        seed: int = 42,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        split_chars: int = 600,
        max_new_tokens: int = 750,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._instruction = instruction
        self._cfg_scale = cfg_scale
        self._seed = seed
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._split_chars = split_chars
        self._max_new_tokens = max_new_tokens
        self._timeout = timeout

    def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        ref_audio: np.ndarray | bytes | None = None,
        ref_text: str | None = None,
        instruction: str | None = None,
    ) -> SynthesisResult:
        """Generate speech and return raw PCM.

        Args:
            text: Text to speak.
            voice_id: Saved voice to clone (from POST /v1/voices).
            ref_audio: Reference WAV for cloning (numpy array or WAV bytes).
            ref_text: Exact transcript of ref_audio (required if ref_audio provided).
            instruction: Override delivery direction.

        Returns:
            SynthesisResult with raw s16le PCM.
        """
        url = f"{self._base_url}/v1/audio/speech"
        instruction = instruction or self._instruction

        # Build multipart form data
        data: dict[str, Any] = {
            "text": text,
            "instruction": instruction,
            "cfg_scale": str(self._cfg_scale),
            "seed": str(self._seed),
            "split_chars": str(self._split_chars),
            "max_new_tokens": str(self._max_new_tokens),
        }

        if self._temperature is not None:
            data["temperature"] = str(self._temperature)
        if self._top_k is not None:
            data["top_k"] = str(self._top_k)
        if self._top_p is not None:
            data["top_p"] = str(self._top_p)
        if self._repetition_penalty is not None:
            data["repetition_penalty"] = str(self._repetition_penalty)

        if voice_id:
            data["voice_id"] = voice_id

        files: dict[str, Any] = {}
        if ref_audio is not None:
            wav_bytes = self._pcm_to_wav(ref_audio)
            files["ref_audio"] = ("reference.wav", wav_bytes, "audio/wav")
            if ref_text is None:
                raise ValueError("ref_text is required when ref_audio is provided")
            data["ref_text"] = ref_text

        resp = httpx.post(
            url,
            data=data,
            files=files if files else None,
            timeout=self._timeout,
        )

        if resp.status_code == 409:
            raise RuntimeError("Server busy — another generation is running")
        if resp.status_code == 404:
            raise RuntimeError(f"Unknown voice_id: {voice_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"Server error {resp.status_code}: {resp.text[:200]}")

        pcm = resp.content
        sample_rate = int(resp.headers.get("X-Sample-Rate", "24000"))
        duration = len(pcm) / 2 / sample_rate if sample_rate > 0 else 0.0

        return SynthesisResult(
            pcm=pcm,
            sample_rate=sample_rate,
            duration_secs=round(duration, 3),
            text=text,
            voice_id=voice_id or "",
        )

    def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream PCM chunks as they are generated.

        Yields raw s16le PCM chunks (~2s each).
        """
        url = f"{self._base_url}/v1/audio/speech"

        data: dict[str, Any] = {
            "text": text,
            "instruction": self._instruction,
            "cfg_scale": str(self._cfg_scale),
            "seed": str(self._seed),
            "split_chars": str(self._split_chars),
            "max_new_tokens": str(self._max_new_tokens),
        }
        if voice_id:
            data["voice_id"] = voice_id

        with httpx.stream(
            "POST", url, data=data, timeout=self._timeout
        ) as resp:
            resp.raise_for_status()
            tail = b""
            for chunk in resp.iter_bytes(chunk_size=8192):
                chunk = tail + chunk
                n = len(chunk) // 2
                tail = chunk[n * 2:]
                if n > 0:
                    yield chunk[: n * 2]
            if tail:
                yield tail

    async def synthesize_stream_async(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Asynchronously stream raw s16le PCM as Breeze generates it.

        This is the preferred path for the live voice agent because it does
        not block the asyncio event loop while waiting for Breeze output.
        """
        url = f"{self._base_url}/v1/audio/speech"

        data: dict[str, Any] = {
            "text": text,
            "instruction": self._instruction,
            "cfg_scale": str(self._cfg_scale),
            "seed": str(self._seed),
            "split_chars": str(self._split_chars),
            "max_new_tokens": str(self._max_new_tokens),
        }

        if self._temperature is not None:
            data["temperature"] = str(self._temperature)
        if self._top_k is not None:
            data["top_k"] = str(self._top_k)
        if self._top_p is not None:
            data["top_p"] = str(self._top_p)
        if self._repetition_penalty is not None:
            data["repetition_penalty"] = str(self._repetition_penalty)
        if voice_id:
            data["voice_id"] = voice_id

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, data=data) as resp:
                if resp.status_code == 409:
                    raise RuntimeError(
                        "Server busy — another generation is running"
                    )
                if resp.status_code == 404:
                    raise RuntimeError(f"Unknown voice_id: {voice_id}")
                resp.raise_for_status()

                tail = b""
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    chunk = tail + chunk
                    n = len(chunk) // 2
                    tail = chunk[n * 2:]
                    if n > 0:
                        yield chunk[: n * 2]

                if tail:
                    yield tail

    def _pcm_to_wav(self, audio: np.ndarray | bytes) -> bytes:
        """Convert PCM numpy array or bytes to WAV bytes."""
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)
            buf = io.BytesIO()
            sf.write(buf, audio, 24000, format="WAV")
            return buf.getvalue()
        # Already bytes — assume WAV
        return audio if isinstance(audio, bytes) else bytes(audio)


# ---------------------------------------------------------------------------
# WebSocket Adapter (full session with streaming)
# ---------------------------------------------------------------------------


class BreezeWSAdapter:
    """Full WebSocket session for incremental text, direction changes, cancellation.

    Messages sent:
      start    → open session
      text     → add text (drained on sentence boundaries)
      flush    → speak buffered text now
      instruction → change delivery from next piece
      cancel   → interrupt, clear buffer
      end      → speak remaining, report done

    Messages received:
      ready    → connection up, send start
      started  → session open
      speaking → about to generate this piece
      queued   → waiting for GPU
      done     → everything spoken
      error    → something wrong
    """

    def __init__(
        self,
        base_url: str,
        ws_port: int = 0,  # 0 = auto-discover from /health
        instruction: str = "Speak clearly and naturally.",
        cfg_scale: float = 1.0,
        seed: int = 42,
        temperature: float | None = None,
        voice_id: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._ws_port = ws_port
        self._instruction = instruction
        self._cfg_scale = cfg_scale
        self._seed = seed
        self._temperature = temperature
        self._voice_id = voice_id

    async def __aenter__(self) -> "BreezeWSAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        """Connect to WebSocket, discover port, send start."""
        if self._ws_port == 0:
            self._ws_port = await self._discover_ws_port()

        ws_url = f"ws://{self._base_url.split('://')[-1].split(':')[0]}:{self._ws_port}"
        self._ws = await websockets.connect(ws_url, max_size=None)

        # Wait for ready message
        msg = await self._ws.recv()
        ready = json.loads(msg)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Unexpected ready message: {ready}")

        # Send start
        start_msg: dict[str, Any] = {
            "type": "start",
            "cfg_scale": self._cfg_scale,
            "seed": self._seed,
        }
        if self._voice_id:
            start_msg["voice_id"] = self._voice_id
        if self._temperature is not None:
            start_msg["temperature"] = self._temperature

        await self._ws.send(json.dumps(start_msg))

        started = await self._ws.recv()
        started_data = json.loads(started)
        if started_data.get("type") != "started":
            raise RuntimeError(f"Unexpected started message: {started_data}")

    async def close(self) -> None:
        """Cancel and close the session."""
        try:
            await self._ws.send(json.dumps({"type": "cancel"}))
        except Exception:
            pass
        try:
            await self._ws.close()
        except Exception:
            pass

    async def speak(
        self,
        text: str,
        instruction: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Speak text and yield PCM chunks.

        Args:
            text: Text to speak.
            instruction: Override delivery direction (applies from next piece).
        """
        # Send instruction change if provided
        if instruction:
            await self._ws.send(json.dumps({
                "type": "instruction",
                "instruction": instruction,
            }))
            # Wait for confirmation
            while True:
                msg = await self._ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "instruction_set":
                        break
                    elif data.get("type") == "speaking":
                        continue

        # Send text
        await self._ws.send(json.dumps({
            "type": "text",
            "text": text,
        }))

        # Receive audio chunks until done
        pcm_chunks: list[bytes] = []
        while True:
            msg = await self._ws.recv()

            if isinstance(msg, bytes):
                pcm_chunks.append(msg)
                yield msg
            else:
                data = json.loads(msg)
                if data.get("type") == "done":
                    break
                elif data.get("type") == "error":
                    raise RuntimeError(data.get("message", "Unknown error"))
                elif data.get("type") == "speaking":
                    pass  # informational

    async def speak_full(
        self,
        text: str,
        instruction: str | None = None,
    ) -> SynthesisResult:
        """Speak text, collect all chunks, return SynthesisResult."""
        pcm_chunks: list[bytes] = []

        async for chunk in self.speak(text, instruction):
            pcm_chunks.append(chunk)

        pcm = b"".join(pcm_chunks)
        sample_rate = 24000
        duration = len(pcm) / 2 / sample_rate if pcm else 0.0

        return SynthesisResult(
            pcm=pcm,
            sample_rate=sample_rate,
            duration_secs=round(duration, 3),
            text=text,
            voice_id=self._voice_id or "",
        )

    async def cancel(self) -> None:
        """Interrupt current generation, clear buffer."""
        await self._ws.send(json.dumps({"type": "cancel"}))
        while True:
            msg = await self._ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "cancelled":
                    break

    async def _discover_ws_port(self) -> int:
        """Query /health to find the WebSocket port."""
        try:
            resp = httpx.get(f"{self._base_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json().get("ws_port", 0)
        except Exception:
            # Default: HTTP port + 1
            parts = self._base_url.split(":")
            port = int(parts[-1]) if parts[-1].isdigit() else 8080
            return port + 1


# ---------------------------------------------------------------------------
# Voice Management
# ---------------------------------------------------------------------------


class BreezeVoiceManager:
    """Manage saved and cached voices via HTTP API."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def save_voice(
        self,
        name: str,
        ref_audio: np.ndarray | bytes,
        ref_text: str,
    ) -> VoiceInfo:
        """Encode reference audio and save as a reusable voice.

        Args:
            name: Voice name (letters, digits, dash, underscore, max 64 chars).
            ref_audio: Reference WAV (numpy array or WAV bytes).
            ref_text: Exact transcript of the reference audio.

        Returns:
            VoiceInfo with the saved voice metadata.
        """
        wav_bytes = self._pcm_to_wav(ref_audio) if isinstance(ref_audio, np.ndarray) else ref_audio

        resp = httpx.post(
            f"{self._base_url}/v1/voices",
            files={"ref_audio": ("reference.wav", wav_bytes, "audio/wav")},
            data={"ref_text": ref_text, "name": name},
            timeout=self._timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        return VoiceInfo(
            id=data.get("id", name),
            seconds=data.get("seconds", 0),
            frames=data.get("frames", 0),
            encode_ms=data.get("encode_ms", 0),
            saved=data.get("saved", True),
            ref_text=data.get("ref_text", ref_text),
        )

    def list_voices(self) -> list[VoiceInfo]:
        """List all cached and saved voices."""
        resp = httpx.get(f"{self._base_url}/v1/voices", timeout=5)
        resp.raise_for_status()

        voices = resp.json()
        if isinstance(voices, dict):
            voices = voices.get("voices", [])
        return [
            VoiceInfo(
                id=v.get("id", ""),
                seconds=v.get("seconds", 0),
                frames=v.get("frames", 0),
                encode_ms=v.get("encode_ms", 0),
                saved=v.get("saved", False),
                ref_text=v.get("ref_text", ""),
            )
            for v in voices
        ]

    def delete_voice(self, voice_id: str) -> dict[str, Any]:
        """Remove a voice from memory (file kept for saved voices)."""
        resp = httpx.delete(
            f"{self._base_url}/v1/voices/{voice_id}",
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()

    def check_health(self) -> dict[str, Any]:
        """Check server health and get sample rate + WS port."""
        resp = httpx.get(f"{self._base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def _pcm_to_wav(self, audio: np.ndarray) -> bytes:
        """Convert PCM array to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Unified Breeze TTS Class (factory)
# ---------------------------------------------------------------------------


class BreezeTTS:
    """High-level TTS adapter for the voice-agent pipeline.

    Usage:
        tts = BreezeTTS.from_config()           # reads config.yaml
        tts = BreezeTTS(mode="http")            # HTTP one-shot
        tts = BreezeTTS(mode="websocket")       # full session

        result = tts.synthesize("Hello world")
        play(result.pcm, sample_rate=24000)
    """

    def __init__(
        self,
        mode: str = "http",
        http_adapter: BreezeHTTPAdapter | None = None,
        ws_adapter: BreezeWSAdapter | None = None,
        voice_manager: BreezeVoiceManager | None = None,
    ) -> None:
        self._mode = mode
        self._http = http_adapter
        self._ws = ws_adapter
        self._voices = voice_manager

    # ---- Factory methods --------------------------------------------------

    @classmethod
    def from_config(cls, mode: str = "auto") -> "BreezeTTS":
        """Create from config.yaml (tts section)."""
        from runtime.config import TTSConfig

        tts_cfg = settings.tts  # type: ignore[attr-defined]

        # Handle both dict-style and dataclass-style configs
        if isinstance(tts_cfg, dict):
            base_url = tts_cfg.get("base_url", "http://127.0.0.1:8080")
            instruction = tts_cfg.get("instruction", "Speak clearly and naturally.")
            voice_id = tts_cfg.get("voice_id")
            cfg_scale = tts_cfg.get("cfg_scale", 1.0)
        else:
            base_url = getattr(tts_cfg, "base_url", "http://127.0.0.1:8080")
            instruction = getattr(tts_cfg, "instruction", "Speak clearly and naturally.")
            voice_id = getattr(tts_cfg, "voice_id", None)
            cfg_scale = getattr(tts_cfg, "cfg_scale", 1.0)

        if mode == "auto":
            # Use websocket for conversational, http for simple
            mode = "websocket" if voice_id or instruction != "Speak clearly and naturally." else "http"

        if mode == "http":
            http_adapter = BreezeHTTPAdapter(
                base_url=base_url,
                instruction=instruction,
                cfg_scale=cfg_scale,
            )
            return cls(mode="http", http_adapter=http_adapter)

        else:
            ws_adapter = BreezeWSAdapter(
                base_url=base_url,
                instruction=instruction,
                cfg_scale=cfg_scale,
                voice_id=voice_id,
            )
            voice_manager = BreezeVoiceManager(base_url=base_url)
            return cls(
                mode="websocket",
                ws_adapter=ws_adapter,
                voice_manager=voice_manager,
            )

    # ---- Public API (HTTP mode) -------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        ref_audio: np.ndarray | bytes | None = None,
        ref_text: str | None = None,
        instruction: str | None = None,
    ) -> SynthesisResult:
        """Generate speech (HTTP mode).

        Args:
            text: Text to speak.
            voice_id: Saved voice to clone.
            ref_audio: Reference WAV for cloning.
            ref_text: Transcript of ref_audio.
            instruction: Override delivery direction.
        """
        if self._mode == "http" and self._http:
            return self._http.synthesize(
                text, voice_id, ref_audio, ref_text, instruction
            )
        elif self._mode == "websocket" and self._ws:
            return asyncio.run(
                self._ws.speak_full(text, instruction)
            )
        raise RuntimeError("No adapter configured")

    def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream PCM chunks (HTTP mode)."""
        if self._mode == "http" and self._http:
            return self._http.synthesize_stream(text, voice_id)
        raise RuntimeError("HTTP streaming not available in websocket mode")

    async def synthesize_stream_async(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream PCM asynchronously in HTTP mode."""
        if self._mode == "http" and self._http:
            async for chunk in self._http.synthesize_stream_async(text, voice_id):
                yield chunk
            return

        raise RuntimeError(
            "Async HTTP streaming is only available in HTTP mode"
        )

    # ---- Public API (WebSocket mode) --------------------------------------

    async def speak(
        self,
        text: str,
        instruction: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream PCM chunks with session management."""
        if self._mode == "websocket" and self._ws:
            async with self._ws:
                async for chunk in self._ws.speak(text, instruction):
                    yield chunk

    async def speak_full(
        self,
        text: str,
        instruction: str | None = None,
    ) -> SynthesisResult:
        """Full utterance with session."""
        if self._mode == "websocket" and self._ws:
            async with self._ws:
                return await self._ws.speak_full(text, instruction)
        raise RuntimeError("WebSocket mode required")

    async def cancel(self) -> None:
        """Interrupt current generation."""
        if self._mode == "websocket" and self._ws:
            await self._ws.cancel()

    # ---- Voice management -------------------------------------------------

    def save_voice(
        self,
        name: str,
        ref_audio: np.ndarray | bytes,
        ref_text: str,
    ) -> VoiceInfo:
        """Save a reference voice for reuse."""
        if self._voices:
            return self._voices.save_voice(name, ref_audio, ref_text)
        raise RuntimeError("Voice manager not available")

    def list_voices(self) -> list[VoiceInfo]:
        """List all registered voices."""
        if self._voices:
            return self._voices.list_voices()
        return []

    # ---- Playback helpers -------------------------------------------------

    @staticmethod
    def play(result: SynthesisResult, blocking: bool = True) -> None:
        """Play PCM result via sounddevice."""
        audio = np.frombuffer(result.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=result.sample_rate, blocking=blocking)
        if blocking:
            sd.wait()

    @staticmethod
    def save_wav(result: SynthesisResult, path: str | Path) -> None:
        """Save PCM result as WAV file."""
        audio = np.frombuffer(result.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(str(path), audio, result.sample_rate)

    # ---- Context manager for WS mode --------------------------------------

    def __enter__(self) -> "BreezeTTS":
        if self._mode == "websocket" and self._ws:
            asyncio.get_event_loop().run_until_complete(self._ws.connect())
        return self

    def __exit__(self, *args: object) -> None:
        if self._mode == "websocket" and self._ws:
            asyncio.get_event_loop().run_until_complete(self._ws.close())


# Import settings after class definitions to avoid circular imports
from runtime.config import settings  # noqa: E402, isort: skip
