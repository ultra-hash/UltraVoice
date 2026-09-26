"""
Whisper Speech-to-Text adapter.

Supports two modes:
  1. OpenAI-compatible API  (remote)
  2. Local Whisper.cpp / faster-whisper (on-device)

Designed for real-time voice agents: processes 20-64ms audio
chunks and returns transcriptions with word-level timestamps
when available.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import httpx

from runtime.config import settings


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class WhisperModel(str, Enum):
    """Whisper model sizes — smallest to largest."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


@dataclass(frozen=True)
class TranscriptionResult:
    """Single transcription output."""

    text: str
    language: str = "en"
    duration_secs: float = 0.0
    start_ms: int = 0
    end_ms: int = 0
    words: list[dict] = field(default_factory=list)  # [{word, start, end}]
    confidence: float = 0.0
    no_speech_prob: float = 0.0


# ---------------------------------------------------------------------------
# Local Whisper (faster-whisper) — optional, falls through gracefully
# ---------------------------------------------------------------------------


class LocalWhisperAdapter:
    """Runs Whisper locally via faster-whisper.

    pip install faster-whisper
    """

    def __init__(
        self,
        model_size: WhisperModel = WhisperModel.BASE,
        device: str = "auto",
        compute_type: str = "float16",
        beam_size: int = 5,
        language: str = "en",
        task: str = "transcribe",
    ) -> None:
        try:
            from faster_whisper import WhisperModel as FWModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required for local mode. "
                "pip install faster-whisper"
            ) from exc

        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._language = language
        self._task = task

        device_id = None if device == "auto" else int(device)
        self._model = FWModel(
            model_size.value,
            device=device,
            device_index=device_id,
            compute_type=compute_type,
            cpu_threads=max(1, __import__("os").cpu_count() // 2),
            num_workers=1,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe PCM audio array."""
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio, sample_rate, format="WAV")
        wav_buf.seek(0)

        segments, info = self._model.transcribe(
            wav_buf,
            beam_size=self._beam_size,
            language=self._language if self._language else None,
            task=self._task,
            initial_prompt=initial_prompt,
            vad_filter=True,
        )

        text_parts = []
        words = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": int(w.start * 1000),
                        "end": int(w.end * 1000),
                    })

        full_text = " ".join(text_parts).strip()
        duration = audio.size / sample_rate if audio.size > 0 else 0.0

        return TranscriptionResult(
            text=full_text,
            language=info.language or "en",
            duration_secs=round(duration, 3),
            words=words,
            confidence=round(seg.scores[0] if (seg := next(segments, None)) else 0.0, 3),
        )


# ---------------------------------------------------------------------------
# Remote API Adapter (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


class RemoteWhisperAdapter:
    """Talks to a Whisper-compatible API server.

    Supports both OpenAI's /v1/audio/transcriptions and
    whisper.cpp /api/transcribe endpoints.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: WhisperModel = WhisperModel.LARGE_V3,
        language: str = "en",
        task: str = "transcribe",
        temperature: float = 0.0,
        response_format: str = "json",
        timeout: float = 30.0,
        endpoint: str = "/inference",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model.value
        self._language = language
        self._task = task
        self._temperature = temperature
        self._response_format = response_format
        self._timeout = timeout
        self._endpoint = endpoint

    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert PCM numpy array to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        return buf.getvalue()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe via HTTP POST to Whisper API."""
        wav_bytes = self._audio_to_wav_bytes(audio, sample_rate)

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self._model,
            "language": self._language,
            "task": self._task,
            "response_format": self._response_format,
            "temperature": str(self._temperature),
        }
        if initial_prompt:
            data["initial_prompt"] = initial_prompt

        resp = httpx.post(
            f"{self._base_url}{self._endpoint}",
            files=files,
            data=data,
            headers=headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()

        result_data = resp.json()
        return self._parse_api_response(result_data, len(audio), sample_rate)

    def _parse_api_response(
        self,
        data: dict,
        audio_length: int,
        sample_rate: int,
    ) -> TranscriptionResult:
        """Normalize API response into TranscriptionResult."""
        text = data.get("text", "").strip()
        duration = audio_length / sample_rate if audio_length > 0 else 0.0

        words = []
        if "segments" in data:
            for seg in data["segments"]:
                if "words" in seg:
                    for w in seg["words"]:
                        words.append({
                            "word": w.get("word", ""),
                            "start": int(w.get("start", 0) * 1000),
                            "end": int(w.get("end", 0) * 1000),
                        })
        elif "words" in data:
            for w in data["words"]:
                words.append({
                    "word": w.get("word", ""),
                    "start": int(w.get("start", 0) * 1000),
                    "end": int(w.get("end", 0) * 1000),
                })

        return TranscriptionResult(
            text=text,
            language=data.get("language", "en"),
            duration_secs=round(duration, 3),
            words=words,
            confidence=data.get("confidence", 0.0),
            no_speech_prob=data.get("no_speech_prob", 0.0),
        )


# ---------------------------------------------------------------------------
# Unified Whisper STT Class (factory)
# ---------------------------------------------------------------------------


class WhisperSTT:
    """High-level STT adapter that works with your voice-agent pipeline.

    Usage:
        stt = WhisperSTT.from_config()          # reads config.yaml
        stt = WhisperSTT(mode="local")           # explicit local
        stt = WhisperSTT(mode="remote", base_url="...")  # explicit API

        result = stt.transcribe(pcm_chunk, sample_rate=16000)
        print(result.text)
    """

    def __init__(
        self,
        adapter: LocalWhisperAdapter | RemoteWhisperAdapter,
        prompt_history: list[str] = None,
    ) -> None:
        self._adapter = adapter
        self._prompt_history: list[str] = prompt_history or []
        self._history_window = 3  # last N transcriptions as context

    # ---- Factory methods --------------------------------------------------

    @classmethod
    def from_config(cls, mode: str = "auto") -> WhisperSTT:
        """Create from config.yaml (stt section)."""
        stt_cfg = settings.stt  # type: ignore[attr-defined]

        # Handle both dict-style and dataclass-style configs
        if isinstance(stt_cfg, dict):
            model = WhisperModel(stt_cfg.get("model", "base"))
            language = stt_cfg.get("language", "en")
            task = stt_cfg.get("task", "transcribe")
            base_url = stt_cfg.get("base_url", "")
            api_key = stt_cfg.get("api_key", "")
        else:
            model = WhisperModel(getattr(stt_cfg, "model", "base"))
            language = getattr(stt_cfg, "language", "en")
            task = getattr(stt_cfg, "task", "transcribe")
            base_url = getattr(stt_cfg, "base_url", "")
            api_key = getattr(stt_cfg, "api_key", "")

        if mode == "auto":
            # If base_url points to localhost and faster-whisper is available,
            # prefer local. Otherwise use remote.
            has_local = __import__("importlib").util.find_spec("faster_whisper") is not None
            mode = "local" if has_local else "remote"

        if mode == "local":
            adapter = LocalWhisperAdapter(
                model_size=model,
                language=language,
                task=task,
            )
        else:
            adapter = RemoteWhisperAdapter(
                base_url=base_url,
                api_key=api_key,
                model=model,
                language=language,
                task=task,
            )

        return cls(adapter)

    # ---- Public API -------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray | bytes | BinaryIO,
        sample_rate: int = 16000,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio (PCM array, WAV bytes, or file-like).

        Args:
            audio: 16kHz mono PCM numpy array or WAV bytes/file.
            sample_rate: Audio sample rate (default 16kHz for VAD pipeline).
            initial_prompt: Optional prompt for disambiguation.

        Returns:
            TranscriptionResult with text, timestamps, and metadata.
        """
        if isinstance(audio, bytes):
            # WAV bytes → numpy
            audio = self._wav_bytes_to_pcm(audio, sample_rate)
        elif hasattr(audio, "read"):
            audio = self._wav_file_to_pcm(audio, sample_rate)

        prompt = initial_prompt or self._build_prompt()
        result = self._adapter.transcribe(audio, sample_rate, prompt)
        self._update_history(result.text)
        return result

    def transcribe_file(
        self,
        path: str | Path,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribe a full audio file."""
        audio, sr = sf.read(str(path), dtype="float32")
        # mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.transcribe(audio, sample_rate=sr)

    def transcribe_stream(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribe a sequence of audio chunks as one continuous utterance.

        Useful when VAD detects a speech segment spanning multiple windows.
        """
        combined = np.concatenate(audio_chunks)
        return self.transcribe(combined, sample_rate=sample_rate)

    # ---- Helper methods ---------------------------------------------------

    def _wav_bytes_to_pcm(self, wav_bytes: bytes, sample_rate: int) -> np.ndarray:
        """Decode WAV bytes to float32 PCM numpy array."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _wav_file_to_pcm(self, fh: BinaryIO, sample_rate: int) -> np.ndarray:
        """Decode WAV file-like object to float32 PCM numpy array."""
        buf = io.BytesIO(fh.read())
        return sf.read(buf, dtype="float32", sr=sample_rate)[0]

    def _build_prompt(self) -> str | None:
        """Build a prompt from recent transcription history."""
        if not self._prompt_history:
            return None
        recent = self._prompt_history[-self._history_window:]
        return " ".join(recent)

    def _update_history(self, text: str) -> None:
        """Append to history, keeping bounded."""
        if text.strip():
            self._prompt_history.append(text.strip())
            self._prompt_history = self._prompt_history[-self._history_window:]

    # ---- Context manager for use in VAD pipeline --------------------------

    def __enter__(self) -> WhisperSTT:
        return self

    def __exit__(self, *args: object) -> None:
        pass
