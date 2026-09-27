# -*- coding: utf-8 -*-
"""
UltraVoice - Live Mic -> VAD -> Whisper -> Qwen -> audio.cpp -> Speaker

Uses the Windows/system default audio input and output devices.

Pipeline:
    Mic -> VAD -> Whisper STT -> Qwen LLM -> audio.cpp TTS -> Speaker

Important:
- Capture uses the system default microphone.
- Playback uses the system default speaker.
- Audio is captured at the device's native/default sample rate and
  converted to 16 kHz mono float32 before VAD/STT.
- VAD has a short pre-roll so the first syllable is not lost.
- Short/no-speech Whisper results are ignored.
- Only one LLM/TTS turn is processed at a time.
"""

from __future__ import annotations

import argparse
import json
import asyncio
import queue
import re
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.adapters.whisper_stt import WhisperSTT
from runtime.adapters.qwen_llm import QwenLLM, ChatOptions
from runtime.adapters.audio_cpp_tts import AudioCppTTS


TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 32

# ---------------------------------------------------------------------------
# Streaming / latency tuning
# ---------------------------------------------------------------------------

# How much generated PCM to pre-buffer before speaker playback starts.
# Lower = less initial latency, higher = more protection from TTS underflow.
PLAYBACK_START_BUFFER_MS = 2000


# Size of each speaker write. 80 ms is a reasonable balance between smooth
# playback and responsiveness.
PLAYBACK_WRITE_CHUNK_MS = 80

# Persistent queue capacity between audio.cpp receiver and speaker playback.
AUDIO_QUEUE_MAXSIZE = 180

# Current playback pipeline keeps audio.cpp PCM at native 24 kHz.
TTS_PCM_SAMPLE_RATE = 24000

# Qwen response limit for each turn.
LLM_MAX_TOKENS = 180

# Qwen text is accumulated until we have a complete word and enough useful
# text to send to audio.cpp. Never deliberately cut inside a word.
# Current test setting. Larger chunks may reduce per-chunk TTS overhead.
# For the next throughput experiment, try 80 / 120.
TTS_TEXT_MIN_CHARS = 150
TTS_TEXT_MAX_CHARS = 600

# TTS text cleanup. Removes Markdown syntax before text reaches audio.cpp.
CLEAN_TTS_TEXT = True

# Diagnostic polling interval used to detect periods where audio.cpp is not
# delivering PCM quickly enough for the already-running speaker stream.
PLAYBACK_STARVATION_CHECK_MS = 80

# Placeholder for a future pitch-preserving time-stretch stage. Keeping this
# at 1.00 means the PCM is played at its native rate with no modification.


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert mono audio to 16 kHz float32."""
    audio = np.asarray(audio).reshape(-1)

    if audio.size == 0:
        return np.empty(0, dtype=np.float32)

    audio = audio.astype(np.float32, copy=False)

    if sample_rate == TARGET_SAMPLE_RATE:
        return audio

    new_len = max(1, round(len(audio) * TARGET_SAMPLE_RATE / sample_rate))

    # np.interp is sufficient for speech preprocessing at this stage.
    old_x = np.arange(len(audio), dtype=np.float64)
    new_x = np.linspace(0, len(audio) - 1, new_len, dtype=np.float64)

    return np.interp(new_x, old_x, audio).astype(np.float32)


def audio_level(audio: np.ndarray) -> tuple[float, float]:
    """Return RMS and peak level."""
    if audio.size == 0:
        return 0.0, 0.0

    x = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    return rms, peak


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

class VADDetector:
    """Silero VAD wrapper operating internally at 16 kHz."""

    def __init__(self, threshold: float = 0.45) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._threshold = threshold
        self._model = load_silero_vad()
        self._model.eval()

    def is_speech(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> bool:
        audio = resample_to_16k(audio, sample_rate)

        # Silero VAD requires at least 512 samples at 16 kHz.
        if len(audio) < 512:
            return False

        with self._torch.no_grad():
            probability = self._model(
                self._torch.from_numpy(audio).unsqueeze(0),
                TARGET_SAMPLE_RATE,
            ).item()

        return probability >= self._threshold



def sanitize_tts_text(text: str) -> str:
    """Remove common Markdown artifacts before text reaches TTS."""
    if not CLEAN_TTS_TEXT:
        return text

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


class TTSDiagnostics:
    """Per-turn metrics for audio.cpp PCM production and playback buffering."""

    def __init__(self) -> None:
        self.first_pcm_at: float | None = None
        self.last_pcm_at: float | None = None
        self.total_pcm_bytes = 0
        self.min_buffer_seconds = float("inf")
        self.max_buffer_seconds = 0.0
        self.starvation_events = 0
        self.playback_started_at: float | None = None
        self.playback_started_buffer_seconds = 0.0
        self.text = ""

    def record_pcm(self, byte_count: int) -> None:
        now = time.perf_counter()
        if self.first_pcm_at is None:
            self.first_pcm_at = now
        self.last_pcm_at = now
        self.total_pcm_bytes += byte_count

    def record_buffer(self, seconds: float) -> None:
        self.min_buffer_seconds = min(self.min_buffer_seconds, seconds)
        self.max_buffer_seconds = max(self.max_buffer_seconds, seconds)

    def set_text(self, text: str) -> None:
        self.text = text

    def summary(self) -> str:
        bytes_per_second = TTS_PCM_SAMPLE_RATE * 2
        generated_seconds = self.total_pcm_bytes / bytes_per_second

        if self.first_pcm_at is not None and self.last_pcm_at is not None:
            production_wall = max(
                0.0,
                self.last_pcm_at - self.first_pcm_at,
            )
        else:
            production_wall = 0.0

        ratio = (
            generated_seconds / production_wall
            if production_wall > 0.001
            else 0.0
        )

        min_buffer = (
            self.min_buffer_seconds
            if self.min_buffer_seconds != float("inf")
            else 0.0
        )

        if ratio >= 1.15:
            interpretation = "producer faster than realtime"
        elif ratio >= 0.95:
            interpretation = "near realtime"
        elif ratio > 0.0:
            interpretation = "producer slower than realtime"
        else:
            interpretation = "no PCM timing data"

        word_count = len(re.findall(r"\b[\w’'-]+\b", self.text))
        speech_wpm = (word_count / generated_seconds * 60.0) if generated_seconds > 0.05 else 0.0

        return (
            f"words={word_count}, "
            f"speech_rate={speech_wpm:.0f} WPM, "
            f"PCM={generated_seconds:.2f}s, "
            f"production={production_wall:.2f}s, "
            f"producer_rate={ratio:.2f}x realtime, "
            f"buffer_min={min_buffer:.2f}s, "
            f"buffer_max={self.max_buffer_seconds:.2f}s, "
            f"starvation_events={self.starvation_events}, "
            f"{interpretation}"
        )

# ---------------------------------------------------------------------------
# Live Voice Agent
# ---------------------------------------------------------------------------

class LiveVoiceAgent:
    """Real-time default-device voice agent."""

    def __init__(
        self,
        system: str = "You are a friendly and concise voice assistant.",
        no_output: bool = False,
        vad_threshold: float = 0.45,
        silence_duration_ms: int = 650,
        min_speech_ms: int = 300,
        max_speech_seconds: float = 30.0,
        pre_roll_ms: int = 250,
    ) -> None:
        self._system = system
        self._no_output = no_output
        self._running = False
        self._processing = False

        self._vad_threshold = vad_threshold
        self._silence_duration_ms = silence_duration_ms
        self._min_speech_ms = min_speech_ms
        self._max_speech_seconds = max_speech_seconds
        self._pre_roll_ms = pre_roll_ms

        # We capture using the default device's native/default rate.
        default_input = sd.query_devices(kind="input")
        self._capture_rate = int(round(default_input["default_samplerate"]))

        # 32 ms chunks at the capture rate.
        self._chunk_samples = max(
            1,
            round(self._capture_rate * CHUNK_MS / 1000),
        )

        self._silence_frames_threshold = max(
            1,
            round(self._silence_duration_ms / CHUNK_MS),
        )

        self._max_samples = int(
            self._capture_rate * self._max_speech_seconds
        )

        self._pre_roll_chunks = max(
            1,
            round(self._pre_roll_ms / CHUNK_MS),
        )

        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=100
        )

        self._vad = VADDetector(threshold=vad_threshold)

        print("\nLoading adapters...")
        self._stt = WhisperSTT.from_config()
        self._llm = QwenLLM.from_config()
        self._tts = AudioCppTTS.from_config()
        print("  Adapters loaded OK")
        self._print_tuning_summary()

    # ------------------------------------------------------------------
    # Startup tuning summary
    # ------------------------------------------------------------------

    def _print_tuning_summary(self) -> None:
        """Print all important voice-pipeline tuning values for this run."""
        print("\n  ------------------------------------------------------------")
        print("  ACTIVE TUNING PARAMETERS")
        print("  ------------------------------------------------------------")
        print(f"  Capture sample rate:       {self._capture_rate} Hz")
        print(f"  Capture chunk:             {CHUNK_MS} ms")
        print(f"  VAD threshold:             {self._vad_threshold:.2f}")
        print(f"  Silence duration:          {self._silence_duration_ms} ms")
        print(f"  Pre-roll:                  {self._pre_roll_ms} ms")
        print(f"  Min speech:                {self._min_speech_ms} ms")
        print(f"  Max speech:                {self._max_speech_seconds:.1f} s")
        print(f"  LLM max tokens:            {LLM_MAX_TOKENS}")
        print(f"  audio.cpp text min chars:     {TTS_TEXT_MIN_CHARS}")
        print(f"  audio.cpp text max chars:     {TTS_TEXT_MAX_CHARS}")
        print(f"  audio.cpp PCM sample rate:    {TTS_PCM_SAMPLE_RATE} Hz")
        print(f"  Playback start buffer:     {PLAYBACK_START_BUFFER_MS} ms")
        print(f"  Playback write chunk:      {PLAYBACK_WRITE_CHUNK_MS} ms")
        print(f"  Audio queue max size:      {AUDIO_QUEUE_MAXSIZE}")
        print("  Speech-rate metric:         word_count / generated PCM duration")
        print(f"  Clean TTS Markdown:         {CLEAN_TTS_TEXT}")
        print(f"  Starvation check interval: {PLAYBACK_STARVATION_CHECK_MS} ms")
        print("  TTS transport:              audio.cpp HTTP")
        print("  ------------------------------------------------------------\n")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str, level: str = "INFO") -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"  [{timestamp}] [{level}] {message}", flush=True)

    # ------------------------------------------------------------------
    # Mic callback
    # ------------------------------------------------------------------

    def _mic_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._log(f"Audio status: {status}", "WARN")

        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        try:
            self._audio_queue.put_nowait(audio)
        except queue.Full:
            # Drop the oldest block rather than blocking PortAudio.
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(audio)
            except (queue.Empty, queue.Full):
                pass

    # ------------------------------------------------------------------
    # Persistent audio.cpp streaming
    # ------------------------------------------------------------------

    async def _audio_cpp_receiver(
        self,
        session,
        audio_queue: asyncio.Queue[bytes | None],
        diagnostics: TTSDiagnostics,
    ) -> None:
        """Receive audio.cpp PCM continuously until the current turn is done."""
        while True:
            message = await session.receive()

            if isinstance(message, bytes):
                if message:
                    diagnostics.record_pcm(len(message))
                    await audio_queue.put(message)
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                self._log(
                    f"Unexpected audio.cpp message: {message!r}",
                    "WARN",
                )
                continue

            kind = data.get("type")

            if kind == "done":
                self._log("audio.cpp turn complete")
                return

            if kind == "error":
                raise RuntimeError(
                    data.get("message", "audio.cpp WebSocket error")
                )

            # ready / started / speaking / queued are informational.

    async def _pcm_playback_worker(
        self,
        audio_queue: asyncio.Queue[bytes | None],
        diagnostics: TTSDiagnostics,
    ) -> None:
        """Play audio.cpp PCM with a small jitter buffer to prevent underflow.

        audio.cpp can temporarily pause between PCM packets while generating the
        next audio frames. Starting playback immediately on the first packet
        makes those pauses audible as clipped/choppy words. We therefore
        accumulate a short startup buffer, then keep a continuous output
        stream running while the receiver continues filling the queue.
        """
        if self._no_output:
            while True:
                pcm = await audio_queue.get()
                try:
                    if pcm is None:
                        return
                finally:
                    audio_queue.task_done()

        # 24 kHz, mono, s16le = 48,000 bytes/sec.
        BYTES_PER_SECOND = TTS_PCM_SAMPLE_RATE * 2
        START_BUFFER_MS = PLAYBACK_START_BUFFER_MS
        START_BUFFER_BYTES = BYTES_PER_SECOND * START_BUFFER_MS // 1000
        WRITE_BYTES = BYTES_PER_SECOND * PLAYBACK_WRITE_CHUNK_MS // 1000

        buffered = bytearray()
        started = False
        stream: sd.RawOutputStream | None = None
        last_queue_log = time.perf_counter()
        min_queue_secs = float("inf")
        max_queue_secs = 0.0

        async def write_buffered(final: bool = False) -> None:
            nonlocal buffered
            if not buffered or stream is None:
                return

            while len(buffered) >= WRITE_BYTES:
                block = bytes(buffered[:WRITE_BYTES])
                del buffered[:WRITE_BYTES]
                await asyncio.to_thread(stream.write, block)

            if final and buffered:
                block = bytes(buffered)
                buffered.clear()
                await asyncio.to_thread(stream.write, block)

        try:
            while True:
                pcm = await audio_queue.get()
                try:
                    if pcm is None:
                        # Flush anything left at the end of the response.
                        # If the whole response was shorter than the startup
                        # buffer, open the output stream now so that audio is
                        # still played instead of being discarded.
                        if buffered:
                            if stream is None:
                                stream = sd.RawOutputStream(
                                    device=None,
                                    samplerate=TTS_PCM_SAMPLE_RATE,
                                    channels=1,
                                    dtype="int16",
                                    blocksize=0,
                                )
                                stream.start()
                                started = True
                                self._log(
                                    f"Streaming audio output ready "
                                    f"(short response buffer="
                                    f"{len(buffered) / BYTES_PER_SECOND:.2f}s)"
                                )

                            await write_buffered(final=True)
                        self._log(
                            f"Audio buffer summary: min={min_queue_secs:.2f}s, "
                            f"max={max_queue_secs:.2f}s; {diagnostics.summary()}"
                        )
                        return

                    buffered.extend(pcm)

                    queue_secs = len(buffered) / BYTES_PER_SECOND
                    min_queue_secs = min(min_queue_secs, queue_secs)
                    max_queue_secs = max(max_queue_secs, queue_secs)
                    diagnostics.record_buffer(queue_secs)
                    now = time.perf_counter()
                    if now - last_queue_log >= 1.0:
                        self._log(
                            f"Audio buffer: {queue_secs:.2f}s "
                            f"(min={min_queue_secs:.2f}s, max={max_queue_secs:.2f}s)"
                        )
                        last_queue_log = now

                    if not started:
                        if len(buffered) < START_BUFFER_BYTES:
                            continue

                        stream = sd.RawOutputStream(
                            device=None,
                            samplerate=TTS_PCM_SAMPLE_RATE,
                            channels=1,
                            dtype="int16",
                            blocksize=0,
                        )
                        stream.start()
                        started = True
                        diagnostics.playback_started_at = time.perf_counter()
                        diagnostics.playback_started_buffer_seconds = (
                            len(buffered) / BYTES_PER_SECOND
                        )

                        self._log(
                            f"Streaming audio output ready "
                            f"(buffer={len(buffered) / BYTES_PER_SECOND:.2f}s)"
                        )

                    await write_buffered(final=False)

                finally:
                    audio_queue.task_done()

        finally:
            if stream is not None:
                # There can be a short tail smaller than WRITE_BYTES.
                if buffered:
                    try:
                        await write_buffered(final=True)
                    except Exception:
                        pass

                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Word-safe Qwen -> audio.cpp text buffering
    # ------------------------------------------------------------------

    @staticmethod
    def _take_audio_cpp_text(
        buffer: str,
        force: bool = False,
    ) -> tuple[list[str], str]:
        """Return complete-word text chunks suitable for audio.cpp.

        We never intentionally cut a word. Sentences flush immediately.
        Otherwise, once the buffer reaches the maximum size, we cut at the
        last whitespace before that limit.
        """
        chunks: list[str] = []
        text = buffer

        sentence_end = re.compile(
            r"^(.+?(?<!\d)[.!?。！？]+(?:[\"'”’)]*)(?:\s+|$))",
            re.DOTALL,
        )

        while True:
            match = sentence_end.match(text)
            if not match:
                break

            chunk = match.group(1).strip()
            if chunk:
                chunks.append(chunk)

            text = text[match.end():].lstrip()

        # Keep latency low without ever splitting a word.
        while len(text) >= TTS_TEXT_MAX_CHARS:
            cut = text.rfind(" ", 0, TTS_TEXT_MAX_CHARS + 1)
            if cut < TTS_TEXT_MIN_CHARS:
                break

            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()

        # Also release a useful phrase once it is complete and ends at a
        # whitespace boundary. This avoids waiting for punctuation forever.
        if len(text) >= TTS_TEXT_MIN_CHARS and text.endswith((" ", "\n")):
            chunks.append(text.strip())
            text = ""

        if force and text.strip():
            chunks.append(text.strip())
            text = ""

        return chunks, text

    # ------------------------------------------------------------------
    # STT -> streaming Qwen -> persistent audio.cpp -> speaker
    # ------------------------------------------------------------------

    async def _process_turn(self, audio: np.ndarray) -> None:
        """Process one complete speech turn."""
        audio_16k = resample_to_16k(audio, self._capture_rate)
        duration = len(audio_16k) / TARGET_SAMPLE_RATE

        if duration < self._min_speech_ms / 1000:
            self._log(f"Ignoring short audio: {duration:.2f}s", "DEBUG")
            return

        rms, peak = audio_level(audio_16k)
        self._log(
            f"Transcribing {duration:.2f}s "
            f"(RMS={rms:.4f}, peak={peak:.4f})..."
        )

        if peak < 0.003 or rms < 0.0005:
            self._log(
                "Audio level is extremely low; ignoring segment. "
                "Check the Windows default microphone.",
                "WARN",
            )
            return

        audio_16k = np.ascontiguousarray(audio_16k, dtype=np.float32)

        # ---------------- STT ----------------
        stt_start = time.perf_counter()
        try:
            stt_result = await asyncio.to_thread(
                self._stt.transcribe,
                audio_16k,
                TARGET_SAMPLE_RATE,
            )
            transcribed = (stt_result.text or "").strip()
            self._log(
                f'STT: "{transcribed}" '
                f"({time.perf_counter() - stt_start:.2f}s)"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(f"STT error: {exc}", "ERROR")
            return

        if not transcribed:
            self._log("Empty transcription", "DEBUG")
            return

        if transcribed.upper() == "[BLANK_AUDIO]":
            self._log("Whisper returned BLANK_AUDIO", "WARN")
            return

        # ---------------- Qwen + audio.cpp concurrently ----------------
        self._log("Generating response (streaming)...")
        llm_start = time.perf_counter()
        first_text_at: float | None = None
        response_parts: list[str] = []
        audio_cpp_text_buffer = ""

        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=AUDIO_QUEUE_MAXSIZE
        )
        diagnostics = TTSDiagnostics()

        playback_task = asyncio.create_task(
            self._pcm_playback_worker(audio_queue, diagnostics),
            name="audio_cpp-pcm-playback",
        )

        try:
            async with self._tts.persistent_session() as audio_cpp:
                receiver_task = asyncio.create_task(
                    self._audio_cpp_receiver(
                        audio_cpp,
                        audio_queue,
                        diagnostics,
                    ),
                    name="audio_cpp-pcm-receiver",
                )

                try:
                    # Every normal Qwen content fragment goes into the SAME
                    # audio.cpp session. There is no new request per sentence.
                    async for token in self._llm.chat_stream(
                        transcribed,
                        system=self._system,
                        options=ChatOptions(max_tokens=LLM_MAX_TOKENS),
                    ):
                        if not token:
                            continue

                        if first_text_at is None:
                            first_text_at = time.perf_counter()
                            self._log(
                                f"LLM first text: "
                                f"{first_text_at - llm_start:.2f}s"
                            )

                        response_parts.append(token)

                        # Do not pass arbitrary token fragments directly to
                        # audio.cpp. Hold them until we have a complete word /
                        # phrase, or a complete sentence.
                        audio_cpp_text_buffer += token
                        text_chunks, audio_cpp_text_buffer = self._take_audio_cpp_text(
                            audio_cpp_text_buffer
                        )

                        for chunk in text_chunks:
                            tts_chunk = sanitize_tts_text(chunk).strip()
                            if not tts_chunk:
                                continue
                            self._log(
                                f"audio.cpp text chunk: {tts_chunk[:120]}"
                            )
                            await audio_cpp.send_text(tts_chunk)

                    # Flush the final partial phrase, but only after Qwen is
                    # finished. At this point even the final short phrase is
                    # a complete response fragment.
                    final_chunks, audio_cpp_text_buffer = self._take_audio_cpp_text(
                        audio_cpp_text_buffer,
                        force=True,
                    )
                    for chunk in final_chunks:
                        tts_chunk = sanitize_tts_text(chunk).strip()
                        if not tts_chunk:
                            continue
                        self._log(
                            f"audio.cpp final text chunk: {tts_chunk[:120]}"
                        )
                        await audio_cpp.send_text(tts_chunk)

                    response = "".join(response_parts).strip()
                    diagnostics.set_text(response)

                    self._log(
                        f"LLM: {response[:160].replace(chr(10), ' ')}... "
                        f"({time.perf_counter() - llm_start:.2f}s)"
                    )

                    if not response:
                        self._log("Empty LLM response", "WARN")
                        return

                    # Flush remaining audio.cpp text and wait for all PCM.
                    await audio_cpp.end()
                    await receiver_task

                    # Receiver has handed off all PCM; playback consumes it
                    # through the same stream, eliminating per-sentence
                    # open/close/play gaps.
                    await audio_queue.join()
                    self._log(f"TTS diagnostics: {diagnostics.summary()}")

                except asyncio.CancelledError:
                    receiver_task.cancel()
                    await asyncio.gather(
                        receiver_task,
                        return_exceptions=True,
                    )
                    raise
                except Exception:
                    if not receiver_task.done():
                        receiver_task.cancel()
                        await asyncio.gather(
                            receiver_task,
                            return_exceptions=True,
                        )
                    raise

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(
                f"LLM/audio.cpp streaming error: {exc}",
                "ERROR",
            )
        finally:
            if not playback_task.done():
                try:
                    await audio_queue.put(None)
                except (asyncio.CancelledError, RuntimeError):
                    pass

            await asyncio.gather(
                playback_task,
                return_exceptions=True,
            )

    # ------------------------------------------------------------------
    # Audio/VAD loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Audio/VAD loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        current_audio: list[np.ndarray] = []
        pre_roll: deque[np.ndarray] = deque(maxlen=self._pre_roll_chunks)

        total_samples = 0
        silence_count = 0
        speech_active = False
        last_status = time.monotonic()

        while self._running:
            try:
                audio = await asyncio.to_thread(
                    self._audio_queue.get,
                    True,
                    1.0,
                )
            except queue.Empty:
                continue

            if audio is None:
                break

            pre_roll.append(audio)

            # VAD works on native capture audio and internally converts to
            # 16 kHz, so the detector always receives the correct rate.
            is_speech = self._vad.is_speech(
                audio,
                self._capture_rate,
            )

            if is_speech:
                if not speech_active:
                    speech_active = True
                    silence_count = 0
                    current_audio = list(pre_roll)
                    total_samples = sum(len(x) for x in current_audio)

                    self._log("SPEECH START")

                else:
                    silence_count = 0
                    current_audio.append(audio)
                    total_samples += len(audio)

                # Hard limit protects against a microphone being left open.
                if total_samples >= self._max_samples:
                    self._log(
                        f"Maximum speech duration reached "
                        f"({total_samples / self._capture_rate:.1f}s)"
                    )
                    await self._finish_turn(current_audio)

                    current_audio = []
                    total_samples = 0
                    silence_count = 0
                    speech_active = False
                    pre_roll.clear()

            elif speech_active:
                # Keep trailing silence in the segment so Whisper gets
                # natural utterance boundaries.
                current_audio.append(audio)
                total_samples += len(audio)
                silence_count += 1

                if silence_count >= self._silence_frames_threshold:
                    duration = total_samples / self._capture_rate

                    self._log(
                        f"SPEECH END ({duration:.2f}s, "
                        f"{total_samples} samples)"
                    )

                    await self._finish_turn(current_audio)

                    current_audio = []
                    total_samples = 0
                    silence_count = 0
                    speech_active = False
                    pre_roll.clear()

            # Periodic status without spamming the console.
            now = time.monotonic()
            if now - last_status >= 5:
                self._log(
                    f"Queue={self._audio_queue.qsize()} "
                    f"speech_active={speech_active}"
                )
                last_status = now

        if current_audio:
            await self._finish_turn(current_audio)

    async def _finish_turn(self, chunks: list[np.ndarray]) -> None:
        """Send one speech segment to the processing pipeline."""

        if not chunks:
            return

        segment = np.concatenate(chunks)

        # Don't start multiple expensive inference pipelines at once.
        # This is especially important for a single local GPU.
        if self._processing:
            self._log(
                "Previous turn still processing; dropping this turn",
                "WARN",
            )
            return

        self._processing = True

        try:
            await self._process_turn(segment)
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True

        default_input = sd.query_devices(kind="input")
        default_output = sd.query_devices(kind="output")

        print("\n" + "=" * 68)
        print("  ULTRAVOICE - LIVE VOICE AGENT")
        print("=" * 68)
        print(f"  Input:    {default_input['name']}")
        print(f"  Output:   {default_output['name']}")
        print(f"  Capture:  {self._capture_rate} Hz")
        print(f"  STT:      Whisper (local)")
        print(f"  LLM:      Qwen3.6-35B (local)")
        print(f"  TTS:      audio.cpp (local)")
        print(f"  VAD:      threshold={self._vad_threshold}")
        print(f"  Silence:  {self._silence_duration_ms} ms")
        print(f"  Pre-roll: {self._pre_roll_ms} ms")
        print("\n  Using Windows/system default audio devices.")
        print("  Press Ctrl+C to stop")
        print("=" * 68)

        self._log(
            f"Opening default microphone at {self._capture_rate} Hz..."
        )

        stream = sd.InputStream(
            device=None,
            channels=CHANNELS,
            samplerate=self._capture_rate,
            blocksize=self._chunk_samples,
            dtype="float32",
            callback=self._mic_callback,
        )

        stream.start()
        self._log("Listening... speak now.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_loop())

        except KeyboardInterrupt:
            print("\n\nStopping...")

        finally:
            self._running = False

            try:
                self._audio_queue.put_nowait(None)
            except queue.Full:
                pass

            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

            # Cancel any active async tasks before closing the event loop.
            # This prevents Python 3.14 from reporting "Task was destroyed while
            # it is pending" and avoids websocket callbacks touching a closed loop.
            try:
                pending = asyncio.all_tasks(loop)
                current = asyncio.current_task(loop=loop)
                for task in pending:
                    if task is not current:
                        task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass

            loop.close()

            self._log("Stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UltraVoice - Mic to Speaker"
    )

    parser.add_argument(
        "-s",
        "--system",
        default="You are a friendly and concise voice assistant.",
        help="System prompt",
    )

    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip speaker playback",
    )

    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.45,
        help="VAD threshold (default: 0.45)",
    )

    parser.add_argument(
        "--silence-ms",
        type=int,
        default=650,
        help="Silence duration to end a turn (default: 650)",
    )

    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=300,
        help="Ignore speech shorter than this (default: 300)",
    )

    parser.add_argument(
        "--pre-roll-ms",
        type=int,
        default=250,
        help="Audio retained before VAD triggers (default: 250)",
    )
    args = parser.parse_args()

    agent = LiveVoiceAgent(
        system=args.system,
        no_output=args.no_output,
        vad_threshold=args.vad_threshold,
        silence_duration_ms=args.silence_ms,
        min_speech_ms=args.min_speech_ms,
        pre_roll_ms=args.pre_roll_ms,
    )

    agent.run()


if __name__ == "__main__":
    main()
