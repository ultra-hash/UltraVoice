# -*- coding: utf-8 -*-
"""
UltraVoice - Live Mic -> VAD -> Whisper -> Qwen -> Breeze -> Speaker

Uses the Windows/system default audio input and output devices.

Pipeline:
    Mic -> VAD -> Whisper STT -> Qwen LLM -> Breeze TTS -> Speaker

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
from runtime.adapters.breeze_tts import BreezeTTS


TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 32


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
        self._tts = BreezeTTS.from_config(mode="http")
        print("  Adapters loaded OK")

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
    # Streaming Qwen -> sentence queue -> Breeze
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tts_chunks(
        buffer: str,
        force: bool = False,
    ) -> tuple[list[str], str]:
        """Extract complete speech-sized chunks from streamed LLM text."""
        chunks: list[str] = []
        text = buffer

        sentence_re = re.compile(
            r"(.+?[.!?。！？]+(?:[\"'”’)]*)"
            r"(?=\s+|$))",
            re.DOTALL,
        )

        while True:
            match = sentence_re.match(text)
            if not match:
                break

            chunk = match.group(1).strip()
            if chunk:
                chunks.append(chunk)

            text = text[match.end():].lstrip()

        # Avoid waiting indefinitely for punctuation.
        max_chars = 120
        if len(text) >= max_chars:
            cut = text.rfind(" ", 0, max_chars + 1)
            if cut >= 40:
                chunks.append(text[:cut].strip())
                text = text[cut:].lstrip()

        if force and text.strip():
            chunks.append(text.strip())
            text = ""

        return chunks, text

    async def _tts_synth_worker(
        self,
        tts_queue: asyncio.Queue[str | None],
        audio_queue: asyncio.Queue[tuple[int, object] | None],
    ) -> None:
        """Synthesize sentences sequentially, independent of playback."""
        sequence = 0

        while True:
            chunk = await tts_queue.get()
            try:
                if chunk is None:
                    return

                sequence += 1
                index = sequence
                self._log(f'TTS synth #{index} start: "{chunk[:120]}"')
                started = time.perf_counter()

                try:
                    # Keep the known-good Breeze one-shot synthesis API for
                    # this step. The important change is that playback is no
                    # longer inside this worker.
                    result = await asyncio.to_thread(
                        self._tts.synthesize,
                        chunk,
                    )

                    self._log(
                        f"TTS synth #{index} done: "
                        f"{result.duration_secs:.2f}s "
                        f"({time.perf_counter() - started:.2f}s wall)"
                    )

                    await audio_queue.put((index, result))

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._log(
                        f"TTS synth #{index} error: {exc}",
                        "ERROR",
                    )

            finally:
                tts_queue.task_done()

    async def _playback_worker(
        self,
        audio_queue: asyncio.Queue[tuple[int, object] | None],
    ) -> None:
        """Play completed TTS sentences in strict sequence order."""
        expected = 1
        pending: dict[int, object] = {}

        while True:
            item = await audio_queue.get()
            try:
                if item is None:
                    return

                index, result = item
                pending[index] = result

                while expected in pending:
                    current = pending.pop(expected)

                    if self._no_output:
                        self._log(
                            f"Output disabled for TTS chunk #{expected}"
                        )
                    else:
                        audio_out = (
                            np.frombuffer(
                                current.pcm,
                                dtype=np.int16,
                            )
                            .astype(np.float32)
                            / 32768.0
                        )

                        self._log(
                            f"Playing TTS chunk #{expected}..."
                        )

                        await asyncio.to_thread(
                            sd.play,
                            audio_out,
                            current.sample_rate,
                        )
                        await asyncio.to_thread(sd.wait)

                        self._log(
                            f"Playback chunk #{expected} complete"
                        )

                    expected += 1

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(
                    f"Playback error: {exc}",
                    "ERROR",
                )
            finally:
                audio_queue.task_done()

    # ------------------------------------------------------------------
    # STT -> LLM -> TTS
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

        # A very low signal usually means the selected Windows default
        # microphone is muted/incorrect, so avoid wasting an STT request.
        if peak < 0.003 or rms < 0.0005:
            self._log(
                "Audio level is extremely low; ignoring segment. "
                "Check the Windows default microphone.",
                "WARN",
            )
            return

        # Make sure Whisper always receives contiguous float32 PCM.
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
            stt_time = time.perf_counter() - stt_start

            self._log(
                f'STT: "{transcribed}" ({stt_time:.2f}s)'
            )

        except Exception as exc:
            self._log(f"STT error: {exc}", "ERROR")
            return

        if not transcribed:
            self._log("Empty transcription", "DEBUG")
            return

        if transcribed.upper() == "[BLANK_AUDIO]":
            self._log(
                "Whisper returned BLANK_AUDIO despite VAD detecting speech. "
                "Audio diagnostics above can be used to verify the mic signal.",
                "WARN",
            )
            return

        # ---------------- STREAMING LLM + OVERLAPPED TTS ----------------
        self._log("Generating response (streaming)...")
        llm_start = time.perf_counter()
        first_text_at: float | None = None
        response_parts: list[str] = []
        text_buffer = ""

        # Two independent stages:
        #
        #   Qwen -> sentence queue -> Breeze synthesis -> audio queue
        #                                                     -> playback
        #
        # This means sentence N+1 can be synthesized while sentence N is
        # playing. Playback never blocks Breeze synthesis anymore.
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=4)
        audio_queue: asyncio.Queue[tuple[int, object] | None] = asyncio.Queue(
            maxsize=8
        )

        synth_task = asyncio.create_task(
            self._tts_synth_worker(tts_queue, audio_queue),
            name="tts-synthesis-worker",
        )
        playback_task = asyncio.create_task(
            self._playback_worker(audio_queue),
            name="tts-playback-worker",
        )

        try:
            async for token in self._llm.chat_stream(
                transcribed,
                system=self._system,
                options=ChatOptions(max_tokens=150),
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
                text_buffer += token

                chunks, text_buffer = self._extract_tts_chunks(text_buffer)

                for chunk in chunks:
                    self._log(
                        f"Queueing TTS sentence: {chunk[:120]}"
                    )
                    await tts_queue.put(chunk)

            chunks, text_buffer = self._extract_tts_chunks(
                text_buffer,
                force=True,
            )

            for chunk in chunks:
                self._log(
                    f"Queueing final TTS sentence: {chunk[:120]}"
                )
                await tts_queue.put(chunk)

            response = "".join(response_parts).strip()

            self._log(
                f"LLM: {response[:160].replace(chr(10), ' ')}... "
                f"({time.perf_counter() - llm_start:.2f}s)"
            )

            if not response:
                self._log("Empty LLM response", "WARN")
                return

            # All Qwen sentences have been submitted. Let Breeze finish
            # synthesizing and let the playback worker drain them in order.
            await tts_queue.join()
            await audio_queue.join()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(
                f"LLM/TTS pipeline error: {exc}",
                "ERROR",
            )
        finally:
            # Stop synthesis worker only after all queued text is synthesized.
            if not synth_task.done():
                await tts_queue.put(None)

            await asyncio.gather(
                synth_task,
                return_exceptions=True,
            )

            # Then stop playback after every synthesized result has played.
            if not playback_task.done():
                await audio_queue.put(None)

            await asyncio.gather(
                playback_task,
                return_exceptions=True,
            )

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
        print(f"  TTS:      Breeze (local)")
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

            try:
                loop.run_until_complete(asyncio.sleep(0))
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
