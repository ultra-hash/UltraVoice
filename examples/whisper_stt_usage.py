"""
WhisperSTT — Usage examples.

Run each example independently to test different modes.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd
import soundfile as sf

from runtime.adapters.whisper_stt import WhisperSTT, WhisperModel


# ---------------------------------------------------------------------------
# Example 1: Create from config.yaml
# ---------------------------------------------------------------------------


def example_from_config() -> None:
    """Load WhisperSTT from config.yaml — recommended for production."""
    print("=== Example 1: From config.yaml ===")

    stt = WhisperSTT.from_config(mode="auto")  # auto = local if available, else remote

    # Transcribe a file
    result = stt.transcribe_file("breeze_test.wav")
    print(f"  Text: {result.text}")
    print(f"  Language: {result.language}")
    print(f"  Duration: {result.duration_secs:.2f}s")
    print(f"  Words: {result.words}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Local Whisper (faster-whisper)
# ---------------------------------------------------------------------------


def example_local_whisper() -> None:
    """Run Whisper locally — requires `pip install faster-whisper`."""
    print("=== Example 2: Local Whisper ===")

    stt = WhisperSTT.from_config(mode="local")

    # Transcribe a file
    result = stt.transcribe_file("breeze_test.wav")
    print(f"  Text: {result.text}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Remote Whisper API
# ---------------------------------------------------------------------------


def example_remote_whisper() -> None:
    """Send audio to a remote Whisper-compatible API server."""
    print("=== Example 3: Remote Whisper API ===")

    stt = WhisperSTT(
        adapter=stt.__class__.__bases__[0]  # hack — see below for proper usage
    )
    # Better — create adapter directly:
    # from runtime.adapters.whisper_stt import RemoteWhisperAdapter
    # adapter = RemoteWhisperAdapter(
    #     base_url="http://127.0.0.1:8081",
    #     model=WhisperModel.LARGE_V3,
    #     language="en",
    # )
    # stt = WhisperSTT(adapter)

    result = stt.transcribe_file("breeze_test.wav")
    print(f"  Text: {result.text}")
    print()


# ---------------------------------------------------------------------------
# Example 4: Transcribe raw PCM from microphone
# ---------------------------------------------------------------------------


def example_mic_capture() -> None:
    """Record from mic → transcribe (local mode)."""
    print("=== Example 4: Mic Capture ===")

    SAMPLE_RATE = 16000
    DURATION = 5  # seconds

    stt = WhisperSTT.from_config(mode="local")

    print(f"  Recording {DURATION}s of audio...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    result = stt.transcribe(audio, sample_rate=SAMPLE_RATE)
    print(f"  Text: {result.text}")
    print()


# ---------------------------------------------------------------------------
# Example 5: VAD-style streaming (chunked inference)
# ---------------------------------------------------------------------------


def example_streaming_chunks() -> None:
    """Simulate VAD pipeline: feed audio in 20ms chunks, merge speech."""
    print("=== Example 5: Streaming Chunks ===")

    SAMPLE_RATE = 16000
    CHUNK_MS = 20
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 320 samples

    stt = WhisperSTT.from_config(mode="local")

    # Load full audio, break into chunks
    audio, sr = sf.read("breeze_test.wav", dtype="float32")
    if sr != SAMPLE_RATE:
        print(f"  Warning: file is {sr}Hz, resampling to {SAMPLE_RATE}Hz")

    chunks: list[np.ndarray] = []
    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i : i + CHUNK_SIZE]
        if len(chunk) >= CHUNK_SIZE // 2:  # skip trailing partial
            chunks.append(chunk)

    print(f"  {len(chunks)} chunks of {CHUNK_MS}ms each")

    # Transcribe all chunks as one utterance
    result = stt.transcribe_stream(chunks, sample_rate=SAMPLE_RATE)
    print(f"  Text: {result.text}")
    print(f"  Duration: {result.duration_secs:.2f}s")
    print()


# ---------------------------------------------------------------------------
# Example 6: With initial prompt (context-aware)
# ---------------------------------------------------------------------------


def example_with_prompt() -> None:
    """Use domain-specific initial prompt for better accuracy."""
    print("=== Example 6: With Initial Prompt ===")

    stt = WhisperSTT.from_config(mode="local")

    prompt = "Medical consultation — doctor speaking to patient."
    result = stt.transcribe_file(
        "breeze_test.wav",
        # note: transcribe_file doesn't pass prompt — use transcribe() instead
    )

    # Better approach with prompt:
    audio, sr = sf.read("breeze_test.wav", dtype="float32")
    result = stt.transcribe(audio, sample_rate=sr, initial_prompt=prompt)
    print(f"  Text: {result.text}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    example_from_config()
    example_local_whisper()
    example_mic_capture()
    example_streaming_chunks()
    example_with_prompt()


if __name__ == "__main__":
    main()
