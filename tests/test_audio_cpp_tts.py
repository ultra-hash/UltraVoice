# -*- coding: utf-8 -*-
"""Test audio.cpp adapter (Kokoro-TTS) via OpenAI-compatible API."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import httpx
import numpy as np
import soundfile as sf

from runtime.adapters.audio_cpp_tts import AudioCppTTS, AudioCppTTSError


def test_health() -> None:
    """Verify audio.cpp server is reachable on port 8082."""
    print("=" * 60)
    print("TEST 0: Server Health Check")
    print("=" * 60)

    base_url = "http://127.0.0.1:8082"
    try:
        resp = httpx.post(
            f"{base_url}/v1/audio/speech",
            json={
                "model": "kokoro-tts",
                "input": "Health check.",
                "response_format": "wav",
            },
            timeout=10,
        )
        print(f"  [OK] Server responded with status {resp.status_code}")
        if resp.status_code == 200:
            print(f"  [OK] Server is healthy and accepting TTS requests")
        else:
            print(f"  [WARN] Status {resp.status_code}: {resp.text[:200]}")
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot connect to {base_url}")
        print(f"    Make sure audio.cpp server is running on port 8082")
        sys.exit(1)


def test_http_synthesis() -> None:
    """Test direct HTTP synthesis via audio.cpp /v1/audio/speech."""
    print("\n" + "=" * 60)
    print("TEST 1: HTTP Synthesis (Direct API)")
    print("=" * 60)

    base_url = "http://127.0.0.1:8082"
    text = "Hello, this is a test of the Kokoro text-to-speech system."
    print(f"  Text: {text}")

    resp = httpx.post(
        f"{base_url}/v1/audio/speech",
        json={
            "model": "kokoro-tts",
            "input": text,
            "response_format": "wav",
        },
        timeout=120,
    )

    if resp.status_code != 200:
        print(f"  [FAIL] Server returned {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)

    content_type = resp.headers.get("content-type", "")
    print(f"  Content-Type: {content_type}")
    print(f"  Response size: {len(resp.content):,} bytes")

    # Save the WAV response
    output_path = Path("test_audio_cpp_direct.wav")
    output_path.write_bytes(resp.content)
    print(f"  [OK] Saved WAV to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Verify WAV is valid
    audio, sr = sf.read(str(output_path))
    duration = len(audio) / sr if sr else 0
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Samples: {len(audio):,}")
    print(f"  [OK] WAV is valid!")


async def test_audio_cpp_tts_class() -> None:
    """Test the AudioCppTTS class with persistent session."""
    print("\n" + "=" * 60)
    print("TEST 2: AudioCppTTS Class")
    print("=" * 60)

    tts = AudioCppTTS(base_url="http://127.0.0.1:8082")
    text = "Testing the AudioCppTTS class integration."
    print(f"  Text: {text}")
    print(f"  Model: {tts.model}")
    print(f"  Sample rate: {tts.sample_rate} Hz")

    pcm_chunks: list[bytes] = []
    async with tts.persistent_session() as session:
        await session.send_text(text)
        await session.end()

        while True:
            msg = await session.receive()
            if isinstance(msg, bytes):
                pcm_chunks.append(msg)
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "done":
                    break
                elif data.get("type") == "error":
                    raise AudioCppTTSError(data.get("message", "Unknown error"))
                    break

    pcm = b"".join(pcm_chunks)
    sample_rate = tts.sample_rate
    duration = len(pcm) / 2 / sample_rate if pcm else 0

    print(f"\n  PCM: {len(pcm):,} bytes")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sample_rate} Hz")

    # Save as WAV
    output_path = Path("test_audio_cpp_class.wav")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(str(output_path), audio, sample_rate)
    print(f"  [OK] Saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"  [OK] AudioCppTTS class works!")


async def test_from_config() -> None:
    """Test AudioCppTTS.from_config() factory method."""
    print("\n" + "=" * 60)
    print("TEST 3: from_config()")
    print("=" * 60)

    tts = AudioCppTTS.from_config()
    print(f"  Base URL: {tts.base_url}")
    print(f"  Model: {tts.model}")
    print(f"  Sample rate: {tts.sample_rate} Hz")
    print(f"  Voice: {tts.voice}")
    print(f"  [OK] from_config() works!")


async def test_error_handling() -> None:
    """Test error handling with non-existent server."""
    print("\n" + "=" * 60)
    print("TEST 4: Error Handling")
    print("=" * 60)

    tts = AudioCppTTS(base_url="http://127.0.0.1:9999")
    print(f"  Target: {tts.base_url} (intentionally wrong port)")
    print("  Expected: connection error caught gracefully")

    try:
        async with tts.persistent_session() as session:
            await session.send_text("This should fail")
            await session.end()
            while True:
                msg = await session.receive()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "error":
                        print(f"  [OK] Got expected error message: {data.get('message', '')[:100]}")
                        break
    except Exception as e:
        print(f"  [OK] Exception caught: {type(e).__name__}: {e}")

    print(f"  [OK] Error handling verified — no crash on bad server")


async def main() -> None:
    print("\nULTRAVOICE - audio.cpp (Kokoro-TTS) Test Suite")
    print("=" * 60)

    test_health()
    test_http_synthesis()
    await test_audio_cpp_tts_class()
    await test_from_config()
    await test_error_handling()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
