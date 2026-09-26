# -*- coding: utf-8 -*-
"""Test Whisper STT with breeze_test.wav"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import httpx

# Set stdout to UTF-8 to avoid encoding issues
import os
if hasattr(sys.stdout, 'buffer'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'


# Test 0: Audio file info
def test_audio_info() -> None:
    """Verify the audio file is valid and inspect its properties."""
    print("=" * 60)
    print("TEST 0: Audio File Info")
    print("=" * 60)

    path = Path("breeze_test.wav")
    if not path.exists():
        print(f"  [FAIL] {path} not found!")
        sys.exit(1)

    audio, sr = sf.read(str(path), dtype="float32")
    duration = len(audio) / sr if sr else 0
    channels = audio.ndim if audio.ndim > 1 else 1

    print(f"  [OK] File exists: {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  [OK] Sample rate: {sr} Hz")
    print(f"  [OK] Duration: {duration:.2f}s")
    print(f"  [OK] Channels: {channels}")
    print(f"  [OK] Samples: {len(audio):,}")
    print(f"  [OK] Range: [{audio.min():.4f}, {audio.max():.4f}]")


# Test 1: WAV bytes to PCM conversion
def test_wav_to_pcm() -> None:
    """Verify WAV bytes can be decoded to float32 PCM."""
    print("\n" + "=" * 60)
    print("TEST 1: WAV Bytes -> PCM Conversion")
    print("=" * 60)

    with open("breeze_test.wav", "rb") as f:
        wav_bytes = f.read()

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"  [OK] Decoded {len(pcm):,} samples")
    print(f"  [OK] Range: [{pcm.min():.4f}, {pcm.max():.4f}]")
    print(f"  [OK] dtype: {pcm.dtype}")


# Test 2: Whisper API connectivity
def test_api_connectivity() -> None:
    """Check if the Whisper API server responds."""
    print("\n" + "=" * 60)
    print("TEST 2: Whisper API Connectivity")
    print("=" * 60)

    base_url = "http://127.0.0.1:8081"

    try:
        resp = httpx.post(
            f"{base_url}/inference",
            files={"file": ("test.wav", b"", "audio/wav")},
            data={"model": "base", "language": "en", "response_format": "json"},
            timeout=5,
        )
        print(f"  [OK] Server responded with status {resp.status_code}")
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot connect to {base_url}")
        print(f"    Make sure whisper.cpp server is running on port 8081")
        sys.exit(1)
    except Exception as e:
        print(f"  [WARN] Connection test returned: {type(e).__name__}: {e}")


# Test 3: Full transcription via API
def test_full_transcription() -> None:
    """Transcribe the full breeze_test.wav via Whisper API."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Transcription (Whisper API)")
    print("=" * 60)

    base_url = "http://127.0.0.1:8081"

    audio, sr = sf.read("breeze_test.wav", dtype="float32")
    print(f"  Loading audio: {sr}Hz, {len(audio):,} samples")

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sr, format="WAV")
    wav_bytes = wav_buf.getvalue()
    print(f"  Encoded to WAV: {len(wav_bytes) / 1024:.1f} KB")

    print(f"  Sending to {base_url}/inference ...")
    resp = httpx.post(
        f"{base_url}/inference",
        files={
            "file": ("breeze_test.wav", wav_bytes, "audio/wav"),
        },
        data={
            "model": "base",
            "language": "en",
            "task": "transcribe",
            "response_format": "json",
            "temperature": "0.0",
            "no_speech_thold": "0.6",
        },
        timeout=120,
    )

    if resp.status_code != 200:
        print(f"  [FAIL] API returned status {resp.status_code}")
        print(f"    Response: {resp.text[:500]}")
        sys.exit(1)

    result = resp.json()
    text = result.get("text", "").strip()
    duration = result.get("duration", 0)
    language = result.get("language", "unknown")
    segments = result.get("segments", [])
    words = result.get("words", [])

    print(f"\n  {'='*50}")
    print(f"  TRANSCRIPTION RESULT")
    print(f"  {'='*50}")
    print(f"  Text: {text}")
    print(f"  Language: {language}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Segments: {len(segments)}")
    print(f"  Words: {len(words)}")

    if segments:
        print(f"\n  Segments:")
        for seg in segments[:5]:
            seg_text = seg.get("text", "").strip()
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            print(f"    [{seg_start:.1f}s - {seg_end:.1f}s] {seg_text}")

    if words:
        print(f"\n  Words (first 10):")
        for w in words[:10]:
            print(f"    '{w.get('word', '')}' [{w.get('start',0):.1f}s - {w.get('end',0):.1f}s]")

    print(f"\n  [OK] Transcription complete!")
    return text


# Test 4: Using the WhisperSTT class
def test_whisper_stt_class() -> None:
    """Test the WhisperSTT class directly."""
    print("\n" + "=" * 60)
    print("TEST 4: WhisperSTT Class")
    print("=" * 60)

    from runtime.adapters.whisper_stt import WhisperSTT, RemoteWhisperAdapter, WhisperModel

    adapter = RemoteWhisperAdapter(
        base_url="http://127.0.0.1:8081",
        model=WhisperModel.BASE,
        language="en",
        task="transcribe",
    )
    stt = WhisperSTT(adapter)

    print("  Transcribing breeze_test.wav via WhisperSTT class...")
    result = stt.transcribe_file("breeze_test.wav")

    print(f"\n  Text: {result.text}")
    print(f"  Language: {result.language}")
    print(f"  Duration: {result.duration_secs:.2f}s")
    print(f"  Words: {len(result.words)}")
    print(f"  [OK] Class works correctly!")


# Main
def main() -> None:
    print("\nULTRAVOICE - Whisper STT Test Suite")
    print("=" * 60)

    test_audio_info()
    test_wav_to_pcm()
    test_api_connectivity()
    text = test_full_transcription()
    test_whisper_stt_class()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
