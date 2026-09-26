# -*- coding: utf-8 -*-
"""Test Breeze TTS — HTTP synthesis primary test."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

from runtime.adapters.breeze_tts import (
    BreezeHTTPAdapter,
    BreezeTTS,
    BreezeVoiceManager,
)


def test_health() -> None:
    print("=" * 60)
    print("TEST 0: Server Health Check")
    print("=" * 60)
    resp = httpx.get("http://127.0.0.1:8082/health", timeout=5)
    health = resp.json()
    print(f"  [OK] HTTP on 8082: rate={health.get('sample_rate')}, ws={health.get('ws_port')}")


def test_http_synthesis() -> None:
    print("\n" + "=" * 60)
    print("TEST 1: HTTP Synthesis")
    print("=" * 60)

    adapter = BreezeHTTPAdapter(
        base_url="http://127.0.0.1:8082",
        instruction="Speak clearly and naturally.",
        cfg_scale=1.0,
        seed=42,
    )

    text = "Hello, this is a test of the Breeze text-to-speech system."
    print(f"  Text: {text}")

    result = adapter.synthesize(text)
    sample_rate = result.sample_rate
    duration = result.duration_secs
    num_samples = len(result.pcm) // 2
    dtype = np.frombuffer(result.pcm, dtype=np.int16)

    print(f"\n  {'='*50}")
    print(f"  SYNTHESIS RESULT")
    print(f"  {'='*50}")
    print(f"  PCM: {len(result.pcm):,} bytes")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Samples: {num_samples:,}")
    print(f"  Range: [{dtype.min()}, {dtype.max()}]")

    output_path = Path("test_breeze_http.wav")
    audio = np.frombuffer(result.pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(str(output_path), audio, sample_rate)
    print(f"  [OK] Saved to {output_path}")
    print(f"  [OK] HTTP synthesis complete!")


def test_class() -> None:
    print("\n" + "=" * 60)
    print("TEST 2: BreezeTTS Class")
    print("=" * 60)

    tts = BreezeTTS.from_config(mode="http")
    text = "Testing the BreezeTTS class integration."
    print(f"  Text: {text}")
    print(f"  Mode: {tts._mode}")

    result = tts.synthesize(text)
    print(f"\n  Text: {result.text}")
    print(f"  Duration: {result.duration_secs:.2f}s")
    print(f"  PCM: {len(result.pcm):,} bytes")

    output_path = Path("test_breeze_class.wav")
    BreezeTTS.save_wav(result, output_path)
    print(f"  [OK] Saved to {output_path}")
    print(f"  [OK] Class works!")


def test_voices() -> None:
    print("\n" + "=" * 60)
    print("TEST 3: Voice Manager")
    print("=" * 60)

    manager = BreezeVoiceManager(base_url="http://127.0.0.1:8082")
    voices = manager.list_voices()

    if not voices:
        print(f"  [INFO] No voices registered")
    else:
        print(f"  Found {len(voices)} voice(s):")
        for v in voices:
            saved_str = "saved" if v.saved else "cached"
            print(f"    - {v.id} ({v.seconds:.1f}s) [{saved_str}]")

    print(f"  [OK] Voice manager works!")


def main() -> None:
    print("\nULTRAVOICE - Breeze TTS Test Suite")
    print("=" * 60)

    test_health()
    test_http_synthesis()
    test_class()
    test_voices()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
