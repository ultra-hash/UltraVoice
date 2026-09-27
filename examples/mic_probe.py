# -*- coding: utf-8 -*-
"""Minimal Windows default microphone probe.

No Whisper, VAD, LLM, TTS, asyncio, or speaker playback.
It verifies whether sounddevice itself receives non-zero microphone samples.
"""

from __future__ import annotations

import queue
import time
import numpy as np
import sounddevice as sd

CHANNELS = 1
CHUNK_MS = 32

q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=100)


def callback(indata, frames, time_info, status):
    if status:
        print(f"AUDIO STATUS: {status}", flush=True)
    audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
    try:
        q.put_nowait(audio)
    except queue.Full:
        try:
            q.get_nowait()
            q.put_nowait(audio)
        except (queue.Empty, queue.Full):
            pass


def main():
    dev = sd.query_devices(kind="input")
    rate = int(round(dev["default_samplerate"]))
    block = max(1, round(rate * CHUNK_MS / 1000))

    print("DEFAULT INPUT:", dev["name"])
    print("RATE:", rate)
    print("BLOCK:", block)
    print("CHANNELS:", CHANNELS)
    print("Speak into the microphone for 10 seconds...", flush=True)

    stream = sd.InputStream(
        device=None,
        channels=CHANNELS,
        samplerate=rate,
        blocksize=block,
        dtype="float32",
        callback=callback,
    )
    stream.start()

    started = time.monotonic()
    blocks = 0
    max_rms = 0.0
    max_peak = 0.0
    next_log = started + 1.0

    try:
        while time.monotonic() - started < 10:
            try:
                audio = q.get(timeout=1.0)
            except queue.Empty:
                continue
            blocks += 1
            rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            max_rms = max(max_rms, rms)
            max_peak = max(max_peak, peak)

            now = time.monotonic()
            if now >= next_log:
                print(
                    f"{blocks=}; current_rms={rms:.6f}; current_peak={peak:.6f}; "
                    f"max_rms={max_rms:.6f}; max_peak={max_peak:.6f}",
                    flush=True,
                )
                next_log += 1.0
    finally:
        stream.stop()
        stream.close()

    print("\nFINAL:")
    print(f"blocks={blocks}")
    print(f"max_rms={max_rms:.6f}")
    print(f"max_peak={max_peak:.6f}")


if __name__ == "__main__":
    main()
