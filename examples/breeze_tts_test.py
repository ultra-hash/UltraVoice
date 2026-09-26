from __future__ import annotations

import asyncio
import json
import wave

import numpy as np
import sounddevice as sd
import websockets


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

WS_PORT = 8083
BREEZE_URL = f"ws://127.0.0.1:{WS_PORT}"

TEXT = "Hello, this is a test of my local Breeze text to speech system."
INSTRUCTION = "Speak naturally and clearly."

VOICE_ID = ""
CFG_SCALE = 1.0
SEED = 42

SAMPLE_RATE = 24000
CHANNELS = 1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main():

    print(f"Connecting to Breeze: {BREEZE_URL}")

    pcm_chunks = []
    sample_rate = SAMPLE_RATE

    async with websockets.connect(BREEZE_URL) as ws:

        # ----------------------------------------------------------
        # Wait for server ready message
        # ----------------------------------------------------------

        message = await ws.recv()

        if isinstance(message, bytes):
            raise RuntimeError("Expected JSON ready message, got audio.")

        ready = json.loads(message)

        print("Server:", ready)

        if ready.get("type") != "ready":
            raise RuntimeError(f"Unexpected first message: {ready}")

        sample_rate = ready.get("sample_rate", SAMPLE_RATE)

        print(f"Sample rate: {sample_rate} Hz")

        # ----------------------------------------------------------
        # Start generation
        # ----------------------------------------------------------

        start_message = {
            "type": "start",
            "voice_id": VOICE_ID,
            "instruction": INSTRUCTION,
            "ref_text": "",
            "cfg_scale": CFG_SCALE,
            "seed": SEED,
        }

        await ws.send(json.dumps(start_message))

        # ----------------------------------------------------------
        # Send text
        # ----------------------------------------------------------

        await ws.send(json.dumps({
            "type": "end",
            "text": TEXT,
        }))

        print("Text sent. Receiving audio...")

        # ----------------------------------------------------------
        # Receive audio
        # ----------------------------------------------------------

        while True:

            message = await ws.recv()

            # Binary = PCM audio
            if isinstance(message, bytes):

                pcm_chunks.append(message)

                total_bytes = sum(len(x) for x in pcm_chunks)

                duration = total_bytes / 2 / sample_rate

                print(
                    f"\rReceived: {duration:.2f}s",
                    end="",
                    flush=True,
                )

                continue

            # JSON = server event
            event = json.loads(message)

            event_type = event.get("type")

            if event_type == "queued":
                print("\nQueued - waiting for GPU...")

            elif event_type == "done":
                print("\nGeneration complete.")
                break

            elif event_type == "cancelled":
                print("\nGeneration cancelled.")
                break

            elif event_type == "error":
                raise RuntimeError(
                    event.get("message", "Unknown Breeze error")
                )

            else:
                print(f"\nServer event: {event}")

    # ----------------------------------------------------------------
    # Combine PCM
    # ----------------------------------------------------------------

    pcm = b"".join(pcm_chunks)

    print(f"Received {len(pcm):,} bytes of PCM")

    if not pcm:
        raise RuntimeError("Breeze returned no audio.")

    # ----------------------------------------------------------------
    # Save raw PCM
    # ----------------------------------------------------------------

    with open("breeze_test.pcm", "wb") as f:
        f.write(pcm)

    # ----------------------------------------------------------------
    # Save WAV
    # ----------------------------------------------------------------

    with wave.open("breeze_test.wav", "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    print("Saved:")
    print("  breeze_test.pcm")
    print("  breeze_test.wav")

    # ----------------------------------------------------------------
    # Play
    # ----------------------------------------------------------------

    print("Playing audio...")

    audio = np.frombuffer(pcm, dtype=np.int16)

    sd.play(
        audio,
        samplerate=sample_rate,
        blocking=True,
    )

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())