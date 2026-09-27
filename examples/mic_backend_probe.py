import time
import queue
import numpy as np
import sounddevice as sd

DEVICES = [7, 25]

for device in DEVICES:
    d = sd.query_devices(device)
    rate = int(d["default_samplerate"])
    channels = int(d["max_input_channels"])

    print("\n" + "=" * 70)
    print(f"DEVICE {device}: {d['name']}")
    print(f"HOST API: {sd.query_hostapis(d['hostapi'])['name']}")
    print(f"RATE: {rate}")
    print(f"CHANNELS: {channels}")
    print("=" * 70)

    q = queue.Queue()
    blocks = 0
    max_rms = 0.0
    max_peak = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal_dummy = None
        if status:
            print("STATUS:", status, flush=True)
        q.put(indata.copy())

    try:
        stream = sd.InputStream(
            device=device,
            channels=channels,
            samplerate=rate,
            blocksize=0,
            dtype="float32",
            callback=callback,
        )
        stream.start()
        print("Speak for 5 seconds...", flush=True)

        end = time.time() + 5
        while time.time() < end:
            try:
                audio = q.get(timeout=1.0)
            except queue.Empty:
                continue

            blocks += 1
            x = np.asarray(audio, dtype=np.float32)
            rms = float(np.sqrt(np.mean(x * x)))
            peak = float(np.max(np.abs(x)))
            max_rms = max(max_rms, rms)
            max_peak = max(max_peak, peak)

        stream.stop()
        stream.close()

        print(f"blocks={blocks}")
        print(f"max_rms={max_rms:.8f}")
        print(f"max_peak={max_peak:.8f}")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
