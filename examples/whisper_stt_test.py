import io
import sounddevice as sd
import soundfile as sf
import httpx

SAMPLE_RATE = 16000
CHANNELS = 1

print("Speak...")

audio = sd.rec(
    int(5 * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
)

sd.wait()

wav = io.BytesIO()

sf.write(
    wav,
    audio,
    SAMPLE_RATE,
    format="WAV",
)

wav.seek(0)

response = httpx.post(
    "http://127.0.0.1:8081/inference",
    files={
        "file": (
            "mic.wav",
            wav,
            "audio/wav",
        )
    },
    data={
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "no_speech_thold": "0.6",
        "response_format": "json",
    },
    timeout=120,
)

response.raise_for_status()

print(response.json()["text"])