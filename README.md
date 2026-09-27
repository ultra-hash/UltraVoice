# 🎙️ UltraVoice

Real-time interactive voice agent with streaming LLM, STT, and TTS.

The live agent streams LLM tokens to audio.cpp (Kokoro-TTS) incrementally
— no waiting for the full response. A jitter buffer smooths TTS's variable
PCM delivery.

## Architecture

```
ultravoice/
├── config.yaml                      # Main configuration (YAML + env vars)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── runtime/
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # YAML config loader with env var substitution
│   ├── live_agent.py                # Real-time mic-to-speaker voice agent (CLI)
│   └── adapters/
│       ├── __init__.py              # Adapter package exports
│       ├── whisper_stt.py           # STT: Whisper API / local faster-whisper
│       ├── qwen_llm.py              # LLM: Qwen via OpenAI-compatible API
│       ├── breeze_tts.py            # TTS: Breeze-TTS-2 HTTP + WebSocket
│       └── audio_cpp_tts.py         # TTS: audio.cpp backend adapter
│
├── tests/
│   ├── __init__.py
│   ├── test_whisper_stt.py          # STT verification tests
│   ├── test_qwen_llm.py             # LLM verification tests
│   ├── test_breeze_tts.py           # TTS verification tests
│   ├── test_audio_cpp_tts.py        # audio.cpp / Kokoro-TTS tests
│   └── test_pipeline.py             # End-to-end pipeline tests
│
├── examples/
│   ├── whisper_stt_usage.py         # STT usage examples
│   ├── whisper_stt_test.py          # STT quick test script
│   ├── pipeline_example.py          # Pipeline usage examples
│   ├── breeze_tts_test.py           # TTS quick test script
│   ├── mic_probe.py                 # Microphone detection utility
│   └── mic_backend_probe.py         # Backend audio utility
│

└── audio samples / outputs (git-ignored)
    ├── breeze_test.wav
    ├── pipeline_output.wav
    └── pipeline_test*.wav
```

```
┌─────────────┐    ┌──────────┐    ┌───────┐    ┌──────────┐    ┌──────────┐
│  Capture    │───▶│   STT    │───▶│  LLM  │───▶│  TTS     │───▶│ Playback │
│  (Mic + VAD)│    │(Whisper) │    │ (Qwen)│    │(audio.cpp)│    │ (Speakers)│
└─────────────┘    └──────────┘    └───────┘    └──────────┘    └──────────┘
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start backend servers

| Service | Port | Description |
|---------|------|-------------|
| Whisper STT | `8081` | whisper.cpp (`/inference`) |
| Qwen LLM | `8080` | Qwen3.6-35B (OpenAI-compatible) |
| Breeze TTS | `8082` / `8083` | HTTP + WebSocket (optional) |
| audio.cpp | `8082` | Kokoro-TTS (used by live agent) |

> **Note**: Breeze TTS and audio.cpp both default to port `8082`. Only one
> should be running at a time. The live agent uses **audio.cpp** by default.

### 3. Configure

Edit `config.yaml` (defaults are pre-filled for local servers):

```yaml
llm:
  base_url: "http://127.0.0.1:8080"
  model: "Qwen3.6-35B-A3B-UD-Q4_K_M"
  temperature: 0.7
  max_tokens: 512

stt:
  base_url: "http://127.0.0.1:8081"
  model: "base"
  language: "en"

tts:
  provider: "breeze"              # HTTP/WS adapter (optional)
  base_url: "http://127.0.0.1:8082"
  instruction: "Speak clearly and naturally."
  cfg_scale: 1.0

  # audio.cpp (Kokoro-TTS) — used by live_agent.py
  audio_cpp_base_url: "http://127.0.0.1:8082"
  audio_cpp_model: "kokoro-tts"
  audio_cpp_sample_rate: 24000
```

Environment variables are substituted for API keys:
```bash
export QWEN_API_KEY="sk-..."
export WHISPER_API_KEY="sk-..."
export BREEZE_API_KEY="..."
```

## Components

### `runtime/adapters/`

| Module | Classes | Description |
|--------|---------|-------------|
| **whisper_stt.py** | `WhisperSTT`, `LocalWhisperAdapter`, `RemoteWhisperAdapter`, `WhisperModel`, `TranscriptionResult` | Speech-to-Text — OpenAI-compatible API or local faster-whisper |
| **qwen_llm.py** | `QwenLLM`, `ChatMessage`, `ChatOptions`, `CompletionResult` | LLM — Qwen via OpenAI-compatible API (vLLM, llama.cpp, Ollama) |
| **breeze_tts.py** | `BreezeTTS`, `BreezeHTTPAdapter`, `BreezeWSAdapter`, `BreezeVoiceManager`, `SynthesisResult`, `VoiceInfo` | Text-to-Speech — Breeze-TTS-2 HTTP + WebSocket |
| **audio_cpp_tts.py** | `AudioCppTTS`, `AudioCppTTSError` | Text-to-Speech — audio.cpp backend (Kokoro-TTS) |

### `runtime/`

| Module | Description |
|--------|-------------|
| **config.py** | YAML config loader with environment variable substitution |
| **live_agent.py** | Real-time mic-to-speaker voice agent (CLI entry point) |
| **\_\_init\_\_.py** | Package exports |

## Live Voice Agent

Real-time mic-to-speaker interaction with Silero VAD, streaming Qwen LLM →
audio.cpp (Kokoro-TTS), and jitter-buffered playback.

### Quick Start

```bash
# Run interactive voice agent
python -m runtime.live_agent

# Test without audio output (useful for debugging)
python -m runtime.live_agent --no-output

# Custom system prompt
python -m runtime.live_agent -s "You are a helpful coding assistant."

# Adjust VAD sensitivity
python -m runtime.live_agent --vad-threshold 0.50

# Tune silence detection (default: 650ms)
python -m runtime.live_agent --silence-ms 800

# Increase minimum speech duration (default: 300ms)
python -m runtime.live_agent --min-speech-ms 400

# Adjust pre-roll (audio kept before VAD triggers, default: 250ms)
python -m runtime.live_agent --pre-roll-ms 300
```

### Architecture

```
[Mic] → VAD → Buffer → STT → LLM → TTS → [Speaker]
```

**Key design decisions:**

| Feature | Implementation |
|---------|---------------|
| **Audio capture** | System default mic at native sample rate, 32 ms chunks |
| **VAD** | Silero VAD (16 kHz, threshold: 0.45) |
| **STT** | Remote whisper.cpp via `http://127.0.0.1:8081/inference` |
| **LLM** | Qwen3.6-35B with `enable_thinking: False` — normal content only |
| **TTS streaming** | Persistent audio.cpp WebSocket session — LLM tokens sent incrementally |
| **Text buffering** | Word-safe chunking — never cuts mid-word; sentences flush immediately |
| **Playback** | Jitter buffer (2 s startup) smooths audio.cpp's variable PCM delivery |
| **Turn management** | Single-threaded — drops speech if previous turn still processing |

### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capture_rate` | auto (device) | Mic sample rate |
| `chunk_ms` | 32 | Audio capture chunk size |
| `vad_threshold` | 0.45 | Speech probability threshold |
| `silence_duration_ms` | 650 | Silence before turn ends |
| `min_speech_ms` | 300 | Ignore speech shorter than this |
| `max_speech_seconds` | 30 | Hard limit on speech duration |
| `pre_roll_ms` | 250 | Audio retained before VAD triggers |
| `playback_start_buffer_ms` | 2000 | PCM pre-buffer before speaker starts |
| `playback_write_chunk_ms` | 80 | Playback write granularity |
| `tts_text_min_chars` | 150 | Min chars before releasing an audio.cpp text chunk |
| `tts_text_max_chars` | 600 | Max chars before forcing an audio.cpp text chunk |
| `tts_pcm_sample_rate` | 24000 | audio.cpp native PCM sample rate |
| `audio_queue_maxsize` | 180 | PCM queue capacity between receiver and playback |
| `llm_max_tokens` | 180 | Per-turn LLM response limit |
| `clean_tts_text` | true | Remove Markdown artifacts before TTS |

### Diagnostics

Each turn reports TTS diagnostics:

```
words=23, speech_rate=115 WPM, PCM=1.20s, production=2.34s,
producer_rate=0.51x realtime, buffer_min=0.42s, buffer_max=1.85s,
starvation_events=0, producer slower than realtime
```

| Metric | Meaning |
|--------|---------|
| `speech_rate` | Estimated WPM (words / generated PCM duration) |
| `producer_rate` | PCM production vs. realtime (≥1.0 = faster) |
| `buffer_min/max` | Jitter buffer range during playback |
| `starvation_events` | Times buffer dropped below safe level |

## Testing & Verification

### STT — Whisper (`tests/test_whisper_stt.py`)

| Test | What it verifies | Status |
|------|------------------|--------|
| Audio File Info | WAV validity, sample rate, channels, duration | ✅ Verified |
| WAV → PCM Conversion | Binary WAV decoding to float32 numpy array | ✅ Verified |
| Whisper API Connectivity | Server responds on configured endpoint | ✅ Verified |
| Full Transcription | End-to-end via `http://127.0.0.1:8081/inference` | ✅ Verified |
| WhisperSTT Class | `transcribe_file()`, `transcribe()`, `transcribe_stream()` | ✅ Verified |

**Verified Configuration**
```
Whisper server:  http://127.0.0.1:8081 (whisper.cpp)
Endpoint:        /inference
Model:           base
Language:        en
Test audio:      breeze_test.wav (202.5 KB, 4.32s, 24kHz mono)
Transcription:   "Hello, this is a test of my local Breeze text-to-speech system."
```

### Breeze TTS (`tests/test_breeze_tts.py`)

| Test | What it verifies | Status |
|------|------------------|--------|
| Server Health Check | HTTP server on port 8082, WS port discovery | ✅ Verified |
| HTTP Synthesis | `POST /v1/audio/speech` — text → s16le PCM | ✅ Verified |
| BreezeTTS Class | `from_config()`, `synthesize()`, `save_wav()` | ✅ Verified |
| Voice Manager | `list_voices()` via `GET /v1/voices` | ✅ Verified |

**Verified Configuration**
```
Breeze server:  http://127.0.0.1:8082
Endpoint:       /v1/audio/speech
WS port:        8083 (auto-discovered from /health)
Sample rate:    24000 Hz
PCM format:     signed 16-bit little endian (s16le)
```

---

### audio.cpp / Kokoro-TTS (`tests/test_audio_cpp_tts.py`)

| Test | What it verifies | Status |
|------|------------------|--------|
| Server Health Check | HTTP server on port 8082, Kokoro-TTS endpoint | ✅ Verified |
| HTTP Synthesis | `POST /v1/audio/speech` — text → WAV | ✅ Verified |
| AudioCppTTS Class | `persistent_session()`, `send_text()`, `end()` | ✅ Verified |
| from_config() | Config.yaml loading | ✅ Verified |
| Error Handling | Connection errors caught gracefully (no crash) | ✅ Verified |

**Verified Configuration**
```
audio.cpp server:  http://127.0.0.1:8082
Endpoint:          /v1/audio/speech
Model:             kokoro-tts
Sample rate:       24000 Hz
PCM format:        signed 16-bit little endian (s16le)
Mode:              offline (per-request WAV generation)
```

### Qwen LLM (`tests/test_qwen_llm.py`)

| Test | What it verifies | Status |
|------|------------------|--------|
| Health Check | Server reachable, model list | ✅ Verified |
| Simple Chat | Non-streaming chat completion | ✅ Verified |
| System Prompt | System instruction handling | ✅ Verified |
| Streaming Chat | Token-by-token streaming | ✅ Verified |
| Chat History | Multi-turn conversation | ✅ Verified |
| from_config() | Config.yaml loading | ✅ Verified |
| Token Counting | Rough token estimation | ✅ Verified |
| Sampling Options | Temperature, max_tokens, top_p | ✅ Verified |

**Verified Configuration**
```
LLM server:    http://127.0.0.1:8080
Endpoint:      /v1/chat/completions
Model:         Qwen3.6-35B-A3B-UD-Q4_K_M
API:           OpenAI-compatible (via openai library)
Reasoning:     `enable_thinking: False` — normal content only
```

### Voice Pipeline (`tests/test_pipeline.py`)

End-to-end: **Speech → Text → AI → Speech**

| Stage | What it does | Status |
|-------|--------------|--------|
| STT | Transcribe audio → text | ✅ Verified |
| LLM | Generate AI response | ✅ Verified |
| TTS | Synthesize speech from text | ✅ Verified |

**Test Results**
```
Input:  breeze_test.wav (207 KB, 4.32s, 24kHz)
STT:    "Hello, this is a test of my local Breeze text-to-speech system." (0.18s)
LLM:    "Hello! Test received loud and clear. How can I help you today?" (10.88s)
TTS:    4.56s @ 24kHz (6.48s wall)
Total:  17.53s end-to-end
```

### Running Tests

```bash
# STT tests
python tests/test_whisper_stt.py

# LLM tests
python tests/test_qwen_llm.py

# TTS tests
python tests/test_breeze_tts.py

# audio.cpp / Kokoro-TTS tests
python tests/test_audio_cpp_tts.py

# Pipeline test
python tests/test_pipeline.py
```

## Usage Examples

### STT (Whisper)

```python
from runtime.adapters.whisper_stt import WhisperSTT

stt = WhisperSTT.from_config()

# Transcribe a file
result = stt.transcribe_file("audio.wav")
print(result.text)

# Transcribe PCM array (from mic / VAD)
result = stt.transcribe(pcm_array, sample_rate=16000)
print(result.text)

# Stream chunks (VAD pipeline)
result = stt.transcribe_stream(chunks, sample_rate=16000)
print(result.text)
```

### LLM (Qwen)

```python
from runtime.adapters.qwen_llm import QwenLLM, ChatOptions

llm = QwenLLM.from_config()

# Non-streaming chat
result = llm.chat("Hello, who are you?")
print(result.text)           # Main response
print(result.reasoning_content)  # Internal reasoning (reasoning models)

# Streaming chat
for token in llm.chat_stream("Tell me a story"):
    print(token, end="")

# With history
result, history = llm.chat_with_history("My name is Alice.")
result, history = llm.chat_with_history("What is my name?", history=history)
print(result.text)  # "Your name is Alice."
```

### TTS (Breeze)

```python
from runtime.adapters.breeze_tts import BreezeTTS

tts = BreezeTTS.from_config(mode="http")

# Synthesize text
result = tts.synthesize("Hello world")
print(f"{result.duration_secs}s of audio")

# Save as WAV
BreezeTTS.save_wav(result, "output.wav")

# Play via sounddevice
BreezeTTS.play(result, blocking=True)
```

### TTS (audio.cpp / Kokoro-TTS)

```python
from runtime.adapters.audio_cpp_tts import AudioCppTTS
import asyncio

# Create adapter
async def main():
    tts = AudioCppTTS.from_config()
    text = "Hello from Kokoro!"
    print(f"Text: {text}")

    # Use persistent session for streaming
    async with tts.persistent_session() as session:
        await session.send_text(text)
        await session.end()

        # Receive PCM chunks
        pcm_chunks = []
        while True:
            msg = await session.receive()
            if isinstance(msg, bytes):
                pcm_chunks.append(msg)
            elif isinstance(msg, str):
                import json
                data = json.loads(msg)
                if data.get("type") == "done":
                    break

    pcm = b"".join(pcm_chunks)
    print(f"PCM: {len(pcm):,} bytes")
    # Save or play...

asyncio.run(main())
```

### Voice Pipeline

```python
from runtime.adapters.breeze_tts import BreezeTTS
from runtime.adapters.qwen_llm import QwenLLM
from runtime.adapters.whisper_stt import WhisperSTT
from tests.test_pipeline import VoicePipeline

pipeline = VoicePipeline.from_config()

# Run on audio file
result = pipeline.run("input.wav", system="Be friendly and concise.")
pipeline.print_summary(result)

# Save or play output
pipeline.save_audio(result, "output.wav")
# pipeline.play(result, blocking=True)
```

## Adapter API Reference

### WhisperSTT

```python
class WhisperSTT:
    @classmethod
    def from_config(cls, mode: str = "auto") -> WhisperSTT

    def transcribe(self, audio: np.ndarray | bytes, sample_rate: int = 16000, initial_prompt: str | None = None) -> TranscriptionResult
    def transcribe_file(self, path: str | Path, sample_rate: int = 16000) -> TranscriptionResult
    def transcribe_stream(self, audio_chunks: list[np.ndarray], sample_rate: int = 16000) -> TranscriptionResult
```

### QwenLLM

```python
class QwenLLM:
    @classmethod
    def from_config(cls) -> QwenLLM

    def chat(self, user_message: str, system: str | None = None, history: list[ChatMessage] | None = None, options: ChatOptions | None = None) -> CompletionResult
    def chat_with_history(self, user_message: str, system: str | None = None, history: list[dict] | None = None, options: ChatOptions | None = None) -> tuple[CompletionResult, list[dict]]
    async def chat_stream(self, user_message: str, system: str | None = None, history: list[ChatMessage] | None = None, options: ChatOptions | None = None) -> AsyncIterator[str]
    async def chat_stream_full(self, user_message: str, system: str | None = None, history: list[ChatMessage] | None = None, options: ChatOptions | None = None) -> CompletionResult
```

### BreezeTTS

```python
class BreezeTTS:
    @classmethod
    def from_config(cls, mode: str = "auto") -> BreezeTTS

    def synthesize(self, text: str, voice_id: str | None = None, ref_audio: np.ndarray | bytes | None = None, ref_text: str | None = None, instruction: str | None = None) -> SynthesisResult
    async def synthesize_stream_async(self, text: str, voice_id: str | None = None) -> AsyncIterator[bytes]
    async def speak(self, text: str, instruction: str | None = None) -> AsyncIterator[bytes]
    async def speak_full(self, text: str, instruction: str | None = None) -> SynthesisResult
    async def cancel(self) -> None
    @asynccontextmanager
    async def persistent_session(self) -> AsyncIterator[BreezeWSAdapter]
```

### AudioCppTTS

```python
class AudioCppTTS:
    @classmethod
    def from_config(cls) -> AudioCppTTS

    def __init__(self, base_url: str = "http://127.0.0.1:8082",
                  model: str = "kokoro-tts",
                  sample_rate: int = 24000,
                  voice: str | None = None,
                  speed: float | None = None,
                  instruction: str | None = None,
                  timeout: float = 120.0)

    def persistent_session(self) -> _Session
    async def send_text(self, text: str) -> None
    async def end(self) -> None
    async def receive(self) -> bytes | str
```

## Data Classes

### TranscriptionResult
```python
@dataclass
class TranscriptionResult:
    text: str
    language: str = "en"
    duration_secs: float = 0.0
    start_ms: int = 0
    end_ms: int = 0
    words: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    no_speech_prob: float = 0.0
```

### CompletionResult
```python
@dataclass
class CompletionResult:
    text: str
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_time_secs: float = 0.0
    reasoning_content: str = ""      # Reasoning models only
    raw: dict = field(default_factory=dict)
```

### SynthesisResult
```python
@dataclass(frozen=True)
class SynthesisResult:
    pcm: bytes                        # raw s16le PCM
    sample_rate: int = 24000
    duration_secs: float = 0.0
    text: str = ""
    voice_id: str = ""
    frames: int = 0                   # 1 frame = 80 ms = 1920 samples
```

## Configuration Schema

```yaml
app:
  name: "ultravoice"
  version: "0.1.0"
  debug: false

audio:
  sample_rate: 16000
  channels: 1
  sample_width: 2
  chunk_duration_ms: 20
  input_device: null
  output_device: null
  silence_threshold: 0.01
  silence_duration_ms: 600
  playback_buffer_size: 3

llm:
  provider: "qwen"
  model: "Qwen3.6-35B-A3B-UD-Q4_K_M"
  temperature: 0.7
  max_tokens: 512
  api_key: "${QWEN_API_KEY}"
  base_url: "http://127.0.0.1:8080"

stt:
  provider: "whisper"
  model: "base"
  language: "en"
  task: "transcribe"
  api_key: "${WHISPER_API_KEY}"
  base_url: "http://127.0.0.1:8081"

tts:
  provider: "breeze"
  model: "breeze-2"
  instruction: "Speak clearly and naturally."
  voice_id: null
  cfg_scale: 1.0
  temperature: null
  api_key: "${BREEZE_API_KEY}"
  base_url: "http://127.0.0.1:8082"

  # audio.cpp (Kokoro-TTS) settings — used by live_agent.py
  audio_cpp_base_url: "http://127.0.0.1:8082"
  audio_cpp_model: "kokoro-tts"
  audio_cpp_sample_rate: 24000
  voice: null
  speed: null

server:
  host: "0.0.0.0"
  port: 8765

logging:
  level: "INFO"
```

## License

MIT
