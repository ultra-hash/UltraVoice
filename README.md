# 🎙️ Ultravoice

Real-time interactive voice agent with streaming LLM, STT, and TTS.

## Architecture

```
┌─────────────┐    ┌──────────┐    ┌───────┐    ┌───────┐    ┌──────────┐
│  Capture    │───▶│   STT    │───▶│  LLM  │───▶│  TTS  │───▶│ Playback │
│  (Mic + VAD)│    │ (Whisper)│    │(Qwen) │    │(Breeze)│    │ (Speakers)│
└─────────────┘    └──────────┘    └───────┘    └───────┘    └──────────┘
       ▲                                                    │
       │              ┌─────────────────────────────────────┘
       │              ▼
       └────── TurnController (state machine)
              ┌──────────┐
              │ Event Bus │◀── all components connected
              └──────────┘
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml`:

```yaml
llm:
  api_key: "your-qwen-api-key"

stt:
  api_key: "your-whisper-api-key"

tts:
  api_key: "your-breeze-api-key"
```

Or use environment variables:
```bash
export QWEN_API_KEY="sk-..."
export WHISPER_API_KEY="sk-..."
export BREEZE_API_KEY="..."
```

### 3. Run

```bash
# Direct
python -m runtime.main

# Or via uvicorn
uvicorn runtime.main:app --host 0.0.0.0 --port 8765
```

## API

### HTTP

| Method | Path           | Description          |
|--------|----------------|----------------------|
| GET    | `/health`      | Health check         |
| POST   | `/session`     | Create a session     |

### WebSocket

| Path                        | Description                        |
|-----------------------------|------------------------------------|
| `/ws/{session_id}`         | Connect to existing session        |
| `/ws`                       | Auto-create session                |

Send/receive raw PCM (16kHz, 16-bit, mono) over the WebSocket.

## Components

### `runtime/core/`
- **event_bus.py** — Async pub/sub event backbone
- **session_manager.py** — Session lifecycle management
- **turn_controller.py** — User↔Agent turn protocol state machine

### `runtime/adapters/`
- **whisper_stt.py** — Speech-to-Text (Whisper API/local)
- **qwen_llm.py** — LLM (Qwen via OpenAI-compatible API)
- **breeze_tts.py** — Text-to-Speech (Breeze)

### `runtime/audio/`
- **capture.py** — Mic capture with Silero VAD
- **sentence_buffer.py** — TTS sentence queue with pre-buffering
- **playback.py** — Speaker output streaming

## Testing & Verification

### Running Tests

```bash
# Run all STT verification tests
python tests/test_whisper_stt.py
```

### Test Suite (`tests/test_whisper_stt.py`)

| Test | What it verifies | Status |
|---|---|---|
| **Audio File Info** | WAV file validity, sample rate, channels, duration | Verified |
| **WAV → PCM Conversion** | Binary WAV decoding to float32 numpy array | Verified |
| **Whisper API Connectivity** | Server responds on configured endpoint | Verified |
| **Full Transcription** | End-to-end transcription via `http://127.0.0.1:8081/inference` | Verified |
| **WhisperSTT Class** | Class integration — `transcribe_file()`, `transcribe()`, `transcribe_stream()` | Verified |

### Verified Configuration

```
Whisper server:  http://127.0.0.1:8081 (whisper.cpp)
Endpoint:        /inference
Model:           base
Language:        en
Sample rate:     16000 Hz (config), 24000 Hz (test audio)
```

### Test Results

**Audio file:** `breeze_test.wav` (202.5 KB, 4.32s, 24kHz mono)

**Transcription result:**
```
Text:   Hello, this is a test of my local Breeze text-to-speech system.
Language: en
Duration: 4.32s
```

All 5 tests passed. The STT pipeline correctly:
1. Reads WAV files and converts to float32 PCM
2. Encodes audio to WAV bytes for the API
3. Sends to the whisper.cpp server endpoint
4. Parses the JSON response into `TranscriptionResult`
5. Supports streaming chunks via `transcribe_stream()`

### Example Usage

```python
from runtime.adapters.whisper_stt import WhisperSTT, WhisperModel

# From config.yaml
stt = WhisperSTT.from_config()

# Transcribe a file
result = stt.transcribe_file("audio.wav")
print(result.text)  # "Hello, this is a test..."

# Transcribe PCM array (from mic / VAD)
result = stt.transcribe(pcm_array, sample_rate=16000)
print(result.text)

# Stream chunks (VAD pipeline)
result = stt.transcribe_stream(chunks, sample_rate=16000)
print(result.text)
```

---

### Breeze TTS (`tests/test_breeze_tts.py`)

| Test | What it verifies | Status |
|---|---|---|
| **Server Health Check** | HTTP server on port 8082, WS port discovery | Verified |
| **HTTP Synthesis** | `POST /v1/audio/speech` — text → s16le PCM | Verified |
| **BreezeTTS Class** | `from_config()`, `synthesize()`, `save_wav()` | Verified |
| **Voice Manager** | `list_voices()` via `GET /v1/voices` | Verified |

**Verified Configuration**
```
Breeze server:  http://127.0.0.1:8082
Endpoint:       /v1/audio/speech
WS port:        8083 (auto-discovered from /health)
Sample rate:    24000 Hz
PCM format:     signed 16-bit little endian (s16le)
```

**Test Result**
```
Text: Hello, this is a test of the Breeze text-to-speech system.
PCM:  192,000 bytes (4.00s @ 24kHz)
```

TTS pipeline correctly:
1. Sends text + params via multipart form POST
2. Receives raw s16le PCM (no WAV header)
3. Converts PCM → float32 for playback / WAV saving
4. Supports voice cloning via `ref_audio` + `ref_text`
5. WebSocket mode available for streaming text sessions (code present, needs WS-enabled server)

**Example Usage**

```python
from runtime.adapters.breeze_tts import BreezeTTS

# From config.yaml
tts = BreezeTTS.from_config(mode="http")

# Synthesize text
result = tts.synthesize("Hello world")
print(f"{result.duration_secs}s of audio")

# Save as WAV
BreezeTTS.save_wav(result, "output.wav")

# Play via sounddevice
BreezeTTS.play(result, blocking=True)
```

---

### Qwen LLM (`tests/test_qwen_llm.py`)

| Test | What it verifies | Status |
|---|---|---|
| **Health Check** | Server reachable, model list | Verified |
| **Simple Chat** | Non-streaming chat completion | Verified |
| **System Prompt** | System instruction handling | Verified |
| **Streaming Chat** | Token-by-token streaming via OpenAI client | Verified |
| **Chat History** | Multi-turn conversation with state | Verified |
| **from_config()** | Config.yaml loading | Verified |
| **Token Counting** | Rough token estimation utility | Verified |
| **Sampling Options** | Temperature, max_tokens, top_p | Verified |

**Verified Configuration**
```
LLM server:    http://127.0.0.1:8080
Endpoint:      /v1/chat/completions
Model:         Qwen3.6-35B-A3B-UD-Q4_K_M
API:           OpenAI-compatible (via openai library)
Note:          Reasoning model — outputs reasoning_content in streaming
```

**Test Results**
```
Test 1 - Simple Chat:
  Prompt:  What is 2+2?
  Response:  Here's a thinking process... (reasoning: 301 chars)
  Tokens:  17 in, 100 out | Time: 4-7s

Test 4 - Chat History:
  You:  My name is Alice.
  Bot:  (reasoning...) 
  You:  What is my name?
  Bot:  Your name is Alice.
```

LLM adapter correctly:
1. Connects to OpenAI-compatible server via the `openai` Python library
2. Handles reasoning models (Qwen, DeepSeek) that return `reasoning_content` separately
3. Supports streaming via `chat_stream()` — yields tokens as they arrive
4. Manages conversation history via `chat_with_history()`
5. Works from config.yaml with env var substitution
6. No API key required for local servers (uses placeholder)

**Example Usage**

```python
from runtime.adapters.qwen_llm import QwenLLM, ChatOptions

# From config.yaml
llm = QwenLLM.from_config()

# Non-streaming chat
result = llm.chat("Hello, who are you?")
print(result.text)         # Main response
print(result.reasoning_content)  # Internal reasoning (reasoning models)

# Streaming chat
for token in llm.chat_stream("Tell me a story"):
    print(token, end="")

# With history
result, history = llm.chat_with_history("My name is Alice.")
result, history = llm.chat_with_history("What is my name?", history=history)
print(result.text)  # "Your name is Alice."

# Custom options
result = llm.chat("Count to 10",
    options=ChatOptions(temperature=0.3, max_tokens=50))
```

---

### Voice Pipeline (`tests/test_pipeline.py`)

End-to-end pipeline: **Speech → Text → AI → Speech**

| Stage | What it does | Status |
|---|---|---|
| **STT** | Transcribe audio → text (Whisper) | Verified |
| **LLM** | Generate AI response (Qwen) | Verified |
| **TTS** | Synthesize speech from text (Breeze) | Verified |

**Test Results**
```
Input:  breeze_test.wav (207KB, 4.32s, 24kHz)
STT:    "Hello, this is a test of my local Breeze text-to-speech system." (0.18s)
LLM:    "Hello! Test received loud and clear. How can I help you today?" (10.88s)
TTS:    4.56s @ 24kHz (6.48s wall) 
Total:  17.53s end-to-end
```

**Usage**

```python
from tests.test_pipeline import VoicePipeline

# Create pipeline
pipeline = VoicePipeline.from_config()

# Run on audio file
result = pipeline.run("input.wav",
    system="You are a friendly assistant. Be concise.")

# Print summary
pipeline.print_summary(result)

# Save or play output
pipeline.save_audio(result, "output.wav")
# pipeline.play(result, blocking=True)

# Or pass raw audio bytes
import pathlib
audio_bytes = pathlib.Path("input.wav").read_bytes()
result = pipeline.run(audio_bytes, system="Be helpful and brief.")
```

**Streaming Mode**

```python
import asyncio

async for partial in pipeline.run_stream("input.wav", system="Be concise."):
    print(partial.llm_response, end="")
    # Partial audio available as partial.output_pcm
```

---

### Live Voice Agent (`examples/live_agent.py`)

Real-time mic-to-speaker voice interaction with Silero VAD.

**Architecture:**
```
[Mic] → VAD → Buffer → STT → LLM → TTS → [Speaker]
              ↓
        Speech Detection
```

**Usage:**

```bash
# Run interactive voice agent
python examples/live_agent.py

# Use specific microphone
python examples/live_agent.py --device 1

# Test without audio output
python examples/live_agent.py --no-output

# Custom system prompt
python examples/live_agent.py -s "You are a helpful coding assistant."

# List audio devices
python examples/live_agent.py --list
```

**Features:**
- Silero VAD for voice activity detection
- Automatic turn detection (speech start/end)
- Configurable silence threshold and duration
- Works with any audio device
- Graceful Ctrl+C shutdown

## License

MIT
