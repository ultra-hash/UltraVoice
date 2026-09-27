"""Adapters — external service integrations."""

from runtime.adapters.audio_cpp_tts import AudioCppTTS
from runtime.adapters.breeze_tts import (
    BreezeHTTPAdapter,
    BreezeTTS,
    BreezeVoiceManager,
    BreezeWSAdapter,
    SynthesisResult,
    VoiceInfo,
)
from runtime.adapters.qwen_llm import (
    ChatMessage,
    ChatOptions,
    CompletionResult,
    QwenLLM,
)
from runtime.adapters.whisper_stt import (
    LocalWhisperAdapter,
    RemoteWhisperAdapter,
    TranscriptionResult,
    WhisperModel,
    WhisperSTT,
)

__all__ = [
    # audio.cpp (Kokoro-TTS)
    "AudioCppTTS",
    # Breeze TTS
    "BreezeHTTPAdapter",
    "BreezeTTS",
    "BreezeVoiceManager",
    "BreezeWSAdapter",
    "SynthesisResult",
    "VoiceInfo",
    # Qwen LLM
    "ChatMessage",
    "ChatOptions",
    "CompletionResult",
    "QwenLLM",
    # Whisper STT
    "LocalWhisperAdapter",
    "RemoteWhisperAdapter",
    "TranscriptionResult",
    "WhisperModel",
    "WhisperSTT",
]
