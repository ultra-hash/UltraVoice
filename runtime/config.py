"""Configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    chunk_duration_ms: int = 20
    input_device: int | None = None
    output_device: int | None = None
    silence_threshold: float = 0.01
    silence_duration_ms: int = 600
    playback_buffer_size: int = 3


@dataclass
class STTConfig:
    provider: str = "whisper"
    model: str = "base"
    language: str = "en"
    task: str = "transcribe"
    api_key: str = ""
    base_url: str = ""


@dataclass
class LLMConfig:
    provider: str = "qwen"
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    api_key: str = ""
    base_url: str = ""


@dataclass
class TTSConfig:
    provider: str = "breeze"
    model: str = "breeze-2"
    instruction: str = "Speak clearly and naturally."
    voice_id: str | None = None
    cfg_scale: float = 1.0
    temperature: float | None = None
    api_key: str = ""
    base_url: str = ""


@dataclass
class AppConfig:
    name: str = "ultravoice"
    version: str = "0.1.0"
    debug: bool = False


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


# ---------------------------------------------------------------------------
# Settings (singleton)
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _resolve_env_vars(obj: dict[str, Any]) -> dict[str, Any]:
    """Replace ${ENV_VAR} placeholders with actual environment values."""
    result = {}
    for key, value in obj.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_key = value[2:-1]
            result[key] = __import__("os").environ.get(env_key, "")
        elif isinstance(value, dict):
            result[key] = _resolve_env_vars(value)
        else:
            result[key] = value
    return result


def _dict_to_settings(d: dict[str, Any]) -> Settings:
    """Flatten nested config into Settings dataclass."""
    return Settings(
        app=AppConfig(**d.get("app", {})),
        audio=AudioConfig(**d.get("audio", {})),
        stt=STTConfig(**d.get("stt", {})),
        llm=LLMConfig(**d.get("llm", {})),
        tts=TTSConfig(**d.get("tts", {})),
        server=ServerConfig(**d.get("server", {})),
        logging=LoggingConfig(**d.get("logging", {})),
    )


def load_config(path: str | Path | None = None) -> Settings:
    """Load config.yaml, resolve env vars, return Settings."""
    load_dotenv()

    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        return Settings()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    resolved = _resolve_env_vars(raw)
    return _dict_to_settings(resolved)


# Global singleton
settings = load_config()
