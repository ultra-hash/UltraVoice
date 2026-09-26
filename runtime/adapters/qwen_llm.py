"""
Qwen LLM adapter using the OpenAI Python library.

Connects to any OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, etc.).
Supports streaming and non-streaming chat completions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional

from openai import OpenAI, AsyncOpenAI, NOT_GIVEN


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message."""

    role: str                     # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None


@dataclass
class CompletionResult:
    """Result from a chat completion call."""

    text: str
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_time_secs: float = 0.0
    reasoning_content: str = ""  # For reasoning models (Qwen, DeepSeek)
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChatOptions:
    """Sampling options for chat completion."""

    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    top_k: int = 0               # 0 = model default
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop: list[str] | None = None


# ---------------------------------------------------------------------------
# Qwen LLM Class (OpenAI-compatible)
# ---------------------------------------------------------------------------


class QwenLLM:
    """LLM adapter using the OpenAI Python library.

    Works with any OpenAI-compatible server: vLLM, llama.cpp, Ollama,
    Together AI, OpenRouter, OpenAI itself, etc.

    Usage:
        llm = QwenLLM.from_config()                    # reads config.yaml
        llm = QwenLLM(base_url="http://127.0.0.1:8080")

        result = llm.chat("Tell me a joke", system="You are helpful.")
        print(result.text)

        # Streaming
        async for chunk in llm.chat_stream("Tell me a story"):
            print(chunk.text, end="")
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: float = 120.0,
        default_options: ChatOptions | None = None,
    ) -> None:
        # Use placeholder if no real key provided (local servers often don't require auth)
        key = api_key if api_key and api_key not in ("not-needed", "", "null") else "local-key"
        self._client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=timeout,
        )
        self._async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=key,
            timeout=timeout,
        )
        self._model = model
        self._timeout = timeout
        self._default_options = default_options or ChatOptions()

    # ---- Factory --------------------------------------------------------

    @classmethod
    def from_config(cls) -> "QwenLLM":
        """Create from config.yaml (llm section)."""
        from runtime.config import settings

        llm_cfg = settings.llm

        # Handle both dict-style and dataclass-style configs
        if isinstance(llm_cfg, dict):
            base_url = llm_cfg.get("base_url", "http://127.0.0.1:8080")
            model = llm_cfg.get("model", "")
            api_key = llm_cfg.get("api_key", "not-needed")
            temperature = llm_cfg.get("temperature", 0.7)
            max_tokens = llm_cfg.get("max_tokens", 512)
        else:
            base_url = getattr(llm_cfg, "base_url", "http://127.0.0.1:8080")
            model = getattr(llm_cfg, "model", "")
            api_key = getattr(llm_cfg, "api_key", "not-needed")
            temperature = getattr(llm_cfg, "temperature", 0.7)
            max_tokens = getattr(llm_cfg, "max_tokens", 512)

        options = ChatOptions(temperature=temperature, max_tokens=max_tokens)
        return cls(base_url=base_url, model=model, api_key=api_key, default_options=options)

    # ---- Non-streaming chat ---------------------------------------------

    def chat(
        self,
        user_message: str,
        system: str | None = None,
        history: list[ChatMessage] | None = None,
        options: ChatOptions | None = None,
    ) -> CompletionResult:
        """Non-streaming chat completion.

        Args:
            user_message: The user's input text.
            system: System prompt.
            history: Previous conversation messages.
            options: Sampling options (overrides defaults).
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})

        opts = options or self._default_options
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": opts.max_tokens,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            # Qwen3.6 defaults to thinking with this server/template.
            # Disable it explicitly so the final answer is emitted as
            # normal assistant content instead of consuming the token
            # budget in reasoning_content.
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        }

        if opts.top_k > 0:
            params["top_k"] = opts.top_k
        if opts.presence_penalty != 0.0:
            params["presence_penalty"] = opts.presence_penalty
        if opts.frequency_penalty != 0.0:
            params["frequency_penalty"] = opts.frequency_penalty
        if opts.seed is not None:
            params["seed"] = opts.seed
        if opts.stop:
            params["stop"] = opts.stop

        import time
        start = time.time()

        response = self._client.chat.completions.create(**params)

        elapsed = time.time() - start
        choice = response.choices[0]

        # Handle reasoning models (Qwen, DeepSeek) that split content from reasoning
        text = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or ""

        return CompletionResult(
            text=text,
            model=response.model,
            finish_reason=choice.finish_reason or "",
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            response_time_secs=round(elapsed, 3),
            reasoning_content=reasoning,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def chat_with_history(
        self,
        user_message: str,
        system: str | None = None,
        history: list[dict] | None = None,
        options: ChatOptions | None = None,
    ) -> tuple[CompletionResult, list[dict]]:
        """Chat that returns updated history for continuation.

        Args:
            user_message: User input.
            system: System prompt.
            history: Existing conversation history (will be updated).
            options: Sampling options.

        Returns:
            (result, updated_history)
        """
        if history is None:
            history = []

        # Build full message list including system
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # Call low-level chat
        import time
        opts = options or self._default_options
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": opts.max_tokens,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
        }

        start = time.time()
        response = self._client.chat.completions.create(**params)
        elapsed = time.time() - start
        choice = response.choices[0]

        # Reasoning is deliberately kept separate from the assistant response.
        # Never fall back to reasoning_content for user-visible/TTS output.
        text = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or ""

        result = CompletionResult(
            text=text,
            model=response.model,
            finish_reason=choice.finish_reason or "",
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            response_time_secs=round(elapsed, 3),
            reasoning_content=reasoning,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

        # Update history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": text})

        return result, history

    # ---- Streaming chat -------------------------------------------------

    async def chat_stream(
        self,
        user_message: str,
        system: str | None = None,
        history: list[ChatMessage] | None = None,
        options: ChatOptions | None = None,
    ) -> AsyncIterator[str]:
        """Stream only assistant response content.

        ``reasoning_content`` from reasoning models such as Qwen is
        intentionally ignored and is never yielded to the caller/TTS.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})

        opts = options or self._default_options
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": opts.max_tokens,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            "stream": True,
            # Critical for Qwen3.6 + llama.cpp: request non-thinking mode.
            # This makes the model emit the user-facing answer in
            # delta.content. We still never forward reasoning_content.
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        }

        if opts.top_k > 0:
            params["top_k"] = opts.top_k
        if opts.presence_penalty != 0.0:
            params["presence_penalty"] = opts.presence_penalty
        if opts.frequency_penalty != 0.0:
            params["frequency_penalty"] = opts.frequency_penalty
        if opts.seed is not None:
            params["seed"] = opts.seed
        if opts.stop:
            params["stop"] = opts.stop

        stream = await self._async_client.chat.completions.create(**params)

        reasoning_chunks = 0

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # ONLY normal assistant content enters the voice pipeline.
            # Qwen reasoning_content is intentionally ignored.
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_chunks += 1

            if delta.content:
                yield delta.content

    async def chat_stream_full(
        self,
        user_message: str,
        system: str | None = None,
        history: list[ChatMessage] | None = None,
        options: ChatOptions | None = None,
    ) -> CompletionResult:
        """Stream chat and collect the normal assistant response."""
        full_text = ""
        async for token in self.chat_stream(
            user_message, system, history, options
        ):
            full_text += token

        opts = options or self._default_options
        return CompletionResult(
            text=full_text,
            model=self._model,
            finish_reason="stop",
            response_time_secs=0.0,
        )

    # ---- Utility --------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Rough token count (4 chars ≈ 1 token)."""
        return max(1, len(text) // 4)

    def health_check(self) -> dict[str, Any]:
        """Check if the LLM server is responsive."""
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            return {"status": "ok", "models": available, "model": self._model}
        except Exception as e:
            return {"status": "error", "error": str(e)}