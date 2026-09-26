# -*- coding: utf-8 -*-
"""Test Qwen LLM adapter using the OpenAI library."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.adapters.qwen_llm import (
    ChatMessage,
    ChatOptions,
    QwenLLM,
)


def test_health_check() -> None:
    """Verify LLM server is reachable and lists models."""
    print("=" * 60)
    print("TEST 0: Health Check")
    print("=" * 60)

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
    )

    status = llm.health_check()
    print(f"  [OK] Status: {status['status']}")
    print(f"  Requested model: {status.get('model')}")
    models = status.get("models", [])
    if models:
        print(f"  Available models ({len(models)}):")
        for m in models[:5]:
            print(f"    - {m}")
    else:
        print(f"  [INFO] No models listed (may only have one)")

    assert status["status"] == "ok", "Health check failed"
    print(f"  [OK] Health check passed!")


def test_simple_chat() -> None:
    """Test basic non-streaming chat."""
    print("\n" + "=" * 60)
    print("TEST 1: Simple Chat")
    print("=" * 60)

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
        timeout=60.0,
    )

    result = llm.chat("What is 2+2?", options=ChatOptions(max_tokens=100))

    print(f"\n  Prompt:     What is 2+2?")
    print(f"  Response:   {result.text[:200]}")
    if result.reasoning_content:
        print(f"  (reasoning: {len(result.reasoning_content)} chars)")
    print(f"  Model:      {result.model}")
    print(f"  Tokens:     {result.prompt_tokens} in, {result.completion_tokens} out")
    print(f"  Time:       {result.response_time_secs}s")
    print(f"  Finish:     {result.finish_reason}")

    # Reasoning models may put content in reasoning_content
    text = result.text or result.reasoning_content
    assert len(text) > 0, "Empty response"
    print(f"  [OK] Simple chat works!")


def test_system_prompt() -> None:
    """Test chat with system instruction."""
    print("\n" + "=" * 60)
    print("TEST 2: System Prompt")
    print("=" * 60)

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
        timeout=60.0,
    )

    system = "You are a helpful assistant that speaks like a pirate."
    result = llm.chat("Hello there!", system=system, options=ChatOptions(max_tokens=100))

    print(f"\n  System:  {system}")
    text = result.text or result.reasoning_content
    print(f"  Response: {text[:150]}")

    assert len(text) > 0, "Empty response"
    print(f"  [OK] System prompt works!")


def test_streaming_chat() -> None:
    """Test streaming chat completion."""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming Chat")
    print("=" * 60)

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
        timeout=60.0,
    )

    full_text = ""
    token_count = 0
    for token in llm.chat_stream("Count from 1 to 5", options=ChatOptions(max_tokens=50)):
        print(token, end="", flush=True)
        full_text += token
        token_count += 1

    print(f"\n\n  Tokens received: {token_count}")
    print(f"  Total text:      {len(full_text)} chars")

    assert len(full_text) > 0, "Empty stream"
    print(f"  [OK] Streaming works!")


def test_chat_history() -> None:
    """Test multi-turn conversation."""
    print("\n" + "=" * 60)
    print("TEST 4: Chat History")
    print("=" * 60)

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
        timeout=60.0,
    )

    history: list[dict] = []

    # Turn 1
    result, history = llm.chat_with_history(
        "My name is Alice.",
        options=ChatOptions(max_tokens=100),
    )
    print(f"  You:  My name is Alice.")
    text1 = result.text or result.reasoning_content
    print(f"  Bot:  {text1[:150]}")

    # Turn 2
    result, history = llm.chat_with_history(
        "What is my name?",
        history=history,
        options=ChatOptions(max_tokens=100),
    )
    print(f"  You:  What is my name?")
    text2 = result.text or result.reasoning_content
    print(f"  Bot:  {text2[:150]}")

    print(f"  History length: {len(history)} messages")
    assert len(result.text) > 0, "Empty response"
    print(f"  [OK] Chat history works!")


def test_from_config() -> None:
    """Create LLM from config.yaml."""
    print("\n" + "=" * 60)
    print("TEST 5: from_config()")
    print("=" * 60)

    llm = QwenLLM.from_config()
    print(f"  Model from config: {llm._model}")
    print(f"  Base URL: {llm._client.base_url}")

    # Quick test
    result = llm.chat("Say hi in one word", options=ChatOptions(max_tokens=20))
    text = result.text or result.reasoning_content
    print(f"  Response: {text}")

    assert len(text) > 0, "Empty response"
    print(f"  [OK] from_config() works!")


def test_token_counting() -> None:
    """Test rough token estimation."""
    print("\n" + "=" * 60)
    print("TEST 6: Token Counting")
    print("=" * 60)

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
    )

    text = "Hello world, this is a test of the token counting utility."
    count = llm.count_tokens(text)
    actual_len = len(text)

    print(f"  Text: \"{text}\"")
    print(f"  Chars:  {actual_len}")
    print(f"  Est. tokens: {count}")
    print(f"  (approximation: {actual_len / count:.1f} chars/token)")

    assert count > 0, "Token count should be > 0"
    print(f"  [OK] Token counting works!")


def test_temperature_and_options() -> None:
    """Test different sampling options."""
    print("\n" + "=" * 60)
    print("TEST 7: Sampling Options")
    print("=" * 60)

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
        timeout=60.0,
    )

    # Low temperature → more deterministic
    result_cool = llm.chat(
        "What is the capital of France?",
        options=ChatOptions(temperature=0.1, max_tokens=50),
    )
    text_cool = result_cool.text or result_cool.reasoning_content
    print(f"  Temp 0.1: {text_cool[:150]}")

    # High temperature → more creative
    result_warm = llm.chat(
        "Write a very short story about a cat.",
        options=ChatOptions(temperature=1.0, max_tokens=80),
    )
    text_warm = result_warm.text or result_warm.reasoning_content
    print(f"\n  Temp 1.0: {text_warm[:100]}...")

    print(f"  [OK] Sampling options work!")


def main() -> None:
    print("\nULTRAVOICE - Qwen LLM Test Suite")
    print("=" * 60)

    test_health_check()
    test_simple_chat()
    test_system_prompt()
    test_streaming_chat()
    test_chat_history()
    test_from_config()
    test_token_counting()
    test_temperature_and_options()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
