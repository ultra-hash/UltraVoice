"""
STT → LLM → TTS Pipeline

End-to-end pipeline: audio → transcribe → generate → synthesize → output
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf

from runtime.adapters.whisper_stt import WhisperSTT, TranscriptionResult
from runtime.adapters.qwen_llm import QwenLLM, CompletionResult
from runtime.adapters.breeze_tts import BreezeTTS, SynthesisResult


# ---------------------------------------------------------------------------
# Pipeline Result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Complete result from a full pipeline run."""

    # STT
    input_audio_bytes: bytes = b""
    transcription: str = ""
    stt_duration_secs: float = 0.0

    # LLM
    prompt: str = ""
    llm_response: str = ""
    llm_response_time_secs: float = 0.0
    llm_tokens_used: int = 0

    # TTS
    output_pcm: bytes = b""
    output_duration_secs: float = 0.0
    output_sample_rate: int = 0
    tts_duration_secs: float = 0.0

    # Overall
    total_duration_secs: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------


class VoicePipeline:
    """STT → LLM → TTS voice pipeline.

    Usage:
        pipeline = VoicePipeline.from_config()

        # Transcribe a file and get audio output
        result = pipeline.run("input.wav", system="You are helpful.")
        pipeline.play(result)
        pipeline.save_audio(result, "output.wav")

        # Stream (LLM tokens arrive as they're generated)
        async for partial in pipeline.run_stream("input.wav", system="Be concise."):
            print(partial.llm_response, end="")
            # Optionally synthesize partial audio as it arrives
    """

    def __init__(
        self,
        stt: WhisperSTT,
        llm: QwenLLM,
        tts: BreezeTTS,
        debug: bool = False,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._debug = debug

    @classmethod
    def from_config(cls, stt_mode: str = "remote") -> "VoicePipeline":
        """Create pipeline from config.yaml."""
        stt = WhisperSTT.from_config()
        llm = QwenLLM.from_config()
        tts = BreezeTTS.from_config(mode="http")
        return cls(stt=stt, llm=llm, tts=tts, debug=True)

    def _log(self, msg: str) -> None:
        if self._debug:
            print(f"  [pipeline] {msg}")

    # ---- Non-streaming (full) pipeline ----------------------------------

    def run(
        self,
        audio_input: str | bytes | np.ndarray,
        system: str | None = None,
        sample_rate: int = 16000,
    ) -> PipelineResult:
        """Run the full pipeline synchronously.

        Args:
            audio_input: File path, raw bytes, or PCM array.
            system: System prompt for LLM.
            sample_rate: Sample rate for PCM arrays (default 16kHz for STT).

        Returns:
            PipelineResult with all stages completed.
        """
        overall_start = time.time()
        result = PipelineResult()

        # ---- STT ----------------------------------------------------------
        self._log("Stage 1/3: Transcribing audio...")
        stt_start = time.time()
        if isinstance(audio_input, str):
            # File path
            result.input_audio_bytes = Path(audio_input).read_bytes()
            stt_result = self._stt.transcribe_file(audio_input)
        elif isinstance(audio_input, bytes):
            result.input_audio_bytes = audio_input
            stt_result = self._stt.transcribe(
                np.frombuffer(audio_input, dtype=np.float32), sample_rate
            )
        else:
            # PCM array
            stt_result = self._stt.transcribe(audio_input, sample_rate)

        result.transcription = stt_result.text
        result.stt_duration_secs = time.time() - stt_start
        self._log(f"  Transcribed: \"{result.transcription}\" ({result.stt_duration_secs:.2f}s)")

        if not result.transcription.strip():
            self._log("  [WARN] Empty transcription — skipping LLM and TTS")
            result.total_duration_secs = time.time() - overall_start
            return result

        # ---- LLM ----------------------------------------------------------
        self._log("Stage 2/3: Generating response...")
        llm_start = time.time()
        result.prompt = result.transcription
        llm_result = self._llm.chat(
            result.transcription,
            system=system,
            options=self._llm._default_options,
        )

        # For reasoning models, use only the actual response (not reasoning)
        if llm_result.reasoning_content:
            # Remove reasoning blocks from text
            llm_text = llm_result.text
            # Remove common reasoning patterns
            for pattern in ["Here's a thinking process:", "Thinking Process:"]:
                if pattern in llm_text:
                    llm_text = llm_text.split(pattern)[0].strip()
            result.llm_response = llm_text or llm_result.text  # fallback to full text if nothing found
        else:
            result.llm_response = llm_result.text

        result.llm_response_time_secs = llm_result.response_time_secs
        result.llm_tokens_used = (
            llm_result.prompt_tokens + llm_result.completion_tokens
        )
        self._log(f"  Response: {result.llm_response[:100]}...")
        self._log(f"  Tokens: {result.llm_tokens_used} | Time: {result.llm_response_time_secs}s")

        # ---- TTS ----------------------------------------------------------
        self._log("Stage 3/3: Synthesizing speech...")
        tts_start = time.time()
        tts_result = self._tts.synthesize(result.llm_response)

        result.output_pcm = tts_result.pcm
        result.output_duration_secs = tts_result.duration_secs
        result.output_sample_rate = tts_result.sample_rate
        result.tts_duration_secs = time.time() - tts_start
        self._log(f"  Synthesized: {result.output_duration_secs:.2f}s @ {result.output_sample_rate}Hz ({result.tts_duration_secs:.2f}s wall)")

        result.total_duration_secs = time.time() - overall_start
        self._log(f"Total: {result.total_duration_secs:.2f}s")

        return result

    # ---- Streaming LLM + synthesize as tokens arrive --------------------

    async def run_stream(
        self,
        audio_input: str | bytes | np.ndarray,
        system: str | None = None,
        sample_rate: int = 16000,
        min_tokens_for_audio: int = 20,
    ) -> AsyncIterator[PipelineResult]:
        """Stream LLM tokens and optionally synthesize partial audio.

        Yields PipelineResult objects as they become available.

        Args:
            audio_input: Audio file path, bytes, or PCM array.
            system: System prompt.
            sample_rate: For PCM arrays.
            min_tokens_for_audio: Synthesize audio every N tokens.

        Yields:
            PipelineResult (partial results, final yield has complete audio).
        """
        overall_start = time.time()
        result = PipelineResult()

        # ---- STT (same as non-streaming) ----------------------------------
        self._log("Stage 1/3: Transcribing audio...")
        stt_start = time.time()
        if isinstance(audio_input, str):
            result.input_audio_bytes = Path(audio_input).read_bytes()
            stt_result = self._stt.transcribe_file(audio_input)
        elif isinstance(audio_input, bytes):
            result.input_audio_bytes = audio_input
            stt_result = self._stt.transcribe(
                np.frombuffer(audio_input, dtype=np.float32), sample_rate
            )
        else:
            stt_result = self._stt.transcribe(audio_input, sample_rate)

        result.transcription = stt_result.text
        result.stt_duration_secs = time.time() - stt_start
        self._log(f"  Transcribed: \"{result.transcription}\" ({result.stt_duration_secs:.2f}s)")

        if not result.transcription.strip():
            result.total_duration_secs = time.time() - overall_start
            yield result
            return

        # ---- LLM (stream) + TTS (partial) ---------------------------------
        self._log("Stage 2/3: Streaming response...")
        llm_start = time.time()
        token_count = 0
        full_response = ""

        async for token in self._llm.chat_stream(
            result.transcription,
            system=system,
            options=self._llm._default_options,
        ):
            full_response += token
            token_count += 1
            result.llm_response = full_response
            result.llm_response_time_secs = time.time() - llm_start
            result.prompt = result.transcription

            # Synthesize audio periodically (every N tokens)
            if token_count % min_tokens_for_audio == 0:
                self._log(f"  Synthesizing {len(full_response)} chars...")
                tts_start = time.time()
                try:
                    tts_result = self._tts.synthesize(full_response)
                    result.output_pcm = tts_result.pcm
                    result.output_duration_secs = tts_result.duration_secs
                    result.output_sample_rate = tts_result.sample_rate
                    result.tts_duration_secs = time.time() - tts_start
                except Exception as e:
                    self._log(f"  [WARN] Synthesis error: {e}")

            result.total_duration_secs = time.time() - overall_start
            yield result

        # Final synthesis with full response
        self._log("Final synthesis (full response)...")
        tts_start = time.time()
        try:
            tts_result = self._tts.synthesize(full_response)
            result.output_pcm = tts_result.pcm
            result.output_duration_secs = tts_result.duration_secs
            result.output_sample_rate = tts_result.sample_rate
            result.tts_duration_secs = time.time() - tts_start
        except Exception as e:
            self._log(f"  [WARN] Final synthesis error: {e}")

        result.llm_tokens_used = token_count
        result.total_duration_secs = time.time() - overall_start
        yield result

    # ---- Utility methods ------------------------------------------------

    def play(self, result: PipelineResult, blocking: bool = True) -> None:
        """Play the pipeline output audio via sounddevice."""
        if not result.output_pcm:
            print("  [pipeline] No audio to play")
            return
        import sounddevice as sd
        audio = np.frombuffer(result.output_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, result.output_sample_rate)
        if blocking:
            sd.wait()

    def save_audio(self, result: PipelineResult, path: str) -> None:
        """Save the pipeline output audio to a WAV file."""
        if not result.output_pcm:
            print("  [pipeline] No audio to save")
            return
        import soundfile as sf
        audio = np.frombuffer(result.output_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(str(path), audio, result.output_sample_rate)

    def print_summary(self, result: PipelineResult) -> None:
        """Print a formatted summary of the pipeline result."""
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)

        print(f"\n  {'─'*50}")
        print(f"  INPUT")
        print(f"  {'─'*50}")
        print(f"    Audio size: {len(result.input_audio_bytes):,} bytes")

        print(f"\n  {'─'*50}")
        print(f"  STT  (Speech-to-Text)")
        print(f"  {'─'*50}")
        print(f"    Text: {result.transcription}")
        print(f"    Time: {result.stt_duration_secs:.2f}s")

        print(f"\n  {'─'*50}")
        print(f"  LLM  (Language Model)")
        print(f"  {'─'*50}")
        print(f"    Input:  {result.prompt}")
        print(f"    Output: {result.llm_response[:200]}")
        if result.llm_tokens_used:
            print(f"    Tokens: {result.llm_tokens_used}")
            print(f"    Time:   {result.llm_response_time_secs:.2f}s")

        print(f"\n  {'─'*50}")
        print(f"  TTS  (Text-to-Speech)")
        print(f"  {'─'*50}")
        print(f"    Duration: {result.output_duration_secs:.2f}s")
        print(f"    Sample rate: {result.output_sample_rate} Hz")
        print(f"    PCM: {len(result.output_pcm):,} bytes")
        print(f"    Time: {result.tts_duration_secs:.2f}s")

        print(f"\n  {'═'*50}")
        print(f"  TOTAL: {result.total_duration_secs:.2f}s")
        print(f"  {'═'*50}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run pipeline on a test audio file."""
    import sys

    # Create pipeline
    pipeline = VoicePipeline.from_config()

    # Determine input
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = Path(__file__).parent.parent / "breeze_test.wav"

    audio_file = Path(audio_file)
    if not audio_file.exists():
        print(f"[ERROR] Audio file not found: {audio_file}")
        sys.exit(1)

    print(f"\nULTRAVOICE - Voice Pipeline")
    print(f"Input: {audio_file}")
    print(f"System prompt: (none)")

    # Run pipeline
    result = pipeline.run(str(audio_file))

    # Print summary
    pipeline.print_summary(result)

    # Save output
    output_path = "pipeline_output.wav"
    pipeline.save_audio(result, output_path)
    print(f"\n  Audio saved to: {output_path}")

    # Play output (uncomment to auto-play)
    # pipeline.play(result, blocking=True)


if __name__ == "__main__":
    main()
