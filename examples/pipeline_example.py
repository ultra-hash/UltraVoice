# -*- coding: utf-8 -*-
"""
Voice Pipeline Example — STT → LLM → TTS

Usage:
    # Run on default test audio
    python examples/pipeline_example.py

    # Run on custom audio
    python examples/pipeline_example.py input.wav

    # With system prompt
    python examples/pipeline_example.py -s "You are helpful and concise."

    # Only run pipeline (no audio output)
    python examples/pipeline_example.py --no-play
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_pipeline import VoicePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Pipeline: STT → LLM → TTS")
    parser.add_argument("audio", nargs="?", default="breeze_test.wav", help="Audio file path")
    parser.add_argument("-s", "--system", default="You are a friendly and concise assistant.", help="System prompt")
    parser.add_argument("--no-play", action="store_true", help="Skip playing audio output")
    parser.add_argument("-o", "--output", default="pipeline_output.wav", help="Output WAV file path")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"\nULTRAVOICE - Voice Pipeline")
    print(f"  Audio:  {audio_path}")
    print(f"  System: {args.system}")
    print(f"  Output: {args.output}")
    print()

    # Create and run pipeline
    pipeline = VoicePipeline.from_config()
    result = pipeline.run(str(audio_path), system=args.system)

    # Print results
    pipeline.print_summary(result)

    # Save output
    pipeline.save_audio(result, args.output)
    print(f"\n  Audio saved to: {args.output}")

    # Play if not disabled
    if not args.no_play:
        print(f"  Playing output...")
        pipeline.play(result, blocking=True)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
