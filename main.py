from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "quran.txt"
MODEL_DIR = BASE_DIR / "models"
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def ensure_venv_runtime() -> None:
    current_python = Path(sys.executable).resolve()
    if VENV_PYTHON.exists() and current_python != VENV_PYTHON.resolve():
        raise SystemExit(
            subprocess.call([str(VENV_PYTHON), str(BASE_DIR / "main.py"), *sys.argv[1:]])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qur'an recitation checker backend with CLI and web API modes."
    )
    parser.add_argument(
        "--mode",
        choices=("api", "cli"),
        default="api",
        help="Start the FastAPI backend or the local terminal microphone checker.",
    )
    parser.add_argument("--surah", type=int, default=1, help="Starting surah number.")
    parser.add_argument("--ayah", type=int, default=1, help="Starting ayah number.")
    parser.add_argument(
        "--model",
        default=os.getenv("QURAN_ASR_MODEL", "small"),
        help="faster-whisper model size (tiny, base, small, medium, large-v3, etc.).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=1.6,
        help="Audio chunk duration for streaming input.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Use bundled recitation audio instead of microphone for end-to-end verification.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API host binding.")
    parser.add_argument("--port", type=int, default=8000, help="API port.")
    return parser.parse_args()


def clear_terminal() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def format_live_output(result: MatchResult, recognized_full_text: str) -> str:
    lines = [
        f"Tracking:   Surah {result.surah}, Ayah {result.ayah}",
        f"Recited:    {recognized_full_text or '-'}",
        f"Expected:   {result.expected_text or '-'}",
        f"Similarity: {result.similarity:.1f}%",
        f"Status:     {result.status}",
        f"Missing:    {' '.join(result.missing_words) if result.missing_words else '-'}",
        f"Incorrect:  {', '.join(result.incorrect_pairs) if result.incorrect_pairs else '-'}",
    ]

    if result.word_timestamps:
        timestamp_text = ", ".join(
            f"{word}@{start:.2f}-{end:.2f}s" for word, start, end in result.word_timestamps[-6:]
        )
        lines.append(f"Timestamps: {timestamp_text}")

    if result.just_advanced:
        lines.append("Advance:    Ayah completed, moved to the next ayah.")

    return "\n".join(lines)


def build_source(args: argparse.Namespace):
    from stream_audio import (
        MicrophoneUnavailableError,
        audio_file_chunks,
        default_fallback_audio_path,
        microphone_chunks,
    )

    if args.self_test:
        fallback = default_fallback_audio_path(BASE_DIR)
        if fallback is None:
            raise FileNotFoundError("No bundled fallback audio file was found for self-test mode.")
        print(f"Audio source: bundled fallback file -> {fallback}")
        return audio_file_chunks(fallback, chunk_seconds=args.chunk_seconds)

    try:
        print("Audio source: microphone")
        return microphone_chunks(chunk_seconds=args.chunk_seconds)
    except MicrophoneUnavailableError as exc:
        fallback = default_fallback_audio_path(BASE_DIR)
        if fallback is None:
            raise RuntimeError(str(exc)) from exc
        print(f"Microphone unavailable ({exc}). Falling back to {fallback}")
        return audio_file_chunks(fallback, chunk_seconds=args.chunk_seconds)


def main() -> int:
    configure_stdio()
    ensure_venv_runtime()

    args = parse_args()

    if args.mode == "api":
        import uvicorn

        uvicorn.run(
            "api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0

    from asr import FasterWhisperASR
    from matcher import MatchResult, RecitationMatcher
    from quran_loader import ensure_quran_dataset, load_quran

    ensure_quran_dataset(DATASET_PATH, source_root=BASE_DIR / "dataset")
    quran_entries = load_quran(DATASET_PATH)

    matcher = RecitationMatcher(quran_entries, start_surah=args.surah, start_ayah=args.ayah)
    asr = FasterWhisperASR(model_size=args.model, model_dir=MODEL_DIR)

    print("Initializing Whisper model. The first run may download model files automatically.")
    asr.load()

    source = build_source(args)

    print("Listening started. Press Ctrl+C to stop.\n")
    try:
        for chunk in source:
            transcript = asr.transcribe_chunk(chunk)
            result = matcher.update(transcript.text, transcript.word_timestamps)
            clear_terminal()
            print(format_live_output(result, matcher.recognized_text))
            if args.self_test and result.is_complete:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
