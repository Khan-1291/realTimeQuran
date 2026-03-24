from __future__ import annotations

import io
import queue
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000


class MicrophoneUnavailableError(RuntimeError):
    pass


def microphone_chunks(
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = 1.6,
    channels: int = 1,
) -> Iterator[np.ndarray]:
    chunk_frames = max(1, int(sample_rate * chunk_seconds))
    buffer_queue: queue.Queue[np.ndarray] = queue.Queue()

    try:
        input_info = sd.query_devices(kind="input")
    except Exception as exc:
        raise MicrophoneUnavailableError(str(exc)) from exc

    if not input_info:
        raise MicrophoneUnavailableError("No input audio device detected")

    cache = np.empty((0,), dtype=np.float32)

    def callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            return
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        buffer_queue.put(mono)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=0,
            callback=callback,
        ):
            while True:
                piece = buffer_queue.get()
                cache = np.concatenate((cache, piece))
                while cache.shape[0] >= chunk_frames:
                    yield cache[:chunk_frames]
                    cache = cache[chunk_frames:]
    except Exception as exc:
        raise MicrophoneUnavailableError(str(exc)) from exc


def audio_file_chunks(
    file_path: Path,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = 1.6,
) -> Iterator[np.ndarray]:
    chunk_frames = max(1, int(sample_rate * chunk_seconds))
    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout="mono",
        rate=sample_rate,
    )
    buffer = np.empty((0,), dtype=np.float32)

    with av.open(str(file_path)) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {file_path}")

        for frame in container.decode(audio_stream):
            converted = resampler.resample(frame)
            converted_frames = converted if isinstance(converted, list) else [converted]
            for converted_frame in converted_frames:
                samples = converted_frame.to_ndarray()
                if samples.ndim > 1:
                    mono = samples[0]
                else:
                    mono = samples
                mono = np.asarray(mono, dtype=np.float32)
                buffer = np.concatenate((buffer, mono))
                while buffer.shape[0] >= chunk_frames:
                    yield buffer[:chunk_frames]
                    buffer = buffer[chunk_frames:]

    if buffer.size:
        yield buffer


def decode_audio_bytes(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if not audio_bytes:
        return np.empty((0,), dtype=np.float32)

    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout="mono",
        rate=sample_rate,
    )
    chunks: list[np.ndarray] = []

    with av.open(io.BytesIO(audio_bytes), mode="r") as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio stream found in uploaded audio payload")

        for frame in container.decode(audio_stream):
            converted = resampler.resample(frame)
            converted_frames = converted if isinstance(converted, list) else [converted]
            for converted_frame in converted_frames:
                samples = converted_frame.to_ndarray()
                mono = samples[0] if samples.ndim > 1 else samples
                chunks.append(np.asarray(mono, dtype=np.float32))

    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks)


def default_fallback_audio_path(base_dir: Path) -> Path | None:
    candidates = [
        base_dir / "dataset" / "quran" / "bismillah.mp3",
        base_dir / "dataset" / "quran" / "001001.mp3",
        base_dir / "dataset" / "quran" / "001002.mp3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
