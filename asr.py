from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class TranscriptResult:
    text: str = ""
    word_timestamps: list[tuple[str, float, float]] = field(default_factory=list)


class FasterWhisperASR:
    def __init__(self, model_size: str = "small", model_dir: Path | None = None) -> None:
        self.model_size = model_size
        self.model_dir = Path(model_dir) if model_dir else None
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return

        download_root = str(self.model_dir) if self.model_dir else None
        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            download_root=download_root,
        )

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        initial_prompt: str | None = None,
    ) -> TranscriptResult:
        self.load()
        assert self._model is not None

        if audio_chunk.size == 0:
            return TranscriptResult()

        audio = np.asarray(audio_chunk, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / max(peak, 1.0)

        with self._lock:
            segments, _ = self._model.transcribe(
                audio,
                language="ar",
                vad_filter=True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                word_timestamps=True,
                condition_on_previous_text=True,
                initial_prompt=initial_prompt,
            )

        texts: list[str] = []
        timestamps: list[tuple[str, float, float]] = []
        for segment in segments:
            segment_text = (segment.text or "").strip()
            if segment_text:
                texts.append(segment_text)
            for word in segment.words or []:
                token = (word.word or "").strip()
                if token:
                    timestamps.append((token, float(word.start), float(word.end)))

        return TranscriptResult(text=" ".join(texts).strip(), word_timestamps=timestamps)
