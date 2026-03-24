from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from quran_loader import QuranEntry, normalize_text, tokenize_text


@dataclass
class MatchResult:
    surah: int
    ayah: int
    expected_text: str
    similarity: float
    missing_words: list[str] = field(default_factory=list)
    incorrect_pairs: list[str] = field(default_factory=list)
    status: str = "waiting"
    just_advanced: bool = False
    is_complete: bool = False
    session_complete: bool = False
    word_timestamps: list[tuple[str, float, float]] = field(default_factory=list)


class RecitationMatcher:
    def __init__(
        self,
        quran_entries: Iterable[QuranEntry],
        start_surah: int = 1,
        start_ayah: int = 1,
    ) -> None:
        self.entries = sorted(quran_entries, key=lambda item: (item.surah, item.ayah))
        self.index_lookup = {(entry.surah, entry.ayah): i for i, entry in enumerate(self.entries)}
        self.current_index = self.index_lookup.get((start_surah, start_ayah), 0)
        self.recognized_words: list[str] = []
        self.latest_word_timestamps: list[tuple[str, float, float]] = []
        self.finished = False

    @property
    def current_entry(self) -> QuranEntry:
        return self.entries[self.current_index]

    @property
    def recognized_text(self) -> str:
        return " ".join(self.recognized_words).strip()

    def update(
        self,
        transcript_text: str,
        word_timestamps: list[tuple[str, float, float]] | None = None,
        replace_recognized_text: bool = False,
    ) -> MatchResult:
        if self.finished:
            return self.current_state()

        new_words = tokenize_text(transcript_text)
        if new_words:
            if replace_recognized_text:
                self.recognized_words = new_words[:]
            else:
                self._merge_words(new_words)

        if word_timestamps:
            self.latest_word_timestamps = word_timestamps

        result = self._build_result()
        if result.is_complete:
            self._advance_ayah()
            result.just_advanced = True
            result.session_complete = self.finished
        return result

    def current_state(self) -> MatchResult:
        result = self._build_result()
        result.session_complete = self.finished
        return result

    def _merge_words(self, new_words: list[str]) -> None:
        if not self.recognized_words:
            self.recognized_words = new_words[:]
            return

        max_overlap = min(len(self.recognized_words), len(new_words), 10)
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if self.recognized_words[-size:] == new_words[:size]:
                overlap = size
                break

        suffix = new_words[overlap:]
        if suffix:
            self.recognized_words.extend(suffix)

    def _build_result(self) -> MatchResult:
        entry = self.current_entry
        expected_words = tokenize_text(entry.text)
        recited_words = self.recognized_words[: len(expected_words) + 4]

        similarity = fuzz.ratio(" ".join(recited_words), " ".join(expected_words))
        missing_words, incorrect_pairs = self._diff_words(expected_words, recited_words)

        status = "listening"
        if not recited_words:
            status = "waiting for speech"
        elif incorrect_pairs:
            status = "incorrect word detected"
        elif missing_words:
            status = "missing words detected"
        else:
            status = "match looks good"

        is_complete = bool(
            expected_words
            and len(recited_words) >= max(1, len(expected_words) - 1)
            and similarity >= 88
            and len(missing_words) <= 1
        )

        return MatchResult(
            surah=entry.surah,
            ayah=entry.ayah,
            expected_text=normalize_text(entry.text),
            similarity=similarity,
            missing_words=missing_words,
            incorrect_pairs=incorrect_pairs,
            status=status,
            is_complete=is_complete,
            session_complete=self.finished,
            word_timestamps=self.latest_word_timestamps,
        )

    def _advance_ayah(self) -> None:
        entry = self.current_entry
        if self.current_index < len(self.entries) - 1:
            consumed = len(tokenize_text(entry.text))
            if consumed > 0:
                self.recognized_words = self.recognized_words[consumed:]
            self.current_index += 1
        else:
            self.finished = True

        self.latest_word_timestamps = []

    @staticmethod
    def _diff_words(expected_words: list[str], recited_words: list[str]) -> tuple[list[str], list[str]]:
        missing_words: list[str] = []
        incorrect_pairs: list[str] = []

        for tag, src_start, src_end, dest_start, dest_end in Levenshtein.opcodes(
            expected_words, recited_words
        ):
            if tag == "delete":
                missing_words.extend(expected_words[src_start:src_end])
            elif tag == "replace":
                expected = " ".join(expected_words[src_start:src_end])
                actual = " ".join(recited_words[dest_start:dest_end])
                incorrect_pairs.append(f"{actual} -> {expected}")

        if len(recited_words) < len(expected_words):
            prefix = expected_words[len(recited_words) :]
            if not missing_words:
                missing_words.extend(prefix)

        return missing_words, incorrect_pairs
