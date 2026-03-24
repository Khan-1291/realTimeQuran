from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_ARABIC_WORDS = re.compile(r"[^\u0621-\u063A\u0641-\u064A0-9\s]")
WHITESPACE = re.compile(r"\s+")


MANUAL_FATIHA = (
    "1|1|\u0628\u0633\u0645 \u0627\u0644\u0644\u0647 \u0627\u0644\u0631\u062d\u0645\u0646 \u0627\u0644\u0631\u062d\u064a\u0645\n"
    "1|2|\u0627\u0644\u062d\u0645\u062f \u0644\u0644\u0647 \u0631\u0628 \u0627\u0644\u0639\u0627\u0644\u0645\u064a\u0646\n"
    "1|3|\u0627\u0644\u0631\u062d\u0645\u0646 \u0627\u0644\u0631\u062d\u064a\u0645\n"
    "1|4|\u0645\u0627\u0644\u0643 \u064a\u0648\u0645 \u0627\u0644\u062f\u064a\u0646\n"
    "1|5|\u0625\u064a\u0627\u0643 \u0646\u0639\u0628\u062f \u0648\u0625\u064a\u0627\u0643 \u0646\u0633\u062a\u0639\u064a\u0646\n"
    "1|6|\u0627\u0647\u062f\u0646\u0627 \u0627\u0644\u0635\u0631\u0627\u0637 \u0627\u0644\u0645\u0633\u062a\u0642\u064a\u0645\n"
    "1|7|\u0635\u0631\u0627\u0637 \u0627\u0644\u0630\u064a\u0646 \u0623\u0646\u0639\u0645\u062a \u0639\u0644\u064a\u0647\u0645 \u063a\u064a\u0631 \u0627\u0644\u0645\u063a\u0636\u0648\u0628 \u0639\u0644\u064a\u0647\u0645 \u0648\u0644\u0627 \u0627\u0644\u0636\u0627\u0644\u064a\u0646\n"
)


@dataclass(frozen=True)
class QuranEntry:
    surah: int
    ayah: int
    text: str


@dataclass(frozen=True)
class SurahSummary:
    surah: int
    name: str
    ayah_count: int


def normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = ARABIC_DIACRITICS.sub("", cleaned)
    cleaned = cleaned.replace("ـ", "")
    cleaned = cleaned.replace("ٱ", "ا")
    cleaned = cleaned.replace("آ", "ا")
    cleaned = cleaned.replace("أ", "ا")
    cleaned = cleaned.replace("إ", "ا")
    cleaned = cleaned.replace("ة", "ه")
    cleaned = cleaned.replace("ى", "ي")
    cleaned = NON_ARABIC_WORDS.sub(" ", cleaned)
    cleaned = WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [token for token in normalized.split(" ") if token]


def ensure_quran_dataset(dataset_path: Path, source_root: Path | None = None) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    source_file = source_root / "quran-simple.txt" if source_root else None
    should_refresh_from_source = False
    if dataset_path.exists():
        existing_entries = load_quran(dataset_path)
        unique_surahs = {entry.surah for entry in existing_entries}
        if len(unique_surahs) >= 114:
            return dataset_path
        should_refresh_from_source = bool(source_file and source_file.exists())

    if source_file and source_file.exists() and (should_refresh_from_source or not dataset_path.exists()):
        try:
            raw_text = source_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = source_file.read_text(encoding="latin-1")
        dataset_path.write_text(raw_text, encoding="utf-8")
    else:
        dataset_path.write_text(MANUAL_FATIHA, encoding="utf-8")
    return dataset_path


def load_quran(dataset_path: Path) -> list[QuranEntry]:
    entries: list[QuranEntry] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        parts = row.split("|", 2)
        if len(parts) != 3:
            continue
        surah, ayah, text = parts
        entries.append(QuranEntry(int(surah), int(ayah), text.strip()))
    if not entries:
        raise ValueError(f"No Qur'an entries were loaded from {dataset_path}")
    return entries


def get_surah_entries(entries: list[QuranEntry], surah: int) -> list[QuranEntry]:
    surah_entries = [entry for entry in entries if entry.surah == surah]
    if not surah_entries:
        raise ValueError(f"Surah {surah} was not found in the dataset")
    return surah_entries


def build_surah_summaries(entries: list[QuranEntry]) -> list[SurahSummary]:
    counts: dict[int, int] = {}
    for entry in entries:
        counts[entry.surah] = counts.get(entry.surah, 0) + 1

    return [
        SurahSummary(surah=surah, name=f"Surah {surah}", ayah_count=ayah_count)
        for surah, ayah_count in sorted(counts.items())
    ]
