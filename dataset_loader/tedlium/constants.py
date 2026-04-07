from __future__ import annotations

from typing import Literal

# Defaults Segment Tedlium
DEFAULT_SEGMENT_REPO_ID: str = "distil-whisper/tedlium"
DEFAULT_SEGMENT_SAMPLE_RATE: int = 16_000
DEFAULT_SEGMENT_IGNORE_SET: tuple[str, ...] = (
    "ignore_time_segment_in_scoring",
    "inter_segment_gap",
    "<unk>",
)

# Defaults Tedlium
TedliumSet = Literal["train", "dev", "test"]
DEFAULT_SAMPLE_RATE: int = 16_000
DEFAULT_IGNORE_SET: tuple[str, ...] = (
    "ignore_time_segment_in_scoring",
    "inter_segment_gap",
    "<unk>",
)
DATA_PARQUET: dict[str, str] = {
    "train": "train.parquet",
    "dev": "dev.parquet",
    "test": "test.parquet",
}


__all__ = [
    "DEFAULT_SEGMENT_REPO_ID",
    "DEFAULT_SEGMENT_SAMPLE_RATE",
    "DEFAULT_SEGMENT_IGNORE_SET",
    "TedliumSet",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_IGNORE_SET",
    "DATA_PARQUET",
]
