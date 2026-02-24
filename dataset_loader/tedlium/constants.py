from typing import Literal

# Types
TedliumTask = Literal["asr", "diarization"]

# Defaults
DEFAULT_REPO_ID = "distil-whisper/tedlium"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_IGNORE_SET = set(("ignore_time_segment_in_scoring", "inter_segment_gap"))
DEFAULT_TASK = ("asr",)

__all__ = [
    "TedliumTask",
    "DEFAULT_REPO_ID",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_IGNORE_SET",
    "DEFAULT_TASK",
]
