from __future__ import annotations

from typing import Literal

# TYPES
ZerothKoreanTask = Literal["asr", "diarization"]

# DEFAULTS
DEFAULT_REPO_ID = "kresnik/zeroth_korean"
DEFAULT_CONFIG_NAME = "default"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)

__all__ = [
    "ZerothKoreanTask",
    "DEFAULT_REPO_ID",
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TASK",
]
