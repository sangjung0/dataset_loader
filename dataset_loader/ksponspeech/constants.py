from __future__ import annotations

from typing import Literal

# TYPES
KSPonSpeechTask = Literal["asr"]

# DEFAULTS
DEFAULT_REPO_ID = "DragonLine/ksponspeech"
DEFAULT_CONFIG_NAME = "default"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)

__all__ = [
    "KSPonSpeechTask",
    "DEFAULT_REPO_ID",
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TASK",
]
