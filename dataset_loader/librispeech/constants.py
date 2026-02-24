from typing import Literal

# Types
LibriTask = Literal["asr"]
LibriSpeechSet = Literal[
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

# Defaults
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)
DEFAULT_DOWNLOAD_URLS = {
    "dev-clean": "https://openslr.trmal.net/resources/12/dev-clean.tar.gz",
    "dev-other": "https://openslr.trmal.net/resources/12/dev-other.tar.gz",
    "test-clean": "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
    "test-other": "https://openslr.trmal.net/resources/12/test-other.tar.gz",
    "train-clean-100": "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://openslr.trmal.net/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://openslr.trmal.net/resources/12/train-other-500.tar.gz",
}


__all__ = [
    "LibriTask",
    "LibriSpeechSet",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TASK",
    "DEFAULT_DOWNLOAD_URLS",
]
