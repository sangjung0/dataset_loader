from typing import Literal

# Types
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
DEFAULT_DOWNLOAD_URLS = {
    "dev-clean": "https://openslr.trmal.net/resources/12/dev-clean.tar.gz",
    "dev-other": "https://openslr.trmal.net/resources/12/dev-other.tar.gz",
    "test-clean": "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
    "test-other": "https://openslr.trmal.net/resources/12/test-other.tar.gz",
    "train-clean-100": "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://openslr.trmal.net/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://openslr.trmal.net/resources/12/train-other-500.tar.gz",
}
DATA_PARQUET = {
    "train-clean-100": "train-clean-100.parquet",
    "train-clean-360": "train-clean-360.parquet",
    "train-other-500": "train-other-500.parquet",
    "dev-clean": "dev-clean.parquet",
    "dev-other": "dev-other.parquet",
    "test-clean": "test-clean.parquet",
    "test-other": "test-other.parquet",
}


__all__ = [
    "LibriSpeechSet",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_DOWNLOAD_URLS",
    "DATA_PARQUET",
]
