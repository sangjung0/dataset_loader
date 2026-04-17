from __future__ import annotations

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Any
from typing_extensions import override

from dataset_loader.abstract import ParquetDataset

from dataset_loader.librispeech.librispeech_sample import LibriSpeechSample


class LibriSpeechDataset(ParquetDataset[LibriSpeechSample]):
    def __init__(
        self: LibriSpeechDataset,
        *,
        parquet: pd.DataFrame,
        sr: int,
    ):
        super().__init__(parquet=parquet)
        self._sr: int = sr

    @property
    @override
    def args(self: LibriSpeechDataset) -> dict[str, Any]:
        return {**super().args, "sr": self._sr}

    @property
    def sr(self: LibriSpeechDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: LibriSpeechDataset, value: int) -> None:
        if value <= 0:
            raise ValueError("Sample rate must be a positive integer")
        self._sr = value

    @override
    def get(self: LibriSpeechDataset, idx: int) -> LibriSpeechSample:
        if self.is_cleaned:
            raise RuntimeError("Cannot get sample from a cleaned dataset.")
        data = self.dataset.iloc[idx].to_dict()

        def load_audio_func() -> npt.NDArray[np.float32]:
            audio_path = data["audio_path"]
            wav, _ = librosa.load(audio_path, sr=self._sr)
            return wav.astype(np.float32)

        _id = data.pop("id")
        result: dict[str, Any] = {
            "load_audio_func": load_audio_func,
            "ref": data["ref"],
        }

        return LibriSpeechSample(id=_id, data=result)


__all__ = ["LibriSpeechDataset"]
