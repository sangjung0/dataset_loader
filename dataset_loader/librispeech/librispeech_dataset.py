from __future__ import annotations

import librosa
import numpy as np
import pandas as pd

from typing_extensions import override

from dataset_loader.interface import Sample
from dataset_loader.abstract import ParquetDataset


class LibriSpeechDataset(ParquetDataset):
    def __init__(
        self: LibriSpeechDataset,
        *,
        parquet: pd.DataFrame,
        sr: int,
        use_cache: int = 0,
    ):
        super().__init__(parquet=parquet, use_cache=use_cache)
        self._sr: int = sr

    @ParquetDataset.args.getter
    def args(self: LibriSpeechDataset) -> dict:
        return {**super().args, "sr": self._sr}

    @property
    def sr(self: LibriSpeechDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: LibriSpeechDataset, value: int) -> None:
        if isinstance(value, int) and value > 0:
            self._sr = value
        else:
            raise ValueError("Sample rate must be a positive integer")

    @override
    def _get(self: LibriSpeechDataset, idx: int) -> Sample:
        data = self._parquet.iloc[idx].to_dict()

        def load_audio_func() -> np.ndarray:
            audio_path = data["audio_path"]
            wav, _ = librosa.load(audio_path, sr=self._sr)
            return wav

        _id = data.pop("id")
        result = {
            "load_audio_func": load_audio_func,
            "ref": data["ref"],
        }

        return Sample(id=_id, data=result)


__all__ = ["LibriSpeechDataset"]
