from __future__ import annotations
import re

import librosa
import numpy as np
import pandas as pd

from typing import Sequence
from typing_extensions import override

from dataset_loader.interface import Sample
from dataset_loader.abstract import ParquetDataset


class TedliumDataset(ParquetDataset):
    def __init__(
        self: TedliumDataset,
        *,
        parquet: pd.DataFrame,
        sr: int,
        use_cache: int = 0,
        ignore_set: Sequence[str] = [],
    ):
        super().__init__(parquet=parquet, use_cache=use_cache)
        self._sr = sr
        self._ignore_set = list(ignore_set)

    @ParquetDataset.args.getter
    def args(self):
        return {**super().args, "ignore_set": self._ignore_set, "sr": self._sr}

    @property
    def sr(self: TedliumDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: TedliumDataset, value: int) -> None:
        if isinstance(value, int) and value > 0:
            self._sr = value
        else:
            raise ValueError("Sample rate must be a positive integer")

    @override
    def _get(self: TedliumDataset, idx: int) -> Sample:
        data = self._parquet.iloc[idx].to_dict()

        def load_audio_func() -> np.ndarray:
            audio_path = data["audio_path"]
            wav, _ = librosa.load(audio_path, sr=self._sr)
            return wav

        diarization = data.pop("stm")
        ref = data["text"]
        for ignore in self._ignore_set:
            ref = re.sub(rf"{re.escape(ignore)}\s*", "", ref)
        ref = ref.strip()
        _id = data.pop("id")

        result = {
            "load_audio_func": load_audio_func,
            "ref": ref,
            "diarization": diarization,
            **data,
        }

        return Sample(id=_id, data=result)


__all__ = ["TedliumDataset"]
