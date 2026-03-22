from __future__ import annotations
import re

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Sequence
from typing_extensions import override

from dataset_loader.base import Sample
from dataset_loader.abstract import ParquetDataset


class TedliumDataset(ParquetDataset):
    def __init__(
        self,
        *,
        parquet: pd.DataFrame,
        sr: int,
        ignore_set: Sequence[str] = [],
    ):
        super().__init__(parquet=parquet)
        self._sr = sr
        self._ignore_set = list(ignore_set)

    @ParquetDataset.args.getter
    def args(self):
        return {**super().args, "ignore_set": self._ignore_set, "sr": self._sr}

    @property
    def sr(self) -> int:
        return self._sr

    @sr.setter
    def sr(self, value: int) -> None:
        if isinstance(value, int) and value > 0:
            self._sr = value
        else:
            raise ValueError("Sample rate must be a positive integer")

    @override
    def get(self, idx: int) -> Sample:
        if self.is_cleaned:
            raise RuntimeError("Cannot get sample from a cleaned dataset")

        data = self._parquet.iloc[idx].to_dict()

        def load_audio_func() -> npt.NDArray[np.float32]:
            audio_path = data["audio_path"]
            wav, _ = librosa.load(audio_path, sr=self._sr)
            return wav.astype(np.float32)

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
