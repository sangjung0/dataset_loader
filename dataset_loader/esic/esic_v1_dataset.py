from __future__ import annotations

import re
import numpy as np
import pandas as pd

from typing_extensions import override

from sjpy.audio import load_from_mp4_file

from dataset_loader.interface import Sample
from dataset_loader.abstract import ParquetDataset

from dataset_loader.esic.constants import VERBATIM


class ESICv1Dataset(ParquetDataset):
    def __init__(
        self: ESICv1Dataset,
        *,
        parquet: pd.DataFrame,
        sr: int,
        use_cache: int = 0,
    ):
        super().__init__(parquet=parquet, use_cache=use_cache)
        self._sr: int = sr

    @ParquetDataset.args.getter
    def args(self: ESICv1Dataset) -> dict:
        return {**super().args, "sr": self._sr}

    @property
    def sr(self: ESICv1Dataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: ESICv1Dataset, value: int) -> None:
        if isinstance(value, int) and value > 0:
            self._sr = value
        else:
            raise ValueError("Sample rate must be a positive integer")

    @override
    def _get(self: ESICv1Dataset, idx: int) -> Sample:
        data = self._parquet.iloc[idx].to_dict()

        def load_audio_func() -> np.ndarray:
            mp4_path = data["mp4_path"]
            wav, _ = load_from_mp4_file(mp4_path, self._sr)
            return wav

        _id = data.pop("id")
        result = {
            "load_audio_func": load_audio_func,
            "ref": re.sub(r"\s+", " ", data[VERBATIM]).strip(),
            **data,
        }
        return Sample(id=_id, data=result)


__all__ = ["ESICv1Dataset"]
