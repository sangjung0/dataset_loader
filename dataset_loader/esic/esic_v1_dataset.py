from __future__ import annotations

import re
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Any, cast
from typing_extensions import override

from sjpy.audio import load_from_mp4_file

from dataset_loader.abstract import ParquetDataset

from dataset_loader.esic.constants import VERBATIM
from dataset_loader.esic.esic_v1_sample import ESICv1Sample


class ESICv1Dataset(ParquetDataset[ESICv1Sample]):
    def __init__(self, *, parquet: pd.DataFrame, sr: int):
        super().__init__(parquet=parquet)
        self._sr: int = sr

    @property
    @override
    def args(self) -> dict[str, Any]:
        return {**super().args, "sr": self._sr}

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
    def get(self, idx: int) -> ESICv1Sample:
        if self.is_cleaned:
            raise RuntimeError("Cannot get sample from a cleaned dataset.")

        data = cast(dict[str, Any], self.dataset.iloc[idx].to_dict())

        def load_audio_func() -> npt.NDArray[np.float32]:
            mp4_path = data["mp4_path"]
            wav, _ = load_from_mp4_file(mp4_path, self._sr)
            return wav

        _id: str = data.pop("id")
        result: dict[str, Any] = {
            "load_audio_func": load_audio_func,
            "ref": re.sub(r"\s+", " ", data[VERBATIM]).strip(),
            **data,
        }
        return ESICv1Sample(id=_id, data=result)


__all__ = ["ESICv1Dataset"]
