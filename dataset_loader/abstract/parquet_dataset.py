from __future__ import annotations

import numpy as np
import pandas as pd

from abc import ABC
from typing import Sequence, Any
from typing_extensions import override, Self
from collections.abc import Mapping

from dataset_loader.base import Dataset


class ParquetDataset(Dataset, ABC):
    def __init__(self, *, parquet: pd.DataFrame):
        super().__init__()
        self._parquet: pd.DataFrame = parquet

    @Dataset.args.getter
    @override
    def args(self) -> dict[str, Any]:
        return {**super().args, "parquet": self._parquet}

    @Dataset.length.getter
    @override
    def length(self) -> int:
        if self.is_cleaned:
            raise RuntimeError("Cannot get length of a cleaned dataset.")
        return len(self._parquet)

    @override
    def select(self, indices: Sequence[int]) -> Self:
        if self.is_cleaned:
            raise RuntimeError("Cannot select from a cleaned dataset.")
        selected_parquet = self._parquet.iloc[list(indices)].reset_index(drop=True)
        args = self.args
        args["parquet"] = selected_parquet
        return self.__class__(**args)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        if self.is_cleaned:
            raise RuntimeError("Cannot slice a cleaned dataset.")
        sliced_parquet = self._parquet.iloc[start:stop:step].reset_index(drop=True)
        args = self.args
        args["parquet"] = sliced_parquet
        return self.__class__(**args)

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if self.is_cleaned:
            raise RuntimeError("Cannot sample from a cleaned dataset.")
        elif rng is None or size == len(self) - start:
            return self.slice(start, start + size)
        else:
            indices = range(len(self))[start:]
            indices = rng.choice(indices, size=size, replace=False)
            return self.select(list(indices))

    @override
    def clean(self) -> None:
        super().clean()
        self._parquet = pd.DataFrame()

    @override
    def to_dict(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot convert a cleaned dataset to dict.")
        args = super().to_dict()
        args["parquet"] = self._parquet.to_dict(orient="records")
        return args

    @classmethod
    @override
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = {**data}
        data["parquet"] = pd.DataFrame(data["parquet"])
        return cls(**data)


__all__ = ["ParquetDataset"]
