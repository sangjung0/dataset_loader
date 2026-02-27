from __future__ import annotations

import numpy as np
import pandas as pd

from abc import ABC
from typing_extensions import override

from dataset_loader.interface import Sample, Dataset


class ParquetDataset(Dataset, ABC):
    def __init__(
        self: ParquetDataset,
        *,
        parquet: pd.DataFrame,
        use_cache: int = 0,
    ):
        super().__init__(use_cache=use_cache)

        self._parquet: pd.DataFrame = parquet

    @Dataset.args.getter
    @override
    def args(self: ParquetDataset) -> dict:
        return {**super().args, "parquet": self._parquet}

    @Dataset.length.getter
    @override
    def length(self: ParquetDataset) -> int:
        if self.is_cleaned:
            raise RuntimeError("Cannot get length of a cleaned dataset.")
        return len(self._parquet)

    @override
    def select(
        self: ParquetDataset, indices: list[int], *, use_cache: int = 0
    ) -> ParquetDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot select from a cleaned dataset.")
        selected_parquet = self._parquet.iloc[indices].reset_index(drop=True)
        args = self.args
        args["parquet"] = selected_parquet
        args["use_cache"] = use_cache
        return self.__class__(**args)

    @override
    def slice(
        self: ParquetDataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        *,
        use_cache: int = 0,
    ) -> ParquetDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot slice a cleaned dataset.")
        sliced_parquet = self._parquet.iloc[start:stop:step].reset_index(drop=True)
        args = self.args
        args["parquet"] = sliced_parquet
        args["use_cache"] = use_cache
        return self.__class__(**args)

    @override
    def _sample(
        self: ParquetDataset,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
        use_cache: int = 0,
    ) -> Sample:
        if self.is_cleaned:
            raise RuntimeError("Cannot sample from a cleaned dataset.")
        elif rng is None or size == len(self) - start:
            return self.slice(start, start + size, use_cache=use_cache)
        else:
            indices = range(len(self))[start:]
            indices = rng.choice(indices, size=size, replace=False)
            return self.select(indices, use_cache=use_cache)

    @override
    def clean(self: ParquetDataset) -> None:
        super().clean()
        self._parquet = pd.DataFrame()

    @override
    def to_dict(self: ParquetDataset) -> dict:
        if self.is_cleaned:
            raise RuntimeError("Cannot convert a cleaned dataset to dict.")
        args = super().to_dict()
        args["parquet"] = self._parquet.to_dict(orient="records")
        return args

    @classmethod
    @override
    def from_dict(cls: type[ParquetDataset], data: dict) -> ParquetDataset:
        data["parquet"] = pd.DataFrame(data["parquet"])
        return cls(**data)


__all__ = ["ParquetDataset"]
