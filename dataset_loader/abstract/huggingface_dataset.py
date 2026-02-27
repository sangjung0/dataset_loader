from __future__ import annotations

import numpy as np

from typing import Sequence
from typing_extensions import override
from datasets import Dataset as DT
from abc import ABC

from dataset_loader.interface import Dataset


class HuggingfaceDataset(Dataset, ABC):
    def __init__(
        self: HuggingfaceDataset,
        *,
        dataset: DT,
        use_cache: int = 0,
    ):
        super().__init__(use_cache=use_cache)
        self._dataset = dataset

    @Dataset.args.getter
    @override
    def args(self: HuggingfaceDataset) -> dict:
        if self.is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {**super().args, "dataset": self._dataset}

    @Dataset.length.getter
    @override
    def length(self: HuggingfaceDataset) -> int:
        if self.is_cleaned:
            raise RuntimeError("Cannot get length of a cleaned dataset")
        return len(self._dataset)

    @override
    def select(
        self: HuggingfaceDataset, indices: Sequence[int], *, use_cache: int = 0
    ) -> HuggingfaceDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot select from a cleaned dataset")
        dataset = self._dataset.select(indices)
        args = self.args
        args["dataset"] = dataset
        args["use_cache"] = use_cache
        return type(self)(**args)

    @override
    def slice(
        self: HuggingfaceDataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        *,
        use_cache: int = 0,
    ) -> HuggingfaceDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot slice a cleaned dataset")
        return self.select(
            range(len(self._dataset))[start:stop:step], use_cache=use_cache
        )

    @override
    def _sample(
        self: HuggingfaceDataset,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
        use_cache: int = 0,
    ) -> HuggingfaceDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot sample from a cleaned dataset")
        elif rng is None or size == len(self) - start:
            return self.slice(start, start + size, use_cache=use_cache)
        else:
            indices = range(len(self))[start:]
            index = rng.choice(indices, size=size, replace=False)
            return self.select(index, use_cache=use_cache)

    @override
    def clean(self: HuggingfaceDataset) -> None:
        if self.is_cleaned:
            return
        self._dataset = None
        super().clean()

    @override
    def to_dict(self: HuggingfaceDataset) -> dict:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["dataset"] = self._dataset.to_dict()
        return args

    @classmethod
    @override
    def from_dict(cls: type[HuggingfaceDataset], data: dict) -> HuggingfaceDataset:
        data["dataset"] = DT.from_dict(data["dataset"])
        return cls(**data)


__all__ = ["HuggingfaceDataset"]
