from __future__ import annotations

import numpy as np

from abc import ABC
from typing import Any, TypeVar
from typing_extensions import override, Self
from collections.abc import Mapping, Iterable
from datasets import Dataset as DT

from dataset_loader.base import Dataset, Sample

S = TypeVar("S", bound=Sample)


class HuggingfaceDataset(Dataset[DT, S], ABC):
    def __init__(self, *, dataset: DT):
        super().__init__()
        self._dataset: DT | None = dataset

    @property
    @override
    def dataset(self) -> DT:
        if self.is_cleaned or self._dataset is None:
            raise RuntimeError("Cannot get dataset of a cleaned dataset.")
        return self._dataset

    @property
    @override
    def args(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {**super().args, "dataset": self.dataset}

    @property
    @override
    def length(self) -> int:
        if self.is_cleaned or self.dataset is None:
            raise RuntimeError("Cannot get length of a cleaned dataset")
        return len(self.dataset)

    @override
    def select(self, indices: Iterable[int]) -> Self:
        if self.is_cleaned or self.dataset is None:
            raise RuntimeError("Cannot select from a cleaned dataset")
        dataset = self.dataset.select(indices)
        args = self.args
        args["dataset"] = dataset
        return type(self)(**args)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        if self.is_cleaned or self.dataset is None:
            raise RuntimeError("Cannot slice a cleaned dataset")
        return self.select(range(len(self.dataset))[start:stop:step])

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if self.is_cleaned or self.dataset is None:
            raise RuntimeError("Cannot sample from a cleaned dataset")
        elif rng is None or size == len(self) - start:
            return self.slice(start, start + size)
        else:
            indices = range(len(self))[start:]
            index = rng.choice(indices, size=size, replace=False)
            return self.select(list(index))

    @override
    def clean(self: Self) -> None:
        if self.is_cleaned:
            return
        self._dataset = None
        super().clean()

    @override
    def to_dict(self: Self) -> dict[str, Any]:
        if self.is_cleaned or self.dataset is None:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["dataset"] = self.dataset.to_dict()
        return args

    @classmethod
    @override
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = {**data}
        data["dataset"] = DT.from_dict(data["dataset"])
        return cls(**data)

    @classmethod
    @override
    def __setstate__(cls, state: Mapping[str, Any]) -> Self:
        dataset = super().__setstate__(state)
        if isinstance(dataset, cls):
            return dataset
        raise TypeError(f"Dataset must be an instance of {cls.__name__}")


__all__ = ["HuggingfaceDataset"]
