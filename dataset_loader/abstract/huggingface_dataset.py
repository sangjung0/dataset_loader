from __future__ import annotations

import numpy as np

from typing import Sequence
from typing_extensions import override
from datasets import Dataset as DT, Audio
from abc import ABC

from dataset_loader.interface import Dataset
from dataset_loader.interface.constants import Task


class HuggingfaceDataset(Dataset, ABC):
    def __init__(
        self: HuggingfaceDataset, *, dataset: DT, sr: int, task: tuple[Task, ...]
    ):
        super().__init__(task=task)
        self._dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))
        self._sr = sr

    @Dataset.args.getter
    @override
    def args(self: HuggingfaceDataset) -> dict:
        return {**super().args, "dataset": self._dataset, "sr": self._sr}

    @Dataset.length.getter
    @override
    def length(self: HuggingfaceDataset) -> int:
        return len(self._dataset)

    @property
    def sr(self: HuggingfaceDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: HuggingfaceDataset, value: int) -> None:
        self._sr = value
        self._dataset = self._dataset.cast_column("audio", Audio(sampling_rate=value))

    @override
    def to_dict(self: HuggingfaceDataset) -> dict:
        return {**super().to_dict(), "dataset": self._dataset.to_dict(), "sr": self._sr}

    @override
    def select(self: HuggingfaceDataset, indices: Sequence[int]) -> HuggingfaceDataset:
        dataset = self._dataset.select(indices)
        args = self.args
        args["dataset"] = dataset
        return type(self)(**args)

    @override
    def slice(
        self: HuggingfaceDataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> HuggingfaceDataset:
        return self.select(range(len(self._dataset))[start:stop:step])

    @override
    def _sample(
        self: HuggingfaceDataset,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> HuggingfaceDataset:
        if rng is None or size == len(self) - start:
            return self.slice(start, start + size)
        else:
            indices = range(len(self))[start:]
            index = rng.choice(indices, size=size, replace=False)
            return self.select(index)

    @classmethod
    @override
    def from_dict(cls: type[HuggingfaceDataset], data: dict) -> HuggingfaceDataset:
        data["dataset"] = DT.from_dict(data["dataset"])
        return cls(**data)


__all__ = ["HuggingfaceDataset"]
