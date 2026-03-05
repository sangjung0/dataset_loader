from __future__ import annotations

import numpy as np

from typing import Any
from typing_extensions import override
from collections.abc import Mapping, Sequence

from dataset_loader.base import Dataset, Sample


class DummyDataset(Dataset):
    def __init__(self: DummyDataset, *, samples: list[Sample]):
        super().__init__()
        self._samples = list(samples)

    @Dataset.args.getter
    def args(self: DummyDataset) -> dict:
        return {
            **super().args,
            "samples": [s.to_dict() for s in self._samples],
        }

    @Dataset.length.getter
    def length(self: DummyDataset) -> int:
        if self.is_cleaned:
            raise RuntimeError("Cannot get length of a cleaned dataset")
        return len(self._samples)

    @override
    def select(self: DummyDataset, indices: Sequence[int]) -> DummyDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot select from a cleaned dataset")
        return DummyDataset(samples=[self._samples[i] for i in indices])

    @override
    def slice(
        self: DummyDataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> DummyDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot slice a cleaned dataset")
        return DummyDataset(samples=self._samples[start:stop:step])

    @override
    def _sample(
        self: DummyDataset,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> DummyDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot sample from a cleaned dataset")
        elif rng is None or size == len(self) - start:
            return self.slice(start, start + size)
        else:
            indices = range(len(self))[start:]
            indices = rng.choice(indices, size=size, replace=False)
            return self.select(list(indices))

    @override
    def get(self: DummyDataset, idx: int) -> Sample:
        try:
            return self._samples[idx]
        except IndexError as e:
            raise IndexError("Index out of range") from e

    @override
    def clean(self: DummyDataset) -> None:
        super().clean()
        self._samples.clear()

    @classmethod
    @override
    def from_dict(cls, data: Mapping[str, Any]) -> DummyDataset:
        data = {**data}
        data["samples"] = [Sample.from_dict(s) for s in data["samples"]]
        return cls(**data)


__all__ = ["DummyDataset"]
