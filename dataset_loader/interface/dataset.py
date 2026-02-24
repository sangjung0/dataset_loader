from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, overload, Sequence

if TYPE_CHECKING:
    from dataset_loader.interface.constants import Task
    from dataset_loader.interface.sample import Sample
    from dataset_loader.interface.concat_dataset import ConcatDataset


class Dataset(ABC):
    def __init__(self: Dataset, *, task: tuple[Task, ...]):
        self.task = task

    @property
    def args(self: Dataset) -> dict:
        return {"task": self.task}

    @property
    @abstractmethod
    def length(self: Dataset) -> int: ...

    @property
    def name(self: Dataset) -> str:
        return self.__class__.__name__

    def __iter__(self: Dataset) -> Generator[Sample, Any, None]:
        yield from self.iter()

    @overload
    def __getitem__(self: Dataset, key: int) -> Sample: ...
    @overload
    def __getitem__(self: Dataset, key: slice) -> Dataset: ...
    @overload
    def __getitem__(self: Dataset, key: Sequence[int]) -> Dataset: ...
    def __getitem__(
        self: Dataset, key: int | slice | Sequence[int]
    ) -> Sample | Dataset:
        return self.getitem(key)

    @overload
    def __add__(self: Dataset, other: Dataset) -> ConcatDataset: ...
    @overload
    def __add__(self: Dataset, other: ConcatDataset) -> ConcatDataset: ...
    def __add__(self: Dataset, other: Dataset | ConcatDataset) -> ConcatDataset:
        return self.concat(other)

    def __len__(self: Dataset) -> int:
        return self.length

    def iter(self: Dataset) -> Generator[Sample, Any, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def getitem(self: Dataset, key: int) -> Sample: ...
    @overload
    def getitem(self: Dataset, key: slice) -> Dataset: ...
    @overload
    def getitem(self: Dataset, key: Sequence[int]) -> Dataset: ...
    def getitem(self: Dataset, key: int | slice | Sequence[int]) -> Sample | Dataset:
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        elif isinstance(key, Sequence):
            return self.select(key)
        elif isinstance(key, int):
            n = len(self)
            if key < 0:
                key += n
            if not (0 <= key < n):
                raise IndexError("Index out of range")
            return self.get(key)
        else:
            raise TypeError("Invalid key type")

    @overload
    def concat(self: Dataset, other: Dataset) -> ConcatDataset: ...
    @overload
    def concat(self: Dataset, other: ConcatDataset) -> ConcatDataset: ...
    def concat(self: Dataset, other: Dataset | ConcatDataset) -> ConcatDataset:
        from dataset_loader.interface.concat_dataset import ConcatDataset

        if self.task != other.task:
            raise ValueError("Datasets have different tasks")
        elif isinstance(other, ConcatDataset):
            return ConcatDataset(datasets=[self] + other._datasets, task=self.task)
        elif isinstance(other, Dataset):
            return ConcatDataset(datasets=[self, other], task=self.task)
        else:
            raise TypeError("Invalid type for concatenation")

    def to_dict(self: Dataset) -> dict:
        return {"task": self.task}

    @overload
    def sample(self: Dataset, size: int) -> Dataset: ...
    @overload
    def sample(self: Dataset, size: int, start: int) -> Dataset: ...
    @overload
    def sample(
        self: Dataset, size: int, start: int, rng: np.random.Generator
    ) -> Dataset: ...
    def sample(
        self: Dataset,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Dataset:
        if start < 0 or start >= len(self):
            raise IndexError("Invalid start index")
        elif size <= 0:
            size = len(self) - start
        else:
            size = min(size, len(self) - start)
        return self._sample(size=size, start=start, rng=rng)

    def samples_to_list(self) -> list[Sample]:
        return list(self.iter())

    @abstractmethod
    def select(self: Dataset, indices: Sequence[int]) -> Dataset: ...

    @abstractmethod
    def slice(
        self: Dataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Dataset: ...

    @abstractmethod
    def get(self: Dataset, idx: int) -> Sample: ...

    @abstractmethod
    def _sample(
        self: Dataset,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Dataset: ...

    @classmethod
    @abstractmethod
    def from_dict(cls: type[Dataset], data: dict) -> Dataset:
        raise NotImplementedError


__all__ = ["Dataset"]
