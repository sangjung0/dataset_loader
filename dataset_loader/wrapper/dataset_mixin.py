from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, overload, Sequence, Generic, TypeVar
from typing_extensions import Self

from dataset_loader.interface import Sample, ConcatDataset, Dataset

T = TypeVar("T", bound="Sample")
U = TypeVar("U", bound="ConcatDataset")


class DatasetMixin(ABC, Generic[T, U]):
    @property
    @abstractmethod
    def dataset(self) -> Dataset: ...

    @property
    def args(self) -> dict:
        return {"dataset": self.dataset}

    @property
    def length(self) -> int:
        return len(self.dataset)

    @property
    def name(self) -> str:
        return self.dataset.name

    def __iter__(self) -> Generator[T, Any, None]:
        yield from self.iter()

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    @overload
    def __getitem__(self, key: Sequence[int]) -> Self: ...
    def __getitem__(self, key: int | slice | Sequence[int]) -> T | Self:
        return self.getitem(key)

    @overload
    def __add__(self, other: Self) -> U: ...
    @overload
    def __add__(self, other: U) -> U: ...
    def __add__(self, other: Self | U) -> U:
        return self.concat(other)

    def __len__(self) -> int:
        return self.length

    def iter(self) -> Generator[T, Any, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def getitem(self, key: int) -> T: ...
    @overload
    def getitem(self, key: slice) -> Self: ...
    @overload
    def getitem(self, key: Sequence[int]) -> Self: ...
    @abstractmethod
    def getitem(self, key: int | slice | Sequence[int]) -> T | Self: ...

    @overload
    def concat(self, other: Self) -> U: ...
    @overload
    def concat(self, other: U) -> U: ...
    @abstractmethod
    def concat(self, other: Self | U) -> U: ...

    def to_dict(self) -> dict:
        return {"dataset": self.dataset.to_dict()}

    @overload
    def sample(self, size: int) -> Self: ...
    @overload
    def sample(self, size: int, start: int) -> Self: ...
    @overload
    def sample(self, size: int, start: int, rng: np.random.Generator) -> Self: ...
    def sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if start < 0 or start >= len(self):
            raise IndexError("Invalid start index")
        elif size <= 0:
            size = len(self) - start
        else:
            size = min(size, len(self) - start)
        return self._sample(size=size, start=start, rng=rng)

    def samples_to_list(self) -> list[T]:
        return list(self.iter())

    def select(self, indices: Sequence[int]) -> Self:
        dataset = self.dataset.select(indices)
        return self.__class__(dataset=dataset)

    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        dataset = self.dataset.slice(start=start, stop=stop, step=step)
        return self.__class__(dataset=dataset)

    @abstractmethod
    def get(self, idx: int) -> T: ...

    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        dataset = self.dataset.sample(size=size, start=start, rng=rng)
        return self.__class__(dataset=dataset)


__all__ = ["DatasetMixin"]
