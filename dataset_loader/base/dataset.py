from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, overload, TypeVar, cast
from typing_extensions import Self, override
from collections.abc import Mapping, Generator, Iterable

from dataset_loader.protocol import DatasetProtocol
from dataset_loader.base.sample import Sample


if TYPE_CHECKING:
    from dataset_loader.base.concat_dataset import ConcatDataset


T = TypeVar("T")


class Dataset(DatasetProtocol[T, Sample], ABC):
    """
    다양한 Dataset을 위한 공통 추상 클래스이다. \n
    다른 라이브러리의 Dataset을 래핑하거나, 자체적으로 Dataset을 구현할 때 이 클래스를 상속하여 구현한다.

    """

    def __init__(self) -> None:
        self._is_cleaned = False

    @property
    @abstractmethod
    @override
    def dataset(self) -> T:
        raise NotImplementedError

    @property
    @override
    def args(self) -> dict[str, Any]:
        return {}

    @property
    @override
    def is_cleaned(self) -> bool:
        return self._is_cleaned

    @property
    @abstractmethod
    @override
    def length(self) -> int:
        raise NotImplementedError

    @override
    def __len__(self) -> int:
        return self.length

    @property
    @override
    def name(self) -> str:
        return self.__class__.__name__

    @override
    def __iter__(self) -> Generator[Sample, None, None]:
        yield from self.iter()

    @override
    def iter(self) -> Generator[Sample, None, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def __getitem__(self, key: int) -> Sample: ...
    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Self: ...
    @override
    def __getitem__(self, key: int | slice | Iterable[int]) -> Sample | Self:
        return self.getitem(key)

    @overload
    def getitem(self, key: int) -> Sample: ...
    @overload
    def getitem(self, key: slice | Iterable[int]) -> Self: ...
    @override
    def getitem(self, key: int | slice | Iterable[int]) -> Sample | Self:
        if self.is_cleaned:
            raise RuntimeError("Cannot access a cleaned dataset")
        elif isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        elif isinstance(key, Iterable):
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

    @abstractmethod
    @override
    def select(self, indices: Iterable[int]) -> Self:
        raise NotImplementedError

    @abstractmethod
    @override
    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Self:
        raise NotImplementedError

    @override
    def sample(
        self,
        size: int = -1,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if self.is_cleaned:
            raise RuntimeError("Cannot sample from a cleaned dataset")
        elif start < 0 or start >= len(self):
            raise IndexError("Invalid start index")
        elif size < 0:
            size = len(self) - start
        else:
            size = min(size, len(self) - start)
        return self._sample(size=size, start=start, rng=rng)

    @abstractmethod
    def _sample(
        self,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        raise NotImplementedError

    @override
    def __add__(self, other: DatasetProtocol[Any, Any]) -> ConcatDataset[Any]:
        return self.concat(other)

    @override
    def concat(self, other: DatasetProtocol[Any, Any]) -> ConcatDataset[Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot concatenate a cleaned dataset")

        from dataset_loader.base.concat_dataset import ConcatDataset

        if isinstance(other, ConcatDataset):
            return ConcatDataset(datasets=[self] + other._datasets)
        elif isinstance(other, Dataset):
            return ConcatDataset(datasets=[self, other])
        else:
            raise TypeError("Invalid type for concatenation")

    @abstractmethod
    @override
    def get(self, idx: int) -> Sample:
        raise NotImplementedError

    @abstractmethod
    @override
    def clean(self) -> None:
        if self.is_cleaned:
            return
        self._is_cleaned = True

    @override
    def to_dict(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return self.args

    @classmethod
    @abstractmethod
    @override
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        raise NotImplementedError

    @override
    def __getstate__(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return {
            **self.to_dict(),
            **self.__get_import__(),
        }

    @classmethod
    @override
    def __setstate__(cls, state: Mapping[str, Any]) -> Dataset[T]:
        if all(k in state for k in ("module", "qualname", "type")):
            state, dataset_cls = cls.__set_import__(state)
        elif all(k not in state for k in ("module", "qualname", "type")):
            dataset_cls = cls
        else:
            raise ValueError("Invalid config data: missing module, qualname, or type")

        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"{dataset_cls} is not a subclass of Dataset")
        elif dataset_cls == Dataset[T]:
            raise TypeError("Cannot instantiate Dataset directly")

        return dataset_cls.from_dict(state)

    @override
    def __get_import__(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot extract import data from a cleaned dataset")
        return {
            "module": self.__class__.__module__,
            "qualname": self.__class__.__qualname__,
            "type": self.__class__.__name__,
        }

    @classmethod
    @override
    def __set_import__(
        cls, import_info: Mapping[str, Any]
    ) -> tuple[dict[str, Any], type[Dataset[T]]]:
        from sjpy.reference import import_from, ImportData

        _class: type[Dataset[T]] = import_from(cast(ImportData, import_info))

        if not isinstance(_class, type):
            raise TypeError(f"{_class} is not a class")
        elif not issubclass(_class, Dataset):
            raise TypeError(f"{_class} is not a subclass of Dataset or DatasetWrapper")
        elif import_info["type"] != _class.__name__:
            raise TypeError(
                f"Type mismatch: expected {_class.__name__}, got {import_info['type']}"
            )

        d = {
            k: v
            for k, v in import_info.items()
            if k not in ("module", "qualname", "type")
        }
        return d, _class


__all__ = ["Dataset"]
