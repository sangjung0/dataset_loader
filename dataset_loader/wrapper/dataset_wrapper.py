from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, overload, TypeVar, cast
from typing_extensions import Self, override
from collections.abc import Mapping, Generator, Iterable

from dataset_loader.protocol import (
    DatasetProtocol,
    SampleProtocol,
    ConcatDatasetProtocol,
)

S = TypeVar("S", bound=SampleProtocol)
D = TypeVar("D", bound=DatasetProtocol[Any, Any])


class DatasetWrapper(DatasetProtocol[D, S], ABC):
    """
    Dataset의 래핑 클래스이다. \n
    다양한 도메인에 대해서 대응하기 위한 인터페이스를 제공한다.

    Attributes:
        dataset (Dataset): 래핑할 Dataset 객체
        args (dict): DatasetWrapper를 생성하는 데 필요한 인자들
        length (int): Dataset의 길이
        name (str): Dataset의 이름
    """

    def __init__(self, dataset: D):
        self._dataset = dataset

    @property
    @override
    def dataset(self) -> D:
        return self._dataset

    @property
    @override
    def args(self) -> dict[str, Any]:
        return {"dataset": self.dataset}

    @property
    @override
    def is_cleaned(self) -> bool:
        return self.dataset.is_cleaned

    @property
    @override
    def length(self) -> int:
        return len(self.dataset)

    @override
    def __len__(self) -> int:
        return self.length

    @property
    @override
    def name(self) -> str:
        return self.dataset.name

    @override
    def __iter__(self) -> Generator[S, None, None]:
        yield from self.iter()

    @override
    def iter(self) -> Generator[S, None, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def __getitem__(self, key: int) -> S: ...
    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Self: ...
    @override
    def __getitem__(self, key: int | slice | Iterable[int]) -> S | Self:
        return self.getitem(key)

    @overload
    def getitem(self, key: int) -> S: ...
    @overload
    def getitem(self, key: slice | Iterable[int]) -> Self: ...
    @override
    def getitem(self, key: int | slice | Iterable[int]) -> S | Self:
        result = self.dataset[key]
        if isinstance(result, SampleProtocol):
            return cast(S, result)
        return self.__class__(dataset=cast(D, result))

    @override
    def select(self, indices: Iterable[int]) -> Self:
        dataset = self.dataset.select(indices)
        return self.__class__(dataset=dataset)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        dataset = self.dataset.slice(start=start, stop=stop, step=step)
        return self.__class__(dataset=dataset)

    @override
    def sample(
        self,
        size: int = -1,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        dataset = self.dataset.sample(size=size, start=start, rng=rng)
        return self.__class__(dataset=dataset)

    @override
    def __add__(
        self, other: DatasetProtocol[Any, S]
    ) -> DatasetWrapper[D, S] | DatasetWrapper[ConcatDatasetProtocol[D, S], S]:
        return self.concat(other)

    @abstractmethod
    @override
    def concat(
        self, other: DatasetProtocol[Any, S]
    ) -> DatasetWrapper[D, S] | DatasetWrapper[ConcatDatasetProtocol[D, S], S]:
        raise NotImplementedError

    @override
    def get(self, idx: int) -> S:
        return cast(S, self.dataset.get(idx))

    @override
    def clean(self) -> None:
        self.dataset.clean()

    @override
    def to_dict(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["class"] = self.dataset.__class__
        args["dataset"] = self.dataset.to_dict()

        return args

    @classmethod
    @override
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = {**data}
        _class = data.pop("class")
        data["dataset"] = _class.from_dict(data["dataset"])
        return cls(**data)

    @override
    def __getstate__(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return {
            **self.__get_import__(),
            "dataset": self.dataset.__getstate__(),
        }

    @classmethod
    @override
    def __setstate__(cls, state: Mapping[str, Any]) -> DatasetWrapper[Any, Any]:
        if all(k in state for k in ("module", "qualname", "type")):
            state, wrapper_cls = cls.__set_import__(state)
        else:
            raise ValueError("Invalid pointer data: missing module, qualname, or type")

        if not issubclass(wrapper_cls, DatasetWrapper):
            raise TypeError(f"{wrapper_cls} is not a subclass of DatasetWrapper")
        elif wrapper_cls == DatasetWrapper:
            raise TypeError("Cannot instantiate DatasetWrapper directly")
        elif cls != DatasetWrapper and wrapper_cls != cls:
            raise TypeError(
                f"Invalid type for deserialization, expected {cls}, got {wrapper_cls}"
            )

        from sjpy.reference import import_from

        dataset_cls = import_from(state["dataset"])
        state["dataset"] = dataset_cls.__setstate__(state["dataset"])
        return wrapper_cls(**state)

    def __get_import__(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return {
            "module": self.__class__.__module__,
            "qualname": self.__class__.__qualname__,
            "type": self.__class__.__name__,
        }

    @classmethod
    @override
    def __set_import__(
        cls, import_info: Mapping[str, Any]
    ) -> tuple[dict[str, Any], type[DatasetWrapper[Any, Any]]]:
        from sjpy.reference import import_from, ImportData

        _class: type[DatasetWrapper[Any, Any]] = import_from(
            cast(ImportData, import_info)
        )

        if not isinstance(_class, type):
            raise TypeError(f"{_class} is not a class")
        elif not issubclass(_class, DatasetWrapper):
            raise TypeError(f"{_class} is not a subclass of DatasetWrapper")
        elif import_info["type"] != _class.__name__:
            raise TypeError(
                f"Type mismatch: expected {_class.__name__}, got {import_info['type']}"
            )

        state = {
            k: v
            for k, v in import_info.items()
            if k not in ("module", "qualname", "type")
        }
        return state, _class


__all__ = ["DatasetWrapper"]
