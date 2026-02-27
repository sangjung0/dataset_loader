from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, overload, Sequence, Generic, TypeVar

from dataset_loader.interface import Sample, ConcatDataset, Dataset

T = TypeVar("T", bound="Sample")
U = TypeVar("U", bound="ConcatDataset")


class DatasetWrapper(ABC, Generic[T, U]):
    """
    Dataset의 래핑 클래스이다. \n
    다양한 도메인에 대해서 대응하기 위한 인터페이스를 제공한다.

    Attributes:
        dataset (Dataset): 래핑된 Dataset 객체
        tasks (tuple[str, ...]): Dataset이 지원하는 태스크의 이름
        args (dict): DatasetWrapper를 생성하는 데 필요한 인자들
        length (int): Dataset의 길이
        name (str): Dataset의 이름
    """

    @property
    @abstractmethod
    def dataset(self: DatasetWrapper) -> Dataset: ...

    @property
    def is_cleaned(self: DatasetWrapper) -> bool:
        return self.dataset.is_cleaned

    @property
    def tasks(self: DatasetWrapper) -> tuple[str, ...]:
        return self.dataset.task

    @property
    def args(self: DatasetWrapper) -> dict:
        return {"dataset": self.dataset}

    @property
    def length(self: DatasetWrapper) -> int:
        return len(self.dataset)

    def __len__(self: DatasetWrapper) -> int:
        return self.length

    @property
    def name(self: DatasetWrapper) -> str:
        return self.dataset.name

    def __iter__(self: DatasetWrapper) -> Generator[T, Any, None]:
        yield from self.iter()

    def iter(self: DatasetWrapper) -> Generator[T, Any, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def __getitem__(self: DatasetWrapper, key: int) -> T: ...
    @overload
    def __getitem__(
        self: DatasetWrapper, key: slice | Sequence[int]
    ) -> DatasetWrapper: ...
    def __getitem__(
        self: DatasetWrapper, key: int | slice | Sequence[int]
    ) -> T | DatasetWrapper:
        return self.getitem(key)

    @overload
    def getitem(self: DatasetWrapper, key: int, *, use_cache: int = 0) -> T: ...
    @overload
    def getitem(
        self: DatasetWrapper, key: slice | Sequence[int], *, use_cache: int = 0
    ) -> DatasetWrapper: ...
    @abstractmethod
    def getitem(
        self: DatasetWrapper, key: int | slice | Sequence[int], *, use_cache: int = 0
    ) -> T | DatasetWrapper:
        """
        key == int -> type[Sample] \n
        key == slice -> DatasetWrapper (using slice method) \n
        key == Sequence[int] -> DatasetWrapper (using select method) \n

        Args:
            key (int | slice | Sequence[int]): 인덱스 또는 슬라이스 또는 인덱스 시퀀스
            use_cache (int): 선택된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            result (type[Sample] | DatasetWrapper): 요청된 샘플 또는 데이터셋

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            TypeError: key의 타입이 int, slice, Sequence[int]이 아닌 경우
        """
        raise NotImplementedError("getitem method must be implemented in subclass")

    def __add__(self: DatasetWrapper, other: DatasetWrapper | U) -> U:
        return self.concat(other)

    @abstractmethod
    def concat(
        self: DatasetWrapper, other: DatasetWrapper | U, *, use_cache: int = 0
    ) -> U:
        """
        self와 other을 연결하여 새로운 ConcatDataset을 반환한다.

        Args:
            other (DatasetWrapper | type[ConcatDataset]): 연결할 다른 데이터셋
            use_cache (int): 연결된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            type[ConcatDataset]: 연결된 데이터셋
        Raises:
            ValueError: self와 other의 task가 다른 경우
            TypeError: other의 타입이 DatasetWrapper 또는 ConcatDataset이 아닌 경우
        """
        raise NotImplementedError("concat method must be implemented in subclass")

    def sample(
        self,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
        use_cache: int = 0,
    ) -> DatasetWrapper:
        """
        데이터셋에서 무작위로 샘플링하여 새로운 DatasetWrapper을 반환한다.
        Args:
            size (int): 샘플링할 샘플의 수. 음수이면 start부터 끝까지 샘플링한다.
            start (int): 샘플링을 시작할 인덱스. 기본값은 0이다.
            rng (np.random.Generator | np.random.RandomState | None): 무작위 수 생성기. 기본값은 None이며, 이 경우 단순 슬라이싱이 수행된다.
            use_cache (int): 샘플링된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            DatasetWrapper: 샘플링된 데이터셋 래퍼
        Raises:
            IndexError: start가 유효한 인덱스 범위를 벗어난 경우
        """
        dataset = self.dataset.sample(
            size=size, start=start, rng=rng, use_cache=use_cache
        )
        return self.__class__(dataset=dataset)

    def select(self, indices: Sequence[int], *, use_cache: int = 0) -> DatasetWrapper:
        dataset = self.dataset.select(indices, use_cache=use_cache)
        return self.__class__(dataset=dataset)

    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        *,
        use_cache: int = 0,
    ) -> DatasetWrapper:
        dataset = self.dataset.slice(
            start=start, stop=stop, step=step, use_cache=use_cache
        )
        return self.__class__(dataset=dataset)

    @abstractmethod
    def get(self, idx: int) -> T: ...

    def clean(self: DatasetWrapper) -> None:
        self.dataset.clean()

    def extract_import_data(self: DatasetWrapper) -> dict:
        """
        데이터셋에서 임포트 데이터를 추출하는 메서드이다. 이 메서드는 데이터셋이 외부 리소스나 참조를 포함하는 경우, 해당 정보를 추출하여 반환한다.
        Returns:
            dict: 임포트 데이터가 포함된 딕셔너리
        """
        return {
            "module": self.__class__.__module__,
            "qualname": self.__class__.__qualname__,
            "type": self.__class__.__name__,
        }

    @staticmethod
    def import_from_pointer(data: dict) -> tuple[dict, type[DatasetWrapper]]:
        """
        to_pointer 메서드로 직렬화된 딕셔너리를 가져와서 필요한 모듈을 동적으로 임포트하는 정적 메서드이다. 이 메서드는 데이터셋의 위치나 참조 정보를 포함하는 딕셔너리를 처리할 때 유용하다.
        Args:
            data (dict): to_pointer 메서드로 직렬화된 딕셔너리
        Returns:
            tuple[dict, type[DatasetWrapper]]: 임포트 데이터가 제거된 딕셔너리와 클래스 타입
        """
        import sys
        from importlib import import_module
        from functools import reduce

        module, qual = data["module"], data["qualname"]
        if module in ("__main__", "__mp_main__"):
            m = sys.modules[module]
        else:
            m = import_module(module)
        cls = reduce(getattr, qual.split("."), m)
        if not issubclass(cls, DatasetWrapper):
            raise TypeError(f"{cls} is not a subclass of DatasetWrapper")
        elif data["type"] != cls.__name__:
            raise TypeError(
                f"Type mismatch: expected {cls.__name__}, got {data['type']}"
            )

        d = data.copy()
        del d["module"]
        del d["qualname"]
        del d["type"]
        return d, cls

    def to_dict(self: DatasetWrapper) -> dict:
        """
        데이터셋을 딕셔너리로 변환한다. 이 메서드는 데이터셋의 상태를 직렬화하거나 저장할 때 유용하다.
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["class"] = self.dataset.__class__
        args["dataset"] = self.dataset.to_dict()
        args["method"] = "from_dict"

        return args

    @classmethod
    def from_dict(cls: type[DatasetWrapper], data: dict) -> DatasetWrapper:
        """
        to_dict 메서드로 직렬화된 딕셔너리에서 Dataset 객체를 생성하는 클래스 메서드이다. 이 메서드는 데이터셋을 저장하거나 전송한 후 다시 로드할 때 유용하다.
        Args:
            data (dict): to_dict 메서드로 직렬화된 딕셔너리
        Returns:
            DatasetWrapper: 딕셔너리에서 생성된 데이터셋 래퍼 객체
        """
        method = data.pop("method")
        if method == "from_dict":
            _class = data.pop("class")
            data["dataset"] = _class.from_dict(data["dataset"])
        elif method == "from_pointer":
            data["dataset"] = Dataset.from_pointer(data["dataset"])
        else:
            raise ValueError(f"Invalid method for deserialization: {method}")

        return cls(**data)

    def to_pointer(self: DatasetWrapper) -> dict:
        """
        데이터셋을 포인터 딕셔너리로 변환한다. 이 메서드는 데이터셋의 위치나 참조를 나타내는 정보를 포함하는 딕셔너리를 반환한다.
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return {
            **self.extract_import_data(),
            "dataset": self.dataset.to_pointer(),
            "type": self.__class__.__name__,
            "method": "from_pointer",
        }

    @classmethod
    def from_pointer(cls: type[DatasetWrapper], data: dict) -> DatasetWrapper:
        """
        to_pointer 메서드로 직렬화된 딕셔너리에서 DatasetWrapper 객체를 생성하는 클래스 메서드이다. 이 메서드는 데이터셋의 위치나 참조 정보를 저장하거나 전송한 후 다시 로드할 때 유용하다.
        Args:
            data (dict): to_pointer 메서드로 직렬화된 딕셔너리
        Returns:
            DatasetWrapper: 딕셔너리에서 생성된 데이터셋 래퍼 객체
        """
        if all(k in data for k in ("module", "qualname", "type")):
            data, wrapper_cls = cls.import_from_pointer(data)
        elif all(k not in data for k in ("module", "qualname", "type")):
            wrapper_cls = cls
        else:
            raise ValueError("Invalid pointer data: missing module, qualname, or type")

        if not issubclass(wrapper_cls, DatasetWrapper):
            raise TypeError(f"{wrapper_cls} is not a subclass of DatasetWrapper")
        elif wrapper_cls == DatasetWrapper:
            raise TypeError("Cannot instantiate DatasetWrapper directly")
        return wrapper_cls.from_dict(data)


__all__ = ["DatasetWrapper"]
