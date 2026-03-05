from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, overload, TypeVar, Generic
from typing_extensions import Self
from collections.abc import Mapping, Sequence

from dataset_loader.protocol import (
    DatasetProtocol,
    SampleProtocol,
    ConcatDatasetProtocol,
)
from dataset_loader.base import Dataset


Dts = TypeVar("Dts", bound=DatasetProtocol)
Spl = TypeVar("Spl", bound=SampleProtocol)


class DatasetWrapper(DatasetProtocol, Generic[Dts, Spl], ABC):
    """
    Dataset의 래핑 클래스이다. \n
    다양한 도메인에 대해서 대응하기 위한 인터페이스를 제공한다.

    Attributes:
        dataset (Dataset): 래핑할 Dataset 객체
        args (dict): DatasetWrapper를 생성하는 데 필요한 인자들
        length (int): Dataset의 길이
        name (str): Dataset의 이름
    """

    def __init__(self, dataset: Dts):
        self._dataset = dataset

    @property
    def dataset(self) -> Dts:
        return self._dataset

    @property
    def is_cleaned(self) -> bool:
        return self.dataset.is_cleaned

    @property
    def args(self) -> dict[str, Any]:
        return {"dataset": self.dataset}

    @property
    def length(self) -> int:
        return len(self.dataset)

    def __len__(self) -> int:
        return self.length

    @property
    def name(self) -> str:
        return self.dataset.name

    def __iter__(self) -> Generator[Spl, None, None]:
        yield from self.iter()

    def iter(self) -> Generator[Spl, None, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def __getitem__(self, key: int) -> Spl: ...
    @overload
    def __getitem__(self, key: slice | Sequence[int]) -> Self: ...
    def __getitem__(self, key: int | slice | Sequence[int]) -> Spl | Self:
        return self.getitem(key)

    @overload
    def getitem(self, key: int) -> Spl: ...
    @overload
    def getitem(self, key: slice | Sequence[int]) -> Self: ...
    @abstractmethod
    def getitem(self, key: int | slice | Sequence[int]) -> Spl | Self:
        """
        key == int -> Spl \n
        key == slice -> Self (using slice method) \n
        key == Sequence[int] -> Self (using select method) \n

        Args:
            key (int | slice | Sequence[int]): 인덱스 또는 슬라이스 또는 인덱스 시퀀스
        Returns:
            result (Spl | Self): 요청된 샘플 또는 데이터셋

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            TypeError: key의 타입이 int, slice, Sequence[int]이 아닌 경우
        """
        raise NotImplementedError("getitem method must be implemented in subclass")

    def __add__(
        self, other: DatasetProtocol | ConcatDatasetProtocol
    ) -> ConcatDatasetProtocol:
        return self.concat(other)

    @abstractmethod
    def concat(
        self, other: DatasetProtocol | ConcatDatasetProtocol
    ) -> ConcatDatasetProtocol:
        """
        self와 other을 연결하여 새로운 ConcatDataset을 반환한다.

        Args:
            other (DatasetWrapper | type[ConcatDataset]): 연결할 다른 데이터셋
        Returns:
            type[ConcatDataset]: 연결된 데이터셋
        Raises:
            ValueError: self와 other의 task가 다른 경우
            TypeError: other의 타입이 DatasetWrapper 또는 ConcatDataset이 아닌 경우
        """
        raise NotImplementedError("concat method must be implemented in subclass")

    def sample(
        self,
        size: int = -1,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        """
        데이터셋에서 무작위로 샘플링하여 새로운 DatasetWrapper을 반환한다.
        Args:
            size (int): 샘플링할 샘플의 수. 음수이면 start부터 끝까지 샘플링한다.
            start (int): 샘플링을 시작할 인덱스. 기본값은 0이다.
            rng (np.random.Generator | np.random.RandomState | None): 무작위 수 생성기. 기본값은 None이며, 이 경우 단순 슬라이싱이 수행된다.
        Returns:
            Self: 샘플링된 데이터셋 래퍼
        Raises:
            IndexError: start가 유효한 인덱스 범위를 벗어난 경우
        """
        dataset = self.dataset.sample(size=size, start=start, rng=rng)
        return self.__class__(dataset=dataset)

    def select(self, indices: Sequence[int]) -> Self:
        """
        데이터셋에서 지정된 인덱스에 해당하는 샘플을 선택하여 새로운 DatasetWrapper을 반환한다.
        Args:
            indices (Sequence[int]): 선택할 샘플의 인덱스 시퀀스
        Returns:
            Self: 선택된 데이터셋 래퍼
        Raises:
            IndexError: indices 중 하나라도 유효한 인덱스 범위를 벗어난 경우
        """
        dataset = self.dataset.select(indices)
        return self.__class__(dataset=dataset)

    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        """
        데이터셋에서 지정된 범위에 해당하는 샘플을 슬라이스하여 새로운 DatasetWrapper을 반환한다.
        Args:
            start (int | None): 슬라이스의 시작 인덱스. 기본값은 None이며, 이 경우 0부터 시작한다.
            stop (int | None): 슬라이스의 끝 인덱스. 기본값은 None이며, 이 경우 데이터셋의 끝까지 슬라이스한다.
            step (int | None): 슬라이스의 간격. 기본값은 None이며, 이 경우 1 간격으로 슬라이스한다.
        Returns:
            Self: 슬라이스된 데이터셋 래퍼
        """
        dataset = self.dataset.slice(start=start, stop=stop, step=step)
        return self.__class__(dataset=dataset)

    @abstractmethod
    def get(self, idx: int) -> Spl:
        """
        데이터셋에서 지정된 인덱스에 해당하는 샘플을 반환한다.
        Args:
            idx (int): 샘플의 인덱스
        Returns:
            type[Spl]: 요청된 샘플
        Raises:
            IndexError: idx가 유효한 인덱스 범위를 벗어난 경우
        """
        raise NotImplementedError

    def clean(self) -> None:
        """
        데이터셋을 정리한다. 이 메서드는 데이터셋이 외부 리소스나 참조를 포함하는 경우, 해당 리소스나 참조를 해제하거나 정리하는 데 사용된다.
        데이터셋이 이미 정리된 경우, 이 메서드는 아무 작업도 수행하지 않는다.
        """
        self.dataset.clean()

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        to_dict 메서드로 직렬화된 딕셔너리에서 Dataset 객체를 생성하는 클래스 메서드이다. 이 메서드는 데이터셋을 저장하거나 전송한 후 다시 로드할 때 유용하다.
        Args:
            data (dict): to_dict 메서드로 직렬화된 딕셔너리
        Returns:
            Self: 딕셔너리에서 생성된 데이터셋 래퍼 객체
        """
        data = {**data}
        method = data.pop("method")
        if method == "from_dict":
            _class = data.pop("class")
            data["dataset"] = _class.from_dict(data["dataset"])
        elif method == "from_config":
            data["dataset"] = Dataset.from_config(data["dataset"])
        else:
            raise ValueError(f"Invalid method for deserialization: {method}")

        return cls(**data)

    def to_config(self) -> dict[str, Any]:
        """
        데이터셋을 설정 딕셔너리로 변환한다. 이 메서드는 데이터셋의 위치나 참조를 나타내는 정보를 포함하는 딕셔너리를 반환한다.
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return {
            **self.extract_import_data(),
            "dataset": self.dataset.to_config(),
            "type": self.__class__.__name__,
            "method": "from_config",
        }

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> DatasetWrapper:
        """
        to_config 메서드로 직렬화된 딕셔너리에서 DatasetWrapper 객체를 생성하는 클래스 메서드이다. 이 메서드는 데이터셋의 위치나 참조 정보를 저장하거나 전송한 후 다시 로드할 때 유용하다.
        Args:
            data (dict): to_config 메서드로 직렬화된 딕셔너리
        Returns:
            Self: 딕셔너리에서 생성된 데이터셋 래퍼 객체
        """
        if all(k in data for k in ("module", "qualname", "type")):
            data, wrapper_cls = cls.import_from_config(data)
        elif all(k not in data for k in ("module", "qualname", "type")):
            wrapper_cls = cls
        else:
            raise ValueError("Invalid pointer data: missing module, qualname, or type")

        if not issubclass(wrapper_cls, DatasetWrapper):
            raise TypeError(f"{wrapper_cls} is not a subclass of DatasetWrapper")
        elif wrapper_cls == DatasetWrapper:
            raise TypeError("Cannot instantiate DatasetWrapper directly")
        return wrapper_cls.from_dict(data)

    def extract_import_data(self) -> dict[str, Any]:
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

    @classmethod
    def import_from_config(
        cls, data: Mapping[str, Any]
    ) -> tuple[dict, type[DatasetWrapper]]:
        """
        to_config 메서드로 직렬화된 딕셔너리를 가져와서 필요한 모듈을 동적으로 임포트하는 정적 메서드이다. 이 메서드는 데이터셋의 위치나 참조 정보를 포함하는 딕셔너리를 처리할 때 유용하다.
        Args:
            data (dict): to_config 메서드로 직렬화된 딕셔너리
        Returns:
            tuple[dict, type[Self]]: 임포트 데이터가 제거된 딕셔너리와 클래스 타입
        """
        import sys
        from importlib import import_module
        from functools import reduce

        module, qual = data["module"], data["qualname"]
        if module in ("__main__", "__mp_main__"):
            m = sys.modules[module]
        else:
            m = import_module(module)
        _class = reduce(getattr, qual.split("."), m)
        if not isinstance(_class, type):
            raise TypeError(f"{_class} is not a class")
        if not issubclass(_class, DatasetWrapper):
            raise TypeError(f"{_class} is not a subclass of DatasetWrapper")
        elif data["type"] != _class.__name__:
            raise TypeError(
                f"Type mismatch: expected {_class.__name__}, got {data['type']}"
            )

        _class: type[DatasetWrapper]
        d = {k: v for k, v in data.items() if k not in ("module", "qualname", "type")}
        return d, _class


__all__ = ["DatasetWrapper"]
