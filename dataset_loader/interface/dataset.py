from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, overload, Sequence
from cachetools import LRUCache

from dataset_loader.interface.sample import Sample

if TYPE_CHECKING:
    from dataset_loader.interface.concat_dataset import ConcatDataset


class Dataset(ABC):
    """
    다양한 Dataset을 위한 공통 추상 클래스이다. \n
    다른 라이브러리의 Dataset을 래핑하거나, 자체적으로 Dataset을 구현할 때 이 클래스를 상속하여 구현한다.

    Attributes:
        use_cache (int): 데이터셋이 캐시를 사용하는 정도를 나타내는 정수이다. 0은 캐시를 사용하지 않음을 의미하고, 1 이상은 캐시를 사용함을 의미한다. 값의 크기는 캐싱하는 Sample의 개수이다.
    """

    def __init__(self: Dataset, *, use_cache: int = 0):
        if not isinstance(use_cache, int) or use_cache < 0:
            raise ValueError("use_cache must be a non-negative integer")

        self._is_cleaned = False
        self._use_cache = use_cache
        self._cache: LRUCache[int, Sample] = LRUCache(maxsize=use_cache)

    @property
    def args(self: Dataset) -> dict:
        return {"use_cache": self._use_cache}

    @property
    def is_cleaned(self: Dataset) -> bool:
        return self._is_cleaned

    @property
    def use_cache(self: Dataset) -> int:
        return self._use_cache

    @property
    @abstractmethod
    def length(self: Dataset) -> int:
        raise NotImplementedError

    def __len__(self: Dataset) -> int:
        return self.length

    @property
    def name(self: Dataset) -> str:
        return self.__class__.__name__

    def __iter__(self: Dataset) -> Generator[Sample, Any, None]:
        yield from self.iter()

    def iter(self: Dataset) -> Generator[Sample, Any, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def __getitem__(self: Dataset, key: int) -> Sample: ...
    @overload
    def __getitem__(self: Dataset, key: slice | Sequence[int]) -> Dataset: ...
    def __getitem__(
        self: Dataset, key: int | slice | Sequence[int]
    ) -> Sample | Dataset:
        return self.getitem(key)

    @overload
    def getitem(self: Dataset, key: int, *, use_cache: int = 0) -> Sample: ...
    @overload
    def getitem(
        self: Dataset, key: slice | Sequence[int], *, use_cache: int = 0
    ) -> Dataset: ...
    def getitem(
        self: Dataset, key: int | slice | Sequence[int], *, use_cache: int = 0
    ) -> Sample | Dataset:
        """
        key == int -> Sample \n
        key == slice -> Dataset (using slice method) \n
        key == Sequence[int] -> Dataset (using select method) \n

        Args:
            key (int | slice | Sequence[int]): 인덱스 또는 슬라이스 또는 인덱스 시퀀스
            use_cache (int): 선택된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            result (Sample | Dataset): 요청된 샘플 또는 데이터셋

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            TypeError: key의 타입이 int, slice, Sequence[int]이 아닌 경우
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot access a cleaned dataset")
        elif isinstance(key, slice):
            return self.slice(
                start=key.start, stop=key.stop, step=key.step, use_cache=use_cache
            )
        elif isinstance(key, Sequence):
            return self.select(key, use_cache=use_cache)
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
    def select(self: Dataset, indices: Sequence[int], *, use_cache: int = 0) -> Dataset:
        """
        주어진 인덱스 시퀀스에 해당하는 샘플들로 구성된 새로운 Dataset을 반환한다.
        Args:
            indices (Sequence[int]): 선택할 샘플의 인덱스 시퀀스
            use_cache (int): 선택된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            Dataset: 선택된 샘플들로 구성된 데이터셋
        """
        raise NotImplementedError

    @abstractmethod
    def slice(
        self: Dataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        *,
        use_cache: int = 0,
    ) -> Dataset:
        """
        주어진 슬라이스에 해당하는 샘플들로 구성된 새로운 Dataset을 반환한다.
        Args:
            start (int | None): 슬라이스의 시작 인덱스. 기본값은 None이며, 이 경우 0부터 시작한다.
            stop (int | None): 슬라이스의 끝 인덱스. 기본값은 None이며, 이 경우 데이터셋의 끝까지 포함한다.
            step (int | None): 슬라이스의 단계. 기본값은 None이며, 이 경우 1씩 증가한다.
            use_cache (int): 선택된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            Dataset: 슬라이스에 해당하는 샘플들로 구성된 데이터셋
        Raises:
            IndexError: start 또는 stop이 유효한 인덱스 범위를 벗어난 경우
            ValueError: stop이 start보다 작은 경우
        """
        raise NotImplementedError

    def sample(
        self: Dataset,
        size: int = -1,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
        use_cache: int = 0,
    ) -> Dataset:
        """
        데이터셋에서 무작위로 샘플링하여 새로운 Dataset을 반환한다.
        Args:
            size (int): 샘플링할 샘플의 수. 음수이면 start부터 끝까지 샘플링한다.
            start (int): 샘플링을 시작할 인덱스. 기본값은 0이다.
            rng (np.random.Generator | np.random.RandomState | None): 무작위 수 생성기. 기본값은 None이며, 이 경우 단순 슬라이싱이 수행된다.
            use_cache (int): 샘플링된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            Dataset: 샘플링된 데이터셋
        Raises:
            IndexError: start가 유효한 인덱스 범위를 벗어난 경우
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot sample from a cleaned dataset")
        elif start < 0 or start >= len(self):
            raise IndexError("Invalid start index")
        elif size < 0:
            size = len(self) - start
        else:
            size = min(size, len(self) - start)
        return self._sample(size=size, start=start, rng=rng, use_cache=use_cache)

    @abstractmethod
    def _sample(
        self: Dataset,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
        use_cache: int = 0,
    ) -> Dataset:
        raise NotImplementedError

    def __add__(self: Dataset, other: Dataset | ConcatDataset) -> ConcatDataset:
        return self.concat(other)

    def concat(
        self: Dataset, other: Dataset | ConcatDataset, *, use_cache: int = 0
    ) -> ConcatDataset:
        """
        self와 other을 연결하여 새로운 ConcatDataset을 반환한다.

        Args:
            other (Dataset | ConcatDataset): 연결할 다른 데이터셋
            use_cache (int): 연결된 데이터셋에서 사용할 캐시 크기. 기본값은 0이며, 이 경우 캐시를 사용하지 않는다.
        Returns:
            ConcatDataset: 연결된 데이터셋
        Raises:
            TypeError: other의 타입이 Dataset 또는 ConcatDataset이 아닌 경우
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot concatenate a cleaned dataset")

        from dataset_loader.interface.concat_dataset import ConcatDataset

        if isinstance(other, ConcatDataset):
            return ConcatDataset(datasets=[self] + other._datasets, use_cache=use_cache)
        elif isinstance(other, Dataset):
            return ConcatDataset(datasets=[self, other], use_cache=use_cache)
        else:
            raise TypeError("Invalid type for concatenation")

    @abstractmethod
    def clean(self: Dataset) -> None:
        """
        데이터셋이 사용한 리소스를 정리하거나 해제하는 메서드이다. 이 메서드는 데이터셋이 더 이상 필요하지 않을 때 호출하여 리소스를 효율적으로 관리할 수 있도록 한다.
        """
        if self.is_cleaned:
            return
        self._is_cleaned = True
        self._cache.clear()

    def extract_import_data(self: Dataset) -> dict:
        """
        데이터셋에서 임포트 데이터를 추출하는 메서드이다. 이 메서드는 데이터셋이 외부 리소스나 참조를 포함하는 경우, 해당 정보를 추출하여 반환한다.
        Returns:
            dict: 임포트 데이터가 포함된 딕셔너리
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot extract import data from a cleaned dataset")
        return {
            "module": self.__class__.__module__,
            "qualname": self.__class__.__qualname__,
        }

    @staticmethod
    def import_from_pointer(data: dict) -> tuple[dict, type[Dataset]]:
        """
        to_pointer 메서드로 직렬화된 딕셔너리를 가져와서 필요한 모듈을 동적으로 임포트하는 정적 메서드이다. 이 메서드는 데이터셋의 위치나 참조 정보를 포함하는 딕셔너리를 처리할 때 유용하다.
        Args:
            data (dict): to_pointer 메서드로 직렬화된 딕셔너리
        Returns:
            tuple[dict, type[Dataset]]: 임포트 데이터가 제거된 딕셔너리와 클래스 타입
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
        if not (issubclass(cls, Dataset)):
            raise TypeError(f"{cls} is not a subclass of Dataset or DatasetWrapper")
        elif data["type"] != cls.__name__:
            raise TypeError(
                f"Type mismatch: expected {cls.__name__}, got {data['type']}"
            )

        d = data.copy()
        del d["module"]
        del d["qualname"]
        del d["type"]
        return d, cls

    def get(self: Dataset, idx: int) -> Sample:
        """
        주어진 인덱스에 해당하는 샘플을 반환한다.
        Args:
            idx (int): 가져올 샘플의 인덱스
        Returns:
            Sample: 요청된 샘플
        Raises:
            IndexError: idx가 유효한 인덱스 범위를 벗어난 경우
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot access a cleaned dataset")
        elif idx in self._cache:
            return self._cache[idx]
        else:
            sample = self._get(idx)
            if self._use_cache > 0:
                self._cache[idx] = sample
            return sample

    @abstractmethod
    def _get(self: Dataset, idx: int) -> Sample:
        raise NotImplementedError

    def to_dict(self: Dataset) -> dict:
        """
        데이터셋을 딕셔너리로 변환한다. 이 메서드는 데이터셋의 상태를 직렬화하거나 저장할 때 유용하다.
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return self.args

    @classmethod
    @abstractmethod
    def from_dict(cls: type[Dataset], data: dict) -> Dataset:
        """
        to_dict 메서드로 직렬화된 딕셔너리에서 Dataset 객체를 생성하는 클래스 메서드이다. 이 메서드는 데이터셋을 저장하거나 전송한 후 다시 로드할 때 유용하다.
        Args:
            data (dict): to_dict 메서드로 직렬화된 딕셔너리
        Returns:
            Dataset: 딕셔너리에서 생성된 데이터셋 객체
        """
        raise NotImplementedError

    def to_pointer(self: Dataset) -> dict:
        """
        데이터셋을 포인터 딕셔너리로 변환한다. 이 메서드는 데이터셋의 위치나 참조를 나타내는 정보를 포함하는 딕셔너리를 반환한다.
        """
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        return {
            **self.to_dict(),
            **self.extract_import_data(),
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_pointer(cls: type[Dataset], data: dict) -> Dataset:
        """
        to_pointer 메서드로 직렬화된 딕셔너리에서 Dataset 객체를 생성하는 클래스 메서드이다. 이 메서드는 데이터셋의 위치나 참조 정보를 저장하거나 전송한 후 다시 로드할 때 유용하다.
        Args:
            data (dict): to_pointer 메서드로 직렬화된 딕셔너리
        Returns:
            Dataset: 딕셔너리에서 생성된 데이터셋 객체
        """
        if all(k in data for k in ("module", "qualname", "type")):
            data, dataset_cls = cls.import_from_pointer(data)
        elif all(k not in data for k in ("module", "qualname", "type")):
            dataset_cls = cls
        else:
            raise ValueError("Invalid pointer data: missing module, qualname, or type")

        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"{dataset_cls} is not a subclass of Dataset")
        elif dataset_cls == Dataset:
            raise TypeError("Cannot instantiate Dataset directly")
        return dataset_cls.from_dict(data)


__all__ = ["Dataset"]
