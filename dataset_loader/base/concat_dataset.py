from __future__ import annotations

import numpy as np

from typing import Any, TypeVar
from typing_extensions import override, Self
from collections.abc import Sequence, Iterable, Mapping, MutableSequence

from dataset_loader.protocol import DatasetProtocol, ConcatDatasetProtocol

from dataset_loader.base.dataset import Dataset
from dataset_loader.base.sample import Sample

D = TypeVar("D", bound=Dataset[Any])


class ConcatDataset(Dataset[MutableSequence[D]], ConcatDatasetProtocol[Any, Any]):
    """
    여러 Dataset을 하나로 합치는 기능을 제공하는 클래스이다. 이 클래스는 Dataset을 상속하여 구현되며, 내부적으로 여러 Dataset을 리스트로 관리한다. \n
    ConcatDataset은 각 Dataset의 샘플을 순차적으로 연결하여 하나의 큰 Dataset처럼 동작한다.

    Attributes:
        datasets (list[Dataset]): 합쳐진 Dataset들의 리스트
    Raises:
        ValueError: datasets가 비어있을 경우 발생한다.
    """

    def __init__(self, *, datasets: Sequence[D]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")

        dts = []
        for dataset in datasets:
            if dataset.is_cleaned:
                raise ValueError("Cannot concatenate a cleaned dataset")
            dts.append(dataset)

        super().__init__()
        self._datasets: list[D] = list(dts)

    @property
    @override
    def dataset(self) -> list[D]:
        return self._datasets.copy()

    @property
    @override
    def args(self) -> dict[str, Any]:
        return {**super().args, "datasets": self._datasets}

    @property
    @override
    def length(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    @property
    @override
    def name(self) -> str:
        return "-".join(self.names)

    @property
    def names(self) -> list[str]:
        """
        ConcatDataset에 포함된 Dataset들의 이름을 반환하는 속성입니다.

        Returns:
            Sequence[str]: ConcatDataset에 포함된 Dataset들의 이름을 나타내는 문자열 리스트입니다. 각 이름은 ConcatDataset에 포함된 Dataset의 name 속성에서 가져와야 합니다.
        """
        ...
        return [ds.name for ds in self._datasets]

    @override
    def select(self, indices: Iterable[int]) -> Self:
        if self.is_cleaned:
            raise RuntimeError("Cannot select from a cleaned dataset")

        length = len(self)
        normalized_indices = [i if i >= 0 else length + i for i in indices]

        selected_datasets = []
        start = 0
        for ds in self._datasets:
            end = start + len(ds)
            if ds_indices := [
                i - start for i in normalized_indices if start <= i < end
            ]:
                selected_datasets.append(ds.select(ds_indices))
            start = end
        args = self.args
        args["datasets"] = selected_datasets
        return type(self)(**args)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        if self.is_cleaned:
            raise RuntimeError("Cannot slice a cleaned dataset")

        start = start if start is not None else 0
        stop = stop if stop is not None else len(self)
        step = step if step is not None else 1

        if start < 0:
            raise IndexError("Negative start index is not supported")
        if stop < start:
            raise ValueError("Stop index must be greater than or equal to start index")
        if step <= 0:
            raise ValueError("Step must be a positive integer")

        return self.select(list(range(start, stop, step)))

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if rng is not None:
            raise NotImplementedError("Random sampling is not implemented")
        return self.slice(start=start, stop=start + size)

    @override
    def concat(self, other: DatasetProtocol[Any, Any]) -> ConcatDataset[Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot concatenate a cleaned dataset")
        elif isinstance(other, ConcatDataset):
            return ConcatDataset(datasets=self._datasets + other._datasets)
        elif isinstance(other, Dataset):
            return ConcatDataset(datasets=self._datasets + [other])
        else:
            raise TypeError("Invalid type for concatenation")

    @override
    def clean(self) -> None:
        if self.is_cleaned:
            return
        for ds in self._datasets:
            ds.clean()
        self._datasets.clear()
        super().clean()

    @override
    def get(self, idx: int) -> Sample:
        start = 0
        for ds in self._datasets:
            d_idx = idx - start
            if d_idx < len(ds):
                return ds.get(d_idx)
            start += len(ds)
        raise IndexError("Index out of range")

    @override
    def to_dict(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["datasets"] = [ds.to_dict() for ds in self._datasets]
        args["classes"] = [ds.__class__ for ds in self._datasets]
        args["method"] = "from_dict"
        return args

    @classmethod
    @override
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = {**data}
        method = data.pop("method")
        if method == "from_dict":
            classes = data.pop("classes")
            data["datasets"] = [
                _class.from_dict(ds)
                for _class, ds in zip(classes, data.pop("datasets"))
            ]
        elif method == "from_pointer":
            data["datasets"] = [Dataset.__setstate__(ds) for ds in data.pop("datasets")]
        else:
            raise ValueError(f"Invalid method for deserialization: {method}")

        return cls(**data)

    @override
    def __getstate__(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["datasets"] = [ds.__getstate__() for ds in self._datasets]
        args["method"] = "from_pointer"

        return args

    @classmethod
    @override
    def __setstate__(cls, state: Mapping[str, Any]) -> Self:
        dataset = super().__setstate__(state)
        if not isinstance(dataset, cls):
            raise TypeError(
                f"Invalid type for deserialization expected: {cls.__name__}, got: {type(dataset).__name__}"
            )
        return dataset


__all__ = ["ConcatDataset"]
