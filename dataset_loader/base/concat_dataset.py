from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing import TypeVar, Generic, Any
from typing_extensions import override, Self
from collections.abc import Sequence, Iterable, Mapping

from dataset_loader.protocol import ConcatDatasetProtocol, DatasetProtocol

from dataset_loader.base.dataset import Dataset

if TYPE_CHECKING:
    from dataset_loader.base.sample import Sample

T = TypeVar("T", bound=Dataset)


class ConcatDataset(Dataset, ConcatDatasetProtocol, Generic[T]):
    """
    여러 Dataset을 하나로 합치는 기능을 제공하는 클래스이다. 이 클래스는 Dataset을 상속하여 구현되며, 내부적으로 여러 Dataset을 리스트로 관리한다. \n
    ConcatDataset은 각 Dataset의 샘플을 순차적으로 연결하여 하나의 큰 Dataset처럼 동작한다.

    Attributes:
        datasets (list[Dataset]): 합쳐진 Dataset들의 리스트
    Raises:
        ValueError: datasets가 비어있을 경우 발생한다.
    """

    def __init__(self, *, datasets: Sequence[T]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")

        dts = []
        for dataset in datasets:
            if dataset.is_cleaned:
                raise ValueError("Cannot concatenate a cleaned dataset")
            dts.append(dataset)

        super().__init__()
        self._datasets: list[T] = list(dts)

    @property
    @override
    def datasets(self) -> list[T]:
        return self._datasets.copy()

    @Dataset.args.getter
    @override
    def args(self) -> dict[str, Any]:
        return {**super().args, "datasets": self._datasets}

    @Dataset.length.getter
    @override
    def length(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    @Dataset.name.getter
    @override
    def name(self) -> str:
        return "-".join(self.names)

    @property
    @override
    def names(self) -> list[str]:
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
    def concat(self, other: ConcatDatasetProtocol | DatasetProtocol) -> ConcatDataset:
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
        self._datasets = []
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
            data["datasets"] = [Dataset.from_config(ds) for ds in data.pop("datasets")]
        else:
            raise ValueError(f"Invalid method for deserialization: {method}")

        return cls(**data)

    @override
    def to_config(self: ConcatDataset) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["datasets"] = [ds.to_pointer() for ds in self._datasets]
        args["method"] = "from_pointer"

        return args


__all__ = ["ConcatDataset"]
