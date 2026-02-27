from __future__ import annotations
from typing import TYPE_CHECKING

import warnings
import numpy as np

from typing import Sequence, TypeVar, Generic
from typing_extensions import override

from dataset_loader.interface.dataset import Dataset

if TYPE_CHECKING:
    from dataset_loader.interface.sample import Sample

T = TypeVar("T", bound="Dataset")


class ConcatDataset(Dataset, Generic[T]):
    """
    여러 Dataset을 하나로 합치는 기능을 제공하는 클래스이다. 이 클래스는 Dataset을 상속하여 구현되며, 내부적으로 여러 Dataset을 리스트로 관리한다. \n
    ConcatDataset은 각 Dataset의 샘플을 순차적으로 연결하여 하나의 큰 Dataset처럼 동작한다.

    Attributes:
        datasets (list[Dataset]): 합쳐진 Dataset들의 리스트
    Raises:
        ValueError: datasets가 비어있을 경우 발생한다.
    """

    def __init__(
        self: ConcatDataset,
        *,
        datasets: Sequence[T],
        use_cache: int = 0,
    ):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")

        for dataset in datasets:
            if dataset.is_cleaned:
                raise ValueError("Cannot concatenate a cleaned dataset")
            if dataset.use_cache > 0 and use_cache > 0:
                warnings.warn(
                    f"Dataset {dataset} has caching enabled (use_cache={dataset.use_cache}), but the concatenated dataset also has caching enabled (use_cache={use_cache}). This may lead to increased memory usage."
                )

        super().__init__(use_cache=use_cache)
        self._datasets: list[Dataset] = list(datasets)

    @property
    def datasets(self: ConcatDataset) -> list[T]:
        return self._datasets.copy()

    @Dataset.args.getter
    @override
    def args(self: ConcatDataset) -> dict:
        return {**super().args, "datasets": self._datasets}

    @Dataset.length.getter
    @override
    def length(self: ConcatDataset) -> int:
        return sum(len(ds) for ds in self._datasets)

    @Dataset.name.getter
    @override
    def name(self: ConcatDataset) -> list[str]:
        return [ds.name for ds in self._datasets]

    @override
    def select(
        self: ConcatDataset, indices: Sequence[int], *, use_cache: int = 0
    ) -> ConcatDataset:
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
                selected_datasets.append(ds.select(ds_indices, use_cache=use_cache))
            start = end
        args = self.args
        args["datasets"] = selected_datasets
        args["use_cache"] = use_cache
        return type(self)(**args)

    @override
    def slice(
        self: ConcatDataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        *,
        use_cache: int = 0,
    ) -> ConcatDataset:
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

        return self.select(list(range(start, stop, step)), use_cache=use_cache)

    @override
    def _sample(
        self: ConcatDataset,
        size: int,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
        use_cache: int = 0,
    ):
        if rng is not None:
            raise NotImplementedError("Random sampling is not implemented")
        return self.slice(start=start, stop=start + size, use_cache=use_cache)

    @override
    def concat(
        self: ConcatDataset, other: Dataset | ConcatDataset, *, use_cache: int = 0
    ) -> ConcatDataset:
        if self.is_cleaned:
            raise RuntimeError("Cannot concatenate a cleaned dataset")
        elif isinstance(other, ConcatDataset):
            return ConcatDataset(
                datasets=self._datasets + other._datasets, use_cache=use_cache
            )
        elif isinstance(other, Dataset):
            return ConcatDataset(datasets=self._datasets + [other], use_cache=use_cache)
        else:
            raise TypeError("Invalid type for concatenation")

    @override
    def clean(self: ConcatDataset) -> None:
        if self.is_cleaned:
            return
        for ds in self._datasets:
            ds.clean()
        self._datasets = []
        super().clean()

    @override
    def _get(self: ConcatDataset, idx: int) -> Sample:
        start = 0
        for ds in self._datasets:
            d_idx = idx - start
            if d_idx < len(ds):
                return ds.get(d_idx)
            start += len(ds)
        raise IndexError("Index out of range")

    @override
    def to_dict(self: ConcatDataset) -> dict:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["datasets"] = [ds.to_dict() for ds in self._datasets]
        args["classes"] = [ds.__class__ for ds in self._datasets]
        args["method"] = "from_dict"
        return args

    @classmethod
    @override
    def from_dict(cls: type[ConcatDataset], data: dict) -> ConcatDataset:
        method = data.pop("method")
        if method == "from_dict":
            classes = data.pop("classes")
            data["datasets"] = [
                _class.from_dict(ds)
                for _class, ds in zip(classes, data.pop("datasets"))
            ]
        elif method == "from_pointer":
            data["datasets"] = [Dataset.from_pointer(ds) for ds in data.pop("datasets")]
        else:
            raise ValueError(f"Invalid method for deserialization: {method}")

        return cls(**data)

    @override
    def to_pointer(self: ConcatDataset) -> dict:
        if self.is_cleaned:
            raise RuntimeError("Cannot serialize a cleaned dataset")
        args = self.args
        args["datasets"] = [ds.to_pointer() for ds in self._datasets]
        args["method"] = "from_pointer"

        return args


__all__ = ["ConcatDataset"]
