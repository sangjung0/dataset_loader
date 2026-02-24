from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing import Sequence, TypeVar, Generic
from typing_extensions import override

from dataset_loader.interface.dataset import Dataset

if TYPE_CHECKING:
    from dataset_loader.interface.sample import Sample

T = TypeVar("T", bound="Dataset")


class ConcatDataset(Dataset, Generic[T]):
    def __init__(self: ConcatDataset, *, datasets: list[T], **kwargs):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")

        task = (
            tuple(sorted(task := kwargs["task"]))
            if "task" in kwargs
            else tuple(sorted(datasets[0].task))
        )
        if any(tuple(sorted(ds.task)) != task for ds in datasets):
            raise ValueError("All datasets must have the same task")

        kwargs["task"] = task
        super().__init__(**kwargs)
        self._datasets = datasets

    @Dataset.args.getter
    @override
    def args(self: ConcatDataset) -> dict:
        return {**super().args, "datasets": self._datasets}

    @Dataset.length.getter
    @override
    def length(self: ConcatDataset) -> int:
        return sum(len(ds) for ds in self._datasets)

    @override
    def concat(self: ConcatDataset, other: Dataset | ConcatDataset) -> ConcatDataset:
        if isinstance(other, ConcatDataset):
            return ConcatDataset(datasets=self._datasets + other._datasets)
        elif isinstance(other, Dataset):
            return ConcatDataset(datasets=self._datasets + [other])
        else:
            raise TypeError("Invalid type for concatenation")

    @override
    def to_dict(self: ConcatDataset) -> dict:
        return {
            **super().to_dict(),
            "datasets": [ds.to_dict() for ds in self._datasets],
            "module": [ds.__class__.__module__ for ds in self._datasets],
            "qualname": [ds.__class__.__qualname__ for ds in self._datasets],
        }

    @override
    def select(self: ConcatDataset, indices: Sequence[int]) -> ConcatDataset:
        selected_datasets = []
        start = 0
        for ds in self._datasets:
            end = start + len(ds)
            if ds_indices := [i - start for i in indices if start <= i < end]:
                selected_datasets.append(ds.select(ds_indices))
            start = end
        return type(self)(datasets=selected_datasets)

    @override
    def slice(
        self: ConcatDataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> ConcatDataset:
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
    def get(self: ConcatDataset, idx: int) -> Sample:
        start = 0
        for ds in self._datasets:
            d_idx = idx - start
            if d_idx < len(ds):
                return ds.get(d_idx)
            start += len(ds)
        raise IndexError("Index out of range")

    @override
    def _sample(
        self: ConcatDataset,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ):
        if rng is not None:
            raise NotImplementedError("Random sampling is not implemented")
        return self.slice(start=start, stop=start + size)

    @classmethod
    @override
    def from_dict(cls: type[ConcatDataset], data: dict) -> ConcatDataset:
        import sys
        from importlib import import_module
        from functools import reduce

        datasets = []
        for ds, module, qual in zip(data["datasets"], data["module"], data["qualname"]):
            if module in ("__main__", "__mp_main__"):
                m = sys.modules[module]
            else:
                m = import_module(module)

            t = reduce(getattr, qual.split("."), m)
            if not issubclass(t, Dataset):
                raise TypeError(f"{t} is not a subclass of Dataset")
            datasets.append(t.from_dict(ds))

        copy_data = data.copy()
        del copy_data["datasets"]
        del copy_data["module"]
        del copy_data["qualname"]

        return cls(datasets=datasets, **copy_data)


__all__ = ["ConcatDataset"]
