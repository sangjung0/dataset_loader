from __future__ import annotations

from typing import overload, Iterable, Any, Mapping
from typing_extensions import Self, override

from dataset_loader.protocol import DatasetProtocol, SampleProtocol

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.thread_loader_mixin import ThreadLoaderMixin

from dataset_loader.wrapper.image_recognition.ir_sample import IRSample


class IRDataset(
    DatasetWrapper[DatasetProtocol, IRSample],
    ThreadLoaderMixin[IRSample],
):
    @overload
    def getitem(self, key: int) -> IRSample: ...
    @overload
    def getitem(self, key: slice | Iterable[int]) -> Self: ...
    @override
    def getitem(self, key: int | slice | Iterable[int]) -> IRSample | Self:
        result = self.dataset[key]
        if isinstance(result, SampleProtocol):
            return IRSample(sample=result)
        return self.__class__(dataset=result)

    @override
    def concat(self, other: DatasetProtocol[DatasetProtocol, IRSample]) -> IRDataset:
        if isinstance(other, IRDataset):
            return IRDataset(dataset=self.dataset + other.dataset)  # type: ignore
        raise TypeError("Invalid type for concatenation")

    @override
    def get(self, idx: int) -> IRSample:
        sample = self.dataset.get(idx)
        return IRSample(sample=sample)

    def _loader(self, sample: IRSample) -> IRSample:
        return sample.loaded_ir_sample()

    @override
    @classmethod
    def __setstate__(cls, data: Mapping[str, Any]) -> IRDataset:
        dataset = super().__setstate__(data)
        if isinstance(dataset, IRDataset):
            return dataset
        raise TypeError(
            f"Invalid type for deserialization, expected IRDataset, got {type(dataset)} but got {type(dataset)}"
        )


__all__ = ["IRDataset"]
