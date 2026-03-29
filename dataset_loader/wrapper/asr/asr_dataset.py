from __future__ import annotations

from typing import overload, Iterable, Any, Mapping
from typing_extensions import Self, override

from dataset_loader.protocol import DatasetProtocol, SampleProtocol

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.thread_loader_mixin import ThreadLoaderMixin

from dataset_loader.wrapper.asr.asr_sample import ASRSample
from dataset_loader.wrapper.asr.protocol import ASRDatasetProtocol


class ASRDataset(
    DatasetWrapper[ASRDatasetProtocol, ASRSample],
    ThreadLoaderMixin[ASRSample],
):
    @property
    def sr(self) -> int:
        return self.dataset.sr

    @sr.setter
    def sr(self, value: int):
        self.dataset.sr = value

    @overload
    def getitem(self, key: int) -> ASRSample: ...
    @overload
    def getitem(self, key: slice | Iterable[int]) -> Self: ...
    @override
    def getitem(self, key: int | slice | Iterable[int]) -> ASRSample | Self:
        result = self.dataset[key]
        if isinstance(result, SampleProtocol):
            return ASRSample(sample=result)
        return self.__class__(dataset=result)

    @override
    def concat(
        self, other: DatasetProtocol[ASRDatasetProtocol, ASRSample]
    ) -> ASRDataset:
        if isinstance(other, ASRDataset):
            return ASRDataset(dataset=self.dataset + other.dataset)  # type: ignore
        raise TypeError("Invalid type for concatenation")

    @override
    def get(self, idx: int) -> ASRSample:
        sample = self.dataset.get(idx)
        return ASRSample(sample=sample)

    def _loader(self, sample: ASRSample) -> ASRSample:
        return sample.loaded_audio_sample()

    @override
    @classmethod
    def __setstate__(cls, data: Mapping[str, Any]) -> ASRDataset:
        dataset = super().__setstate__(data)
        if isinstance(dataset, ASRDataset):
            return dataset
        raise TypeError(
            f"Invalid type for deserialization, expected ASRDataset, got {type(dataset)} but got {type(dataset)}"
        )


__all__ = ["ASRDataset"]
