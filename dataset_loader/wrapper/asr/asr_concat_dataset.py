from __future__ import annotations

from typing import cast, overload
from typing_extensions import override, Self
from collections.abc import Sequence

from dataset_loader.base import Sample
from dataset_loader.protocol import ConcatDatasetProtocol, DatasetProtocol

from dataset_loader.wrapper.concat_dataset_wrapper import ConcatDatasetWrapper
from dataset_loader.wrapper.asr.protocol import (
    ASRDatasetProtocol,
    ASRConcatDatasetProtocol,
)

from dataset_loader.wrapper.asr.asr_sample import ASRSample
from dataset_loader.wrapper.asr.asr_dataset import ASRDataset


class ASRConcatDataset(
    ConcatDatasetWrapper[ASRConcatDatasetProtocol, ASRSample],
    ASRConcatDatasetProtocol,
    ASRDatasetProtocol,
):
    def __init__(self, dataset: ConcatDatasetProtocol):
        datasets = dataset.datasets
        if any(not isinstance(ds, ASRDatasetProtocol) for ds in datasets):
            raise TypeError("All datasets must be instances of ASRDatasetProtocol")

        datasets = cast(Sequence[ASRDatasetProtocol], datasets)
        if not all(datasets[0].sr == ds.sr for ds in datasets):
            raise ValueError("All datasets must have the same sampling rate")

        dataset = cast(ASRConcatDatasetProtocol, dataset)
        super().__init__(dataset=dataset)

    @property
    def sr(self) -> int:
        return self.dataset.sr

    @sr.setter
    def sr(self, value: int):
        self.dataset.sr = value

    @overload
    def getitem(self, key: int) -> ASRSample: ...
    @overload
    def getitem(self, key: slice | Sequence[int]) -> Self: ...
    @override
    def getitem(self, key: int | slice | Sequence[int]) -> ASRSample | Self:
        result = self.dataset[key]
        if isinstance(result, Sample):
            return ASRSample(sample=result)
        assert isinstance(result, ASRConcatDatasetProtocol)
        return self.__class__(dataset=result)

    @override
    def concat(
        self, other: DatasetProtocol | ConcatDatasetProtocol
    ) -> "ASRConcatDataset":
        from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset

        if isinstance(other, ASRConcatDataset) or isinstance(other, ASRDataset):
            return ASRConcatDataset(self.dataset + other.dataset)
        else:
            raise TypeError("Invalid type for concatenation")

    @override
    def get(self, idx: int) -> ASRSample:
        sample = self.dataset.get(idx)
        return ASRSample(sample=sample)

    def _loader(self, sample: ASRSample) -> ASRSample:
        _ = sample.audio  # Force loading audio
        return sample


__all__ = ["ASRConcatDataset"]
