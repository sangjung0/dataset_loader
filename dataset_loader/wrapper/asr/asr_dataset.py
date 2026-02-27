from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Sequence
from typing_extensions import Self, override

from dataset_loader.interface import Sample

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.thread_loader_mixin import ThreadLoaderMixin
from dataset_loader.wrapper.asr.asr_sample import ASRSample
from dataset_loader.wrapper.asr.asr_dataset_ptc import ASRDatasetPtc

if TYPE_CHECKING:
    from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset


class ASRDataset(
    DatasetWrapper[ASRSample, "ASRConcatDataset"], ThreadLoaderMixin[ASRSample]
):
    def __init__(self, dataset: ASRDatasetPtc):
        self._dataset = dataset

    @property
    @override
    def dataset(self) -> ASRDatasetPtc:
        return self._dataset

    @property
    def sr(self) -> int:
        return self.dataset.sr

    @sr.setter
    def sr(self, value: int):
        self.dataset.sr = value

    def getitem(self, key: int | slice | Sequence[int]) -> ASRSample | Self:
        result = self.dataset.getitem(key)
        if isinstance(result, Sample):
            return ASRSample(sample=result)
        return self.__class__(dataset=result)

    def concat(self, other: Self | "ASRConcatDataset") -> "ASRConcatDataset":
        from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset

        if isinstance(other, ASRConcatDataset) or isinstance(other, ASRDataset):
            return ASRConcatDataset(self.dataset + other.dataset)
        else:
            raise TypeError("Invalid type for concatenation")

    def get(self, idx: int) -> ASRSample:
        sample = self.dataset.get(idx)
        return ASRSample(sample=sample)

    def _loader(self, sample: ASRSample) -> ASRSample:
        sample.audio
        return sample


__all__ = ["ASRDataset", "ASRDatasetPtc"]
