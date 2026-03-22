from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping

from typing import Sequence, overload
from typing_extensions import Self, override

from dataset_loader.base import Sample
from dataset_loader.protocol import DatasetProtocol, ConcatDatasetProtocol

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.thread_loader_mixin import ThreadLoaderMixin

from dataset_loader.wrapper.asr.asr_sample import ASRSample
from dataset_loader.wrapper.asr.protocol import (
    ASRDatasetProtocol,
)

if TYPE_CHECKING:
    from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset


class ASRDataset(
    DatasetWrapper[ASRDatasetProtocol, ASRSample], ThreadLoaderMixin[ASRSample]
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
    def getitem(self, key: slice | Sequence[int]) -> Self: ...
    @override
    def getitem(self, key: int | slice | Sequence[int]) -> ASRSample | Self:
        result = self.dataset[key]
        if isinstance(result, Sample):
            return ASRSample(sample=result)
        assert isinstance(result, ASRDatasetProtocol)
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

    @override
    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> ASRDataset:
        return super().from_config(data)


__all__ = ["ASRDataset"]
