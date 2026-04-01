from __future__ import annotations

from typing import Any, overload
from typing_extensions import override, Self
from collections.abc import Iterable, MutableSequence

from dataset_loader.protocol import (
    DatasetProtocol,
    SampleProtocol,
    ConcatDatasetProtocol,
)

from dataset_loader.wrapper.asr.protocol import ASRDatasetProtocol
from dataset_loader.wrapper.asr.asr_dataset_mixin import ASRDatasetMixin
from dataset_loader.wrapper.asr.asr_sample import ASRSample


class ASRConcatDataset(
    ASRDatasetMixin[ConcatDatasetProtocol[ASRDatasetProtocol, ASRSample]]
):
    """
    ASRDataset을 연결하여 새로운 ASRDataset을 만드는 클래스이다.
    """

    def __init__(self, dataset: ConcatDatasetProtocol[ASRDatasetProtocol, ASRSample]):
        for ds in dataset.dataset:
            if not isinstance(ds, ASRDatasetProtocol):
                raise TypeError(
                    f"All datasets must be ASRDatasetProtocol, but got {type(ds)}"
                )

        super().__init__(dataset=dataset)

    @property
    def sr(self) -> int:
        return self.dataset._datasets[0].sr

    @sr.setter
    def sr(self, value: int) -> None:
        for dataset in self.dataset._datasets:
            dataset.sr = value

    @property
    def names(self) -> MutableSequence[str]:
        return self.dataset.names

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
    def concat(self, other: DatasetProtocol[Any, Any]) -> ASRConcatDataset:
        from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset
        from dataset_loader.wrapper.asr.asr_dataset import ASRDataset

        # ASRDataset 또는 ASRConcatDataset인 경우
        if isinstance(other, (ASRDataset, ASRConcatDataset)):
            return ASRConcatDataset(self.dataset + other.dataset)
        # Dataset이면서 ASRDatasetProtocol을 만족하는 경우
        if isinstance(other, ASRDatasetProtocol):
            return ASRConcatDataset(self.dataset + other)

        raise TypeError("Invalid type for concatenation")


__all__ = ["ASRConcatDataset"]
