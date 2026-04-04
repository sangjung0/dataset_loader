from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Any, cast
from typing_extensions import override

from dataset_loader.protocol import DatasetProtocol
from dataset_loader.base import Dataset, ConcatDataset
from dataset_loader.abstract import ASRSample

from dataset_loader.wrapper.asr.protocol import ASRDatasetProtocol
from dataset_loader.wrapper.asr.asr_dataset_mixin import ASRDatasetMixin

if TYPE_CHECKING:
    from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset


class ASRDataset(ASRDatasetMixin[ASRDatasetProtocol]):
    @property
    def sr(self) -> int:
        return self.dataset.sr

    @sr.setter
    def sr(self, value: int) -> None:
        self.dataset.sr = value

    @override
    def concat(self, other: DatasetProtocol[Any, Any]) -> ASRConcatDataset:
        from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset
        from dataset_loader.wrapper.asr.asr_dataset import ASRDataset

        # ASRConcatDataset인 경우
        if isinstance(other, ASRConcatDataset):
            return ASRConcatDataset(other.dataset + self.dataset)
        # ASRDataset인 경우
        if isinstance(other, ASRDataset):
            dataset = cast(
                ConcatDataset[Any, ASRSample],
                self.dataset + other.dataset,
            )
            return ASRConcatDataset(dataset)
        # Dataset이면서 ASRDatasetProtocol을 만족하는 경우
        if isinstance(other, ASRDatasetProtocol) and isinstance(other, Dataset):
            return ASRConcatDataset(other + self.dataset)

        raise TypeError("Invalid type for concatenation")


__all__ = ["ASRDataset"]
