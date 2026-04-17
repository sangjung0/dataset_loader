# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from typing import Any, TypeVar, cast
from typing_extensions import override
from collections.abc import MutableSequence

from dataset_loader.protocol import DatasetProtocol, ConcatDatasetProtocol
from dataset_loader.abstract import ASRSample

from dataset_loader.wrapper.asr.protocol import ASRDatasetProtocol
from dataset_loader.wrapper.asr.asr_dataset_mixin import ASRDatasetMixin

RefT = TypeVar("RefT")
DiarizationT = TypeVar("DiarizationT")


class ASRConcatDataset(ASRDatasetMixin[RefT, DiarizationT]):
    """
    ASRDataset을 연결하여 새로운 ASRDataset을 만드는 클래스이다.
    """

    def __init__(
        self,
        dataset: ConcatDatasetProtocol[
            ASRDatasetProtocol, ASRSample[RefT, DiarizationT]
        ],
    ) -> None:
        for ds in dataset.dataset:
            if not isinstance(ds, ASRDatasetProtocol):
                raise TypeError(
                    f"All datasets must be ASRDatasetProtocol, but got {type(ds)}"
                )

        super().__init__(dataset=dataset)

    @property
    @override
    def dataset(
        self,
    ) -> ConcatDatasetProtocol[ASRDatasetProtocol, ASRSample[RefT, DiarizationT]]:
        return cast(
            ConcatDatasetProtocol[ASRDatasetProtocol, ASRSample[RefT, DiarizationT]],
            super().dataset,
        )

    @property
    def sr(self) -> int:
        return self.dataset.dataset[0].sr

    @sr.setter
    def sr(self, value: int) -> None:
        for dataset in self.dataset.dataset:
            dataset.sr = value

    @property
    def names(self) -> MutableSequence[str]:
        return self.dataset.names

    @override
    def concat(self, other: DatasetProtocol[Any, Any]) -> ASRConcatDataset[Any, Any]:
        from dataset_loader.wrapper.asr.asr_concat_dataset import ASRConcatDataset
        from dataset_loader.wrapper.asr.asr_dataset import ASRDataset

        # ASRDataset 또는 ASRConcatDataset인 경우
        if isinstance(other, (ASRDataset, ASRConcatDataset)):
            other = cast(ASRDatasetProtocol, other)
            return ASRConcatDataset(self.dataset + other.dataset)
        # Dataset이면서 ASRDatasetProtocol을 만족하는 경우
        if isinstance(other, ASRDatasetProtocol):
            return ASRConcatDataset(self.dataset + other)

        raise TypeError("Invalid type for concatenation")


__all__ = ["ASRConcatDataset"]
