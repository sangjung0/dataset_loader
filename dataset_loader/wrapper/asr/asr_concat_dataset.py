from __future__ import annotations

from typing_extensions import override

from dataset_loader.interface import ConcatDataset

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.asr.asr_dataset import ASRDataset, ASRDatasetPtc
from dataset_loader.wrapper.asr.asr_sample import ASRSample


class ASRConcatDataset(
    ASRDataset, DatasetWrapper[ASRSample, ConcatDataset[ASRDatasetPtc]]
):
    def __init__(self, dataset: ConcatDataset[ASRDatasetPtc]):
        super().__init__(dataset=dataset)

        datasets = dataset._datasets
        sr_set = {ds.sr for ds in datasets}
        if len(sr_set) != 1:
            raise ValueError("All datasets must have the same sampling rate")

        self._sr = sr_set.pop()

    @property
    @override
    def dataset(self) -> ConcatDataset[ASRDatasetPtc]:
        return self._dataset

    @ASRDataset.sr.setter
    def sr(self, value: int):
        for ds in self.dataset._datasets:
            ds.sr = value
        self._sr = value


__all__ = ["ASRConcatDataset"]
