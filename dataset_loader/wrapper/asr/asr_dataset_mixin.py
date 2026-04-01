from __future__ import annotations

from abc import ABC
from typing import TypeVar, Any
from typing_extensions import override, Self
from collections.abc import Mapping

from dataset_loader.protocol import DatasetProtocol

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.thread_loader_mixin import ThreadLoaderMixin

from dataset_loader.wrapper.asr.asr_sample import ASRSample


T = TypeVar("T", bound=DatasetProtocol[Any, Any])


class ASRDatasetMixin(
    DatasetWrapper[T, ASRSample],
    ThreadLoaderMixin[ASRSample],
    ABC,
):
    @override
    def get(self, idx: int) -> ASRSample:
        sample = self.dataset.get(idx)
        return ASRSample(sample=sample)

    @override
    def _loader(self, sample: ASRSample) -> ASRSample:
        return sample.loaded_audio_sample()

    @classmethod
    @override
    def __setstate__(cls, data: Mapping[str, Any]) -> Self:
        dataset = super().__setstate__(data)
        if isinstance(dataset, cls):
            return dataset
        raise TypeError(
            f"Invalid type for deserialization, expected {cls.__name__}, got {type(dataset)} but got {type(dataset)}"
        )


__all__ = ["ASRDatasetMixin"]
