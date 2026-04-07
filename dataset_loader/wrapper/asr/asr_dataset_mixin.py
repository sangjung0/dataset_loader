from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar
from typing_extensions import override

from dataset_loader.abstract import ASRSample

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.thread_loader_mixin import ThreadLoaderMixin

from dataset_loader.wrapper.asr.protocol import ASRDatasetProtocol

RefT = TypeVar("RefT")
DiarizationT = TypeVar("DiarizationT")


class ASRDatasetMixin(
    DatasetWrapper[ASRSample[RefT, DiarizationT]],
    ThreadLoaderMixin[ASRSample[RefT, DiarizationT]],
    ASRDatasetProtocol,
    ABC,
):
    @property
    @abstractmethod
    def sr(self) -> int: ...

    @sr.setter
    @abstractmethod
    def sr(self, value: int) -> None: ...

    @override
    def _loader(
        self, sample: ASRSample[RefT, DiarizationT]
    ) -> ASRSample[RefT, DiarizationT]:
        return sample.loaded_audio_sample()


__all__ = ["ASRDatasetMixin"]
