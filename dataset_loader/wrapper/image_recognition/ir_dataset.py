from __future__ import annotations

from typing import Any, TypeVar
from typing_extensions import override

from dataset_loader.protocol import DatasetProtocol
from dataset_loader.abstract import IRSample

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper
from dataset_loader.wrapper.thread_loader_mixin import ThreadLoaderMixin

LabelT = TypeVar("LabelT")


class IRDataset(
    DatasetWrapper[IRSample[LabelT]],
    ThreadLoaderMixin[IRSample[LabelT]],
):
    @override
    def concat(self, other: DatasetProtocol[Any, IRSample[Any]]) -> IRDataset[Any]:
        # IRDataset인 경우
        if isinstance(other, IRDataset):
            return IRDataset(dataset=self.dataset + other.dataset)
        # DatasetProtocol인 경우
        elif isinstance(other, DatasetProtocol):
            return IRDataset(dataset=self.dataset + other)
        raise TypeError("Invalid type for concatenation")

    def _loader(self, sample: IRSample[LabelT]) -> IRSample[LabelT]:
        return sample.loaded_ir_sample()


__all__ = ["IRDataset"]
