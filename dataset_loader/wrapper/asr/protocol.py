from __future__ import annotations

from typing import runtime_checkable, Protocol

from dataset_loader.protocol import DatasetProtocol, ConcatDatasetProtocol


@runtime_checkable
class ASRDatasetProtocol(DatasetProtocol, Protocol):
    @property
    def sr(self) -> int: ...

    @sr.setter
    def sr(self, value: int): ...


@runtime_checkable
class ASRConcatDatasetProtocol(ConcatDatasetProtocol, Protocol):
    @property
    def sr(self) -> int: ...

    @sr.setter
    def sr(self, value: int): ...


__all__ = ["ASRDatasetProtocol", "ASRConcatDatasetProtocol"]
