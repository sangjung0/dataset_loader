from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dataset_loader.protocol import DatasetProtocol, SampleProtocol


@runtime_checkable
class ASRDatasetProtocol(DatasetProtocol[Any, SampleProtocol], Protocol):
    @property
    def sr(self) -> int: ...

    @sr.setter
    def sr(self, value: int) -> None: ...


__all__ = ["ASRDatasetProtocol"]
