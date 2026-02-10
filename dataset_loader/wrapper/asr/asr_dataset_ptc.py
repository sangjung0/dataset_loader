from __future__ import annotations
from typing import TYPE_CHECKING

from dataset_loader.interface import Dataset


class ASRDatasetPtc(Dataset):
    @property
    def sr(self) -> int: ...


__all__ = ["ASRDatasetPtc"]
