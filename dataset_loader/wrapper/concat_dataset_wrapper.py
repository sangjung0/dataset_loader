from __future__ import annotations

from abc import ABC
from typing import TypeVar
from collections.abc import Sequence

from dataset_loader.protocol import (
    SampleProtocol,
    ConcatDatasetProtocol,
)

from dataset_loader.wrapper.dataset_wrapper import DatasetWrapper


Dts = TypeVar("Dts", bound=ConcatDatasetProtocol)
Spl = TypeVar("Spl", bound=SampleProtocol)


class ConcatDatasetWrapper(DatasetWrapper[Dts, Spl], ABC):
    """
    ConcatDataset의 래핑 클래스이다. \n
    다양한 도메인에 대해서 대응하기 위한 인터페이스를 제공한다.

    Attributes:
        dataset (ConcatDatasetProtocol): 래핑할 ConcatDataset 객체
        args (dict): ConcatDatasetWrapper를 생성하는 데 필요한 인자들
        length (int): ConcatDataset의 길이
        name (str): ConcatDataset의 이름
        names (Sequence[str]): ConcatDataset에 포함된 Dataset들의 이름
    """

    @property
    def names(self) -> Sequence[str]:
        return self.dataset.names


__all__ = ["ConcatDatasetWrapper"]
