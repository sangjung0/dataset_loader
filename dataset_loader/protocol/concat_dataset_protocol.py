from __future__ import annotations

from typing import Protocol, runtime_checkable, TypeVar, Any
from typing_extensions import override
from collections.abc import MutableSequence

from dataset_loader.protocol.dataset_protocol import DatasetProtocol
from dataset_loader.protocol.sample_protocol import SampleProtocol

D = TypeVar("D")
S = TypeVar("S", bound=SampleProtocol, covariant=True)


@runtime_checkable
class ConcatDatasetProtocol(DatasetProtocol[MutableSequence[D], S], Protocol):
    """
    이 프로토콜은 Dataset을 여러 개 연결하는 ConcatDataset이 가져야 하는 속성과 메서드를 정의합니다.
    구현할 때는 dataset_loader.base.concat_dataset.ConcatDataset 클래스를 상속하여 구현합니다.
    """

    """ConcatDataset에 포함된 Dataset 객체들을 담는 리스트입니다. 이 리스트는 ConcatDataset이 생성될 때 초기화되어야 합니다. """
    _datasets: list[D]

    @property
    def names(self) -> MutableSequence[str]:
        """
        ConcatDataset에 포함된 각 Dataset의 이름을 반환하는 속성입니다.

        Returns:
            list[str]: ConcatDataset에 포함된 각 Dataset의 이름을 담은 리스트입니다. 각 이름은 ConcatDataset에 포함된 Dataset의 name 속성에서 가져와야 합니다.
        """
        ...

    @override
    def __add__(
        self, other: DatasetProtocol[Any, Any]
    ) -> ConcatDatasetProtocol[Any, Any]: ...

    @override
    def concat(
        self, other: DatasetProtocol[Any, Any]
    ) -> ConcatDatasetProtocol[Any, Any]: ...


__all__ = ["ConcatDatasetProtocol"]
