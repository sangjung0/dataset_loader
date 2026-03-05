from __future__ import annotations

from typing import runtime_checkable, Protocol
from collections.abc import Sequence

from dataset_loader.protocol.dataset_protocol import DatasetProtocol


@runtime_checkable
class ConcatDatasetProtocol(DatasetProtocol, Protocol):
    """
    여러 개의 Dataset을 하나로 합친 Dataset을 나타내는 프로토콜입니다. 이 프로토콜은 ConcatDataset이 가져야 하는 속성과 메서드를 정의합니다.
    ConcatDataset을 구현할 때 이 프로토콜을 사용하지 않습니다.
    구현할 때는 dataset_loader.base.dataset.ConcatDataset 클래스를 상속하여 구현합니다.
    """

    @property
    def datasets(self) -> Sequence[DatasetProtocol]:
        """
        ConcatDataset에 포함된 Dataset들의 리스트를 반환하는 속성입니다.

        Returns:
            Sequence[DatasetProtocol]: ConcatDataset에 포함된 Dataset들의 리스트입니다. 각 Dataset은 DatasetProtocol을 준수해야 합니다.
        """
        ...

    @property
    def names(self) -> Sequence[str]:
        """
        ConcatDataset에 포함된 Dataset들의 이름을 반환하는 속성입니다.

        Returns:
            Sequence[str]: ConcatDataset에 포함된 Dataset들의 이름을 나타내는 문자열 리스트입니다. 각 이름은 ConcatDataset에 포함된 Dataset의 name 속성에서 가져와야 합니다.
        """
        ...


__all__ = ["ConcatDatasetProtocol"]
