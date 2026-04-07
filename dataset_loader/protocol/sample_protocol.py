from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from typing_extensions import Self
from collections.abc import Mapping, MutableMapping


@runtime_checkable
class SampleProtocol(Protocol):
    """
    데이터셋에서 반환되는 샘플을 나타내는 프로토콜입니다. 이 프로토콜은 샘플이 가져야 하는 속성과 메서드를 정의합니다.
    샘플을 구현할 때 이 프로토콜을 사용하지 않습니다.
    구현할 때는 dataset_loader.base.sample.Sample 클래스를 상속하여 구현합니다.
    """

    id: str
    """샘플의 고유 식별자를 반환하는 속성입니다."""
    data: Mapping[str, Any]
    """샘플의 실제 데이터가 포함된 딕셔너리를 반환하는 속성입니다."""

    def to_dict(self) -> MutableMapping[str, Any]:
        """샘플을 딕셔너리로 변환하는 메서드입니다."""
        ...

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """딕셔너리에서 샘플을 생성하는 정적 메서드입니다."""
        ...


__all__ = ["SampleProtocol"]
