from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Sample:
    """
    Dataset에서 데이터를 가져올 때 반환되는 객체를 나타내는 클래스이다.
    각 샘플은 고유한 id와 데이터가 포함된 딕셔너리를 가진다.
    Attributes:
        id (str): 샘플의 고유 식별자
        data (dict[str, Any]): 샘플의 실제 데이터가 포함된 딕셔너리
    """

    id: str = field(compare=True, hash=True, repr=True)
    data: dict[str, Any] = field(compare=False, hash=False, repr=False)

    def to_dict(self: Sample) -> dict:
        return {"id": self.id, "data": self.data}

    @staticmethod
    def from_dict(data: dict) -> Sample:
        return Sample(id=data["id"], data=data["data"])


__all__ = ["Sample"]
