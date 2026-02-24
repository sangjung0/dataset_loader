from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Sample:
    id: str = field(compare=True, hash=True, repr=True)
    data: dict[str, Any] = field(compare=False, hash=False, repr=False)

    def to_dict(self: Sample) -> dict:
        return {"id": self.id, "data": self.data}

    @staticmethod
    def from_dict(data: dict) -> Sample:
        return Sample(id=data["id"], data=data["data"])


__all__ = ["Sample"]
