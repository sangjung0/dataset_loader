from __future__ import annotations

from typing import cast
from dataclasses import dataclass

from dataset_loader.abstract import ASRSample


@dataclass(frozen=True, slots=True)
class ZerothKoreanSample(ASRSample):
    """
    Zeroth Korean 데이터셋의 샘플을 나타내는 클래스.
    """

    @property
    def text(self) -> str:
        return cast(str, self.data["text"])


__all__ = ["ZerothKoreanSample"]
