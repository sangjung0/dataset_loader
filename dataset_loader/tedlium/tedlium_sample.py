from __future__ import annotations


from typing import cast
from dataclasses import dataclass

from dataset_loader.abstract import ASRSample


@dataclass(frozen=True, slots=True)
class TedliumSample(ASRSample):
    """
    TED-LIUM 데이터셋의 샘플을 나타내는 클래스.
    """

    @property
    def channel(self) -> str:
        return cast(str, self.data["channel"])

    @property
    def speakers(self) -> list[str]:
        return cast(list[str], self.data["speakers"])

    @property
    def audio_path(self) -> str:
        return cast(str, self.data["audio_path"])

    @property
    def duration(self) -> float:
        return cast(float, self.data["duration"])

    @property
    def text(self) -> str:
        return cast(str, self.data["text"])


__all__ = ["TedliumSample"]
