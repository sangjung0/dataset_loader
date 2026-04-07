from __future__ import annotations

from typing import cast, TypedDict
from typing_extensions import ReadOnly
from dataclasses import dataclass

from dataset_loader.abstract import ASRSample


class DiarizationLabel(TypedDict):
    start: ReadOnly[float]
    end: ReadOnly[float]
    speaker: ReadOnly[str]
    gender: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class SegmentTedliumSample(ASRSample[str, list[DiarizationLabel]]):
    """
    TED-LIUM 데이터셋의 샘플을 나타내는 클래스.
    """

    @property
    def original_id(self) -> str:
        return cast(str, self.data["original_id"])

    @property
    def file(self) -> str:
        return cast(str, self.data["file"])


__all__ = ["SegmentTedliumSample"]
