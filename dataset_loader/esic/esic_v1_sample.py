from __future__ import annotations

from typing import cast
from dataclasses import dataclass
from pathlib import Path

from dataset_loader.abstract import ASRSample

from dataset_loader.esic.constants import (
    TXT,
    VERT_TS,
    ORTO,
    ORTO_TS,
    VERBATIM,
    PUNCT_VERBATIM,
)


@dataclass(frozen=True, slots=True)
class ESICv1Sample(ASRSample):
    """
    ESIC v1.1 데이터셋의 샘플을 나타내는 클래스.
    """

    @property
    def txt(self) -> str:
        return cast(str, self.data[TXT])

    @property
    def vert_ts(self) -> str:
        return cast(str, self.data[VERT_TS])

    @property
    def orto(self) -> str:
        return cast(str, self.data[ORTO])

    @property
    def orto_ts(self) -> str:
        return cast(str, self.data[ORTO_TS])

    @property
    def verbatim(self) -> str:
        return cast(str, self.data[VERBATIM])

    @property
    def punct_verbatim(self) -> str:
        return cast(str, self.data[PUNCT_VERBATIM])

    @property
    def mp4_path(self) -> Path:
        return cast(Path, self.data["mp4_path"])


__all__ = ["ESICv1Sample"]
