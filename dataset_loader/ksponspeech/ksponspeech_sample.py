from __future__ import annotations

from dataclasses import dataclass

from dataset_loader.abstract import ASRSample


@dataclass(frozen=True, slots=True)
class KSponSpeechSample(ASRSample):
    """
    KSponSpeech 데이터셋의 샘플을 나타내는 클래스.
    """

    ...


__all__ = ["KSponSpeechSample"]
