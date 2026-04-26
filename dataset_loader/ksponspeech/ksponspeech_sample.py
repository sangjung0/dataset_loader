from __future__ import annotations

from dataclasses import dataclass

from dataset_loader.abstract import ASRSample


@dataclass(frozen=True, slots=True)
class KSponSpeechSample(ASRSample[str, None]):
    """
    KSponSpeech 데이터셋의 샘플을 나타내는 클래스.
    """

    @property
    def raw(self) -> str:
        if "raw" not in self.data:
            raise AttributeError("Raw transcript is not available in this sample")
        return self.data["raw"]  # type: ignore[no-any-return]

    @property
    def phonetic(self) -> str:
        if "phonetic" not in self.data:
            raise AttributeError("Phonetic transcript is not available in this sample")
        return self.data["phonetic"]  # type: ignore[no-any-return]

    @property
    def spelling(self) -> str:
        if "spelling" not in self.data:
            raise AttributeError("Spelling transcript is not available in this sample")
        return self.data["spelling"]  # type: ignore[no-any-return]


__all__ = ["KSponSpeechSample"]
