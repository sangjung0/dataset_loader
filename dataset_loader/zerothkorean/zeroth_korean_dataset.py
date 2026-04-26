# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

import re
import numpy as np
import numpy.typing as npt

from typing import Any, cast
from typing_extensions import override
from datasets import Dataset, Audio
from pathvalidate import sanitize_filepath

from dataset_loader.abstract import HuggingfaceDataset

from dataset_loader.zerothkorean.zeroth_korean_sample import ZerothKoreanSample


class ZerothKoreanDataset(HuggingfaceDataset[ZerothKoreanSample]):
    def __init__(self, *, dataset: Dataset, sr: int):
        super().__init__(dataset=dataset)
        self._sr = sr
        self._cast_audio(sr)

    @property
    @override
    def args(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {
            **super().args,
            "sr": self._sr,
        }

    @property
    def sr(self) -> int:
        return self._sr

    @sr.setter
    def sr(self, value: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")
        elif value == self._sr:
            return
        elif value <= 0:
            raise ValueError("Sample rate must be a positive integer")
        self._sr = value
        self._cast_audio(value)

    @override
    def get(self, idx: int) -> ZerothKoreanSample:
        if self.is_cleaned:
            raise RuntimeError("Cannot get sample from a cleaned dataset")
        data = cast(dict[str, Any], self.dataset[idx])
        _id = sanitize_filepath(data["path"])[-255:]

        def load_audio() -> npt.NDArray[np.float32]:
            return data["audio"]["array"]  # type: ignore[no-any-return]

        result: dict[str, Any] = {
            "load_audio_func": load_audio,
            "ref": re.sub(r"\s+", " ", data["text"]).strip(),
            "text": data["text"],
            "diarization": [
                {
                    "start": 0,
                    "end": -1,  # NOTE 임시 값
                    "speaker": data["speaker_id"],
                }
            ],
        }
        return ZerothKoreanSample(id=_id, data=result)

    def _cast_audio(self, sr: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")
        self._dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sr))


__all__ = ["ZerothKoreanDataset"]
