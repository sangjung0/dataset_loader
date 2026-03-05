from __future__ import annotations

import re
import librosa
import numpy as np

from typing import Any
from typing_extensions import override
from datasets import Dataset
from pathvalidate import sanitize_filepath

from dataset_loader.abstract import HuggingfaceDataset
from dataset_loader.base import Sample


class ZerothKoreanDataset(HuggingfaceDataset):
    def __init__(self, *, dataset: Dataset, sr: int):
        super().__init__(dataset=dataset)
        self._sr = sr
        self._original_sr = sr

    @HuggingfaceDataset.args.getter
    @override
    def args(self) -> dict[str, Any]:
        if self._is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {
            **super().args,
            "sr": self._sr,
        }

    @property
    def sr(self: ZerothKoreanDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: ZerothKoreanDataset, value: int) -> None:
        if not (isinstance(value, int) and value > 0):
            raise ValueError("Sample rate must be a positive integer")
        self._sr = value

    @override
    def get(self, idx: int) -> Sample:
        if self.is_cleaned:
            raise RuntimeError("Cannot get sample from a cleaned dataset")
        assert self._dataset is not None
        data = self._dataset[idx]
        _id = sanitize_filepath(data["path"])[-255:]

        def load_audio() -> np.ndarray:
            return self._resample_audio(data["audio"]["array"]).astype(np.float32)

        result = {
            "load_audio_func": load_audio,
            "ref": re.sub(r"\s+", " ", data["text"]).strip(),
            "text": data["text"],
            "diarization": [
                {
                    "start": 0,
                    "end": -1,  # NOTE 임시 값
                    "label": data["speaker_id"],
                }
            ],
        }
        return Sample(id=_id, data=result)

    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        if self._sr != self._original_sr:
            audio = librosa.resample(
                audio, orig_sr=self._original_sr, target_sr=self._sr
            )
        return audio


__all__ = ["ZerothKoreanDataset"]
