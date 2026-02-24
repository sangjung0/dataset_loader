from __future__ import annotations

import librosa
import numpy as np

from typing_extensions import override
from datasets import Dataset
from pathvalidate import sanitize_filepath

from dataset_loader.abstract import HuggingfaceDataset
from dataset_loader.interface import Sample
from dataset_loader.zerothkorean.constants import ZerothKoreanTask


class ZerothKoreanDataset(HuggingfaceDataset):
    def __init__(
        self, *, dataset: Dataset, sr: int, task: tuple[ZerothKoreanTask, ...]
    ):
        super().__init__(dataset=dataset, sr=sr, task=task)
        self._original_sr = sr

    @override
    def get(self, idx: int) -> Sample:
        data = self._dataset[idx]
        _id = sanitize_filepath(data["path"])[-255:]

        def load_audio() -> np.ndarray:
            return self._resample_audio(data["audio"]["array"]).astype(np.float32)

        result = {"load_audio_func": load_audio}
        if "asr" in self.task:
            result["ref"] = data["text"]
        if "diarization" in self.task:
            result["diarization"] = [
                {
                    "start": 0,
                    "end": -1,  # NOTE 임시 값
                    "label": data["speaker_id"],
                }
            ]
        return Sample(id=_id, data=result)

    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        if self._sr != self._original_sr:
            audio = librosa.resample(
                audio, orig_sr=self._original_sr, target_sr=self._sr
            )
        return audio


__all__ = ["ZerothKoreanDataset"]
