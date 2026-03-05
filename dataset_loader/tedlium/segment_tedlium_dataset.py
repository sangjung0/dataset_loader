from __future__ import annotations

import numpy as np

from typing import Any
from typing_extensions import override
from collections.abc import Sequence
from datasets import Dataset, Audio

from sjpy.string import remove_spaces_and_symbols

from dataset_loader.abstract import HuggingfaceDataset
from dataset_loader.base import Sample


class SegmentTedliumDataset(HuggingfaceDataset):
    def __init__(self, *, dataset: Dataset, sr: int, ignore_set: Sequence[str]):
        super().__init__(dataset=dataset)

        self._ignore_set: set[str] = set(ignore_set)
        self._sr = sr
        self._cast_audio(sr)

    @HuggingfaceDataset.args.getter
    @override
    def args(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {
            **super().args,
            "sr": self._sr,
            "ignore_set": self._ignore_set,
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
        elif not (isinstance(value, int) and value > 0):
            raise ValueError("Sample rate must be a positive integer")
        self._sr = value
        self._cast_audio(value)

    def _cast_audio(self, sr: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")
        assert self._dataset is not None
        self._dataset = self._dataset.cast_column("audio", Audio(sampling_rate=sr))

    @override
    def get(self, idx: int) -> Sample:
        if self.is_cleaned:
            raise RuntimeError("Cannot get sample from a cleaned dataset")
        assert self._dataset is not None
        data = self._dataset[idx]

        def load_audio_func() -> np.ndarray:
            samples = data["audio"].get_all_samples()
            wav = samples.data.mean(dim=0).detach().cpu().numpy()
            return wav

        text = data["text"].strip()
        if text in self._ignore_set:
            text = ""

        infos = data["id"].split("-")
        result = {
            "original_id": data["id"],
            "load_audio_func": load_audio_func,
            "file": data["file"],
            "ref": text,
            "diarization": [
                {
                    "start": float(infos[1]),
                    "end": float(infos[2]),
                    "label": data["speaker_id"],
                    "gender": data["gender"],
                }
            ],
        }

        _id = remove_spaces_and_symbols(data["id"])[-255:]
        return Sample(id=_id, data=result)


__all__ = ["SegmentTedliumDataset"]
