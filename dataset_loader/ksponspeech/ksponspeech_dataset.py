from __future__ import annotations

import re
import numpy as np

from typing_extensions import override
from pathvalidate import sanitize_filepath
from datasets import Dataset, Audio

from dataset_loader.abstract import HuggingfaceDataset
from dataset_loader.interface import Sample


class KSPonSpeechDataset(HuggingfaceDataset):
    def __init__(
        self,
        *,
        dataset: Dataset,
        sr: int,
        use_cache: int = 0,
    ):
        d0 = dataset[0]
        if "path" not in d0:
            dataset = dataset.cast_column("audio", Audio(decode=False)).map(
                lambda b: {"path": [a.get("path") for a in b["audio"]]},
                batched=True,
            )

        super().__init__(dataset=dataset, use_cache=use_cache)
        self._sr = sr
        self._cast_audio(sr)

    @HuggingfaceDataset.args.getter
    @override
    def args(self: KSPonSpeechDataset) -> dict:
        if self.is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {
            **super().args,
            "sr": self._sr,
        }

    @property
    def sr(self: KSPonSpeechDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: KSPonSpeechDataset, value: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")
        elif value == self._sr:
            return
        elif not (isinstance(value, int) and value > 0):
            raise ValueError("Sample rate must be a positive integer")

        self._sr = value
        self._cast_audio(value)

    def _cast_audio(self: KSPonSpeechDataset, sr: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")

        self._dataset = self._dataset.cast_column("audio", Audio(sampling_rate=sr))

    @override
    def _get(self, idx: int) -> Sample:
        data = self._dataset[idx]
        _id = sanitize_filepath(data["path"])[-255:]

        def load_audio() -> np.ndarray:
            return (
                data["audio"].get_all_samples().data.mean(dim=0).detach().cpu().numpy()
            )

        result = {
            "load_audio_func": load_audio,
            "ref": re.sub(r"\s+", " ", data["transcripts"]),
        }

        return Sample(id=_id, data=result)


__all__ = ["KSPonSpeechDataset"]
