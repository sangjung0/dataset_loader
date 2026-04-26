# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Any
from typing_extensions import override
from pathvalidate import sanitize_filepath
from datasets import Dataset, Audio

from dataset_loader.abstract import HuggingfaceDataset

from dataset_loader.ksponspeech.ksponspeech_sample import KSponSpeechSample
from dataset_loader.ksponspeech.preprocess import bracket_filter, special_filter


class KSponSpeechDataset(HuggingfaceDataset[KSponSpeechSample]):
    def __init__(self, *, dataset: Dataset, sr: int):
        d0 = dataset[0]
        if "path" not in d0:
            dataset = dataset.cast_column("audio", Audio(decode=False)).map(
                lambda b: {"path": [a.get("path") for a in b["audio"]]},
                batched=True,
            )

        super().__init__(dataset=dataset)
        self._sr = sr
        self._cast_audio(sr)

    @property
    @override
    def args(self) -> dict[str, Any]:
        if self.is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {**super().args, "sr": self._sr}

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

    def _cast_audio(self, sr: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")
        self._dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sr))

    @override
    def get(self, idx: int) -> KSponSpeechSample:
        if self.is_cleaned:
            raise RuntimeError("Cannot get sample from a cleaned dataset")
        data: dict[str, Any] = self.dataset[idx]
        _id = sanitize_filepath(data["path"])[-255:]

        def load_audio() -> npt.NDArray[np.float32]:
            audio: npt.NDArray[np.float32] = (
                data["audio"]
                .get_all_samples()
                .data.mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return audio

        transcript: str = data["transcripts"]
        spelling: str = special_filter(
            bracket_filter(transcript, mode="spelling"), mode="spelling"
        )
        phonetic: str = special_filter(
            bracket_filter(transcript, mode="phonetic"), mode="phonetic"
        )

        result: dict[str, Any] = {
            "load_audio_func": load_audio,
            "ref": spelling,
            "raw": transcript,
            "phonetic": phonetic,
            "spelling": spelling,
        }

        return KSponSpeechSample(id=_id, data=result)


__all__ = ["KSponSpeechDataset"]
