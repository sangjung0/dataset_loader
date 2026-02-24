from __future__ import annotations

import numpy as np
import librosa

from typing_extensions import override
from pathvalidate import sanitize_filepath
from datasets import Dataset, Audio

from dataset_loader.abstract import HuggingfaceDataset
from dataset_loader.interface import Sample

from dataset_loader.ksponspeech.constants import KSPonSpeechTask


class KSPonSpeechDataset(HuggingfaceDataset):
    def __init__(
        self,
        *,
        dataset: Dataset,
        sr: int,
        task: tuple[KSPonSpeechTask, ...],
    ):
        d0 = dataset[0]
        if "path" not in d0:
            dataset = (
                dataset.cast_column("audio", Audio(decode=False))
                .map(
                    lambda b: {"path": [a.get("path") for a in b["audio"]]},
                    batched=True,
                )
                .cast_column("audio", Audio(sampling_rate=sr))
            )
        else:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))

        super().__init__(dataset=dataset, sr=sr, task=task)

    @override
    def get(self, idx: int) -> Sample:
        data = self._dataset[idx]
        _id = sanitize_filepath(data["path"])[-255:]

        def load_audio() -> np.ndarray:
            return (
                data["audio"].get_all_samples().data.mean(dim=0).detach().cpu().numpy()
            )

        result = {
            "load_audio_func": load_audio,
            "ref": data["transcripts"],
        }

        return Sample(id=_id, data=result)


__all__ = ["KSPonSpeechDataset"]
