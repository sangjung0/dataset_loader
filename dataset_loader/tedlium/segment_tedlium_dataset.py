from __future__ import annotations

from typing import get_args
from typing_extensions import override
from datasets import Dataset

from sjpy.string import remove_spaces_and_symbols

from dataset_loader.abstract import HuggingfaceDataset
from dataset_loader.interface import Sample

from dataset_loader.tedlium.constants import TedliumTask


class SegmentTedliumDataset(HuggingfaceDataset):
    def __init__(
        self,
        dataset: Dataset,
        sr: int,
        task: tuple[TedliumTask, ...],
        ignore_set: set[str],
    ):
        for t in task:
            if t not in get_args(TedliumTask):
                raise ValueError(f"Invalid task: {t}")

        super().__init__(dataset=dataset, sr=sr, task=task)
        self._ignore_set = ignore_set

    @HuggingfaceDataset.args.getter
    @override
    def args(self):
        return {
            **super().args,
            "ignore_set": self._ignore_set,
        }

    @override
    def get(self, idx: int) -> Sample:
        data = self._dataset[idx]

        def load_audio_func():
            samples = data["audio"].get_all_samples()
            wav = samples.data.mean(dim=0).detach().cpu().numpy()
            return wav

        text = data["text"]
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
