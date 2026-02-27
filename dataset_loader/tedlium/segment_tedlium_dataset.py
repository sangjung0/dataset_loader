from __future__ import annotations

from typing import Sequence
from typing_extensions import override
from datasets import Dataset, Audio

from sjpy.string import remove_spaces_and_symbols

from dataset_loader.abstract import HuggingfaceDataset
from dataset_loader.interface import Sample


class SegmentTedliumDataset(HuggingfaceDataset):
    def __init__(
        self: SegmentTedliumDataset,
        *,
        dataset: Dataset,
        sr: int,
        use_cache: int = 0,
        ignore_set: Sequence[str],
    ):
        super().__init__(dataset=dataset, use_cache=use_cache)

        self._ignore_set: set[str] = set(ignore_set)
        self._sr = sr
        self._cast_audio(sr)

    @HuggingfaceDataset.args.getter
    @override
    def args(self: SegmentTedliumDataset) -> dict:
        if self.is_cleaned:
            raise RuntimeError("Cannot get args of a cleaned dataset")
        return {
            **super().args,
            "sr": self._sr,
            "ignore_set": self._ignore_set,
        }

    @property
    def sr(self: SegmentTedliumDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: SegmentTedliumDataset, value: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")
        elif value == self._sr:
            return
        elif not (isinstance(value, int) and value > 0):
            raise ValueError("Sample rate must be a positive integer")
        self._sr = value
        self._cast_audio(value)

    def _cast_audio(self: SegmentTedliumDataset, sr: int) -> None:
        if self.is_cleaned:
            raise RuntimeError("Cannot change sample rate of a cleaned dataset")

        self._dataset = self._dataset.cast_column("audio", Audio(sampling_rate=sr))

    @override
    def _get(self, idx: int) -> Sample:
        data = self._dataset[idx]

        def load_audio_func():
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
