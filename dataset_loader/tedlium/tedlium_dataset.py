from __future__ import annotations

import numpy as np

from typing import get_args
from typing_extensions import override
from datasets import Dataset as DT, Audio

from sjpy.string import remove_spaces_and_symbols

from dataset_loader.interface import Sample, Dataset

from dataset_loader.tedlium.constants import TedliumTask


class TedliumDataset(Dataset):
    def __init__(
        self: TedliumDataset,
        *,
        dataset: DT,
        sr: int,
        task: tuple[TedliumTask, ...],
        ignore_set: set[str],
        _pdataset: list[dict] | None = None,
    ):
        for t in task:
            if t not in get_args(TedliumTask):
                raise ValueError(f"Invalid task: {t}")
        super().__init__(task=task)

        self._dataset = dataset
        self._sr = sr
        self._ignore_set = ignore_set
        self._pdataset = self.__prepare(dataset) if _pdataset is None else _pdataset

    @Dataset.args.getter
    @override
    def args(self: TedliumDataset) -> dict:
        return {
            **super().args,
            "dataset": self._dataset,
            "sr": self._sr,
            "ignore_set": self._ignore_set,
            "_pdataset": self._pdataset,
        }

    @Dataset.length.getter
    @override
    def length(self: TedliumDataset) -> int:
        return len(self._pdataset)

    @property
    def sr(self: TedliumDataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: TedliumDataset, value: int) -> None:
        self._sr = value
        self._dataset = self._dataset.cast_column("audio", Audio(sampling_rate=value))

    @override
    def to_dict(self: TedliumDataset) -> dict:
        return {
            **super().to_dict(),
            "dataset": self._dataset,
            "sr": self._sr,
            "ignore_set": self._ignore_set,
            "_pdataset": self._pdataset,
        }

    @override
    def select(self: TedliumDataset, indices: list[int]) -> TedliumDataset:
        selected_dataset = [self._pdataset[i] for i in indices]
        return TedliumDataset(
            dataset=self._dataset,
            sr=self._sr,
            task=self.task,
            ignore_set=self._ignore_set,
            _pdataset=selected_dataset,
        )

    @override
    def slice(
        self: TedliumDataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> TedliumDataset:
        sliced_dataset = self._pdataset[start:stop:step]
        return TedliumDataset(
            dataset=self._dataset,
            sr=self._sr,
            task=self.task,
            ignore_set=self._ignore_set,
            _pdataset=sliced_dataset,
        )

    @override
    def _sample(
        self: TedliumDataset,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Sample:
        if rng is None or size == len(self) - start:
            return self.slice(start, start + size)
        else:
            indices = range(len(self))[start:]
            index = rng.choice(indices, size=size, replace=False)
            return self.select(index)

    def __prepare(self: TedliumDataset, dataset: DT) -> list[dict]:
        file_pairs: dict[str, list[dict]] = {}
        for data in dataset:
            file = data["file"]
            if file not in file_pairs:
                file_pairs[file] = []
            file_pairs[file].append(data)

        for value in file_pairs.values():
            value.sort(key=lambda x: x["audio"].metadata.stream_index)

        pdataset = []
        for file, data_list in file_pairs.items():
            _id = remove_spaces_and_symbols(file)[-255:]

            speaker_info = []
            audios = []
            text = ""
            for data in data_list:
                infos = data["id"].split("-")
                speaker_info.append(
                    {
                        "start": float(infos[1]),
                        "end": float(infos[2]),
                        "label": data["speaker_id"],
                        "gender": data["gender"],
                    }
                )
                audios.append(data["audio"])
                t = data["text"]
                if t not in self._ignore_set:
                    text += data["text"] + " "
            text = text[:-1]

            pdataset.append(
                {
                    "id": _id,
                    "file": file,
                    "audios": audios,
                    "text": text,
                    "speaker_info": speaker_info,
                }
            )

        return pdataset

    @override
    def get(self, idx: int) -> Sample:
        data = self._pdataset[idx]

        def load_audio_func():
            audios = data["audios"]
            sample = audios[0].get_all_samples()
            sr = sample.sample_rate
            wav_list = [sample.data.mean(dim=0).detach().cpu().numpy()]
            for audio in audios[1:]:
                samples = audio.get_all_samples()
                if samples.sample_rate != sr:
                    raise ValueError("Sampling rate mismatch")
                wav_list.append(samples.data.mean(dim=0).detach().cpu().numpy())
            return np.hstack(wav_list)

        result = {
            "load_audio_func": load_audio_func,
            "file": data["file"],
            "ref": data["text"],
            "diarization": data["speaker_info"],
        }

        return Sample(id=data["id"], data=result)

    @classmethod
    @override
    def from_dict(cls: type[TedliumDataset], data: dict) -> TedliumDataset:
        data["dataset"] = DT.from_dict(data["dataset"])
        return cls(**data)


__all__ = ["TedliumDataset"]
