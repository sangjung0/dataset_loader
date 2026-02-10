from __future__ import annotations

import librosa
import numpy as np

from pathlib import Path
from typing import Sequence, get_args
from typing_extensions import override, Self

from dataset_loader.interface import Dataset, Sample
from dataset_loader.librispeech.constants import LibriTask


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        audio_paths: list[Path],
        references: list[str],
        sample_rate: int,
        task: tuple[LibriTask, ...],
    ):
        if len(ids) != len(audio_paths) or len(ids) != len(references):
            raise ValueError(
                "ids, audio_paths, and references must have the same length"
            )
        for t in task:
            if t not in get_args(LibriTask):
                raise ValueError(f"Invalid task: {t}")
        super().__init__(task=task)

        self._ids = ids
        self._audio_paths = audio_paths
        self._references = references
        self._sr = sample_rate

    @Dataset.args.getter
    @override
    def args(self) -> dict:
        return {
            **super().args,
            "ids": self._ids,
            "audio_paths": self._audio_paths,
            "references": self._references,
            "sample_rate": self._sr,
        }

    @Dataset.length.getter
    @override
    def length(self) -> int:
        return len(self._ids)

    @property
    def sr(self) -> int:
        return self._sr

    @sr.setter
    def sr(self, value: int):
        self._sr = value

    @override
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "ids": self._ids,
            "audio_paths": [str(p) for p in self._audio_paths],
            "references": self._references,
            "sample_rate": self._sr,
        }

    @override
    def select(self, indices: Sequence[int]) -> Self:
        return LibriSpeechDataset(
            [self._ids[i] for i in indices],
            [self._audio_paths[i] for i in indices],
            [self._references[i] for i in indices],
            sample_rate=self._sr,
            task=self.task,
        )

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        return LibriSpeechDataset(
            self._ids[start:stop:step],
            self._audio_paths[start:stop:step],
            self._references[start:stop:step],
            sample_rate=self._sr,
            task=self.task,
        )

    @override
    def get(self, idx: int) -> Sample:
        _id = self._ids[idx]
        audio_path = self._audio_paths[idx]
        reference = self._references[idx]

        def load_audio() -> np.ndarray:
            return librosa.load(audio_path, sr=self._sr)[0]

        return Sample(id=_id, data={"load_audio_func": load_audio, "ref": reference})

    def save(self, path: Path, description="LibriSpeechDataset"):
        from sjpy.file.json import JsonSaver

        JsonSaver(description).save(self.to_dict(), path)

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if rng is None or size == len(self._ids) - start:
            return self.slice(start, start + size)
        else:
            data = list(
                zip(
                    self._ids[start:],
                    self._audio_paths[start:],
                    self._references[start:],
                )
            )
            data = rng.choice(data, size=size, replace=False)
            ids, audio_paths, references = zip(*data)
            return LibriSpeechDataset(ids, audio_paths, references, self._sr, self.task)

    @staticmethod
    @override
    def from_dict(data: dict) -> Self:
        return LibriSpeechDataset(
            ids=data["ids"],
            audio_paths=[Path(p) for p in data["audio_paths"]],
            references=data["references"],
            sample_rate=data["sample_rate"],
            task=tuple(data.get("task", ())),
        )

    @staticmethod
    def load(path: Path):
        from sjpy.file.json import load_json

        _, data = load_json(path)
        return LibriSpeechDataset.from_dict(data)


__all__ = ["LibriSpeechDataset"]
