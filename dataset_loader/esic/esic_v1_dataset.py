from __future__ import annotations

import numpy as np

from pathlib import Path
from typing_extensions import override
from typing import Sequence, get_args

from sjpy.audio import load_from_mp4_file
from sjpy.string import normalize_text_only_en
from sjpy.file.json import JsonSaver, load_json

from dataset_loader.interface import Dataset, Sample

from dataset_loader.esic.constants import ESICTask, DEFAULT_TASK


DEFAULT_SAMPLE_RATE = 16_000


class ESICv1Dataset(Dataset):
    def __init__(
        self: ESICv1Dataset,
        *,
        X: list[Path],
        Y: list[Path],
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[ESICTask, ...] = DEFAULT_TASK,
    ):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        for t in task:
            if t not in get_args(ESICTask):
                raise ValueError(f"Invalid task: {t}")
        super().__init__(task=task)
        self._sr = sr
        self._X = X
        self._Y = Y

    @Dataset.args.getter
    @override
    def args(self: ESICv1Dataset) -> dict:
        return {
            **super().args,
            "X": self._X,
            "Y": self._Y,
            "sr": self._sr,
        }

    @Dataset.length.getter
    @override
    def length(self: ESICv1Dataset) -> int:
        return len(self._X)

    @property
    def sr(self: ESICv1Dataset) -> int:
        return self._sr

    @sr.setter
    def sr(self: ESICv1Dataset, value: int) -> None:
        self._sr = value

    @override
    def to_dict(self: ESICv1Dataset) -> dict:
        return {
            **super().to_dict(),
            "X": [str(x) for x in self._X],
            "Y": [str(y) for y in self._Y],
            "sr": self._sr,
        }

    @override
    def select(self: ESICv1Dataset, indices: Sequence[int]) -> ESICv1Dataset:
        return ESICv1Dataset(
            X=[self._X[i] for i in indices],
            Y=[self._Y[i] for i in indices],
            sr=self.sr,
            task=self.task,
        )

    @override
    def slice(
        self: ESICv1Dataset,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> ESICv1Dataset:
        return ESICv1Dataset(
            X=self._X[start:stop:step],
            Y=self._Y[start:stop:step],
            sr=self.sr,
            task=self.task,
        )

    @override
    def get(self: ESICv1Dataset, idx: int) -> Sample:
        x, y = self._X[idx], self._Y[idx]

        def load_audio() -> np.ndarray:
            return load_from_mp4_file(x, self._sr)[0]

        ref = y.read_text(encoding="utf-8")
        sid = normalize_text_only_en(str(Path(*x.parts[-3:-1])))[-255:]
        return Sample(id=sid, data={"load_audio_func": load_audio, "ref": ref})

    def save(self: ESICv1Dataset, path: Path, description="ESICv1Dataset"):
        JsonSaver(description).save(self.to_dict(), path)

    @override
    def _sample(
        self: ESICv1Dataset,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> ESICv1Dataset:
        if rng is None or size == len(self._X) - start:
            return self.slice(start, start + size)
        else:
            data = list(zip(self._X[start:], self._Y[start:]))
            data = rng.choice(data, size=size, replace=False)
            X, Y = zip(*data)
            return ESICv1Dataset(X=list(X), Y=list(Y), sr=self.sr, task=self.task)

    @classmethod
    @override
    def from_dict(cls: type[ESICv1Dataset], data: dict) -> ESICv1Dataset:
        return ESICv1Dataset(
            X=[Path(x) for x in data["X"]],
            Y=[Path(y) for y in data["Y"]],
            sr=data["sr"],
            task=data.get("task", DEFAULT_TASK),
        )

    @staticmethod
    def load(path: Path):
        _, data = load_json(path)
        return ESICv1Dataset.from_dict(data)


__all__ = ["ESICv1Dataset"]
