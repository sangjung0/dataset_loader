from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Any
from typing_extensions import Self, override
from dataclasses import dataclass, field
from collections.abc import Mapping, MutableMapping, Callable

from dataset_loader.protocol import SampleProtocol
from dataset_loader.base.sample import Sample


@dataclass(frozen=True)
class ASRSample(SampleProtocol):
    sample: SampleProtocol = field(compare=False, hash=True, repr=False)

    @property
    @override
    def id(self) -> str:
        return self.sample.id

    @property
    @override
    def data(self) -> Mapping[str, Any]:
        return self.sample.data

    @property
    def audio(self) -> npt.NDArray[np.float32]:
        if "load_audio_func" not in self.sample.data:
            raise AttributeError("Audio data is not available in this sample")
        audio: npt.NDArray[np.float32] = self.sample.data["load_audio_func"]()
        return audio

    @property
    def ref(self) -> str:
        if "ref" not in self.sample.data:
            raise AttributeError("ASR label is not available in this sample")
        ref: str = self.sample.data["ref"]
        return ref

    @property
    def diarization(self) -> list[dict[str, Any]]:
        if "diarization" not in self.sample.data:
            raise AttributeError("Diarization label is not available in this sample")
        diarization: list[dict[str, Any]] = self.sample.data["diarization"]
        return diarization

    def loaded_audio_sample(self) -> ASRSample:
        data = {**self.data}
        audio = self.audio
        data["load_audio_func"] = lambda: audio
        return ASRSample.create(id=self.id, data=data)

    def to_dict(self) -> MutableMapping[str, Any]:
        return self.sample.to_dict()

    @classmethod
    @override
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(sample=Sample.from_dict(data))

    @classmethod
    def create(
        cls,
        id: str,
        *,
        load_audio_func: Callable[[], npt.NDArray[np.float32]] | None = None,
        audio: npt.NDArray[np.float32] | None = None,
        ref: str | None = None,
        diarization: list[Mapping[str, Any]] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> Self:
        if data is None:
            data = {}
        else:
            data = dict(data)

        if "load_audio_func" not in data:
            if load_audio_func is None and audio is None:
                raise ValueError("Either load_audio_func or audio must be provided")
            data["load_audio_func"] = (
                (lambda: audio) if load_audio_func is None else load_audio_func
            )

        if "ref" not in data:
            if ref is None:
                raise ValueError("ref must be provided")
            data["ref"] = ref

        if "diarization" not in data and diarization is not None:
            data["diarization"] = diarization

        sample = Sample(id=id, data=data)
        return cls(sample=sample)


__all__ = ["ASRSample"]
