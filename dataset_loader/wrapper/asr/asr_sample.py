from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Any, Callable
from dataclasses import dataclass, field
from collections.abc import Mapping, MutableMapping

from dataset_loader.protocol import SampleProtocol
from dataset_loader.base.sample import Sample


@dataclass(frozen=True)
class ASRSample(SampleProtocol):
    sample: SampleProtocol = field(compare=False, hash=True, repr=False)

    @property
    def id(self) -> str:
        return self.sample.id

    @property
    def data(self) -> MutableMapping[str, Any]:
        return self.sample.data

    @property
    def audio(self) -> npt.NDArray[np.float32]:
        if "load_audio_func" not in self.sample.data:
            raise AttributeError("Audio data is not available in this sample")
        return self.sample.data["load_audio_func"]()

    @property
    def ref(self) -> str:
        if "ref" not in self.sample.data:
            raise AttributeError("ASR label is not available in this sample")
        return self.sample.data["ref"]

    @property
    def diarization(self) -> list[dict[str, Any]]:
        if "diarization" not in self.sample.data:
            raise AttributeError("Diarization label is not available in this sample")
        return self.sample.data["diarization"]

    def loaded_audio_sample(self) -> ASRSample:
        data = {**self.data}
        audio = self.audio
        data["load_audio_func"] = lambda: audio
        return ASRSample.create(id=self.id, data=data)

    def to_dict(self) -> MutableMapping[str, Any]:
        return self.sample.to_dict()

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ASRSample:
        return ASRSample(sample=Sample.from_dict(data))

    @staticmethod
    def create(
        id: str,
        *,
        load_audio_func: Callable[[], npt.NDArray[np.float32]] | None = None,
        audio: npt.NDArray[np.float32] | None = None,
        ref: str | None = None,
        diarization: list[Mapping[str, Any]] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> ASRSample:
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
        return ASRSample(sample=sample)


__all__ = ["ASRSample"]
