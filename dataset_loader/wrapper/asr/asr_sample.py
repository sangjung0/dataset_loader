from __future__ import annotations

import numpy as np

from typing import Any, Callable
from dataclasses import dataclass, field
from functools import cached_property
from collections.abc import Mapping, MutableMapping

from dataset_loader.protocol import SampleProtocol
from dataset_loader.base.sample import Sample


@dataclass(frozen=True)
class ASRSample(SampleProtocol):
    sample: SampleProtocol = field(compare=False, hash=False, repr=False)

    @property
    def id(self) -> str:
        return self.sample.id

    @property
    def data(self) -> MutableMapping[str, Any]:
        return self.sample.data

    @cached_property
    def audio(self) -> np.ndarray:
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

    def __hash__(self) -> int:
        return hash(self.sample)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ASRSample):
            return False
        return self.sample == other.sample

    def to_dict(self) -> MutableMapping[str, Any]:
        return self.sample.to_dict()

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ASRSample:
        return ASRSample(sample=Sample.from_dict(data))

    @staticmethod
    def create(
        id: str,
        load_audio_func: Callable[[], np.ndarray] | None = None,
        audio: np.ndarray | None = None,
        ref: str | None = None,
        diarization: list[Mapping[str, Any]] | None = None,
    ) -> ASRSample:
        data = {
            "load_audio_func": (load_audio_func if audio is None else (lambda: audio)),
            "ref": ref,
            "diarization": diarization,
        }

        sample = Sample(id=id, data=data)
        return ASRSample(sample=sample)


__all__ = ["ASRSample"]
