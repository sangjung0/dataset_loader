import numpy as np

from typing import Any, Callable
from typing_extensions import Self
from dataclasses import dataclass, field
from functools import cached_property

from dataset_loader.interface.sample import Sample


@dataclass(frozen=True)
class ASRSample:
    sample: Sample = field(compare=False, hash=False, repr=False)

    @property
    def id(self) -> str:
        return self.sample.id

    @cached_property
    def audio(self) -> np.ndarray | Callable[[], np.ndarray]:
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

    def to_dict(self) -> dict:
        return self.sample.to_dict()

    @staticmethod
    def from_dict(data: dict) -> Self:
        return ASRSample(sample=Sample.from_dict(data))

    @staticmethod
    def create(
        id: str,
        load_audio_func: Callable[[], np.ndarray] | None = None,
        audio: np.ndarray | None = None,
        ref: str | None = None,
        diarization: list[dict[str, Any]] | None = None,
    ) -> Self:
        data = {
            "load_audio_func": (load_audio_func if audio is None else (lambda: audio)),
            "ref": ref,
            "diarization": diarization,
        }

        sample = Sample(id=id, data=data)
        return ASRSample(sample=sample)


__all__ = ["ASRSample"]
