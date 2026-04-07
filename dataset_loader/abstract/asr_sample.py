from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Any, TypeVar, TypedDict, Generic, cast
from typing_extensions import Self, ReadOnly
from dataclasses import dataclass
from collections.abc import Mapping, Callable

from dataset_loader.base.sample import Sample

RefT = TypeVar("RefT", covariant=True)
DiarizationT = TypeVar("DiarizationT", covariant=True)


class ASRSampleData(TypedDict, Generic[RefT, DiarizationT]):
    load_audio_func: ReadOnly[Callable[[], npt.NDArray[np.float32]]]
    ref: ReadOnly[RefT]
    diarization: ReadOnly[DiarizationT]


@dataclass(frozen=True, slots=True)
class ASRSample(Sample, Generic[RefT, DiarizationT]):
    @property
    def audio(self) -> npt.NDArray[np.float32]:
        if "load_audio_func" not in self.data:
            raise AttributeError("Audio data is not available in this sample")
        return cast(npt.NDArray[np.float32], self.data["load_audio_func"]())

    @property
    def ref(self) -> RefT:
        if "ref" not in self.data:
            raise AttributeError("ASR label is not available in this sample")
        return cast(RefT, self.data["ref"])

    @property
    def diarization(self) -> DiarizationT:
        if "diarization" not in self.data:
            raise AttributeError("Diarization label is not available in this sample")
        return cast(DiarizationT, self.data["diarization"])

    def loaded_audio_sample(self) -> ASRSample[RefT, DiarizationT]:
        data = {**self.data}
        audio = self.audio
        data["load_audio_func"] = lambda: audio
        return ASRSample.create(id=self.id, data=data)

    @classmethod
    def create(
        cls,
        id: str,
        *,
        load_audio_func: Callable[[], npt.NDArray[np.float32]] | None = None,
        audio: npt.NDArray[np.float32] | None = None,
        ref: Any = None,
        diarization: Any = None,
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

        return cls(id=id, data=data)


__all__ = ["ASRSample", "ASRSampleData"]
