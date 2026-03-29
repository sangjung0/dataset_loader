from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Any, Callable
from dataclasses import dataclass, field
from collections.abc import Mapping, MutableMapping

from dataset_loader.protocol import SampleProtocol
from dataset_loader.base.sample import Sample


@dataclass(frozen=True)
class IRSample(SampleProtocol):
    sample: SampleProtocol = field(compare=False, hash=True, repr=False)

    @property
    def id(self) -> str:
        return self.sample.id

    @property
    def data(self) -> MutableMapping[str, Any]:
        return self.sample.data

    @property
    def raw(self) -> npt.NDArray[np.float32]:
        if "load_raw" not in self.sample.data:
            raise AttributeError("Raw image data is not available in this sample")
        return self.sample.data["load_raw"]()

    @property
    def label(self) -> str | dict[str, Any] | list:
        if "label" not in self.sample.data:
            raise AttributeError("Label is not available in this sample")
        return self.sample.data["label"]

    def loaded_ir_sample(self) -> IRSample:
        data = {**self.data}
        raw = self.raw
        data["load_raw"] = lambda: raw
        return IRSample.create(id=self.id, data=data)

    def to_dict(self) -> MutableMapping[str, Any]:
        return self.sample.to_dict()

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> IRSample:
        return IRSample(sample=Sample.from_dict(data))

    @staticmethod
    def create(
        id: str,
        *,
        load_raw: Callable[[], npt.NDArray[np.float32]] | None = None,
        raw: npt.NDArray[np.float32] | None = None,
        label: str | dict[str, Any] | list | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> IRSample:
        if data is None:
            data = {}
        else:
            data = dict(data)

        if "load_raw" not in data:
            if load_raw is None and raw is None:
                raise ValueError("Either load_raw or raw must be provided")
            data["load_raw"] = (lambda: raw) if load_raw is None else load_raw

        if "label" not in data:
            if label is None:
                raise ValueError("Label must be provided")
            data["label"] = label

        sample = Sample(id=id, data=data)
        return IRSample(sample=sample)


__all__ = ["IRSample"]
