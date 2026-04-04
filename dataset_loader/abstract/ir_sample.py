from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Any
from typing_extensions import Self
from dataclasses import dataclass
from collections.abc import Mapping, Callable

from dataset_loader.base.sample import Sample


@dataclass(frozen=True, slots=True)
class IRSample(Sample):
    @property
    def raw(self) -> npt.NDArray[np.uint8]:
        if "load_raw" not in self.data:
            raise AttributeError("Raw image data is not available in this sample")
        image: npt.NDArray[np.uint8] = self.data["load_raw"]()
        return image

    @property
    def label(self) -> str | dict[str, Any] | list[Any]:
        if "label" not in self.data:
            raise AttributeError("Label is not available in this sample")
        label: str | dict[str, Any] | list[Any] = self.data["label"]
        return label

    def loaded_ir_sample(self) -> IRSample:
        data = {**self.data}
        raw = self.raw
        data["load_raw"] = lambda: raw
        return IRSample.create(id=self.id, data=data)

    @classmethod
    def create(
        cls,
        id: str,
        *,
        load_raw: Callable[[], npt.NDArray[np.uint8]] | None = None,
        raw: npt.NDArray[np.uint8] | None = None,
        label: str | dict[str, Any] | list[Any] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> Self:
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

        return cls(id=id, data=data)

    def __str__(self) -> str:
        return f"IRSample(id={self.id}, label={self.label})"

    def __repr__(self) -> str:
        return f"IRSample(id={self.id}, label={self.label})"


__all__ = ["IRSample"]
