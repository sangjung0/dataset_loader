from __future__ import annotations

import pytest
import numpy as np

from typing import Any
from collections.abc import Mapping
from dataset_loader.base import Sample

from tests.unit.base.dummy_dataset import DummyDataset
from tests.unit.base.mixin_dataset_test import MixinDatasetTest


class TestDataset(MixinDatasetTest):
    @pytest.fixture
    def data(self) -> list[dict[str, Any]]:
        return [
            {
                "id": str(i),
                "load_audio_func": np.array([i]),
                "asr": f"text_{i}",
                "diarization": f"dia_{i}",
            }
            for i in range(50)
        ]

    @pytest.fixture
    def samples(self, data: list[Mapping[str, Any]]) -> list[Sample]:
        return [Sample(id=d["id"], data=d) for d in data]

    @pytest.fixture
    def dataset(self, samples: list[Sample]) -> DummyDataset:
        return DummyDataset(samples=samples)


__all__ = ["TestDataset"]
