from __future__ import annotations

import pytest
import numpy as np

from dataset_loader.interface import Sample

from tests.unit.interface.dummy_dataset import DummyDataset
from tests.unit.interface.mixin_dataset_test import MixinDatasetTest


class TestDataset(MixinDatasetTest):
    @pytest.fixture
    def data(self) -> list[dict]:
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
    def samples(self, data: list[dict]) -> list[Sample]:
        return [Sample(id=d["id"], data=d) for d in data]

    @pytest.fixture(params=tuple({"use_cache": i} for i in range(0, 101, 50)))
    def dataset(
        self, samples: list[Sample], request: pytest.FixtureRequest
    ) -> DummyDataset:
        return DummyDataset(samples=samples, **request.param)


__all__ = ["TestDataset"]
