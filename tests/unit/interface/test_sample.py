from __future__ import annotations

import pytest
import numpy as np

from dataset_loader.interface import Sample


class TestSample:
    @pytest.fixture
    def original(self):
        return Sample(
            id="test_id",
            data={
                "load_audio": np.array([0.0, 1.0, -1.0]),
                "Y": {"diarization": "world"},
            },
        )

    @pytest.fixture
    def original_copy(self):
        return Sample(
            id="test_id",
            data={
                "load_audio": np.array([0.0, 1.0, -1.0]),
                "Y": {"asr": "hello", "diarization": "world"},
            },
        )

    @pytest.fixture
    def other(self):
        return Sample(
            id="different_id",
            data={"load_audio": np.array([0.0, 1.0]), "Y": {"asr": "hi"}},
        )

    def test_equality(self, original: Sample, original_copy: Sample, other: Sample):
        assert original == original_copy
        assert original != other

    def test_hash(self, original: Sample, original_copy: Sample, other: Sample):
        assert hash(original) == hash(original_copy)
        assert hash(original) != hash(other)


__all__ = ["TestSample"]
