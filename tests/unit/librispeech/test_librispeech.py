from __future__ import annotations

import pytest

from dataset_loader.interface import Dataset, Sample
from dataset_loader.librispeech import LibriSpeech, LibriSpeechDataset
from dataset_loader.wrapper.asr import ASRDataset

from tests.unit.interface import MixinDatasetTest

SAMPLE_SIZE = 200


class TestLibriSpeech(MixinDatasetTest):
    @pytest.fixture
    def librispeech(self) -> LibriSpeech:
        return LibriSpeech()

    @pytest.fixture(
        params=(
            {"method": "test-clean", "sample_size": SAMPLE_SIZE},
            {"method": "test-other", "sample_size": SAMPLE_SIZE},
            {"method": "train-clean-100", "sample_size": SAMPLE_SIZE},
            {"method": "train-clean-360", "sample_size": SAMPLE_SIZE},
            {"method": "train-other-500", "sample_size": SAMPLE_SIZE},
            {"method": "dev-clean", "sample_size": SAMPLE_SIZE},
            {"method": "dev-other", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(
        self, librispeech: LibriSpeech, request: pytest.FixtureRequest
    ) -> LibriSpeechDataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: Dataset = getattr(librispeech, method.replace("-", "_"))()
        return dataset.sample(sample_size)

    @pytest.fixture
    def asr_dataset(self, dataset: Dataset) -> ASRDataset:
        return ASRDataset(dataset)

    @pytest.fixture
    def samples(self, dataset: Dataset) -> list[Sample]:
        return [sample for sample in dataset]


__all__ = ["TestLibriSpeech"]
