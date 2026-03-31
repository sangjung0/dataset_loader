from __future__ import annotations

import pytest

from dataset_loader.base import Sample
from dataset_loader.librispeech import LibriSpeech, LibriSpeechDataset
from dataset_loader.wrapper.asr import ASRDataset, ASRSample, ASRDatasetProtocol

from tests.unit.base import MixinDatasetTest
from tests.unit.wrapper.asr import MixinASRDatasetTest

SAMPLE_SIZE = 200


class TestLibriSpeech(MixinASRDatasetTest, MixinDatasetTest):
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
        dataset: LibriSpeechDataset = getattr(librispeech, method.replace("-", "_"))()
        return dataset.sample(sample_size)

    @pytest.fixture
    def samples(self, dataset: LibriSpeechDataset) -> list[Sample]:
        return [sample for sample in dataset]

    @pytest.fixture
    def asr_dataset(self, dataset: LibriSpeechDataset) -> ASRDataset:
        if isinstance(dataset, ASRDatasetProtocol):
            return ASRDataset(dataset=dataset)
        raise TypeError("Dataset must be an instance of ASRDatasetProtocol")

    @pytest.fixture
    def asr_samples(self, asr_dataset: ASRDataset) -> list[ASRSample]:
        return [sample for sample in asr_dataset]


__all__ = ["TestLibriSpeech"]
