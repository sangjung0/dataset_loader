from __future__ import annotations

import pytest

from dataset_loader.interface import Dataset, Sample
from dataset_loader.ksponspeech import KSPonSpeech, KSPonSpeechDataset
from dataset_loader.wrapper.asr import ASRDataset, ASRSample

from tests.unit.interface import MixinDatasetTest
from tests.unit.wrapper.asr import MixinASRDatasetTest

SAMPLE_SIZE = 200


class TestKSPonSpeech(MixinASRDatasetTest, MixinDatasetTest):
    @pytest.fixture
    def ksponspeech(self) -> KSPonSpeech:
        return KSPonSpeech()

    @pytest.fixture(
        params=(
            {"method": "train", "sample_size": SAMPLE_SIZE},
            {"method": "valid", "sample_size": SAMPLE_SIZE},
            {"method": "test", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(
        self, ksponspeech: KSPonSpeech, request: pytest.FixtureRequest
    ) -> KSPonSpeechDataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: Dataset = getattr(ksponspeech, method)()
        return dataset.sample(sample_size)

    @pytest.fixture
    def samples(self, dataset: Dataset) -> list[Sample]:
        return [sample for sample in dataset]

    @pytest.fixture
    def asr_dataset(self, dataset: Dataset) -> ASRDataset:
        return ASRDataset(dataset)

    @pytest.fixture
    def asr_samples(self, asr_dataset: ASRDataset) -> list[ASRSample]:
        return [sample for sample in asr_dataset]


__all__ = ["TestKSPonSpeech"]
