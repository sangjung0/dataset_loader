from __future__ import annotations

import pytest

from dataset_loader.interface import Dataset, Sample
from dataset_loader.ksponspeech import KSPonSpeech, KSPonSpeechDataset
from dataset_loader.wrapper.asr import ASRDataset

from tests.unit.interface import MixinDatasetTest

SAMPLE_SIZE = 200


class TestKSPonSpeech(MixinDatasetTest):
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
    def asr_dataset(self, dataset: Dataset) -> ASRDataset:
        return ASRDataset(dataset)

    @pytest.fixture
    def samples(self, dataset: Dataset) -> list[Sample]:
        return [sample for sample in dataset]


__all__ = ["TestKSPonSpeech"]
