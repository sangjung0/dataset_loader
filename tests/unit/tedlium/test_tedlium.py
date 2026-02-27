from __future__ import annotations

import pytest

from dataset_loader.interface import Dataset, Sample
from dataset_loader.tedlium import Tedlium, TedliumDataset
from dataset_loader.wrapper.asr import ASRDataset, ASRSample

from tests.unit.interface import MixinDatasetTest
from tests.unit.wrapper.asr import MixinASRDatasetTest

SAMPLE_SIZE = 200


class TestTedlium(MixinASRDatasetTest, MixinDatasetTest):
    @pytest.fixture
    def tedlium(self) -> Tedlium:
        return Tedlium()

    @pytest.fixture(
        params=(
            {"method": "train", "sample_size": SAMPLE_SIZE},
            {"method": "dev", "sample_size": SAMPLE_SIZE},
            {"method": "test", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(
        self, tedlium: Tedlium, request: pytest.FixtureRequest
    ) -> TedliumDataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: Dataset = getattr(tedlium, method)()
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


__all__ = ["TestTedlium"]
