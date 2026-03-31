from __future__ import annotations

import pytest

from dataset_loader.base import Sample
from dataset_loader.esic import ESICv1, ESICv1Dataset
from dataset_loader.wrapper.asr import ASRDataset, ASRSample, ASRDatasetProtocol

from tests.unit.base import MixinDatasetTest
from tests.unit.wrapper.asr import MixinASRDatasetTest

SAMPLE_SIZE = 200


class TestESICv1(MixinDatasetTest, MixinASRDatasetTest):
    @pytest.fixture
    def esic_v1(self) -> ESICv1:
        return ESICv1()

    @pytest.fixture(
        params=(
            {"method": "dev", "sample_size": SAMPLE_SIZE},
            {"method": "dev2", "sample_size": SAMPLE_SIZE},
            {"method": "test", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(self, esic_v1: ESICv1, request: pytest.FixtureRequest) -> ESICv1Dataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: ESICv1Dataset = getattr(esic_v1, method)()
        return dataset.sample(sample_size)

    @pytest.fixture
    def samples(self, dataset: ESICv1Dataset) -> list[Sample]:
        return [sample for sample in dataset]

    @pytest.fixture
    def asr_dataset(self, dataset: ESICv1Dataset) -> ASRDataset:
        if isinstance(dataset, ASRDatasetProtocol):
            return ASRDataset(dataset=dataset)
        raise TypeError("Dataset must be an instance of ASRDatasetProtocol")

    @pytest.fixture
    def asr_samples(self, asr_dataset: ASRDataset) -> list[ASRSample]:
        return [sample for sample in asr_dataset]


__all__ = ["TestESICv1"]
