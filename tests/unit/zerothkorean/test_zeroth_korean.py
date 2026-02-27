from __future__ import annotations

import pytest

from dataset_loader.interface import Dataset, Sample
from dataset_loader.zerothkorean import ZerothKorean, ZerothKoreanDataset
from dataset_loader.wrapper.asr import ASRDataset

from tests.unit.interface import MixinDatasetTest

SAMPLE_SIZE = 200


class TestZerothKorean(MixinDatasetTest):
    @pytest.fixture
    def zerothkorean(self) -> ZerothKorean:
        return ZerothKorean()

    @pytest.fixture(
        params=(
            {"method": "train", "sample_size": SAMPLE_SIZE},
            {"method": "test", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(
        self,
        zerothkorean: ZerothKorean,
        request: pytest.FixtureRequest,
    ) -> ZerothKoreanDataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: Dataset = getattr(zerothkorean, method)()
        return dataset.sample(sample_size)

    @pytest.fixture
    def asr_dataset(self, dataset: Dataset) -> ASRDataset:
        return ASRDataset(dataset)

    @pytest.fixture
    def samples(self, dataset: Dataset) -> list[Sample]:
        return [sample for sample in dataset]


__all__ = ["TestZerothKorean"]
