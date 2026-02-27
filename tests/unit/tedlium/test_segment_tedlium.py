from __future__ import annotations

import pytest

from dataset_loader.interface import Dataset, Sample
from dataset_loader.tedlium import SegmentTedlium, SegmentTedliumDataset
from dataset_loader.wrapper.asr import ASRDataset

from tests.unit.interface import MixinDatasetTest

SAMPLE_SIZE = 200


class TestSegmentTedlium(MixinDatasetTest):
    @pytest.fixture
    def segment_tedlium(self) -> SegmentTedlium:
        return SegmentTedlium()

    @pytest.fixture(
        params=(
            {"method": "train", "sample_size": SAMPLE_SIZE},
            {"method": "validation", "sample_size": SAMPLE_SIZE},
            {"method": "test", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(
        self,
        segment_tedlium: SegmentTedlium,
        request: pytest.FixtureRequest,
    ) -> SegmentTedliumDataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: Dataset = getattr(segment_tedlium, method)()
        return dataset.sample(sample_size)

    @pytest.fixture
    def asr_dataset(self, dataset: Dataset) -> ASRDataset:
        return ASRDataset(dataset)

    @pytest.fixture
    def samples(self, dataset: Dataset) -> list[Sample]:
        return [sample for sample in dataset]


__all__ = ["TestSegmentTedlium"]
