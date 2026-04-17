from __future__ import annotations

import pytest

from dataset_loader import (
    Sample,
    ASRSample,
    SegmentTedlium,
    SegmentTedliumDataset,
    SegmentTedliumDiarizationLabel,
    ASRDataset,
)

from tests.unit.base import MixinDatasetTest
from tests.unit.wrapper.asr import MixinASRDatasetTest

SAMPLE_SIZE = 200


class TestSegmentTedlium(
    MixinASRDatasetTest[str, list[SegmentTedliumDiarizationLabel]], MixinDatasetTest
):
    @pytest.fixture
    def segment_tedlium(self) -> SegmentTedlium:
        return SegmentTedlium()

    @pytest.fixture(
        params=(
            # {"method": "train", "sample_size": SAMPLE_SIZE},
            # {"method": "validation", "sample_size": SAMPLE_SIZE},
            # {"method": "test", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(
        self,
        segment_tedlium: SegmentTedlium,
        request: pytest.FixtureRequest,
    ) -> SegmentTedliumDataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: SegmentTedliumDataset = getattr(segment_tedlium, method)()
        return dataset.sample(sample_size)

    @pytest.fixture
    def samples(self, dataset: SegmentTedliumDataset) -> list[Sample]:
        return [sample for sample in dataset]

    @pytest.fixture
    def asr_dataset(
        self, dataset: SegmentTedliumDataset
    ) -> ASRDataset[str, list[SegmentTedliumDiarizationLabel]]:
        return ASRDataset(dataset=dataset)

    @pytest.fixture
    def asr_samples(
        self, asr_dataset: ASRDataset[str, list[SegmentTedliumDiarizationLabel]]
    ) -> list[ASRSample[str, list[SegmentTedliumDiarizationLabel]]]:
        return [sample for sample in asr_dataset]


__all__ = ["TestSegmentTedlium"]
