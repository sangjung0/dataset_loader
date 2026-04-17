from __future__ import annotations

import pytest

from dataset_loader import (
    Sample,
    ASRSample,
    ASRDataset,
    ZerothKorean,
    ZerothKoreanDataset,
    ZerothKoreanDiarizationLabel,
)

from tests.unit.base import MixinDatasetTest
from tests.unit.wrapper.asr import MixinASRDatasetTest

SAMPLE_SIZE = 200


class TestZerothKorean(
    MixinASRDatasetTest[str, list[ZerothKoreanDiarizationLabel]], MixinDatasetTest
):
    @pytest.fixture
    def zerothkorean(self) -> ZerothKorean:
        return ZerothKorean()

    @pytest.fixture(
        params=(
            # {"method": "train", "sample_size": SAMPLE_SIZE},
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
        dataset: ZerothKoreanDataset = getattr(zerothkorean, method)()
        return dataset.sample(sample_size)

    @pytest.fixture
    def samples(self, dataset: ZerothKoreanDataset) -> list[Sample]:
        return [sample for sample in dataset]

    @pytest.fixture
    def asr_dataset(
        self, dataset: ZerothKoreanDataset
    ) -> ASRDataset[str, list[ZerothKoreanDiarizationLabel]]:
        return ASRDataset(dataset=dataset)

    @pytest.fixture
    def asr_samples(
        self, asr_dataset: ASRDataset[str, list[ZerothKoreanDiarizationLabel]]
    ) -> list[ASRSample[str, list[ZerothKoreanDiarizationLabel]]]:
        return [sample for sample in asr_dataset]


__all__ = ["TestZerothKorean"]
