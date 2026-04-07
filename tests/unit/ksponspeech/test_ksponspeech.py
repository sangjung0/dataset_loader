from __future__ import annotations

import pytest

from dataset_loader.base import Sample
from dataset_loader.abstract import ASRSample
from dataset_loader.ksponspeech import KSponSpeech, KSponSpeechDataset
from dataset_loader.wrapper.asr import ASRDataset, ASRDatasetProtocol

from tests.unit.base import MixinDatasetTest
from tests.unit.wrapper.asr import MixinASRDatasetTest

SAMPLE_SIZE = 200


class TestKSPonSpeech(MixinASRDatasetTest[str, None], MixinDatasetTest):
    @pytest.fixture
    def ksponspeech(self) -> KSponSpeech:
        return KSponSpeech()

    @pytest.fixture(
        params=(
            # {"method": "train", "sample_size": SAMPLE_SIZE},
            # {"method": "valid", "sample_size": SAMPLE_SIZE},
            {"method": "test", "sample_size": SAMPLE_SIZE},
        )
    )
    def dataset(
        self, ksponspeech: KSponSpeech, request: pytest.FixtureRequest
    ) -> KSponSpeechDataset:
        method = request.param["method"]
        sample_size = request.param["sample_size"]
        dataset: KSponSpeechDataset = getattr(ksponspeech, method)()
        return dataset.sample(sample_size)

    @pytest.fixture
    def samples(self, dataset: KSponSpeechDataset) -> list[Sample]:
        return [sample for sample in dataset]

    @pytest.fixture
    def asr_dataset(self, dataset: KSponSpeechDataset) -> ASRDataset[str, None]:
        if isinstance(dataset, ASRDatasetProtocol):
            return ASRDataset(dataset=dataset)
        raise TypeError("Dataset must be an instance of ASRDatasetProtocol")

    @pytest.fixture
    def asr_samples(
        self, asr_dataset: ASRDataset[str, None]
    ) -> list[ASRSample[str, None]]:
        return [sample for sample in asr_dataset]


__all__ = ["TestKSPonSpeech"]
