from __future__ import annotations

import pytest

from dataset_loader.wrapper.asr import ASRDataset, ASRSample, ASRConcatDataset

from tests.unit.interface import MixinDatasetTest

THREAD_ITER_TEST_SIZE = 20


class MixinASRDatasetTest:
    """
    Need: asr_dataset, samples
    """

    def test_asr_sample_rate(self, asr_dataset: ASRDataset):
        assert isinstance(asr_dataset.sr, int)
        assert asr_dataset.sr > 0

        original_sr = asr_dataset.sr
        sample = asr_dataset[0]
        audio = sample.audio
        dur = len(audio) / original_sr

        assert dur > 0

        changed_sr = original_sr * 2
        asr_dataset.sr = changed_sr
        assert asr_dataset.sr == changed_sr
        sample = asr_dataset[0]
        changed_audio = sample.audio
        changed_dur = len(changed_audio) / changed_sr

        assert abs(dur - changed_dur) < 1e-3

    def test_asr_thread_iter(self, asr_dataset: ASRDataset):
        for idx, sample in enumerate(
            asr_dataset.thread_iter(num_workers=2, prefetch=4)
        ):
            assert isinstance(sample, ASRSample)
            assert hasattr(sample, "audio")
            assert sample.audio is not None

            if idx >= THREAD_ITER_TEST_SIZE - 1:
                break

    def test_asr__len__(self, asr_dataset: ASRDataset, asr_samples: list[ASRSample]):
        MixinDatasetTest.test__len__(self, asr_dataset, asr_samples)

    def test_asr__iter__(self, asr_dataset: ASRDataset, asr_samples: list[ASRSample]):
        MixinDatasetTest.test__iter__(self, asr_dataset, asr_samples)

    def test_asr__getitem__(
        self, asr_dataset: ASRDataset, asr_samples: list[ASRSample]
    ):
        MixinDatasetTest.test__getitem__(self, asr_dataset, asr_samples)

    def test_asr_select(self, asr_dataset: ASRDataset):
        MixinDatasetTest.test_select(self, asr_dataset)

    def test_asr_slice(self, asr_dataset: ASRDataset):
        MixinDatasetTest.test_slice(self, asr_dataset)

    def test_asr_sample(self, asr_dataset: ASRDataset, asr_samples: list[ASRSample]):
        MixinDatasetTest.test_sample(self, asr_dataset, asr_samples)

    def test_asr_get(self, asr_dataset: ASRDataset):
        MixinDatasetTest.test_get(self, asr_dataset)

    def test_asr__add__(self, asr_dataset: ASRDataset):
        MixinDatasetTest.test__add__(self, asr_dataset, ASRConcatDataset)

    def test_asr_to_dict_and_from_dict(
        self, asr_dataset: ASRDataset, asr_samples: list[ASRSample]
    ):
        MixinDatasetTest.test_to_dict_and_from_dict(self, asr_dataset, asr_samples)

    def test_asr_to_pointer_and_from_pointer(
        self, asr_dataset: ASRDataset, asr_samples: list[ASRSample]
    ):
        MixinDatasetTest.test_to_pointer_and_from_pointer(
            self, asr_dataset, asr_samples, ASRDataset
        )

    def test_sample_identity(self, asr_dataset: ASRDataset):
        MixinDatasetTest.test_sample_identity(self, asr_dataset)
