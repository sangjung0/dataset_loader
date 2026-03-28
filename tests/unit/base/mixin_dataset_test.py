from __future__ import annotations

from dataset_loader.base import Dataset, Sample, ConcatDataset

from tests.unit.protocol import MixinDatasetProtocolTest


class MixinDatasetTest(MixinDatasetProtocolTest):
    def test_dataset__len__(self, dataset: Dataset, samples: list[Sample]):
        type(self).assert__len__(dataset, samples)

    def test_dataset__iter__(self, dataset: Dataset, samples: list[Sample]):
        type(self).assert__iter__(dataset, samples)

    def test_dataset__getitem__(self, dataset: Dataset, samples: list[Sample]):
        type(self).assert__getitem__(dataset, samples)

    def test_dataset_select(self, dataset: Dataset):
        type(self).assert_select(dataset)

    def test_dataset_slice(self, dataset: Dataset):
        type(self).assert_slice(dataset)

    def test_dataset_sample(self, dataset: Dataset, samples: list[Sample]):
        type(self).assert_sample(dataset, samples)

    def test_dataset_get(self, dataset: Dataset):
        type(self).assert_get(dataset)

    def test_dataset__add__(self, dataset: Dataset):
        type(self).assert__add__(dataset, ConcatDataset)

    def test_dataset_to_dict_and_from_dict(
        self, dataset: Dataset, samples: list[Sample]
    ):
        type(self).assert_to_dict_and_from_dict(dataset, samples)

    def test_dataset_to_config_and_from_config(
        self, dataset: Dataset, samples: list[Sample]
    ):
        type(self).assert_to_config_and_from_config(dataset, samples, Dataset)

    def test_dataset_sample_identity(self, dataset: Dataset):
        type(self).assert_sample_identity(dataset)


__all__ = ["MixinDatasetTest"]
