from __future__ import annotations

import pytest
import numpy as np

from dataset_loader.interface import Dataset, Sample


class MixinDatasetTest:
    """
    Need: dataset, samples
    """

    def test__len__(self, dataset: Dataset, samples: list[Sample]):
        assert len(dataset) == len(samples)

    def test__iter__(self, dataset: Dataset, samples: list[Sample]):
        for i, sample in enumerate(dataset):
            assert sample == samples[i]

    def test__getitem__(self, dataset: Dataset, samples: list[Sample]):
        # int
        for i in range(len(samples)):
            assert dataset[i] == samples[i]

        # slice
        sl = slice(len(samples) // 3, len(samples) // 2, 2)
        sliced_dataset = dataset[sl]
        sliced_dataset_list = [sample for sample in sliced_dataset]
        assert isinstance(sliced_dataset, Dataset)
        assert sliced_dataset_list == samples[sl]

        # fancy indexing
        indices = [i for i in range(0, len(samples), 5)]
        indexed_dataset = dataset[indices]
        indexed_dataset_list = [sample for sample in indexed_dataset]
        assert isinstance(indexed_dataset, Dataset)
        assert indexed_dataset_list == [samples[i] for i in indices]

    def test_select(self, dataset: Dataset):
        indices = [i for i in range(0, len(dataset), 5)]
        selected_dataset = dataset.select(indices)
        validate_dataset = dataset[indices]
        assert isinstance(selected_dataset, Dataset)
        assert [sample for sample in selected_dataset] == [
            sample for sample in validate_dataset
        ]

    def test_slice(self, dataset: Dataset):
        sl = slice(len(dataset) // 3, len(dataset) // 2, 2)
        sliced_dataset = dataset.slice(sl.start, sl.stop, sl.step)
        validate_dataset = dataset[sl]
        assert isinstance(sliced_dataset, Dataset)
        assert [sample for sample in sliced_dataset] == [
            sample for sample in validate_dataset
        ]

    def test_sample(self, dataset: Dataset, samples: list[Sample]):
        # without rng
        size = len(dataset) // 5
        start = np.random.randint(0, len(dataset) - size)
        sampled_dataset = dataset.sample(size=size, start=start)
        assert isinstance(sampled_dataset, Dataset)
        assert [s for s in sampled_dataset] == samples[start : start + size]

        with pytest.raises(IndexError):
            dataset.sample(size=size, start=-1)
        with pytest.raises(IndexError):
            dataset.sample(size=size, start=len(samples) + 1)

        # with rng
        rng = np.random.default_rng(seed=42)
        size = len(samples) // 5
        start = np.random.randint(0, len(samples) - size)
        sampled_dataset = dataset.sample(size=size, start=start, rng=rng)
        assert isinstance(sampled_dataset, Dataset)
        assert len(sampled_dataset) == size

    def test_get(self, dataset: Dataset):
        for i in range(len(dataset)):
            assert dataset.get(i) == dataset[i]
        with pytest.raises(IndexError):
            dataset.get(len(dataset))
        assert dataset.get(-1) == dataset[-1]

    def test__add__(self, dataset: Dataset):
        l = len(dataset) // 3
        data_1 = dataset[:l]
        data_2 = dataset[l : 2 * l]
        data_3 = dataset[2 * l :]

        # other: Dataset
        concat_dataset = data_1 + data_2
        assert concat_dataset.__class__.__name__ == "ConcatDataset"
        assert len(concat_dataset) == len(data_1) + len(data_2)
        assert [sample for sample in concat_dataset] == [
            sample for sample in data_1
        ] + [sample for sample in data_2]

        # other: ConcatDataset
        concat_other = concat_dataset
        concat_dataset = concat_other + data_3
        assert concat_dataset.__class__.__name__ == "ConcatDataset"
        assert len(concat_dataset) == len(concat_other) + len(data_3)
        assert [sample for sample in concat_dataset] == (
            [sample for sample in concat_other] + [sample for sample in data_3]
        )

    def test_to_dict_and_from_dict(self, dataset: Dataset, samples: list[Sample]):
        d = dataset.to_dict()
        restored = type(dataset).from_dict(d)
        assert isinstance(restored, type(dataset))
        assert [sample for sample in restored] == samples

    def test_to_pointer_and_from_pointer(self, dataset: Dataset, samples: list[Sample]):
        pointer = dataset.to_pointer()
        restored = type(dataset).from_pointer(pointer)
        restored_2 = Dataset.from_pointer(pointer)
        assert isinstance(restored, type(dataset))
        assert [sample for sample in restored] == samples
        assert isinstance(restored_2, type(dataset))
        assert [sample for sample in restored_2] == samples

    def test_sample_identity(self, dataset: Dataset):
        assert len(dataset) == len(set(sample.id for sample in dataset))


__all__ = ["MixinDatasetTest"]
