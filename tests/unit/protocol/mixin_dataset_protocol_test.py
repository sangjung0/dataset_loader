from __future__ import annotations

import pytest
import numpy as np

from collections.abc import Sequence

from dataset_loader.protocol import DatasetProtocol, SampleProtocol


class MixinDatasetProtocolTest:
    """
    Need: dataset, samples
    """

    @staticmethod
    def assert__len__(dataset: DatasetProtocol, samples: Sequence[SampleProtocol]):
        assert len(dataset) == len(samples)

    @staticmethod
    def assert__iter__(dataset: DatasetProtocol, samples: Sequence[SampleProtocol]):
        for i, sample in enumerate(dataset):
            assert sample == samples[i]

    @staticmethod
    def assert__getitem__(dataset: DatasetProtocol, samples: Sequence[SampleProtocol]):
        # int
        for i in range(len(samples)):
            assert dataset[i] == samples[i]

        # slice
        sl = slice(len(samples) // 3, len(samples) // 2, 2)
        sliced_dataset = dataset[sl]
        sliced_dataset_list = [sample for sample in sliced_dataset]
        assert isinstance(sliced_dataset, dataset.__class__)
        assert sliced_dataset_list == samples[sl]

        # fancy indexing
        indices = [i for i in range(0, len(samples), 5)]
        indexed_dataset = dataset[indices]
        indexed_dataset_list = [sample for sample in indexed_dataset]
        assert isinstance(indexed_dataset, dataset.__class__)
        assert indexed_dataset_list == [samples[i] for i in indices]

    @staticmethod
    def assert_select(dataset: DatasetProtocol):
        indices = [i for i in range(0, len(dataset), 5)]
        selected_dataset = dataset.select(indices)
        validate_dataset = dataset[indices]
        assert isinstance(selected_dataset, dataset.__class__)
        assert [sample for sample in selected_dataset] == [
            sample for sample in validate_dataset
        ]

    @staticmethod
    def assert_slice(dataset: DatasetProtocol):
        sl = slice(len(dataset) // 3, len(dataset) // 2, 2)
        sliced_dataset = dataset.slice(sl.start, sl.stop, sl.step)
        validate_dataset = dataset[sl]
        assert isinstance(sliced_dataset, dataset.__class__)
        assert [sample for sample in sliced_dataset] == [
            sample for sample in validate_dataset
        ]

    @staticmethod
    def assert_sample(dataset: DatasetProtocol, samples: Sequence[SampleProtocol]):
        # without rng
        size = len(dataset) // 5
        start = np.random.randint(0, len(dataset) - size)
        sampled_dataset = dataset.sample(size=size, start=start)
        assert isinstance(sampled_dataset, dataset.__class__)
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
        assert isinstance(sampled_dataset, dataset.__class__)
        assert len(sampled_dataset) == size

    @staticmethod
    def assert_get(dataset: DatasetProtocol):
        for i in range(len(dataset)):
            assert dataset.get(i) == dataset[i]
        with pytest.raises(IndexError):
            dataset.get(len(dataset))
        assert dataset.get(-1) == dataset[-1]

    @staticmethod
    def assert__add__(
        dataset: DatasetProtocol, ConcatDatasetType: type[DatasetProtocol]
    ):
        length = len(dataset) // 3
        data_1 = dataset[:length]
        data_2 = dataset[length : 2 * length]
        data_3 = dataset[2 * length :]

        # other: DatasetProtocol
        concat_dataset = data_1 + data_2
        assert isinstance(concat_dataset, ConcatDatasetType)
        assert len(concat_dataset) == len(data_1) + len(data_2)
        assert [sample for sample in concat_dataset] == [
            sample for sample in data_1
        ] + [sample for sample in data_2]

        # other: ConcatDatasetType
        concat_other = concat_dataset
        concat_dataset = concat_other + data_3
        assert isinstance(concat_dataset, ConcatDatasetType)
        assert len(concat_dataset) == len(concat_other) + len(data_3)
        assert [sample for sample in concat_dataset] == (
            [sample for sample in concat_other] + [sample for sample in data_3]
        )

    @staticmethod
    def assert_to_dict_and_from_dict(
        dataset: DatasetProtocol, samples: Sequence[SampleProtocol]
    ):
        d = dataset.to_dict()
        restored = type(dataset).from_dict(d)
        assert isinstance(restored, type(dataset))
        assert [sample for sample in restored] == samples

    @staticmethod
    def assert_to_config_and_from_config(
        dataset: DatasetProtocol,
        samples: Sequence[SampleProtocol],
        DatasetProtocolType: type[DatasetProtocol],
    ):
        pointer = dataset.__getstate__()
        restored = type(dataset).__setstate__(pointer)
        restored_2 = DatasetProtocolType.__setstate__(pointer)
        assert isinstance(restored, type(dataset))
        assert [sample for sample in restored] == samples
        assert isinstance(restored_2, type(dataset))
        assert [sample for sample in restored_2] == samples

    @staticmethod
    def assert_sample_identity(dataset: DatasetProtocol):
        assert len(dataset) == len(set(sample.id for sample in dataset))


__all__ = ["MixinDatasetProtocolTest"]
