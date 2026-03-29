from __future__ import annotations

from dataset_loader.wrapper.image_recognition import IRDataset, IRSample

from tests.unit.protocol import MixinDatasetProtocolTest

THREAD_ITER_TEST_SIZE = 20


class MixinIRDatasetTest(MixinDatasetProtocolTest):
    """
    Need: ir_dataset, samples
    """

    def test_ir_thread_iter(self, ir_dataset: IRDataset):
        for idx, sample in enumerate(ir_dataset.thread_iter(num_workers=2, prefetch=4)):
            assert isinstance(sample, IRSample)
            assert hasattr(sample, "raw")

            if idx >= THREAD_ITER_TEST_SIZE - 1:
                break

    def test_ir__len__(self, ir_dataset: IRDataset, ir_samples: list[IRSample]):
        type(self).assert__len__(ir_dataset, ir_samples)

    def test_ir__iter__(self, ir_dataset: IRDataset, ir_samples: list[IRSample]):
        type(self).assert__iter__(ir_dataset, ir_samples)

    def test_ir__getitem__(self, ir_dataset: IRDataset, ir_samples: list[IRSample]):
        type(self).assert__getitem__(ir_dataset, ir_samples)

    def test_ir_select(self, ir_dataset: IRDataset):
        type(self).assert_select(ir_dataset)

    def test_ir_slice(self, ir_dataset: IRDataset):
        type(self).assert_slice(ir_dataset)

    def test_ir_sample(self, ir_dataset: IRDataset, ir_samples: list[IRSample]):
        type(self).assert_sample(ir_dataset, ir_samples)

    def test_ir_get(self, ir_dataset: IRDataset):
        type(self).assert_get(ir_dataset)

    def test_ir__add__(self, ir_dataset: IRDataset):
        type(self).assert__add__(ir_dataset, IRDataset)

    def test_ir_to_dict_and_from_dict(
        self, ir_dataset: IRDataset, ir_samples: list[IRSample]
    ):
        type(self).assert_to_dict_and_from_dict(ir_dataset, ir_samples)

    def test_ir_to_pointer_and_from_pointer(
        self, ir_dataset: IRDataset, ir_samples: list[IRSample]
    ):
        type(self).assert_to_config_and_from_config(ir_dataset, ir_samples, IRDataset)

    def test_ir_sample_identity(self, ir_dataset: IRDataset):
        type(self).assert_sample_identity(ir_dataset)


__all__ = ["MixinIRDatasetTest"]
