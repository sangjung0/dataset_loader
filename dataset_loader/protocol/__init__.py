# dataset_loader/protocol/__init__.py

from dataset_loader.protocol.concat_dataset_protocol import ConcatDatasetProtocol
from dataset_loader.protocol.dataset_protocol import DatasetProtocol
from dataset_loader.protocol.sample_protocol import SampleProtocol

__all__ = ["DatasetProtocol", "SampleProtocol", "ConcatDatasetProtocol"]
