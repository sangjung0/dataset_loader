# dataset_loader/interface/asr/__init__.py

from dataset_loader.wrapper.image_recognition.ir_dataset import IRDataset
from dataset_loader.wrapper.image_recognition.ir_sample import IRSample

__all__ = [
    "IRDataset",
    "IRSample",
]
