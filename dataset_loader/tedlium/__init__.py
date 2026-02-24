# dataset_loader/tedlium/__init__.py

from dataset_loader.tedlium.segment_tedlium_dataset import SegmentTedliumDataset
from dataset_loader.tedlium.tedlium_dataset import TedliumDataset
from dataset_loader.tedlium.tedlium import Tedlium

__all__ = ["TedliumDataset", "Tedlium", "SegmentTedliumDataset"]
