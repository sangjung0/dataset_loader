from __future__ import annotations

from typing_extensions import override
from datasets import Audio

from dataset_loader.abstract import HuggingfaceSnapshot

from dataset_loader.tedlium.segment_tedlium_dataset import SegmentTedliumDataset
from dataset_loader.tedlium.constants import (
    DEFAULT_SEGMENT_REPO_ID,
    DEFAULT_SEGMENT_SAMPLE_RATE,
    DEFAULT_SEGMENT_IGNORE_SET,
)


class SegmentTedlium(HuggingfaceSnapshot):
    def __init__(
        self: SegmentTedlium,
        *,
        repo_id: str = DEFAULT_SEGMENT_REPO_ID,
        dir_name: str | None = None,
        path: str | None = None,
    ):
        super().__init__(repo_id=repo_id, dir_name=dir_name, path=path)

    @override
    def load(
        self: SegmentTedlium,
        name: str,
        load_options: dict | None = None,
    ):
        dataset_dict = super().load(name, load_options=load_options)
        dataset = dataset_dict[name]
        return dataset

    def train(
        self: SegmentTedlium,
        *,
        sr: int = DEFAULT_SEGMENT_SAMPLE_RATE,
        use_cache: int = 0,
        ignore_set: set[str] = DEFAULT_SEGMENT_IGNORE_SET,
    ):
        dataset = self.load(
            "train",
            load_options={
                "data_files": {
                    "train": f"{self.path}/release3/train/*.parquet",
                },
            },
        )

        return SegmentTedliumDataset(
            dataset=dataset, sr=sr, use_cache=use_cache, ignore_set=ignore_set
        )

    def validation(
        self,
        *,
        sr: int = DEFAULT_SEGMENT_SAMPLE_RATE,
        use_cache: int = 0,
        ignore_set: set[str] = DEFAULT_SEGMENT_IGNORE_SET,
    ):
        dataset = self.load(
            "validation",
            load_options={
                "data_files": {
                    "validation": f"{self.path}/release3/validation/*.parquet",
                },
            },
        )

        return SegmentTedliumDataset(
            dataset=dataset, sr=sr, use_cache=use_cache, ignore_set=ignore_set
        )

    def test(
        self,
        *,
        sr: int = DEFAULT_SEGMENT_SAMPLE_RATE,
        use_cache: int = 0,
        ignore_set: set[str] = DEFAULT_SEGMENT_IGNORE_SET,
    ):
        dataset = self.load(
            "test",
            load_options={
                "data_files": {
                    "test": f"{self.path}/release3/test/*.parquet",
                },
            },
        )
        return SegmentTedliumDataset(
            dataset=dataset, sr=sr, use_cache=use_cache, ignore_set=ignore_set
        )


__all__ = ["SegmentTedlium"]
