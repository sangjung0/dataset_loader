from __future__ import annotations

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

    def train(
        self: SegmentTedlium,
        *,
        sr: int = DEFAULT_SEGMENT_SAMPLE_RATE,
        use_cache: int = 0,
        ignore_set: set[str] = DEFAULT_SEGMENT_IGNORE_SET,
    ):
        dataset = super().load(
            "train",
            load_options={
                "data_files": {
                    "train": f"{self.path}/release3/train/*.parquet",
                },
            },
        )["train"]

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
        dataset = super().load(
            "validation",
            load_options={
                "data_files": {
                    "validation": f"{self.path}/release3/validation/*.parquet",
                },
            },
        )["validation"]

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
        dataset = super().load(
            "test",
            load_options={
                "data_files": {
                    "test": f"{self.path}/release3/test/*.parquet",
                },
            },
        )["test"]
        return SegmentTedliumDataset(
            dataset=dataset, sr=sr, use_cache=use_cache, ignore_set=ignore_set
        )


__all__ = ["SegmentTedlium"]
