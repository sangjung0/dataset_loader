from __future__ import annotations

from collections.abc import Sequence

from dataset_loader.abstract import HuggingfaceSnapshot

from dataset_loader.tedlium.segment_tedlium_dataset import SegmentTedliumDataset
from dataset_loader.tedlium.constants import (
    DEFAULT_SEGMENT_REPO_ID,
    DEFAULT_SEGMENT_SAMPLE_RATE,
    DEFAULT_SEGMENT_IGNORE_SET,
)


class SegmentTedlium(HuggingfaceSnapshot):
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_SEGMENT_REPO_ID,
        dir_name: str | None = None,
        path: str | None = None,
    ):
        super().__init__(repo_id=repo_id, dir_name=dir_name, path=path)

    def train(
        self,
        *,
        sr: int = DEFAULT_SEGMENT_SAMPLE_RATE,
        ignore_set: Sequence[str] = DEFAULT_SEGMENT_IGNORE_SET,
    ) -> SegmentTedliumDataset:
        dataset = self.load(
            "train",
            load_options={
                "data_files": {
                    "train": f"{self.path}/release3/train/*.parquet",
                },
            },
        )

        return SegmentTedliumDataset(dataset=dataset, sr=sr, ignore_set=ignore_set)

    def validation(
        self,
        *,
        sr: int = DEFAULT_SEGMENT_SAMPLE_RATE,
        ignore_set: Sequence[str] = DEFAULT_SEGMENT_IGNORE_SET,
    ) -> SegmentTedliumDataset:
        dataset = self.load(
            "validation",
            load_options={
                "data_files": {
                    "validation": f"{self.path}/release3/validation/*.parquet",
                },
            },
        )

        return SegmentTedliumDataset(dataset=dataset, sr=sr, ignore_set=ignore_set)

    def test(
        self,
        *,
        sr: int = DEFAULT_SEGMENT_SAMPLE_RATE,
        ignore_set: Sequence[str] = DEFAULT_SEGMENT_IGNORE_SET,
    ) -> SegmentTedliumDataset:
        dataset = self.load(
            "test",
            load_options={
                "data_files": {
                    "test": f"{self.path}/release3/test/*.parquet",
                },
            },
        )
        return SegmentTedliumDataset(dataset=dataset, sr=sr, ignore_set=ignore_set)


__all__ = ["SegmentTedlium"]


if __name__ == "__main__":
    tedlium = SegmentTedlium()
    train_ds = tedlium.train()
    val_ds = tedlium.validation()
    test_ds = tedlium.test()

    print(train_ds[0])
    print(val_ds[0])
    print(test_ds[0])
