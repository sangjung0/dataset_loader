from __future__ import annotations

from dataset_loader.abstract import HuggingfaceSnapshot

from dataset_loader.tedlium.segment_tedlium_dataset import SegmentTedliumDataset
from dataset_loader.tedlium.tedlium_dataset import TedliumDataset
from dataset_loader.tedlium.constants import (
    DEFAULT_REPO_ID,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_IGNORE_SET,
    DEFAULT_TASK,
    TedliumTask,
)


class Tedlium(HuggingfaceSnapshot):
    def __init__(
        self: Tedlium,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        dir_name: str | None = None,
        path: str | None = None,
    ):
        super().__init__(repo_id=repo_id, dir_name=dir_name, path=path)

    def train(
        self: Tedlium,
        *,
        segment: bool = False,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[TedliumTask, ...] = DEFAULT_TASK,
        ignore_set: set[str] = DEFAULT_IGNORE_SET,
    ):
        dataset = super().load(
            "train",
            load_options={
                "data_files": {
                    "train": f"{self.path}/release3/train/*.parquet",
                },
            },
        )["train"]

        if segment:
            return SegmentTedliumDataset(
                dataset=dataset, sr=sr, task=task, ignore_set=ignore_set
            )
        return TedliumDataset(dataset=dataset, sr=sr, task=task, ignore_set=ignore_set)

    def validation(
        self,
        *,
        segment: bool = False,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[TedliumTask, ...] = DEFAULT_TASK,
        ignore_set: set[str] = DEFAULT_IGNORE_SET,
    ):
        dataset = super().load(
            "validation",
            load_options={
                "data_files": {
                    "validation": f"{self.path}/release3/validation/*.parquet",
                },
            },
        )["validation"]

        if segment:
            return SegmentTedliumDataset(
                dataset=dataset, sr=sr, task=task, ignore_set=ignore_set
            )
        return TedliumDataset(dataset=dataset, sr=sr, task=task, ignore_set=ignore_set)

    def test(
        self,
        *,
        segment: bool = False,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[TedliumTask, ...] = DEFAULT_TASK,
        ignore_set: set[str] = DEFAULT_IGNORE_SET,
    ):
        dataset = super().load(
            "test",
            load_options={
                "data_files": {
                    "test": f"{self.path}/release3/test/*.parquet",
                },
            },
        )["test"]
        if segment:
            return SegmentTedliumDataset(
                dataset=dataset, sr=sr, task=task, ignore_set=ignore_set
            )
        return TedliumDataset(dataset=dataset, sr=sr, task=task, ignore_set=ignore_set)


__all__ = ["Tedlium"]
