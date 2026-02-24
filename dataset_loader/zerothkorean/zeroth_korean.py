from __future__ import annotations

from pathlib import Path

from dataset_loader.abstract import HuggingfaceLoader

from dataset_loader.zerothkorean.zeroth_korean_dataset import ZerothKoreanDataset
from dataset_loader.zerothkorean.constants import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_REPO_ID,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TASK,
    ZerothKoreanTask,
)


class ZerothKorean(HuggingfaceLoader):
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        dir_name: str | None = None,
        path: str | Path | None = None,
    ):
        super().__init__(repo_id=repo_id, dir_name=dir_name, path=path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        return super().split_names(config_name)

    def train(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[ZerothKoreanTask, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return ZerothKoreanDataset(
            dataset=super().download(config_name, "train", **kwargs), sr=sr, task=task
        )

    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[ZerothKoreanTask, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return ZerothKoreanDataset(
            dataset=super().download(config_name, "test", **kwargs), sr=sr, task=task
        )


__all__ = ["ZerothKorean"]
