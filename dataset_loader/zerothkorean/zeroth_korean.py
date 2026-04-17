# pyright: reportMissingTypeStubs=false

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing_extensions import override
from collections.abc import Sequence
from datasets import Dataset, IterableDatasetDict

from dataset_loader.abstract import HuggingfaceLoader

from dataset_loader.zerothkorean.zeroth_korean_dataset import ZerothKoreanDataset
from dataset_loader.zerothkorean.constants import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_REPO_ID,
    DEFAULT_SAMPLE_RATE,
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

    @override
    def download(
        self,
        *,
        config_name: str = DEFAULT_CONFIG_NAME,
        split_name: str | Sequence[str] | None = None,
        local_files_only: bool = False,
    ) -> Dataset | IterableDatasetDict:
        return super().download(
            config_name=config_name,
            split_name=split_name,
            local_files_only=local_files_only,
        )

    def train(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        **kwargs: Any,
    ) -> ZerothKoreanDataset:
        dataset = self.load(config_name=config_name, split_name="train", **kwargs)
        if isinstance(dataset, (IterableDatasetDict, list)):
            return ZerothKoreanDataset(dataset=dataset[0], sr=sr)  # type: ignore[return-value, unused-ignore]
        return ZerothKoreanDataset(dataset=dataset, sr=sr)

    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        **kwargs: Any,
    ) -> ZerothKoreanDataset:
        dataset = self.load(config_name=config_name, split_name="test", **kwargs)
        if isinstance(dataset, (IterableDatasetDict, list)):
            return ZerothKoreanDataset(dataset=dataset[0], sr=sr)  # type: ignore[return-value, unused-ignore]
        return ZerothKoreanDataset(dataset=dataset, sr=sr)


__all__ = ["ZerothKorean"]
