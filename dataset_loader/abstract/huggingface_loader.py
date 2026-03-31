from __future__ import annotations

from pathlib import Path
from functools import cached_property, lru_cache
from typing_extensions import override
from collections.abc import Sequence
from datasets import (
    load_dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    Dataset,
    DownloadConfig,
    IterableDatasetDict,
)

from dataset_loader.base import DatasetLoader


class HuggingfaceLoader(DatasetLoader):
    def __init__(
        self,
        *,
        repo_id: str,
        dir_name: str | None = None,
        path: str | Path | None = None,
    ):
        super().__init__(dir_name=dir_name, path=path)
        self._repo_id: str = repo_id

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @cached_property
    def config_names(self) -> list[str]:
        config_names: list[str] = get_dataset_config_names(self.repo_id)
        return config_names

    @lru_cache(maxsize=32)
    def _split_names(self, config_name: str) -> list[str]:
        if config_name not in self.config_names:
            raise ValueError(
                f"Config name '{config_name}' is not valid. Available configs: {self.config_names}"
            )
        split_names: list[str] = get_dataset_split_names(self.repo_id, config_name)
        return split_names

    def split_names(self, config_name: str) -> list[str]:
        return self._split_names(config_name)

    @override
    def download(
        self,
        *,
        config_name: str,
        split_name: str | Sequence[str] | None = None,
        local_files_only: bool = False,
    ) -> Dataset | IterableDatasetDict:
        if config_name not in self.config_names:
            raise ValueError(
                f"Config name '{config_name}' is not valid. Available configs: {self.config_names}"
            )
        if split_name is None:
            split_name = self.split_names(config_name)
        if isinstance(split_name, str):
            split_name = [split_name]
        if not all(s in self.split_names(config_name) for s in split_name):
            raise ValueError(
                f"One or more split names are not valid for config '{config_name}'. Available splits: {self.split_names(config_name)}"
            )

        return load_dataset(
            self.repo_id,
            name=config_name,
            cache_dir=str(self.path),
            split=list(split_name),
            download_config=DownloadConfig(local_files_only=local_files_only),
        )

    @override
    def load(
        self,
        *,
        config_name: str,
        split_name: str | Sequence[str],
        local_files_only: bool = False,
    ) -> Dataset | IterableDatasetDict:
        dir_name = self.repo_id.replace("/", "___")
        path = self.path / dir_name
        if not path.exists():
            raise FileNotFoundError(
                "Dataset not found in cache. Please download it first using the 'download' method."
            )

        return self.download(
            config_name=config_name,
            split_name=split_name,
            local_files_only=local_files_only,
        )


__all__ = ["HuggingfaceLoader"]
