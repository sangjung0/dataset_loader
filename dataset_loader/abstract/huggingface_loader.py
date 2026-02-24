from __future__ import annotations

from functools import lru_cache, cached_property
from typing_extensions import override
from datasets import (
    load_dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    Dataset,
    DownloadConfig,
)

from dataset_loader.interface import DatasetLoader


class HuggingfaceLoader(DatasetLoader):
    def __init__(self: HuggingfaceLoader, *, repo_id: str, **kwargs):
        super().__init__(**kwargs)
        self._repo_id: str = repo_id

    @property
    def repo_id(self: HuggingfaceLoader) -> str:
        return self._repo_id

    @cached_property
    def config_names(self: HuggingfaceLoader) -> list[str]:
        return get_dataset_config_names(self.repo_id)

    @lru_cache(maxsize=32)
    def split_names(self: HuggingfaceLoader, config_name: str) -> list[str]:
        if config_name not in self.config_names:
            raise ValueError(
                f"Config name '{config_name}' is not valid. Available configs: {self.config_names}"
            )
        return get_dataset_split_names(self.repo_id, config_name)

    @override
    def download(
        self: HuggingfaceLoader,
        config_name: str,
        split: str,
        local_files_only: bool = False,
    ) -> Dataset:
        if config_name not in self.config_names:
            raise ValueError(
                f"Config name '{config_name}' is not valid. Available configs: {self.config_names}"
            )
        if split not in self.split_names(config_name):
            raise ValueError(
                f"Split '{split}' is not valid for config '{config_name}'. Available splits: {self.split_names(config_name)}"
            )
        return load_dataset(
            self.repo_id,
            name=config_name,
            cache_dir=self.path,
            split=split,
            download_config=DownloadConfig(local_files_only=local_files_only),
        )


__all__ = ["HuggingfaceLoader"]
