# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false

from __future__ import annotations

from typing import Any
from typing_extensions import override
from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset
from collections.abc import Mapping

from dataset_loader.base import DatasetLoader


class HuggingfaceSnapshot(DatasetLoader):
    def __init__(
        self,
        *,
        repo_id: str,
        dir_name: str | None = None,
        path: str | None = None,
    ):
        super().__init__(dir_name=dir_name, path=path)
        self._repo_id: str = repo_id

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @override
    def download(self, snapshot_options: Mapping[str, Any] | None = None) -> str:
        default_snapshot_options = {
            "repo_id": self.repo_id,
            "repo_type": "dataset",
            "revision": "refs/convert/parquet",
            "local_dir": self.path,
            "allow_patterns": ["**.parquet"],
            "token": True,
        }

        if snapshot_options is None:
            snapshot_options = default_snapshot_options
        else:
            snapshot_options = {
                **default_snapshot_options,
                **snapshot_options,
            }

        path: str = snapshot_download(**snapshot_options)
        return path

    @override
    def load(self, name: str, load_options: Mapping[str, Any] | None = None) -> Dataset:
        default_options = {
            "path": "parquet",
            "data_files": {name: f"{self.path}/*.parquet"},
        }

        if load_options is None:
            load_options = default_options
        else:
            load_options = {
                **default_options,
                **load_options,
            }

        dataset: Dataset = load_dataset(**load_options)[name]
        return dataset


__all__ = ["HuggingfaceSnapshot"]
