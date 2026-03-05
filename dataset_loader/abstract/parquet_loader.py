from __future__ import annotations

import pandas as pd

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from typing_extensions import override
from functools import cached_property
from collections.abc import Mapping

from dataset_loader.base import DatasetLoader


class ParquetLoader(DatasetLoader, ABC):
    def __init__(
        self,
        *,
        dir_name: str | None = None,
        path: str | Path | None = None,
        parquet_name_and_path: Mapping[str, str] | None = None,
    ):
        super().__init__(dir_name=dir_name, path=path)
        if parquet_name_and_path is None:
            parquet_name_and_path = {}
        self._parquet_name_and_path = {**parquet_name_and_path}

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(self._parquet_name_and_path.keys())

    @property
    def parquet_name_and_path(self) -> dict[str, str]:
        return self._parquet_name_and_path.copy()

    @override
    def load(self, *, name: str, prepare_dir: str = ".prepare") -> pd.DataFrame:
        if name not in self.names:
            raise ValueError(f"Invalid config: {name}, expected one of {self.names}")

        parquet_path = self.path / prepare_dir / self._parquet_name_and_path[name]
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        return pd.read_parquet(parquet_path)

    def prepare(
        self,
        *,
        name: str = "all",
        verbose: bool = True,
        prepare_dir: str = ".prepare",
        parse_options: Mapping[str, Any] | None = None,
    ) -> None:
        if name == "all":
            for name in self.names:
                self.prepare(
                    name=name,
                    verbose=verbose,
                    prepare_dir=prepare_dir,
                    parse_options=parse_options,
                )
            return
        elif name not in self.names:
            raise ValueError(f"Invalid config: {name}, expected one of {self.names}")

        if parse_options is None:
            parse_options = {}

        parquet_path = self.path / prepare_dir / self._parquet_name_and_path[name]

        if parquet_path.exists():
            if verbose:
                print(f"Parquet file already exists: {parquet_path}")
        else:
            if verbose:
                print(f"Preparing {name} set and saving to {parquet_path}...")
            data = self._parse_files(name=name, verbose=verbose, **parse_options)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_parquet(parquet_path)

    @abstractmethod
    def _parse_files(self, *, name: str, verbose: bool = True) -> list[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _parse_files method")


__all__ = ["ParquetLoader"]
