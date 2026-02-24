from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dataset_loader.interface.constants import DEFAULT_PATH


class DatasetLoader(ABC):
    def __init__(
        self: DatasetLoader,
        *,
        dir_name: str | None = None,
        path: str | Path | None = None,
    ):
        if path is None:
            path = DEFAULT_PATH
        if dir_name is None:
            dir_name = self.__class__.__name__

        path = Path(path)
        if path.exists():
            if not path.is_dir():
                raise NotADirectoryError(f"Root path is not a directory: {path}")

        self._path: Path = path / dir_name

    @property
    def path(self: DatasetLoader) -> Path:
        return self._path

    @abstractmethod
    def download(self: DatasetLoader, *args, **kwargs):
        pass


__all__ = ["DatasetLoader"]
