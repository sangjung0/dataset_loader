from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dataset_loader.interface.constants import DEFAULT_PATH


class DatasetLoader(ABC):
    """
    다양한 Dataset을 다운받고 로드하는 기능을 제공하는 공통 추상 클래스이다.
    각 데이터셋 로더는 이 클래스를 상속하여 구현한다.

    Attributes:
        dir_name (str | None): 데이터셋이 저장되는 디렉토리 이름. 기본값은 클래스 이름이다.
        path (str | Path | None): 데이터셋이 저장되는 디렉토리 경로 기본값은 ${HOME}/.datasets 이다.
    """

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
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self: DatasetLoader) -> Path:
        return self._path

    @abstractmethod
    def download(self: DatasetLoader, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load(self: DatasetLoader, *args, **kwargs):
        raise NotImplementedError


__all__ = ["DatasetLoader"]
