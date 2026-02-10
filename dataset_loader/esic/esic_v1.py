from __future__ import annotations
from typing import TYPE_CHECKING

import os

from typing import get_args
from pathlib import Path
from functools import lru_cache

from dataset_loader.esic.algorithm import search_dirs, select_file_from_dir
from dataset_loader.esic.constants import (
    ESICTask,
    ESICDataset,
    VERBATIM,
    MP4,
    DEFAULT_DEV,
    DEFAULT_DEV2,
    DEFAULT_TEST,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DOWNLOAD_URL,
    DEFAULT_TASK,
)
from dataset_loader.esic.esic_v1_dataset import ESICv1Dataset

if TYPE_CHECKING:
    pass


class ESICv1:
    """
    ESIC v1.1 데이터셋 로더 및 다운로드 매니저.
    ESIC 데이터셋은 연속적인 오디오로 구성되어 있음.
    """

    def __init__(self, root: Path | str):
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(f"ESICv1 root path not found: {root}")
        elif not root.is_dir():
            raise NotADirectoryError(f"ESICv1 root path is not a directory: {root}")

        self.__root = root

    @property
    def download_url(self) -> str:
        return DEFAULT_DOWNLOAD_URL

    def download(self, url: str | None = None) -> Path:
        import shutil

        from sjpy.download import download
        from sjpy.archive.zip import extract_zip

        if list(self.__root.glob("*")):
            return self.__root

        if url is None:
            env_url = os.getenv("ESICV1_DOWNLOAD_URL")
            url = env_url if env_url is not None else self.download_url

        downloaded = download(url)
        # downloaded = Path("/tmp/temp_26dd497dd89641e090672a9cf86450fd")
        extracted = extract_zip(downloaded) / "ESIC-v1.1.zip"
        target = self.__root / extracted.name
        shutil.move(extracted, target)
        extracted = extract_zip(target, self.__root)
        target.unlink()
        return extracted

    def __get_dirs(
        self, post_path: Path | str, excludes: tuple[str] = ()
    ) -> list[Path]:
        if isinstance(post_path, str):
            if post_path not in get_args(ESICDataset):
                raise ValueError(
                    f"post_path must be one of {get_args(ESICDataset)}, but got {post_path}"
                )
        return search_dirs(self.__root / post_path, excludes=excludes)

    def __generate_items(
        self,
        dirs: list[Path],
        source_file_type: str,
        truth_file_type: str,
        sample_rate: int,
        task: tuple[ESICTask, ...],
    ) -> ESICv1Dataset:
        X = []
        Y = []
        for d in dirs:
            x = select_file_from_dir(d, source_file_type)
            y = select_file_from_dir(d, truth_file_type)
            X.append(x)
            Y.append(y)
        return ESICv1Dataset(X, Y, sr=sample_rate, task=task)

    @lru_cache(maxsize=1)
    def dev(
        self,
        post_path: Path | str = DEFAULT_DEV,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        task: tuple[ESICTask, ...] = DEFAULT_TASK,
    ) -> ESICv1Dataset:
        return self.__generate_items(
            self.__get_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
            sample_rate,
            task,
        )

    @lru_cache(maxsize=1)
    def dev2(
        self,
        post_path: Path | str = DEFAULT_DEV2,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        task: tuple[ESICTask, ...] = DEFAULT_TASK,
    ) -> ESICv1Dataset:
        return self.__generate_items(
            self.__get_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
            sample_rate,
            task,
        )

    @lru_cache(maxsize=1)
    def test(
        self,
        post_path: Path | str = DEFAULT_TEST,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        task: tuple[ESICTask, ...] = DEFAULT_TASK,
    ) -> ESICv1Dataset:
        return self.__generate_items(
            self.__get_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
            sample_rate,
            task,
        )


__all__ = ["ESICv1"]
