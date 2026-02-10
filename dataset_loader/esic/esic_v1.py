from __future__ import annotations
from typing import TYPE_CHECKING

import os

from pathlib import Path
from functools import lru_cache

from dataset_loader.esic.algorithm import search_dirs, select_file_from_dir
from dataset_loader.esic.file_type import VERBATIM, MP4
from dataset_loader.esic.esic_v1_dataset import ESICv1Dataset

if TYPE_CHECKING:
    pass

DEFAULT_DEV = "v1.1/dev"
DEFAULT_DEV2 = "v1.1/dev2"
DEFAULT_TEST = "v1.1/test"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DOWNLOAD_URL = "https://lindat.mff.cuni.cz/repository/server/api/core/items/45b19770-d58e-45d9-aacd-d19875d1c987/allzip?handleId=11234/1-5415"


class ESICv1:
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

    @lru_cache(maxsize=1)
    def dev_dirs(
        self, post_path: Path | str = DEFAULT_DEV, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def dev2_dirs(
        self, post_path: Path | str = DEFAULT_DEV2, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def test_dirs(
        self, post_path: Path | str = DEFAULT_TEST, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    def all_dirs(self, excludes: tuple[str] = ()) -> list[Path]:
        return (
            self.dev_dirs(excludes=excludes)
            + self.dev2_dirs(excludes=excludes)
            + self.test_dirs(excludes=excludes)
        )

    def __generate_items(
        self,
        dirs: list[Path],
        source_file_type: str,
        truth_file_type: str,
        sample_rate: int,
    ) -> ESICv1Dataset:
        X = []
        Y = []
        for d in dirs:
            x = select_file_from_dir(d, source_file_type)
            y = select_file_from_dir(d, truth_file_type)
            X.append(x)
            Y.append(y)
        return ESICv1Dataset(X, Y, sr=sample_rate)

    @lru_cache(maxsize=1)
    def dev(
        self,
        post_path: Path | str = DEFAULT_DEV,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> ESICv1Dataset:
        return self.__generate_items(
            self.dev_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
            sample_rate,
        )

    @lru_cache(maxsize=1)
    def dev2(
        self,
        post_path: Path | str = DEFAULT_DEV2,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> ESICv1Dataset:
        return self.__generate_items(
            self.dev2_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
            sample_rate,
        )

    @lru_cache(maxsize=1)
    def test(
        self,
        post_path: Path | str = DEFAULT_TEST,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> ESICv1Dataset:
        return self.__generate_items(
            self.test_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
            sample_rate,
        )


__all__ = ["ESICv1"]
