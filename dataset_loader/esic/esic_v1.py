from __future__ import annotations

import os

from typing import get_args
from pathlib import Path

from dataset_loader.interface import DatasetLoader

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


class ESICv1(DatasetLoader):
    """
    ESIC v1.1 데이터셋 로더 및 다운로드 매니저.
    ESIC 데이터셋은 연속적인 오디오로 구성되어 있음.
    """

    @property
    def download_url(self: ESICv1) -> str:
        return DEFAULT_DOWNLOAD_URL

    def download(self: ESICv1, url: str | None = None) -> Path:
        import shutil

        from sjpy.download import download
        from sjpy.archive.zip import extract_zip

        if list(self.path.glob("*")):
            return self.path

        if url is None:
            env_url = os.getenv("ESICV1_DOWNLOAD_URL")
            url = env_url if env_url is not None else self.download_url

        downloaded = download(url)
        # downloaded = Path("/tmp/temp_26dd497dd89641e090672a9cf86450fd")
        extracted = extract_zip(downloaded) / "ESIC-v1.1.zip"
        target = self.path / extracted.name
        shutil.move(extracted, target)
        extracted = extract_zip(target, self.path)
        target.unlink()
        return extracted

    def __get_dirs(
        self: ESICv1, post_path: Path | str, excludes: tuple[str] = ()
    ) -> list[Path]:
        if isinstance(post_path, str):
            if post_path not in get_args(ESICDataset):
                raise ValueError(
                    f"post_path must be one of {get_args(ESICDataset)}, but got {post_path}"
                )
        return search_dirs(self.path / post_path, excludes=excludes)

    def __generate_items(
        self: ESICv1,
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
        return ESICv1Dataset(X=list(X), Y=list(Y), sr=sample_rate, task=task)

    def dev(
        self: ESICv1,
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

    def dev2(
        self: ESICv1,
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

    def test(
        self: ESICv1,
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
