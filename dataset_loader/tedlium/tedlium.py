from __future__ import annotations

import pandas as pd

from typing import Sequence
from typing_extensions import override

from dataset_loader.abstract import ParquetLoader

from dataset_loader.tedlium.tedlium_dataset import TedliumDataset

from dataset_loader.tedlium.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_IGNORE_SET,
    DATA_PARQUET,
    TedliumSet,
)


class Tedlium(ParquetLoader):
    """
    Tedlium 데이터셋 로더.
    Tedlium 데이터셋은 연속적인 오디오로 구성되어 있음.
    """

    def __init__(
        self: Tedlium,
        *,
        dir_name: str | None = None,
        path: str | None = None,
        parquet_name_and_path: dict[str, str] | None = None,
    ):
        if parquet_name_and_path is None:
            parquet_name_and_path = DATA_PARQUET
        super().__init__(
            dir_name=dir_name, path=path, parquet_name_and_path=parquet_name_and_path
        )

    @override
    def download(self: Tedlium, *args, **kwargs):
        raise NotImplementedError(
            "Tedlium dataset is not available for download. Please download it manually from the official website and place it in the specified directory."
        )

    @override
    def load(
        self: Tedlium, *, name: TedliumSet, prepare_dir: str = ".prepare"
    ) -> pd.DataFrame:
        data = super().load(name=name, prepare_dir=prepare_dir)
        data["audio_path"] = data["audio_path"].apply(lambda x: self.path / x)
        return data

    @override
    def _parse_files(
        self: Tedlium,
        *,
        name: str,
        verbose: bool = False,
    ) -> list[dict[str, str]]:
        from dataset_loader.tedlium.algorithm import (
            parse_ctl_hashes,
            parse_files,
        )

        if name == "train":
            path = self.path / "TEDLIUM_release-3/data"
            sph_hash = parse_ctl_hashes(path / "ctl" / "sph_md5sum")
            stm_hash = parse_ctl_hashes(path / "ctl" / "stm_md5sum")
            return parse_files(
                path / "sph",
                path / "stm",
                sph_hash=sph_hash,
                stm_hash=stm_hash,
                verbose=verbose,
                dataset_path=self.path,
            )
        else:
            path = self.path / f"TEDLIUM_release-3/legacy/{name}"
            return parse_files(
                path / "sph",
                path / "stm",
                verbose=verbose,
                dataset_path=self.path,
            )

    def train(
        self: Tedlium,
        *,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
        ignore_set: Sequence[str] = DEFAULT_IGNORE_SET,
    ):
        data = self.load(name="train", prepare_dir=prepare_dir)
        return TedliumDataset(
            parquet=data, sr=sr, use_cache=use_cache, ignore_set=ignore_set
        )

    def dev(
        self: Tedlium,
        *,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
        ignore_set: Sequence[str] = DEFAULT_IGNORE_SET,
    ):
        data = self.load(name="dev", prepare_dir=prepare_dir)
        return TedliumDataset(
            parquet=data, sr=sr, use_cache=use_cache, ignore_set=ignore_set
        )

    def test(
        self: Tedlium,
        *,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
        ignore_set: Sequence[str] = DEFAULT_IGNORE_SET,
    ):
        data = self.load(name="test", prepare_dir=prepare_dir)
        return TedliumDataset(
            parquet=data, sr=sr, use_cache=use_cache, ignore_set=ignore_set
        )


__all__ = ["Tedlium"]
