from __future__ import annotations

import os
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from typing_extensions import override

from sjpy.string import normalize_text_only_en

from dataset_loader.abstract import ParquetLoader

from dataset_loader.esic.esic_v1_dataset import ESICv1Dataset
from dataset_loader.esic.algorithm import search_dirs, select_file_from_dir
from dataset_loader.esic.constants import (
    ESICDataset,
    TXT,
    VERT_TS,
    ORTO,
    ORTO_TS,
    VERBATIM,
    PUNCT_VERBATIM,
    MP4,
    DEFAULT_DEV,
    DEFAULT_DEV2,
    DEFAULT_TEST,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DOWNLOAD_URL,
    DATA_PARQUET,
)


class ESICv1(ParquetLoader):
    """
    ESIC v1.1 데이터셋 로더 및 다운로드 매니저.
    ESIC 데이터셋은 연속적인 오디오로 구성되어 있음.
    """

    def __init__(
        self: ESICv1,
        *,
        dir_name: str | None = None,
        path: str | Path | None = None,
        parquet_name_and_path: dict[str, str] | None = None,
        download_url: str = DEFAULT_DOWNLOAD_URL,
    ):
        if parquet_name_and_path is None:
            parquet_name_and_path = DATA_PARQUET
        super().__init__(
            dir_name=dir_name, path=path, parquet_name_and_path=parquet_name_and_path
        )
        self._download_url = download_url

    @property
    def download_url(self: ESICv1) -> str:
        return self._download_url

    @override
    def download(
        self: ESICv1,
        *,
        url: str | None = None,
        verbose: bool = True,
    ) -> Path:
        import shutil

        from sjpy.download import download
        from sjpy.archive.zip import extract_zip

        if list(self.path.glob("*")):
            return self.path

        if url is None:
            env_url = os.getenv("ESICV1_DOWNLOAD_URL")
            url = env_url if env_url is not None else self.download_url

        downloaded = download(url, verbose=verbose)
        # downloaded = Path("/tmp/temp_26dd497dd89641e090672a9cf86450fd")
        extracted = extract_zip(downloaded, verbose=verbose) / "ESIC-v1.1.zip"
        target = self.path / extracted.name
        shutil.move(extracted, target)
        extracted = extract_zip(target, self.path, verbose=verbose)
        target.unlink()
        return extracted

    @override
    def load(
        self: ESICv1, *, name: ESICDataset, prepare_dir: str = ".prepare"
    ) -> pd.DataFrame:
        data = super().load(name=name, prepare_dir=prepare_dir)
        data["mp4_path"] = data["mp4_path"].apply(lambda x: self.path / x)
        return data

    @override
    def _parse_files(
        self: ESICv1,
        *,
        name: str,
        verbose: bool = False,
        excludes: tuple[str] = (),
    ) -> list[dict]:
        data = []
        for d in tqdm(
            search_dirs(self.path / name, excludes=excludes),
            desc=f"Parsing {name}",
            disable=not verbose,
        ):
            _id = normalize_text_only_en(str(Path(*d.parts[-3:])))[-255:]
            txt_path = select_file_from_dir(d, TXT)
            vert_ts_path = select_file_from_dir(d, VERT_TS)
            orto_path = select_file_from_dir(d, ORTO)
            orto_ts_path = select_file_from_dir(d, ORTO_TS)
            verbatim_path = select_file_from_dir(d, VERBATIM)
            punct_verbatim_path = select_file_from_dir(d, PUNCT_VERBATIM)
            mp4_path = select_file_from_dir(d, MP4)
            data.append(
                {
                    "id": _id,
                    TXT: txt_path.read_text(encoding="utf-8"),
                    VERT_TS: vert_ts_path.read_text(encoding="utf-8"),
                    ORTO: orto_path.read_text(encoding="utf-8"),
                    ORTO_TS: orto_ts_path.read_text(encoding="utf-8"),
                    VERBATIM: verbatim_path.read_text(encoding="utf-8"),
                    PUNCT_VERBATIM: punct_verbatim_path.read_text(encoding="utf-8"),
                    "mp4_path": str(mp4_path.relative_to(self.path)),
                }
            )
        return data

    def dev(
        self: ESICv1,
        *,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> ESICv1Dataset:
        data = self.load(name=DEFAULT_DEV, prepare_dir=prepare_dir)
        return ESICv1Dataset(parquet=data, sr=sr, use_cache=use_cache)

    def dev2(
        self: ESICv1,
        *,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> ESICv1Dataset:
        data = self.load(name=DEFAULT_DEV2, prepare_dir=prepare_dir)
        return ESICv1Dataset(parquet=data, sr=sr, use_cache=use_cache)

    def test(
        self: ESICv1,
        *,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> ESICv1Dataset:
        data = self.load(name=DEFAULT_TEST, prepare_dir=prepare_dir)
        return ESICv1Dataset(parquet=data, sr=sr, use_cache=use_cache)


__all__ = ["ESICv1"]
