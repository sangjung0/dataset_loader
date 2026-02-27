from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import Literal, Sequence, Mapping
from typing_extensions import override
from tqdm import tqdm

from dataset_loader.abstract import ParquetLoader

from dataset_loader.librispeech.librispeech_dataset import LibriSpeechDataset
from dataset_loader.librispeech.constants import (
    LibriSpeechSet,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DOWNLOAD_URLS,
    DATA_PARQUET,
)


class LibriSpeech(ParquetLoader):
    """
    LibriSpeech 데이터셋 로더 및 다운로드 매니저.
    LibriSpeech 데이터셋은 연속적인 오디오가 아닌 세그먼트로 나눠져 있음을 유의.
    """

    def __init__(
        self: LibriSpeech,
        *,
        dir_name: str | None = None,
        path: str | Path | None = None,
        parquet_name_and_path: dict[str, Path] | None = None,
        download_urls: dict[str, str] = DEFAULT_DOWNLOAD_URLS,
    ):
        if parquet_name_and_path is None:
            parquet_name_and_path = DATA_PARQUET
        super().__init__(
            dir_name=dir_name, path=path, parquet_name_and_path=parquet_name_and_path
        )
        self._download_urls = download_urls

    @property
    def download_urls(self: LibriSpeech) -> dict[str, str]:
        return self._download_urls.copy()

    @override
    def download(
        self: LibriSpeech,
        *,
        name: Sequence[str | LibriSpeechSet] | str | Literal["all"] = "all",
        url: Mapping[str, str | Path] | str | Path | None = None,
        verbose: bool = True,
    ) -> Path | list[Path]:
        if name == "all":
            return self.download(
                name=list(self._download_urls.keys()), url=url, verbose=verbose
            )
        elif isinstance(name, Sequence) and not isinstance(name, str):
            if url is None:
                return [self.download(name=n, verbose=verbose) for n in name]
            elif isinstance(url, Mapping):
                return [
                    self.download(name=n, url=url[n], verbose=verbose) for n in name
                ]
            else:
                raise ValueError("URL must be a mapping when name is a sequence.")

        if name not in self._download_urls:
            raise ValueError(
                f"Unknown dataset name: {name}, expected one of {list(self._download_urls.keys())}"
            )

        if url is None:
            url = self._download_urls[name]
        elif isinstance(url, Mapping):
            url = url[name]
        return self._download(name=name, url=url, verbose=verbose)

    def _download(
        self: LibriSpeech, *, name: str, url: str, verbose: bool = True
    ) -> Path:
        from sjpy.download import download
        from sjpy.archive.tar import extract_tar
        from sjpy.file.algorithm import move_dir_contents

        target_path = self.path / name
        if list(target_path.glob("*")):
            return target_path

        downloaded = download(url, verbose=verbose)
        # downloaded = Path("/tmp/temp_efa914241bdc425fb5dc366931490768")
        extract_tar(downloaded, target_path.parent, verbose=verbose)
        downloaded.unlink()
        move_dir_contents(target_path.parent / "LibriSpeech", self.path, overwrite=True)
        (target_path.parent / "LibriSpeech").rmdir()

        return target_path

    @override
    def load(
        self: LibriSpeech, *, name: str, prepare_dir: str = ".prepare"
    ) -> pd.DataFrame:
        data = super().load(name=name, prepare_dir=prepare_dir)
        data["audio_path"] = data["audio_path"].apply(lambda x: self.path / x)
        return data

    @override
    def _parse_files(
        self: LibriSpeech,
        *,
        name: str,
        verbose: bool = False,
    ) -> list[dict[str, str]]:
        target = self.path / name
        if not target.exists():
            raise FileNotFoundError(f"LibriSpeech dataset not found at: {target}")

        data = []
        for txt_path in tqdm(
            target.rglob("**/*.txt"), desc=f"Parsing {name}", disable=not verbose
        ):
            lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split(" ", maxsplit=1)
                if len(parts) != 2:
                    continue
                _id = parts[0]
                ref = parts[1]
                paths = parts[0].split("-")
                audio_path = target / paths[0] / paths[1] / f"{parts[0]}.flac"
                data.append(
                    {
                        "id": _id,
                        "audio_path": str(audio_path.relative_to(self.path)),
                        "ref": ref,
                    }
                )
        return data

    def train_clean_100(
        self: LibriSpeech,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> LibriSpeechDataset:
        data = self.load(name="train-clean-100", prepare_dir=prepare_dir)
        return LibriSpeechDataset(parquet=data, sr=sr, use_cache=use_cache)

    def train_clean_360(
        self: LibriSpeech,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> LibriSpeechDataset:
        data = self.load(name="train-clean-360", prepare_dir=prepare_dir)
        return LibriSpeechDataset(parquet=data, sr=sr, use_cache=use_cache)

    def train_other_500(
        self: LibriSpeech,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> LibriSpeechDataset:
        data = self.load(name="train-other-500", prepare_dir=prepare_dir)
        return LibriSpeechDataset(parquet=data, sr=sr, use_cache=use_cache)

    def dev_clean(
        self: LibriSpeech,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> LibriSpeechDataset:
        data = self.load(name="dev-clean", prepare_dir=prepare_dir)
        return LibriSpeechDataset(parquet=data, sr=sr, use_cache=use_cache)

    def dev_other(
        self: LibriSpeech,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> LibriSpeechDataset:
        data = self.load(name="dev-other", prepare_dir=prepare_dir)
        return LibriSpeechDataset(parquet=data, sr=sr, use_cache=use_cache)

    def test_clean(
        self: LibriSpeech,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> LibriSpeechDataset:
        data = self.load(name="test-clean", prepare_dir=prepare_dir)
        return LibriSpeechDataset(parquet=data, sr=sr, use_cache=use_cache)

    def test_other(
        self: LibriSpeech,
        sr: int = DEFAULT_SAMPLE_RATE,
        prepare_dir: str = ".prepare",
        use_cache: int = 0,
    ) -> LibriSpeechDataset:
        data = self.load(name="test-other", prepare_dir=prepare_dir)
        return LibriSpeechDataset(parquet=data, sr=sr, use_cache=use_cache)


__all__ = ["LibriSpeech"]
