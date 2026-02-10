import os

from pathlib import Path
from typing import Literal, get_args

from dataset_loader.librispeech.librispeech_dataset import LibriSpeechDataset
from dataset_loader.librispeech.constants import (
    LibriTask,
    LibriSpeechSet,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TASK,
    DEFAULT_DOWNLOAD_URLS,
)


class LibriSpeech:
    """
    LibriSpeech 데이터셋 로더 및 다운로드 매니저.
    LibriSpeech 데이터셋은 연속적인 오디오가 아닌 세그먼트로 나눠져 있음을 유의.
    """

    def __init__(self, path: Path | str):
        path = Path(path)
        self.__path = path

    def download_urls(self) -> dict[str, str]:
        return DEFAULT_DOWNLOAD_URLS.copy()

    def download(
        self,
        names: list[str | LibriSpeechSet] | Literal["all", "env"] = "all",
        urls: list[str] | None = None,
    ) -> Path | list[Path]:
        if names == "all":
            names = list(DEFAULT_DOWNLOAD_URLS.keys())
        elif names == "env":
            name = os.getenv("LIBRISPEECH_NAME")
            url = os.getenv("LIBRISPEECH_URL")
            if not (name and url):
                raise ValueError(
                    "Environment variables LIBRISPEECH_NAME and LIBRISPEECH_URL must be set."
                )
            return self._download(name, url)
        elif not isinstance(names, list):
            raise ValueError("Names must be 'all', 'env', or a list of dataset names.")

        args = []
        if urls is None:
            for name in names:
                if name in DEFAULT_DOWNLOAD_URLS:
                    args.append((name, DEFAULT_DOWNLOAD_URLS[name]))
                else:
                    raise ValueError(f"Unknown dataset name: {name}")
        elif len(names) != len(urls):
            raise ValueError("Name and URL lists must have the same length.")
        else:
            for name, url in zip(names, urls):
                args.append((name, url))

        return [self._download(name, url) for name, url in args]

    def _download(self, name: str, url: str) -> Path:
        from sjpy.download import download
        from sjpy.archive.tar import extract_tar
        from sjpy.file.algorithm import move_dir_contents

        target_path = self.__path / name
        if list(target_path.glob("*")):
            return target_path

        downloaded = download(url)
        # downloaded = Path("/tmp/temp_efa914241bdc425fb5dc366931490768")
        extract_tar(downloaded, target_path.parent)
        downloaded.unlink()
        move_dir_contents(target_path.parent / "LibriSpeech", self.__path)
        (target_path.parent / "LibriSpeech").rmdir()

        return target_path

    def __load_set(
        self, name: str, sr: int, task=tuple[LibriTask, ...]
    ) -> LibriSpeechDataset:
        for t in task:
            if t not in get_args(LibriTask):
                raise ValueError(f"Task {t} is not compatible with LibriSpeechDataset")

        target = self.__path / name
        if not target.exists():
            raise FileNotFoundError(f"LibriSpeech dataset not found at: {target}")

        ids = []
        refs = []
        audio_paths = []
        original_txt = target.rglob("**/*.txt")
        for txt in original_txt:
            lines = txt.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split(" ", maxsplit=1)
                if len(parts) != 2:
                    continue
                ids.append(parts[0])
                refs.append(parts[1])
                paths = parts[0].split("-")
                audio_path = target / paths[0] / paths[1] / f"{parts[0]}.flac"
                audio_paths.append(audio_path)

        return LibriSpeechDataset(
            ids=ids,
            audio_paths=audio_paths,
            references=refs,
            sample_rate=sr,
            task=task,
        )

    def load_train_clean_100(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[LibriTask, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("train-clean-100", sr=sr, task=task)

    def load_train_clean_360(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[LibriTask, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("train-clean-360", sr=sr, task=task)

    def load_train_other_500(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[LibriTask, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("train-other-500", sr=sr, task=task)

    def load_dev_clean(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[LibriTask, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("dev-clean", sr=sr, task=task)

    def load_dev_other(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[LibriTask, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("dev-other", sr=sr, task=task)

    def load_test_clean(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[LibriTask, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("test-clean", sr=sr, task=task)

    def load_test_other(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[LibriTask, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("test-other", sr=sr, task=task)


__all__ = ["LibriSpeech", "LibriSpeechDataset"]
