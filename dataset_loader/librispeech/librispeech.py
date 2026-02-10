import warnings
import os

from pathlib import Path
from typing import Literal, Sequence

from dataset_loader.interface.types import Task

from dataset_loader.librispeech.librispeech_dataset import LibriSpeechDataset

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)
DEFAULT_DOWNLOAD_URLS = {
    "dev-clean": "https://openslr.trmal.net/resources/12/dev-clean.tar.gz",
    "dev-other": "https://openslr.trmal.net/resources/12/dev-other.tar.gz",
    "test-clean": "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
    "test-other": "https://openslr.trmal.net/resources/12/test-other.tar.gz",
    "train-clean-100": "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://openslr.trmal.net/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://openslr.trmal.net/resources/12/train-other-500.tar.gz",
}
LibriSpeech = Literal[
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]


class LibriSpeech:
    def __init__(self, path: Path | str):
        path = Path(path)
        self.__path = path

    def download_urls(self) -> dict[str, str]:
        return DEFAULT_DOWNLOAD_URLS.copy()

    def download(
        self,
        names: list[str | LibriSpeech] | Literal["all", "env"] = "all",
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
        self, name: str, sr: int, task=tuple[Task, ...]
    ) -> LibriSpeechDataset:
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
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("train-clean-100", sr=sr, task=task)

    def load_train_clean_360(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("train-clean-360", sr=sr, task=task)

    def load_train_other_500(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("train-other-500", sr=sr, task=task)

    def load_dev_clean(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("dev-clean", sr=sr, task=task)

    def load_dev_other(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("dev-other", sr=sr, task=task)

    def load_test_clean(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("test-clean", sr=sr, task=task)

    def load_test_other(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LibriSpeechDataset:
        return self.__load_set("test-other", sr=sr, task=task)


if __name__ != "__main__":
    warnings.warn(
        "[INFO] LibriSpeech 오디오가 연속적이지 않고 세그먼트로 나눠져 있음.",
        category=UserWarning,
        stacklevel=2,
    )

__all__ = ["LibriSpeech", "LibriSpeechDataset"]
