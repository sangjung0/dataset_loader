from __future__ import annotations

from pathlib import Path

from dataset_loader.abstract import HuggingfaceLoader

from dataset_loader.ksponspeech.ksponspeech_dataset import KSPonSpeechDataset
from dataset_loader.ksponspeech.constants import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_REPO_ID,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TASK,
    KSPonSpeechTask,
)


class KSPonSpeech(HuggingfaceLoader):
    def __init__(
        self: KSPonSpeech,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        dir_name: str | None = None,
        path: str | Path | None = None,
    ):
        super().__init__(repo_id=repo_id, dir_name=dir_name, path=path)

    def split_names(
        self: KSPonSpeech, config_name: str = DEFAULT_CONFIG_NAME
    ) -> list[str]:
        return super().split_names(config_name)

    def train(
        self: KSPonSpeech,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[KSPonSpeechTask, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return KSPonSpeechDataset(
            dataset=super().download(config_name, "train", **kwargs), sr=sr, task=task
        )

    def valid(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[KSPonSpeechTask, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return KSPonSpeechDataset(
            dataset=super().download(config_name, "valid", **kwargs), sr=sr, task=task
        )

    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[KSPonSpeechTask, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return KSPonSpeechDataset(
            dataset=super().download(config_name, "test", **kwargs), sr=sr, task=task
        )


__all__ = ["KSPonSpeech"]
