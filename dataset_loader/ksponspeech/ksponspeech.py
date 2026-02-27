from __future__ import annotations

from pathlib import Path

from dataset_loader.abstract import HuggingfaceLoader

from dataset_loader.ksponspeech.ksponspeech_dataset import KSPonSpeechDataset
from dataset_loader.ksponspeech.constants import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_REPO_ID,
    DEFAULT_SAMPLE_RATE,
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
        use_cache: int = 0,
        **kwargs,
    ):
        dataset = self.load(config_name=config_name, split_name="train", **kwargs)
        return KSPonSpeechDataset(dataset=dataset, sr=sr, use_cache=use_cache)

    def valid(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        use_cache: int = 0,
        **kwargs,
    ):
        dataset = self.load(config_name=config_name, split_name="valid", **kwargs)
        return KSPonSpeechDataset(dataset=dataset, sr=sr, use_cache=use_cache)

    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        use_cache: int = 0,
        **kwargs,
    ):
        dataset = self.load(config_name=config_name, split_name="test", **kwargs)
        return KSPonSpeechDataset(dataset=dataset, sr=sr, use_cache=use_cache)


__all__ = ["KSPonSpeech"]
