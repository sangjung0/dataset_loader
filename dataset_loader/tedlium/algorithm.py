from __future__ import annotations

import hashlib
import librosa

from pathlib import Path
from tqdm import tqdm


def get_file_hash(file: Path):
    hash_md5 = hashlib.md5()
    readlines = file.read_bytes()
    hash_md5.update(readlines)
    return hash_md5.hexdigest()


def parse_ctl_hashes(ctl_file: Path):
    readlines = ctl_file.read_text().splitlines()
    hashes = {}
    for line in readlines:
        if line := line.strip():
            hash, filename = line.split()
            hashes[filename] = hash
    return hashes


def parse_files(
    sph_dir: Path,
    stm_dir: Path,
    *,
    sph_hash: dict[str, str] | None = None,
    stm_hash: dict[str, str] | None = None,
    verbose: bool = False,
    dataset_path: Path | str = "/",
) -> list[dict]:
    sph_files = list(sph_dir.glob("*.sph"))
    stm_files = list(stm_dir.glob("*.stm"))
    if len(sph_files) != len(stm_files):
        raise ValueError(
            f"Number of .sph files ({len(sph_files)}) does not match number of .stm files ({len(stm_files)})"
        )

    data = []
    for sph_file in tqdm(sph_files, desc="Parsing files", disable=not verbose):
        stm_file = stm_dir / (sph_file.stem + ".stm")
        if not stm_file.exists():
            raise FileNotFoundError(
                f"STM file {stm_file} does not exist for SPH file {sph_file}"
            )

        if sph_hash is not None and get_file_hash(sph_file) != sph_hash[sph_file.name]:
            raise ValueError(f"Hash mismatch for SPH file {sph_file}")
        if stm_hash is not None and get_file_hash(stm_file) != stm_hash[stm_file.name]:
            raise ValueError(f"Hash mismatch for STM file {stm_file}")

        readlines = stm_file.read_text().splitlines()
        splitlines = [line.split(maxsplit=6) for line in readlines]
        file_ids, channel_ids, speaker_ids = set(), set(), set()
        refs = []
        for splitline in splitlines:
            if len(splitline) < 7:
                raise ValueError(f"Invalid line in STM file {stm_file}: {splitline}")
            file_id, channel_id, speaker_id, start_time, end_time, label, ref = (
                splitline
            )

            file_ids.add(file_id)
            channel_ids.add(channel_id)
            speaker_ids.add(speaker_id)
            refs.append(
                {
                    "start": float(start_time),
                    "end": float(end_time),
                    "label": label,
                    "ref": ref.strip(),
                    "speaker_id": speaker_id,
                }
            )

        if len(file_ids) != 1:
            raise ValueError(f"Multiple file IDs in STM file {stm_file}: {file_ids}")
        if len(channel_ids) != 1:
            raise ValueError(
                f"Multiple channel IDs in STM file {stm_file}: {channel_ids}"
            )

        text = " ".join(ref["ref"] for ref in refs)
        wav, sr = librosa.load(sph_file)
        duration = librosa.get_duration(y=wav, sr=sr)

        data.append(
            {
                "id": file_ids.pop(),
                "channel": channel_ids.pop(),
                "speakers": list(speaker_ids),
                "audio_path": str(sph_file.relative_to(dataset_path)),
                "duration": duration,
                "text": text,
                "stm": refs,
            }
        )

    return data


__all__ = ["get_file_hash", "parse_ctl_hashes", "parse_files"]
