from __future__ import annotations

from typing import Literal

# Types
ESICDataset = Literal["v1.1/dev", "v1.1/dev2", "v1.1/test"]

# Defaults
DEFAULT_DEV = "v1.1/dev"
DEFAULT_DEV2 = "v1.1/dev2"
DEFAULT_TEST = "v1.1/test"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DOWNLOAD_URL = "https://lindat.mff.cuni.cz/repository/server/api/core/items/45b19770-d58e-45d9-aacd-d19875d1c987/allzip?handleId=11234/1-5415"
DATA_PARQUET = {
    DEFAULT_DEV: DEFAULT_DEV + ".parquet",
    DEFAULT_DEV2: DEFAULT_DEV2 + ".parquet",
    DEFAULT_TEST: DEFAULT_TEST + ".parquet",
}

# Constants
TXT = "txt"
VERT_TS = "vert+ts"
ORTO = "o"
ORTO_TS = "o+ts"
VERBATIM = "v"
PUNCT_VERBATIM = "pv"
MP4 = "mp4"

FILE_TYPE = {
    TXT: {"file": "en.OSt.man.txt", "explain": "통역사의 수동 번역"},
    VERT_TS: {
        "file": "en.OSt.man.vert+ts",
        "explain": "시작시간, 종료시간, 원형단어, 기호 포함 단어, 문장 세그먼트 번호, 기타 태그",
    },
    ORTO: {"file": "en.OSt.man.orto.txt", "explain": "특수 태그 제거된 정제 문장"},
    ORTO_TS: {
        "file": "en.OSt.man.orto+ts.txt",
        "explain": "특수 태그 제거 + 타임스탬프",
    },
    VERBATIM: {
        "file": "en.OSt.man.verbatim.txt",
        "explain": "소문자, 구두점 제거, 숫자는 문자로, 태그 제거, 불완전 발화 포함",
    },
    PUNCT_VERBATIM: {
        "file": "en.OSt.man.punct-verbatim.txt",
        "explain": "대소문자 및 구두점 포함, 숫자는 숫자로, 태그 제거, 불완전 발화 포함",
    },
    MP4: {"file": "en.OS.man-diar.mp4", "explain": "원본영상"},
}


__all__ = [
    "ESICDataset",
    "DEFAULT_DEV",
    "DEFAULT_DEV2",
    "DEFAULT_TEST",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_DOWNLOAD_URL",
    "DATA_PARQUET",
    "TXT",
    "VERT_TS",
    "ORTO",
    "ORTO_TS",
    "VERBATIM",
    "PUNCT_VERBATIM",
    "MP4",
    "FILE_TYPE",
]
