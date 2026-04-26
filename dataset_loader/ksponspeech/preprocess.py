# This preprocessing logic is adapted from sooftware/ksponspeech.
# Original project: https://github.com/sooftware/ksponspeech
# Original license: MIT License, Copyright (c) 2020 Soohwan Kim
#
# Modifications:
# - Reimplemented bracket filtering with regex-based dual-transcription handling.
# - Refactored special-symbol filtering.

from __future__ import annotations

import re

from typing import Literal


Mode = Literal["spelling", "phonetic"]
DUAL_TRANSCRIPT_PATTERN = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")


def bracket_filter(
    sentence: str,
    mode: Mode = "spelling",
) -> str:
    group_idx = 1 if mode == "spelling" else 2

    return DUAL_TRANSCRIPT_PATTERN.sub(
        lambda m: m.group(group_idx),
        sentence,
    )


SENTENCE_MARK: list[str] = ["?", "!", "."]
NOISE: list[str] = ["o", "n", "u", "b", "l"]
EXCEPT: list[str] = [
    "/",
    "+",
    "*",
    "-",
    "@",
    "$",
    "^",
    "&",
    "[",
    "]",
    "=",
    ":",
    ";",
    ",",
]
SPACE_PATTERN = re.compile(r"\s\s+")


def special_filter(
    sentence: str, mode: Mode = "phonetic", percent_replace: str = "퍼센트"
) -> str:
    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if (
            ch not in SENTENCE_MARK
            and idx + 1 < len(sentence)
            and ch in NOISE
            and sentence[idx + 1] == "/"
        ):
            continue

        if ch == "#":
            new_sentence += "샾"

        elif ch == "%":
            if mode == "phonetic":
                new_sentence += percent_replace
            elif mode == "spelling":
                new_sentence += "%"

        elif ch not in EXCEPT:
            new_sentence += ch

    new_sentence = re.sub(SPACE_PATTERN, " ", new_sentence.strip())
    return new_sentence


__all__ = ["bracket_filter", "special_filter"]
