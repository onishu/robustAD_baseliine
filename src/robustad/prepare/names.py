"""Defect names encoded in RobustAD filenames.

The templates were read from the released anomaly filenames. A stem that does
not match is returned as unresolved rather than assigned a guessed type.
"""

from __future__ import annotations

import re

from robustad.prepare.spec import METAL_DEFECTS, PCB_DEFECTS

_INDEX = re.compile(r"^(?P<body>.+)_(?P<index>\d+)$")


def parse_anomaly_stem(stem: str) -> dict[str, str | int]:
    matched = _INDEX.match(stem)
    if matched is None:
        return {"defect": "", "condition": "", "index": -1, "parsed": 0}
    body = matched.group("body")
    index = int(matched.group("index"))
    if body.startswith("Test_PCB_") or body.startswith("Train_PCB_"):
        prefix = "Test_PCB_" if body.startswith("Test_PCB_") else "Train_PCB_"
        rest = body[len(prefix) :]
        for defect in PCB_DEFECTS:
            suffix = f"_{defect}"
            if rest.endswith(suffix):
                return {
                    "defect": defect,
                    "condition": rest[: -len(suffix)],
                    "index": index,
                    "parsed": 1,
                }
    if body.startswith("Live_"):
        rest = body[len("Live_") :]
        for defect in METAL_DEFECTS:
            prefix = f"{defect}_"
            if rest.startswith(prefix):
                return {
                    "defect": defect,
                    "condition": rest[len(prefix) :],
                    "index": index,
                    "parsed": 1,
                }
    return {"defect": "", "condition": "", "index": index, "parsed": 0}
