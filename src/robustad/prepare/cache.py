"""Cache headers so a score file cannot be read under a different setting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CacheMismatch(ValueError):
    pass


def cache_key(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_header(path: Path, config: dict) -> str:
    key = cache_key(config)
    payload = {"cache_key": key, "config": config}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return key


def read_header(path: Path, expected: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cache_key") != cache_key(expected):
        raise CacheMismatch("キャッシュの設定識別子が一致しません。")
    return payload
