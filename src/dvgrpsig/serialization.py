from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np


def _normalize(value: Any) -> Any:
    if hasattr(value, "to_canonical"):
        return _normalize(value.to_canonical())
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def canonical_encode(value: Any) -> bytes:
    normalized = _normalize(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_decode(payload: bytes) -> Any:
    def restore(value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"__bytes__"}:
            return bytes.fromhex(value["__bytes__"])
        if isinstance(value, dict):
            return {key: restore(val) for key, val in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return restore(json.loads(payload.decode("utf-8")))

