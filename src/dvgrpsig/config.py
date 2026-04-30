from __future__ import annotations

import json
from pathlib import Path

from dvgrpsig.types import ParameterTrack


_TRACK_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "parameter_tracks.json"


def _load_tracks() -> dict[str, ParameterTrack]:
    payload = json.loads(_TRACK_CONFIG_PATH.read_text(encoding="utf-8"))
    tracks: dict[str, ParameterTrack] = {}
    for name, values in payload.items():
        tracks[name] = ParameterTrack(
            name=name,
            n=int(values["n"]),
            m=int(values["m"]),
            q=int(values["q"]),
            sigma1=int(values["sigma1"]),
            sigma2=int(values["sigma2"]),
            sigma_enc=int(values["sigma_enc"]),
            bz=float(values["bz"]),
            beta_s=int(values["beta_s"]),
            challenge_weight=int(values["challenge_weight"]),
            stern_rounds=int(values["stern_rounds"]),
            trapdoor_offsets=tuple(int(value) for value in values["trapdoor_offsets"]),
        )
    return tracks


_TRACKS = _load_tracks()


def get_parameter_track(name: str) -> ParameterTrack:
    try:
        return _TRACKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRACKS))
        raise KeyError(f"Unknown parameter track {name!r}; available: {available}") from exc
