from __future__ import annotations

from datetime import datetime
from pathlib import Path


_ARTIFACTS_DIRNAME = "artifacts"
_DEBUG_DIRNAME = "debug"
_FINAL_DIRNAME = "final"
_DEBUG_ONLY_TRACKS = frozenset({"toy"})
_FINAL_OUTPUT_TRACKS = ("GPV-S", "GPV-M", "GPV-L")
_FINAL_REPORT_PATH = Path("artifacts/final/report.md")


def debug_artifacts_dir(workspace_root: str | Path) -> Path:
    return Path(workspace_root) / _ARTIFACTS_DIRNAME / _DEBUG_DIRNAME


def final_artifacts_dir(workspace_root: str | Path) -> Path:
    return Path(workspace_root) / _ARTIFACTS_DIRNAME / _FINAL_DIRNAME


def final_report_path() -> Path:
    return _FINAL_REPORT_PATH


def final_report_assets_dir() -> Path:
    return _FINAL_REPORT_PATH.with_name(f"{_FINAL_REPORT_PATH.stem}_assets")


def timestamped_report_backup_path(timestamp: datetime) -> Path:
    suffix = timestamp.strftime("%Y%m%d-%H%M%S")
    return _FINAL_REPORT_PATH.with_name(f"{_FINAL_REPORT_PATH.stem}.backup-{suffix}.md")


def is_debug_only_track(track_name: str) -> bool:
    return track_name in _DEBUG_ONLY_TRACKS


def is_final_output_track(track_name: str) -> bool:
    return track_name in _FINAL_OUTPUT_TRACKS


def filter_requested_tracks(track_names: list[str], *, formal_only: bool) -> list[str]:
    if formal_only:
        return [track_name for track_name in track_names if is_final_output_track(track_name)]
    return track_names


def formal_track_names() -> list[str]:
    return list(_FINAL_OUTPUT_TRACKS)
