from __future__ import annotations

from datetime import datetime
from pathlib import Path

import dvgrpsig.experiment_io as experiment_io


def test_debug_artifacts_dir_is_under_artifacts_directory():
    workspace = Path("C:/tmp/lab")
    assert experiment_io.debug_artifacts_dir(workspace) == workspace / "artifacts" / "debug"


def test_final_artifacts_dir_is_under_artifacts_directory():
    workspace = Path("C:/tmp/lab")
    assert experiment_io.final_artifacts_dir(workspace) == workspace / "artifacts" / "final"


def test_final_report_paths_are_repo_local():
    assert experiment_io.final_report_path() == Path("artifacts/final/report.md")
    assert experiment_io.final_report_assets_dir() == Path("artifacts/final/report_assets")


def test_timestamped_report_backup_path_uses_report_stem_and_given_time():
    timestamp = datetime(2026, 4, 27, 10, 11, 12)
    assert experiment_io.timestamped_report_backup_path(timestamp) == Path(
        "artifacts/final/report.backup-20260427-101112.md"
    )


def test_track_selection_helpers_distinguish_debug_and_formal_tracks():
    assert experiment_io.is_debug_only_track("toy") is True
    assert experiment_io.is_debug_only_track("GPV-S") is False
    assert experiment_io.is_final_output_track("GPV-S") is True
    assert experiment_io.is_final_output_track("GPV-M") is True
    assert experiment_io.is_final_output_track("GPV-L") is True
    assert experiment_io.is_final_output_track("toy") is False


def test_filter_requested_tracks_excludes_toy_when_formal_only():
    assert experiment_io.filter_requested_tracks(["toy", "GPV-S"], formal_only=False) == ["toy", "GPV-S"]
    assert experiment_io.filter_requested_tracks(["toy", "GPV-S"], formal_only=True) == ["GPV-S"]
