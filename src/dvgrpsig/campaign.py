from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from dvgrpsig.experiment_io import final_artifacts_dir, final_report_assets_dir, final_report_path, formal_track_names
from dvgrpsig.experiments import run_bench
from dvgrpsig.pipeline import build_report_bundle


DEFAULT_VERIFIER_COUNTS = (1, 2, 4, 8, 16)
DEFAULT_DISTRIBUTION_COUNTS = (1, 8)


@dataclass(slots=True)
class CampaignConfig:
    workspace_root: str
    tracks: list[str]
    verifier_counts: list[int]
    trials: int
    mode: str
    distribution_verifier_counts: list[int]
    distribution_samples_per_class: int
    report_path: str
    assets_dir: str
    skip_existing: bool


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv_list(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(value)]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_timestamp()}] {message}\n")


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _shard_name(prefix: str, track: str, verifier_count: int) -> str:
    safe_track = track.replace("/", "_")
    return f"{prefix}_{safe_track}_n{verifier_count}.csv"


def _load_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists() and path.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_formal_campaign(config: CampaignConfig) -> dict[str, str]:
    workspace_root = Path(config.workspace_root).resolve()
    artifacts_root = final_artifacts_dir(workspace_root)
    shards_dir = artifacts_root / "shards"
    log_path = artifacts_root / "formal_campaign.log"
    manifest_path = artifacts_root / "formal_campaign_manifest.json"
    report_path = Path(config.report_path)
    assets_dir = Path(config.assets_dir)

    payload = {
        "status": "running",
        "started_at": _utc_timestamp(),
        "config": asdict(config),
    }
    _write_manifest(manifest_path, payload)
    _append_log(log_path, "Formal campaign started.")

    for track in config.tracks:
        for verifier_count in config.verifier_counts:
            shard_path = shards_dir / _shard_name("bench", track, verifier_count)
            if config.skip_existing and shard_path.exists() and shard_path.stat().st_size > 0:
                _append_log(log_path, f"Skip existing bench shard {shard_path.name}.")
                continue
            _append_log(log_path, f"Running bench shard track={track} N={verifier_count} trials={config.trials}.")
            bench_frame = run_bench(
                track_names=[track],
                verifier_counts=[verifier_count],
                trials=config.trials,
                mode=config.mode,
                formal_only=True,
            )
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            bench_frame.to_csv(shard_path, index=False)
            _append_log(log_path, f"Wrote bench shard {shard_path.name} with {len(bench_frame)} rows.")

    _append_log(log_path, "Distribution campaign skipped by current formal experiment plan.")

    bench_paths = sorted(shards_dir.glob("bench_*.csv"))
    bench_frame = _load_csvs(bench_paths)
    distribution_frame = pd.DataFrame()

    bench_output = artifacts_root / "bench_full.csv"
    distribution_output = artifacts_root / "distribution_full.csv"
    bench_frame.to_csv(bench_output, index=False)
    distribution_frame.to_csv(distribution_output, index=False)
    _append_log(log_path, f"Aggregated bench shards into {bench_output.name}.")
    _append_log(log_path, f"Aggregated distribution shards into {distribution_output.name}.")

    report_result = build_report_bundle(
        bench_frame=bench_frame,
        distribution_frame=distribution_frame,
        report_path=report_path,
        assets_dir=assets_dir,
    )
    _append_log(log_path, f"Report written to {report_result['report_path']}.")

    payload.update(
        {
            "status": "completed",
            "completed_at": _utc_timestamp(),
            "bench_output": str(bench_output),
            "distribution_output": str(distribution_output),
            "report_path": str(report_result["report_path"]),
        }
    )
    _write_manifest(manifest_path, payload)
    _append_log(log_path, "Formal campaign completed.")
    return {
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "bench_output": str(bench_output),
        "distribution_output": str(distribution_output),
        "report_path": str(report_result["report_path"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dvgrpsig-formal-campaign")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--tracks", default=",".join(formal_track_names()))
    parser.add_argument("--verifier-counts", default=",".join(str(value) for value in DEFAULT_VERIFIER_COUNTS))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--mode", default="optimized-hybrid")
    parser.add_argument(
        "--distribution-verifier-counts",
        default=",".join(str(value) for value in DEFAULT_DISTRIBUTION_COUNTS),
    )
    parser.add_argument("--distribution-samples-per-class", type=int, default=20_000)
    parser.add_argument("--report-path", default=str(final_report_path()))
    parser.add_argument("--assets-dir", default=str(final_report_assets_dir()))
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = CampaignConfig(
        workspace_root=args.workspace_root,
        tracks=_parse_csv_list(args.tracks),
        verifier_counts=_parse_int_csv_list(args.verifier_counts),
        trials=args.trials,
        mode=args.mode,
        distribution_verifier_counts=_parse_int_csv_list(args.distribution_verifier_counts),
        distribution_samples_per_class=args.distribution_samples_per_class,
        report_path=args.report_path,
        assets_dir=args.assets_dir,
        skip_existing=not args.no_skip_existing,
    )
    result = run_formal_campaign(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
