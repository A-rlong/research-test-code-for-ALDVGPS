from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dvgrpsig.experiment_io import (
    debug_artifacts_dir,
    final_artifacts_dir,
    final_report_assets_dir,
    final_report_path,
)
from dvgrpsig.experiments import finalize_results, run_bench, run_distribution, run_flow
from dvgrpsig.pipeline import build_report_bundle


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dvgrpsig")
    subparsers = parser.add_subparsers(dest="command", required=True)

    flow_parser = subparsers.add_parser("run_flow")
    flow_parser.add_argument("--track", required=True)
    flow_parser.add_argument("--verifier-count", type=int, required=True)
    flow_parser.add_argument("--mode", required=True)
    flow_parser.add_argument("--message", required=True)
    flow_parser.add_argument("--output", required=True)

    bench_parser = subparsers.add_parser("run_bench")
    bench_parser.add_argument("--tracks", required=True)
    bench_parser.add_argument("--verifier-counts", required=True)
    bench_parser.add_argument("--trials", type=int, required=True)
    bench_parser.add_argument("--mode", required=True)
    bench_parser.add_argument("--formal-only", action="store_true")
    bench_parser.add_argument("--workspace-root", default=".")
    bench_parser.add_argument("--output")

    distribution_parser = subparsers.add_parser("run_distribution")
    distribution_parser.add_argument("--tracks", required=True)
    distribution_parser.add_argument("--verifier-counts", required=True)
    distribution_parser.add_argument("--samples-per-class", type=int, required=True)
    distribution_parser.add_argument("--mode", default="optimized-hybrid")
    distribution_parser.add_argument("--formal-only", action="store_true")
    distribution_parser.add_argument("--workspace-root", default=".")
    distribution_parser.add_argument("--output")

    report_parser = subparsers.add_parser("build_report")
    report_parser.add_argument("--bench-csv", required=True)
    report_parser.add_argument("--distribution-csv", required=True)
    report_parser.add_argument("--report-path")
    report_parser.add_argument("--assets-dir")

    args = parser.parse_args(argv)

    if args.command == "run_flow":
        payload = run_flow(
            track_name=args.track,
            verifier_count=args.verifier_count,
            mode=args.mode,
            message=args.message,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if args.command == "run_bench":
        tracks = _parse_csv_list(args.tracks)
        verifier_counts = [int(item) for item in _parse_csv_list(args.verifier_counts)]
        frame = run_bench(
            track_names=tracks,
            verifier_counts=verifier_counts,
            trials=args.trials,
            mode=args.mode,
            formal_only=args.formal_only,
        )
        workspace_root = Path(args.workspace_root)
        default_dir = final_artifacts_dir(workspace_root) if args.formal_only else debug_artifacts_dir(workspace_root)
        output_path = Path(args.output) if args.output else default_dir / "bench.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return 0

    if args.command == "run_distribution":
        tracks = _parse_csv_list(args.tracks)
        verifier_counts = [int(item) for item in _parse_csv_list(args.verifier_counts)]
        frame = run_distribution(
            track_names=tracks,
            verifier_counts=verifier_counts,
            samples_per_class=args.samples_per_class,
            formal_only=args.formal_only,
            mode=args.mode,
        )
        workspace_root = Path(args.workspace_root)
        default_dir = final_artifacts_dir(workspace_root) if args.formal_only else debug_artifacts_dir(workspace_root)
        output_path = Path(args.output) if args.output else default_dir / "distribution.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return 0

    if args.command == "build_report":
        bench_frame = pd.read_csv(args.bench_csv)
        distribution_frame = pd.read_csv(args.distribution_csv)
        report_path = Path(args.report_path) if args.report_path else final_report_path()
        assets_dir = Path(args.assets_dir) if args.assets_dir else final_report_assets_dir()
        build_report_bundle(
            bench_frame=finalize_results(bench_frame),
            distribution_frame=finalize_results(distribution_frame),
            report_path=report_path,
            assets_dir=assets_dir,
        )
        return 0

    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
