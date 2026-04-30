from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dvgrpsig.cli import main


def test_cli_run_flow_writes_json_result(tmp_path: Path) -> None:
    output_path = tmp_path / "run-flow.json"

    exit_code = main(
        [
            "run_flow",
            "--track",
            "toy",
            "--verifier-count",
            "1",
            "--mode",
            "reference-full",
            "--message",
            "hello",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["track"] == "toy"
    assert payload["verify_accepted"] is True
    assert payload["judge_accepted"] is True
    assert payload["zk_proof_bytes"] > 0


def test_cli_run_bench_writes_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "bench.csv"
    exit_code = main(
        [
            "run_bench",
            "--tracks",
            "toy",
            "--verifier-counts",
            "1",
            "--trials",
            "1",
            "--mode",
            "optimized-hybrid",
            "--workspace-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    frame = pd.read_csv(output_path)
    assert {"track", "algorithm", "elapsed_ms", "m_rcpt_bytes"}.issubset(frame.columns)
    assert "simulate" not in set(frame["algorithm"])


def test_cli_run_distribution_writes_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "distribution.csv"
    exit_code = main(
        [
            "run_distribution",
            "--tracks",
            "toy",
            "--verifier-counts",
            "1",
            "--samples-per-class",
            "2",
            "--workspace-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    frame = pd.read_csv(output_path)
    assert {"track", "auc", "mmd_pvalue", "indistinguishable"}.issubset(frame.columns)


def test_cli_build_report_writes_markdown_and_assets(tmp_path: Path) -> None:
    bench_csv = tmp_path / "bench.csv"
    distribution_csv = tmp_path / "distribution.csv"
    report_path = tmp_path / "实验报告.md"
    assets_dir = tmp_path / "实验报告_assets"

    pd.DataFrame(
        [
            {
                "track": "GPV-S",
                "mode": "optimized-hybrid",
                "verifier_count": 1,
                "trial": 0,
                "algorithm": "proxy_sign",
                "elapsed_ms": 12.0,
                "repetitions": 1,
                "success": 1,
                "failure": 0,
                "error": 0,
                "signature_bytes": 120,
                "m_rcpt_bytes": 80,
                "receipt_bytes": 60,
                "open_bytes": 20,
            }
        ]
    ).to_csv(bench_csv, index=False)
    pd.DataFrame(
        [
            {
                "track": "GPV-S",
                "verifier_count": 1,
                "samples_per_class": 4,
                "max_ks_stat": 0.04,
                "min_ks_pvalue": 0.4,
                "max_abs_cohen_d": 0.02,
                "auc": 0.5,
                "auc_ci_low": 0.48,
                "auc_ci_high": 0.52,
                "mmd_stat": 0.001,
                "mmd_pvalue": 0.45,
                "indistinguishable": True,
            }
        ]
    ).to_csv(distribution_csv, index=False)

    exit_code = main(
        [
            "build_report",
            "--bench-csv",
            str(bench_csv),
            "--distribution-csv",
            str(distribution_csv),
            "--report-path",
            str(report_path),
            "--assets-dir",
            str(assets_dir),
        ]
    )
    assert exit_code == 0
    assert report_path.exists()
    assert assets_dir.exists()
