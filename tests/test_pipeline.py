from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from dvgrpsig.pipeline import build_report_bundle


def test_build_report_bundle_backs_up_existing_report_and_writes_formal_only_output(tmp_path: Path) -> None:
    bench = pd.DataFrame(
        [
            {
                "track": "toy",
                "mode": "reference-full",
                "verifier_count": 2,
                "trial": 0,
                "algorithm": "proxy_sign",
                "elapsed_ms": 10.0,
                "repetitions": 1,
                "success": 1,
                "failure": 0,
                "error": 0,
                "signature_bytes": 10,
                "m_rcpt_bytes": 4,
                "receipt_bytes": 20,
                "open_bytes": 8,
            },
            {
                "track": "GPV-S",
                "mode": "optimized-hybrid",
                "verifier_count": 2,
                "trial": 0,
                "algorithm": "proxy_sign",
                "elapsed_ms": 100.0,
                "repetitions": 1,
                "success": 1,
                "failure": 0,
                "error": 0,
                "signature_bytes": 100,
                "m_rcpt_bytes": 40,
                "receipt_bytes": 200,
                "open_bytes": 80,
            },
        ]
    )
    distribution = pd.DataFrame(
        [
            {
                "track": "toy",
                "verifier_count": 2,
                "samples_per_class": 10,
                "max_ks_stat": 0.4,
                "min_ks_pvalue": 0.1,
                "max_abs_cohen_d": 0.2,
                "auc": 0.6,
                "auc_ci_low": 0.55,
                "auc_ci_high": 0.65,
                "mmd_stat": 0.1,
                "mmd_pvalue": 0.05,
                "indistinguishable": False,
            },
            {
                "track": "GPV-M",
                "verifier_count": 2,
                "samples_per_class": 10,
                "max_ks_stat": 0.04,
                "min_ks_pvalue": 0.2,
                "max_abs_cohen_d": 0.02,
                "auc": 0.5,
                "auc_ci_low": 0.48,
                "auc_ci_high": 0.52,
                "mmd_stat": 0.005,
                "mmd_pvalue": 0.5,
                "indistinguishable": True,
            },
        ]
    )
    report_path = tmp_path / "实验报告.md"
    report_path.write_text("# old report", encoding="utf-8")
    assets_dir = tmp_path / "实验报告_assets"

    result = build_report_bundle(
        bench_frame=bench,
        distribution_frame=distribution,
        report_path=report_path,
        assets_dir=assets_dir,
        backup_timestamp=datetime(2026, 4, 27, 11, 12, 13),
    )

    assert result["report_path"] == report_path
    assert result["backup_path"] == tmp_path / "实验报告.backup-20260427-111213.md"
    assert result["backup_path"].read_text(encoding="utf-8") == "# old report"
    report_text = report_path.read_text(encoding="utf-8")
    assert "toy" not in report_text
    assert "Set-S" in report_text
    assert "Set-M" in report_text
    assert result["plot_paths"]
    assert all(path.exists() for path in result["plot_paths"])
    assert sorted(path.name for path in result["plot_paths"]) == [
        "n4_business_by_set.png",
        "set_m_signature_receipt_sizes.png",
        "set_m_timings_by_n.png",
    ]
