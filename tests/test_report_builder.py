from __future__ import annotations

from pathlib import Path

import pandas as pd

from dvgrpsig.report_builder import (
    build_n1_timing_frame,
    build_markdown_report,
    build_parameter_track_frame,
    filter_formal_tracks,
    write_markdown_report,
)


def test_filter_formal_tracks_only_keeps_formal_rows() -> None:
    frame = pd.DataFrame(
        [
            {"track": "toy", "signature_bytes": 111},
            {"track": "GPV-S", "signature_bytes": 222},
            {"track": "GPV-L", "signature_bytes": 333},
        ]
    )
    filtered = filter_formal_tracks(frame)
    assert filtered["track"].tolist() == ["GPV-S", "GPV-L"]


def test_build_parameter_track_frame_lists_formal_tracks() -> None:
    frame = build_parameter_track_frame()
    assert frame.columns.tolist() == ["Parameter", "Set-S", "Set-M", "Set-L"]
    assert frame["Parameter"].tolist() == [
        "n",
        "m",
        "q",
        "sigma_1=sigma_2",
        "sigma_enc",
        "B_z",
        "proof rounds",
    ]
    assert frame.loc[frame["Parameter"] == "n", ["Set-S", "Set-M", "Set-L"]].iloc[0].tolist() == [256, 512, 768]
    assert frame.loc[frame["Parameter"] == "m", ["Set-S", "Set-M", "Set-L"]].iloc[0].tolist() == [11776, 23552, 35328]
    assert frame.loc[frame["Parameter"] == "proof rounds", ["Set-S", "Set-M", "Set-L"]].iloc[0].tolist() == [192, 224, 256]


def test_build_n1_timing_frame_uses_required_operation_order() -> None:
    rows = []
    operations = {
        "keygen": 1.0,
        "audit_keygen": 2.0,
        "proxy_authorize": 3.0,
        "proxy_keygen": 4.0,
        "proxy_sign": 5.0,
        "verify": 6.0,
        "receipt_gen": 7.0,
        "open_receipt": 8.0,
        "judge": 9.0,
        "simulate": 10.0,
        "symmetric_keygen": 11.0,
    }
    for track in ("GPV-S", "GPV-M", "GPV-L"):
        for algorithm, elapsed in operations.items():
            rows.append(
                {
                    "track": track,
                    "mode": "optimized-hybrid",
                    "verifier_count": 1,
                    "trial": 0,
                    "algorithm": algorithm,
                    "elapsed_ms": elapsed,
                    "repetitions": 1,
                    "success": 1,
                    "failure": 0,
                    "error": 0,
                    "signature_bytes": 100,
                    "m_rcpt_bytes": 50,
                    "receipt_bytes": 20,
                    "open_bytes": 8,
                }
            )

    frame = build_n1_timing_frame(pd.DataFrame(rows))

    assert frame.columns.tolist() == ["Operation", "Set-S", "Set-M", "Set-L"]
    assert frame["Operation"].tolist() == [
        "KeyGen",
        "AudKeyGen",
        "ProxyAuth",
        "ProxyKeyGen",
        "ProxySign",
        "Verify",
        "ReceiptGen",
        "Open",
        "Judge",
    ]
    assert "SymmetrickeyGen" not in frame["Operation"].tolist()


def test_build_markdown_report_uses_chinese_narrative_with_english_tables() -> None:
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
                "signature_bytes": 100,
                "m_rcpt_bytes": 40,
                "receipt_bytes": 60,
                "open_bytes": 20,
            },
            {
                "track": "GPV-S",
                "mode": "optimized-hybrid",
                "verifier_count": 4,
                "trial": 0,
                "algorithm": "proxy_sign",
                "elapsed_ms": 22.0,
                "repetitions": 1,
                "success": 1,
                "failure": 0,
                "error": 0,
                "signature_bytes": 220,
                "m_rcpt_bytes": 80,
                "receipt_bytes": 120,
                "open_bytes": 50,
            },
        ]
    )
    distribution = pd.DataFrame(
        [
            {
                "track": "toy",
                "verifier_count": 2,
                "samples_per_class": 10,
                "max_ks_stat": 0.5,
                "min_ks_pvalue": 0.2,
                "max_abs_cohen_d": 0.3,
                "auc": 0.61,
                "auc_ci_low": 0.55,
                "auc_ci_high": 0.67,
                "mmd_stat": 0.1,
                "mmd_pvalue": 0.03,
                "indistinguishable": False,
            },
            {
                "track": "GPV-M",
                "verifier_count": 4,
                "samples_per_class": 10,
                "max_ks_stat": 0.02,
                "min_ks_pvalue": 0.4,
                "max_abs_cohen_d": 0.01,
                "auc": 0.50,
                "auc_ci_low": 0.48,
                "auc_ci_high": 0.52,
                "mmd_stat": 0.001,
                "mmd_pvalue": 0.45,
                "indistinguishable": True,
            },
        ]
    )

    markdown = build_markdown_report(
        bench_frame=bench,
        distribution_frame=distribution,
        report_title="DV Group Proxy Signature Report",
        figure_links={"All Algorithm Timings": "实验报告_assets/timings_all_algorithms.png"},
    )

    assert "# DV Group Proxy Signature Report" in markdown
    assert "本文报告只汇总正式参数轨道" in markdown
    assert "## Parameter Sets" in markdown
    assert "## N=1 Operation Timings" in markdown
    assert "| Operation | Set-S | Set-M | Set-L |" in markdown
    assert "| Parameter | Set-S | Set-M | Set-L |" in markdown
    assert "toy" not in markdown
    assert "Set-S" in markdown
    assert "Set-M" in markdown
    assert "![All Algorithm Timings](实验报告_assets/timings_all_algorithms.png)" in markdown


def test_write_markdown_report_supports_obsidian_target_path(tmp_path: Path) -> None:
    target = tmp_path / "实验报告.md"
    content = "# Report\n\n内容"
    written = write_markdown_report(content, target)
    assert written == target
    assert target.read_text(encoding="utf-8") == content
