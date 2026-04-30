from __future__ import annotations

from pathlib import Path

import pandas as pd

import dvgrpsig.campaign as campaign


def test_run_formal_campaign_writes_shards_and_manifest(tmp_path: Path, monkeypatch) -> None:
    def fake_run_bench(*, track_names, verifier_counts, trials, mode, formal_only):
        assert formal_only is True
        return pd.DataFrame(
            [
                {
                    "track": track_names[0],
                    "mode": mode,
                    "verifier_count": verifier_counts[0],
                    "trial": 0,
                    "algorithm": "proxy_sign",
                    "elapsed_ms": 1.0,
                    "repetitions": 1,
                    "success": 1,
                    "failure": 0,
                    "error": 0,
                    "signature_bytes": 10,
                    "m_rcpt_bytes": 8,
                    "receipt_bytes": 6,
                    "open_bytes": 4,
                }
            ]
        )

    def fake_build_report_bundle(*, bench_frame, distribution_frame, report_path, assets_dir):
        assert distribution_frame.empty
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text("# report", encoding="utf-8")
        Path(assets_dir).mkdir(parents=True, exist_ok=True)
        return {"report_path": Path(report_path), "backup_path": None, "plot_paths": []}

    monkeypatch.setattr(campaign, "run_bench", fake_run_bench)
    monkeypatch.setattr(campaign, "build_report_bundle", fake_build_report_bundle)

    config = campaign.CampaignConfig(
        workspace_root=str(tmp_path),
        tracks=["GPV-S"],
        verifier_counts=[1],
        trials=1,
        mode="optimized-hybrid",
        distribution_verifier_counts=[1],
        distribution_samples_per_class=4,
        report_path=str(tmp_path / "实验报告.md"),
        assets_dir=str(tmp_path / "实验报告_assets"),
        skip_existing=True,
    )

    result = campaign.run_formal_campaign(config)

    assert Path(result["manifest_path"]).exists()
    assert Path(result["log_path"]).exists()
    assert Path(result["bench_output"]).exists()
    assert Path(result["distribution_output"]).exists()
    assert Path(result["report_path"]).exists()
    assert (tmp_path / "artifacts" / "final" / "shards" / "bench_GPV-S_n1.csv").exists()
    assert not (tmp_path / "artifacts" / "final" / "shards" / "distribution_GPV-S_n1.csv").exists()


def test_campaign_defaults_match_formal_experiment_plan() -> None:
    assert campaign.DEFAULT_VERIFIER_COUNTS == (1, 2, 4, 8, 16)
    parser = campaign._build_parser()
    args = parser.parse_args([])
    assert args.trials == 5
