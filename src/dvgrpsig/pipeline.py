from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dvgrpsig.experiments import finalize_results
from dvgrpsig.plots import (
    save_n4_business_by_set_plot,
    save_set_m_signature_receipt_size_plot,
    save_set_m_timing_by_n_plot,
)
from dvgrpsig.report_builder import build_markdown_report, write_markdown_report


def build_report_bundle(
    *,
    bench_frame: pd.DataFrame,
    distribution_frame: pd.DataFrame,
    report_path: str | Path,
    assets_dir: str | Path,
    backup_timestamp: datetime | None = None,
) -> dict[str, Any]:
    report_path = Path(report_path)
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    bench_formal = finalize_results(bench_frame)
    distribution_formal = finalize_results(distribution_frame) if not distribution_frame.empty else distribution_frame

    backup_path = None
    if report_path.exists():
        timestamp = backup_timestamp or datetime.now()
        backup_path = report_path.with_name(f"{report_path.stem}.backup-{timestamp.strftime('%Y%m%d-%H%M%S')}.md")
        backup_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    figure_paths: dict[str, str] = {}
    created_paths: list[Path] = []

    if not bench_formal.empty:
        path, _ = save_set_m_timing_by_n_plot(
            frame=bench_formal,
            output_path=assets_dir / "set_m_timings_by_n.png",
            title="Set-M Timings by N",
        )
        created_paths.append(path)
        figure_paths["Set-M ProxySign Verify ReceiptGen Timings"] = f"{assets_dir.name}/{path.name}"

        path, _ = save_n4_business_by_set_plot(
            frame=bench_formal,
            output_path=assets_dir / "n4_business_by_set.png",
            title="N=4 Business Timings by Set",
        )
        created_paths.append(path)
        figure_paths["N=4 Business Timings by Set"] = f"{assets_dir.name}/{path.name}"

        path, _ = save_set_m_signature_receipt_size_plot(
            frame=bench_formal,
            output_path=assets_dir / "set_m_signature_receipt_sizes.png",
            title="Set-M Signature and Receipt Plaintext Sizes",
        )
        created_paths.append(path)
        figure_paths["Set-M Sigma and M_rcpt Sizes"] = f"{assets_dir.name}/{path.name}"

    markdown = build_markdown_report(
        bench_frame=bench_formal,
        distribution_frame=distribution_formal,
        report_title="DV Group Proxy Signature Report",
        figure_links=figure_paths,
    )
    written_report = write_markdown_report(markdown, report_path)

    return {
        "report_path": written_report,
        "backup_path": backup_path,
        "plot_paths": created_paths,
    }
