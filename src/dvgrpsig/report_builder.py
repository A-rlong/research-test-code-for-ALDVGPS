from __future__ import annotations

from pathlib import Path

import pandas as pd

from dvgrpsig.config import get_parameter_track
from dvgrpsig.experiment_io import formal_track_names
from dvgrpsig.experiments import summarize_bench


FORMAL_TRACKS = set(formal_track_names())
TRACK_LABELS = {
    "GPV-S": "Set-S",
    "GPV-M": "Set-M",
    "GPV-L": "Set-L",
}
OPERATION_LABELS = (
    ("keygen", "KeyGen"),
    ("audit_keygen", "AudKeyGen"),
    ("proxy_authorize", "ProxyAuth"),
    ("proxy_keygen", "ProxyKeyGen"),
    ("proxy_sign", "ProxySign"),
    ("verify", "Verify"),
    ("receipt_gen", "ReceiptGen"),
    ("open_receipt", "Open"),
    ("judge", "Judge"),
)


def filter_formal_tracks(frame: pd.DataFrame, *, track_column: str = "track") -> pd.DataFrame:
    if track_column not in frame.columns:
        return frame.copy()
    return frame.loc[frame[track_column].isin(FORMAL_TRACKS)].reset_index(drop=True)


def build_parameter_track_frame() -> pd.DataFrame:
    tracks = {track_name: get_parameter_track(track_name) for track_name in formal_track_names()}
    rows = []
    for parameter_name, accessor in (
        ("n", lambda track: track.n),
        ("m", lambda track: track.m),
        ("q", lambda track: track.q),
        ("sigma_1=sigma_2", lambda track: f"{track.sigma1}={track.sigma2}"),
        ("sigma_enc", lambda track: track.sigma_enc),
        ("B_z", lambda track: track.bz),
        ("proof rounds", lambda track: track.stern_rounds),
    ):
        row = {"Parameter": parameter_name}
        for track_name in formal_track_names():
            row[TRACK_LABELS[track_name]] = accessor(tracks[track_name])
        rows.append(row)
    return pd.DataFrame(rows)


def build_n1_timing_frame(bench_frame: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_bench(filter_formal_tracks(bench_frame))
    rows = []
    for algorithm, operation_label in OPERATION_LABELS:
        row = {"Operation": operation_label}
        for track_name in formal_track_names():
            match = summary.loc[
                (summary["track"] == track_name)
                & (summary["verifier_count"] == 1)
                & (summary["algorithm"] == algorithm),
                "mean_ms",
            ]
            row[TRACK_LABELS[track_name]] = "" if match.empty else round(float(match.iloc[0]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_markdown_report(
    *,
    bench_frame: pd.DataFrame,
    distribution_frame: pd.DataFrame,
    report_title: str = "DV Group Proxy Signature Report",
    figure_links: dict[str, str] | None = None,
) -> str:
    bench_formal = filter_formal_tracks(bench_frame)
    parameter_frame = build_parameter_track_frame()
    n1_timing_frame = build_n1_timing_frame(bench_formal)

    sections = [
        f"# {report_title}",
        "",
        "本文报告只汇总正式参数轨道，展示为 `Set-S/Set-M/Set-L`。开发阶段调试数据不会进入最终统计、图表和结论。",
        "",
        "审计打开与公开归责的正确性表述按方案要求采用“除可忽略的 LWE 解码失败概率外成立”的口径。本轮实验不运行 Sim 分布检验。",
        "",
        "## Parameter Sets",
        "",
        _dataframe_to_markdown(parameter_frame),
        "",
        "## N=1 Operation Timings",
        "",
        _dataframe_to_markdown(n1_timing_frame),
        "",
    ]

    if figure_links:
        sections.extend(
            [
                "## Figures",
                "",
                "下列图表均为最终正式轨道生成结果，图题和坐标轴保持英文。",
                "",
            ]
        )
        for title, link in figure_links.items():
            sections.extend([f"### {title}", "", f"![{title}]({link})", ""])

    return "\n".join(sections)


def write_markdown_report(markdown: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _round_frame(frame: pd.DataFrame, digits: dict[str, int]) -> pd.DataFrame:
    rounded = frame.copy()
    for column, precision in digits.items():
        if column in rounded.columns:
            rounded[column] = rounded[column].map(lambda value: round(float(value), precision))
    return rounded


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No formal rows available._"
    headers = [str(column) for column in frame.columns]
    divider = ["---"] * len(headers)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for record in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in record) + " |")
    return "\n".join(rows)
