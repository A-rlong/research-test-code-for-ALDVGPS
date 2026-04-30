from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dvgrpsig.experiments import summarize_bench
from dvgrpsig.report_builder import OPERATION_LABELS, TRACK_LABELS, filter_formal_tracks


OPERATION_LABEL_MAP = dict(OPERATION_LABELS)


def save_timing_grid_plot(
    *,
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    algorithms: tuple[str, ...] | None = None,
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bench(filter_formal_tracks(frame))
    if algorithms is not None:
        summary = summary.loc[summary["algorithm"].isin(algorithms)].reset_index(drop=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    tracks = summary["track"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, max(len(tracks), 1), figsize=(6 * max(len(tracks), 1), 4.5), squeeze=False)
    for axis, track_name in zip(axes[0], tracks, strict=False):
        track_frame = summary.loc[summary["track"] == track_name]
        sns.lineplot(
            data=track_frame,
            x="verifier_count",
            y="mean_ms",
            hue="algorithm",
            marker="o",
            ax=axis,
        )
        axis.set_title(track_name)
        axis.set_xlabel("N")
        axis.set_ylabel("Mean Time (ms)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180)
    plt.close(figure)
    return path, summary


def save_size_line_plot(
    *,
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bench(filter_formal_tracks(frame))
    plotted = (
        summary.loc[:, ["track", "verifier_count", "signature_bytes", "m_rcpt_bytes"]]
        .drop_duplicates()
        .melt(id_vars=["track", "verifier_count"], value_vars=["signature_bytes", "m_rcpt_bytes"], var_name="artifact", value_name="bytes")
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    for axis, artifact_name in zip(axes[0], ["signature_bytes", "m_rcpt_bytes"], strict=False):
        artifact_frame = plotted.loc[plotted["artifact"] == artifact_name]
        sns.lineplot(data=artifact_frame, x="verifier_count", y="bytes", hue="track", marker="o", ax=axis)
        axis.set_title(artifact_name)
        axis.set_xlabel("N")
        axis.set_ylabel("Bytes")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180)
    plt.close(figure)
    return path, plotted


def save_ratio_stacked_plot(
    *,
    frame: pd.DataFrame,
    verifier_count: int,
    output_path: str | Path,
    title: str,
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bench(filter_formal_tracks(frame))
    plotted = summary.loc[summary["verifier_count"] == verifier_count, ["track", "algorithm", "mean_ms"]].copy()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11, 5))
    if plotted.empty:
        axis.set_title(title)
        axis.set_xlabel("Track")
        axis.set_ylabel("Time Share")
        axis.text(0.5, 0.5, f"No rows for N={verifier_count}", ha="center", va="center", transform=axis.transAxes)
    else:
        plotted["share"] = plotted.groupby("track")["mean_ms"].transform(lambda column: column / column.sum() if column.sum() else 0.0)
        pivot = plotted.pivot(index="track", columns="algorithm", values="share").fillna(0.0)
        pivot.plot(kind="bar", stacked=True, ax=axis, colormap="tab20")
        axis.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.set_title(title)
    axis.set_xlabel("Track")
    axis.set_ylabel("Time Share")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180)
    plt.close(figure)
    return path, plotted


def save_distribution_metric_plot(
    *,
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    metric_column: str,
    ylabel: str,
) -> tuple[Path, pd.DataFrame]:
    plotted = filter_formal_tracks(frame)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=plotted, x="verifier_count", y=metric_column, hue="track", marker="o", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("N")
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180)
    plt.close(figure)
    return path, plotted


def save_set_m_timing_by_n_plot(
    *,
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str = "Set-M Timings by N",
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bench(filter_formal_tracks(frame))
    algorithms = ("proxy_sign", "verify", "receipt_gen")
    plotted = summary.loc[
        (summary["track"] == "GPV-M") & (summary["algorithm"].isin(algorithms)),
        ["verifier_count", "algorithm", "mean_ms"],
    ].copy()
    plotted["Operation"] = plotted["algorithm"].map(OPERATION_LABEL_MAP)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if plotted.empty:
        axis.text(0.5, 0.5, "No Set-M rows", ha="center", va="center", transform=axis.transAxes)
    else:
        sns.lineplot(data=plotted, x="verifier_count", y="mean_ms", hue="Operation", marker="o", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("N")
    axis.set_ylabel("Mean Time (ms)")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180)
    plt.close(figure)
    return path, plotted


def save_n4_business_by_set_plot(
    *,
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str = "N=4 Business Timings by Set",
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bench(filter_formal_tracks(frame))
    algorithms = ("proxy_authorize", "proxy_keygen", "proxy_sign", "verify", "receipt_gen")
    plotted = summary.loc[
        (summary["verifier_count"] == 4) & (summary["algorithm"].isin(algorithms)),
        ["track", "algorithm", "mean_ms"],
    ].copy()
    plotted["Set"] = pd.Categorical(
        plotted["track"].map(TRACK_LABELS),
        categories=["Set-S", "Set-M", "Set-L"],
        ordered=True,
    )
    plotted["Operation"] = plotted["algorithm"].map(OPERATION_LABEL_MAP)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if plotted.empty:
        axis.text(0.5, 0.5, "No N=4 rows", ha="center", va="center", transform=axis.transAxes)
    else:
        sns.lineplot(data=plotted.sort_values("Set"), x="Set", y="mean_ms", hue="Operation", marker="o", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Parameter Set")
    axis.set_ylabel("Mean Time (ms)")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180)
    plt.close(figure)
    return path, plotted


def save_set_m_signature_receipt_size_plot(
    *,
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str = "Set-M Signature and Receipt Plaintext Sizes",
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bench(filter_formal_tracks(frame))
    size_rows = (
        summary.loc[
            summary["track"] == "GPV-M",
            ["track", "verifier_count", "signature_bytes", "m_rcpt_bytes"],
        ]
        .drop_duplicates(["track", "verifier_count"])
        .sort_values("verifier_count")
        .reset_index(drop=True)
    )
    plotted = size_rows.melt(
        id_vars=["track", "verifier_count"],
        value_vars=["signature_bytes", "m_rcpt_bytes"],
        var_name="artifact",
        value_name="bytes",
    )
    plotted["Artifact"] = plotted["artifact"].map({"signature_bytes": "sigma", "m_rcpt_bytes": "M_rcpt"})
    plotted["MiB"] = plotted["bytes"].astype(float) / (1024 * 1024)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if plotted.empty:
        axis.text(0.5, 0.5, "No Set-M size rows", ha="center", va="center", transform=axis.transAxes)
    else:
        sns.lineplot(data=plotted, x="verifier_count", y="MiB", hue="Artifact", marker="o", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("N")
    axis.set_ylabel("Size (MiB)")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180)
    plt.close(figure)
    return path, plotted
