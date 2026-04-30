from __future__ import annotations

import pandas as pd

from dvgrpsig.experiments import finalize_results, run_bench, run_distribution, run_flow, summarize_bench


def test_run_flow_toy_reference_full_accepts():
    result = run_flow(track_name="toy", verifier_count=2, mode="reference-full", message="hello")
    assert result["verify_accepted"] is True
    assert result["judge_accepted"] is True
    assert result["zk_proof_bytes"] > 0


def test_finalize_results_filters_out_toy_rows():
    frame = pd.DataFrame([{"track": "toy"}, {"track": "GPV-S"}])
    filtered = finalize_results(frame)
    assert filtered["track"].tolist() == ["GPV-S"]


def test_run_bench_toy_returns_raw_dataframe():
    frame = run_bench(
        track_names=["toy"],
        verifier_counts=[1],
        trials=1,
        mode="optimized-hybrid",
        formal_only=False,
    )
    assert frame["track"].tolist() == ["toy"] * len(frame)
    assert set(frame["algorithm"]) >= {
        "keygen",
        "audit_keygen",
        "proxy_authorize",
        "proxy_keygen",
        "proxy_sign",
        "verify",
        "receipt_gen",
        "open_receipt",
        "judge",
    }
    assert "simulate" not in set(frame["algorithm"])
    assert "symmetric_keygen" not in set(frame["algorithm"])
    assert {"elapsed_ms", "success", "failure", "m_rcpt_bytes"}.issubset(frame.columns)
    assert "simulate_signature_bytes" not in frame.columns


def test_summarize_bench_aggregates_trial_rows():
    frame = pd.DataFrame(
        [
            {
                "track": "GPV-S",
                "mode": "optimized-hybrid",
                "verifier_count": 1,
                "trial": 0,
                "algorithm": "proxy_sign",
                "elapsed_ms": 10.0,
                "repetitions": 1,
                "success": 1,
                "failure": 0,
                "error": 0,
                "signature_bytes": 100,
                "m_rcpt_bytes": 50,
                "receipt_bytes": 20,
                "open_bytes": 8,
            },
            {
                "track": "GPV-S",
                "mode": "optimized-hybrid",
                "verifier_count": 1,
                "trial": 1,
                "algorithm": "proxy_sign",
                "elapsed_ms": 14.0,
                "repetitions": 1,
                "success": 1,
                "failure": 0,
                "error": 0,
                "signature_bytes": 120,
                "m_rcpt_bytes": 60,
                "receipt_bytes": 25,
                "open_bytes": 9,
            },
        ]
    )
    summary = summarize_bench(frame)
    row = summary.iloc[0]
    assert row["mean_ms"] == 12.0
    assert row["repetitions"] == 2
    assert row["signature_bytes"] == 110.0
    assert row["m_rcpt_bytes"] == 55.0


def test_run_distribution_toy_returns_metric_dataframe():
    frame = run_distribution(
        track_names=["toy"],
        verifier_counts=[1],
        samples_per_class=2,
        formal_only=False,
    )
    assert frame["track"].tolist() == ["toy"]
    assert {"max_ks_stat", "auc", "mmd_pvalue", "indistinguishable"}.issubset(frame.columns)
