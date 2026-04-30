from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from dvgrpsig.config import get_parameter_track
from dvgrpsig.experiment_io import filter_requested_tracks
from dvgrpsig.protocol import (
    audit_keygen,
    estimate_receipt_plaintext_bytes,
    judge,
    keygen,
    open_receipt,
    proxy_authorize,
    proxy_keygen,
    proxy_sign,
    receipt_gen,
    setup,
    simulate,
    symmetric_keygen,
    verify,
)
from dvgrpsig.reporting import filter_formal_tracks
from dvgrpsig.serialization import canonical_encode
from dvgrpsig.types import Context, ProxyPolicy, Roster, RosterEntry


BENCH_ALGORITHMS = (
    "keygen",
    "audit_keygen",
    "proxy_authorize",
    "proxy_keygen",
    "proxy_sign",
    "verify",
    "receipt_gen",
    "open_receipt",
    "judge",
)
BUSINESS_GROWTH_ALGORITHMS = (
    "proxy_authorize",
    "proxy_keygen",
    "proxy_sign",
    "verify",
    "receipt_gen",
)


def _build_roster(pp, verifier_keypairs):
    entries = []
    for index, keypair in enumerate(verifier_keypairs, start=1):
        entries.append(
            RosterEntry(
                member_index=index,
                member_id=f"verifier-{index}",
                public_key=keypair.public_key,
            )
        )
    roster_root = pp.hash_bytes(
        canonical_encode([entry.public_key.fingerprint for entry in entries]),
        32,
        domain=b"roster-root",
    ).hex()
    return Roster(entries=entries, epoch=f"{pp.track.name}-epoch", roster_root=roster_root)


def _build_context(roster: Roster, trial_index: int, action_type: str = "release") -> Context:
    return Context(
        epoch=roster.epoch,
        roster_root=roster.roster_root,
        session_id=f"session-{trial_index}",
        tx_id=f"tx-{trial_index}",
        action_type=action_type,
        tau="2026-04-27T12:00:00Z",
    )


def _build_policy(trial_index: int) -> ProxyPolicy:
    return ProxyPolicy(
        policy_id=f"policy-{trial_index}",
        description="delegated release policy",
        valid_from="2026-04-27T00:00:00Z",
        valid_to="2026-12-31T23:59:59Z",
        allowed_action_types=("release", "distribution"),
    )


def _timed_keygen(pp, actor_label: str):
    start = time.perf_counter()
    keypair = keygen(pp, actor_label=actor_label)
    return keypair, (time.perf_counter() - start) * 1000.0


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def _packed_bits_to_bytes(bits: int) -> int:
    return _ceil_div(max(bits, 0), 8)


def _mod_q_vector_bytes(length: int, q: int) -> int:
    return _packed_bits_to_bytes(length * q.bit_length())


def _sparse_challenge_bytes(pp) -> int:
    index_bits = max(1, (pp.n - 1).bit_length())
    return _packed_bits_to_bytes(pp.challenge_weight * (index_bits + 1))


def _ternary_vector_bytes(length: int) -> int:
    return _packed_bits_to_bytes(length * 2)


def _permutation_bytes(length: int) -> int:
    return _packed_bits_to_bytes(length * max(1, (length - 1).bit_length()))


def _receipt_proof_component_bytes(pp, proof) -> int:
    if proof.scheme == "fs-sternext-symmetric-capsule":
        return len(canonical_encode(proof.to_canonical()))
    witness_rows, witness_columns = tuple(int(value) for value in proof.witness_shape)
    base_len = witness_rows * witness_columns
    extended_len = 3 * base_len
    per_round_public = 3 * 32 + 1 + 2 * 32
    total = len(proof.scheme.encode("utf-8")) + 32 + 32 + 8 + 8 + 8
    for round_record in proof.rounds:
        branch = int(round_record["response"]["branch"])
        response_bytes = 0
        if branch == 1:
            response_bytes = _ternary_vector_bytes(extended_len) + _mod_q_vector_bytes(extended_len, pp.q)
        elif branch in (2, 3):
            response_bytes = _permutation_bytes(extended_len) + _mod_q_vector_bytes(extended_len, pp.q)
        total += per_round_public + response_bytes
    return total


def _signature_component_bytes(pp, signature) -> int:
    verifier_count = len(signature.u_blocks)
    certificate = signature.certificate
    u_bytes = sum(_mod_q_vector_bytes(block.size, pp.q) for block in signature.u_blocks)
    z_bytes = sum(_mod_q_vector_bytes(block.size, pp.q) for block in signature.z_blocks)
    certificate_bytes = (
        len(canonical_encode(certificate.w_payload))
        + _sparse_challenge_bytes(pp)
        + _mod_q_vector_bytes(certificate.response_y1.size, pp.q)
    )
    return (
        u_bytes
        + _sparse_challenge_bytes(pp)
        + z_bytes
        + certificate_bytes
        + len(canonical_encode(signature.context))
        + verifier_count * 32
    )


def _receipt_plaintext_component_bytes(pp, opened) -> int:
    return (
        len(opened.member_id.encode("utf-8"))
        + 32
        + _receipt_proof_component_bytes(pp, opened.receipt_proof)
        + 32
        + _packed_bits_to_bytes(opened.rho.size)
    )


def run_flow(
    *,
    track_name: str,
    verifier_count: int,
    mode: str,
    message: str,
    trial_index: int = 0,
) -> dict[str, Any]:
    trial = _execute_trial(
        track_name=track_name,
        verifier_count=verifier_count,
        mode=mode,
        message=message,
        trial_index=trial_index,
    )
    return {
        "track": track_name,
        "mode": mode,
        "verifier_count": verifier_count,
        "verify_accepted": trial["verify_accepted"],
        "judge_accepted": trial["judge_accepted"],
        "signature_bytes": trial["signature_bytes"],
        "zk_proof_bytes": trial["zk_proof_bytes"],
        "m_rcpt_bytes": trial["m_rcpt_bytes"],
        "receipt_bytes": trial["receipt_bytes"],
        "open_bytes": trial["open_bytes"],
    }


def _execute_trial(
    *,
    track_name: str,
    verifier_count: int,
    mode: str,
    message: str,
    trial_index: int,
) -> dict[str, Any]:
    pp = setup(get_parameter_track(track_name))
    timings_ms: dict[str, float] = {}

    keygen_timings = []
    authorizer, elapsed_ms = _timed_keygen(pp, f"{track_name}-authorizer-{trial_index}")
    keygen_timings.append(elapsed_ms)
    proxy, elapsed_ms = _timed_keygen(pp, f"{track_name}-proxy-{trial_index}")
    keygen_timings.append(elapsed_ms)
    verifiers = []
    for index in range(1, verifier_count + 1):
        verifier, elapsed_ms = _timed_keygen(pp, f"{track_name}-verifier-{trial_index}-{index}")
        verifiers.append(verifier)
        keygen_timings.append(elapsed_ms)
    timings_ms["keygen"] = sum(keygen_timings) / len(keygen_timings)

    roster = _build_roster(pp, verifiers)
    context = _build_context(roster, trial_index)
    policy = _build_policy(trial_index)
    receipt_plaintext_bytes = (
        estimate_receipt_plaintext_bytes(pp, roster.entries[0]) if mode == "reference-full" else 64
    )

    start = time.perf_counter()
    auditor = audit_keygen(pp, receipt_plaintext_bytes=receipt_plaintext_bytes)
    timings_ms["audit_keygen"] = (time.perf_counter() - start) * 1000.0

    shared_keys = symmetric_keygen(roster)

    start = time.perf_counter()
    certificate = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    timings_ms["proxy_authorize"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    derived = proxy_keygen(pp, proxy, certificate, roster)
    timings_ms["proxy_keygen"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    signature = proxy_sign(
        pp=pp,
        message=message,
        certificate=certificate,
        context=context,
        roster=roster,
        proxy_derived_key=derived,
        shared_keys=shared_keys,
        mode=mode,
    )
    timings_ms["proxy_sign"] = (time.perf_counter() - start) * 1000.0

    verifier_entry = roster.entries[0]
    start = time.perf_counter()
    verify_result = verify(
        pp=pp,
        message=message,
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=verifier_entry,
        verifier_keypair=verifiers[0],
        shared_key=shared_keys[verifier_entry.member_id],
    )
    timings_ms["verify"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    receipt = receipt_gen(
        pp=pp,
        message=message,
        signature=signature,
        context=context,
        roster=roster,
        verifier_entry=verifier_entry,
        verifier_keypair=verifiers[0],
        verify_result=verify_result,
        auditor_public_key=auditor.public_key,
        mode=mode,
    )
    timings_ms["receipt_gen"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    opened = open_receipt(
        pp=pp,
        auditor_keypair=auditor,
        receipt=receipt,
        message=message,
        signature=signature,
        context=context,
        roster=roster,
    )
    timings_ms["open_receipt"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    verdict = judge(
        pp=pp,
        opened=opened,
        message=message,
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
    )
    timings_ms["judge"] = (time.perf_counter() - start) * 1000.0

    return {
        "track": track_name,
        "mode": mode,
        "verifier_count": verifier_count,
        "trial": trial_index,
        "verify_accepted": verify_result.accepted,
        "judge_accepted": verdict.accepted,
        "signature_bytes": _signature_component_bytes(pp, signature),
        "zk_proof_bytes": _receipt_proof_component_bytes(pp, opened.receipt_proof),
        "m_rcpt_bytes": _receipt_plaintext_component_bytes(pp, opened),
        "receipt_bytes": len(canonical_encode(receipt.to_canonical())),
        "open_bytes": len(canonical_encode(opened.payload)),
        "timings_ms": timings_ms,
    }


def run_bench(
    *,
    track_names: list[str],
    verifier_counts: list[int],
    trials: int,
    mode: str,
    formal_only: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_tracks = filter_requested_tracks(track_names, formal_only=formal_only)
    for track_name in selected_tracks:
        for verifier_count in verifier_counts:
            for trial_index in range(trials):
                result = _execute_trial(
                    track_name=track_name,
                    verifier_count=verifier_count,
                    mode=mode,
                    message=f"bench-message-{track_name}-{verifier_count}-{trial_index}",
                    trial_index=trial_index,
                )
                trial_success = int(result["verify_accepted"] and result["judge_accepted"])
                for algorithm, elapsed_ms in result["timings_ms"].items():
                    rows.append(
                        {
                            "track": result["track"],
                            "mode": result["mode"],
                            "verifier_count": result["verifier_count"],
                            "trial": result["trial"],
                            "algorithm": algorithm,
                            "elapsed_ms": elapsed_ms,
                            "repetitions": 1,
                            "success": trial_success,
                            "failure": 1 - trial_success,
                            "error": 0,
                            "signature_bytes": result["signature_bytes"],
                            "m_rcpt_bytes": result["m_rcpt_bytes"],
                            "receipt_bytes": result["receipt_bytes"],
                            "open_bytes": result["open_bytes"],
                        }
                    )
    frame = pd.DataFrame(rows)
    if formal_only:
        frame = finalize_results(frame)
    return frame.sort_values(["track", "verifier_count", "trial", "algorithm"]).reset_index(drop=True)


def _signature_features(signature) -> dict[str, float]:
    u_norms = np.asarray([np.linalg.norm(block.astype(np.float64)) for block in signature.u_blocks], dtype=np.float64)
    z_norms = np.asarray([np.linalg.norm(block.astype(np.float64)) for block in signature.z_blocks], dtype=np.float64)
    theta_bit_density = np.asarray(
        [
            np.unpackbits(np.frombuffer(tag, dtype=np.uint8)).mean()
            for tag in signature.theta_map.values()
        ],
        dtype=np.float64,
    )
    feature_vector = {
        "u_mean_norm": float(u_norms.mean()),
        "u_std_norm": float(u_norms.std()),
        "u_total_norm": float(np.linalg.norm(u_norms)),
        "z_mean_norm": float(z_norms.mean()),
        "z_std_norm": float(z_norms.std()),
        "z_total_norm": float(np.linalg.norm(z_norms)),
        "challenge_weight": float(np.count_nonzero(signature.challenge_c)),
        "challenge_l2": float(np.linalg.norm(signature.challenge_c.astype(np.float64))),
        "theta_density_mean": float(theta_bit_density.mean()),
        "theta_density_std": float(theta_bit_density.std()),
    }
    return feature_vector


def _feature_matrix(signatures: list[Any]) -> tuple[np.ndarray, list[str]]:
    rows = [_signature_features(signature) for signature in signatures]
    columns = sorted(rows[0])
    matrix = np.asarray([[row[column] for column in columns] for row in rows], dtype=np.float64)
    return matrix, columns


def _cohen_d(real_values: np.ndarray, sim_values: np.ndarray) -> float:
    real_var = float(real_values.var(ddof=1))
    sim_var = float(sim_values.var(ddof=1))
    pooled = math.sqrt(max(((real_values.size - 1) * real_var + (sim_values.size - 1) * sim_var) / max(real_values.size + sim_values.size - 2, 1), 1e-12))
    return float((real_values.mean() - sim_values.mean()) / pooled)


def _auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - (n_pos * (n_pos + 1) / 2.0)) / max(n_pos * n_neg, 1)


def _train_holdout_classifier(real_features: np.ndarray, sim_features: np.ndarray) -> tuple[float, float, float]:
    sample_count = min(real_features.shape[0], sim_features.shape[0])
    split = max(sample_count // 2, 1)
    train_real = real_features[:split]
    train_sim = sim_features[:split]
    test_real = real_features[split:]
    test_sim = sim_features[split:]
    if test_real.size == 0 or test_sim.size == 0:
        test_real = train_real
        test_sim = train_sim
    train_features = np.vstack([train_real, train_sim])
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0.0] = 1.0
    train_real_norm = (train_real - mean) / std
    train_sim_norm = (train_sim - mean) / std
    mu_real = train_real_norm.mean(axis=0)
    mu_sim = train_sim_norm.mean(axis=0)
    var = train_features.var(axis=0) + 1e-6
    weights = (mu_real - mu_sim) / var
    bias = -0.5 * float(np.dot(mu_real + mu_sim, weights))

    test_features = np.vstack([test_real, test_sim])
    test_norm = (test_features - mean) / std
    scores = test_norm @ weights + bias
    labels = np.concatenate([np.ones(test_real.shape[0], dtype=np.int64), np.zeros(test_sim.shape[0], dtype=np.int64)])
    auc = _auc_score(labels, scores)

    rng = np.random.default_rng(20260427)
    bootstrap_scores = []
    for _ in range(200):
        indices = rng.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[indices]
        if sampled_labels.min() == sampled_labels.max():
            continue
        bootstrap_scores.append(_auc_score(sampled_labels, scores[indices]))
    if bootstrap_scores:
        ci_low, ci_high = np.percentile(np.asarray(bootstrap_scores, dtype=np.float64), [2.5, 97.5])
    else:
        ci_low = ci_high = auc
    return float(auc), float(ci_low), float(ci_high)


def _median_gamma(feature_matrix: np.ndarray) -> float:
    sample = feature_matrix[: min(feature_matrix.shape[0], 256)]
    if sample.shape[0] < 2:
        return 1.0
    distances = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
    upper = distances[np.triu_indices_from(distances, k=1)]
    median = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    return 1.0 / max(2.0 * median * median, 1e-6)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * np.sum((left - right) ** 2, axis=1))


def _linear_time_mmd(real_features: np.ndarray, sim_features: np.ndarray, gamma: float) -> float:
    paired = min(real_features.shape[0], sim_features.shape[0])
    paired -= paired % 2
    if paired < 2:
        return 0.0
    x1 = real_features[:paired:2]
    x2 = real_features[1:paired:2]
    y1 = sim_features[:paired:2]
    y2 = sim_features[1:paired:2]
    h_values = _rbf_kernel(x1, x2, gamma) + _rbf_kernel(y1, y2, gamma) - _rbf_kernel(x1, y2, gamma) - _rbf_kernel(x2, y1, gamma)
    return float(h_values.mean())


def _mmd_permutation_test(real_features: np.ndarray, sim_features: np.ndarray) -> tuple[float, float]:
    features = np.vstack([real_features, sim_features])
    labels = np.concatenate([np.ones(real_features.shape[0], dtype=np.int64), np.zeros(sim_features.shape[0], dtype=np.int64)])
    gamma = _median_gamma(features)
    observed = _linear_time_mmd(real_features, sim_features, gamma)
    rng = np.random.default_rng(20260427)
    exceedances = 0
    permutations = 200
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        perm_real = features[permuted == 1]
        perm_sim = features[permuted == 0]
        statistic = _linear_time_mmd(perm_real, perm_sim, gamma)
        if statistic >= observed:
            exceedances += 1
    p_value = (exceedances + 1) / (permutations + 1)
    return observed, float(p_value)


def _distribution_summary(real_features: np.ndarray, sim_features: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    ks_rows = []
    for index, feature_name in enumerate(feature_names):
        statistic, p_value = scipy_stats.ks_2samp(real_features[:, index], sim_features[:, index])
        ks_rows.append(
            {
                "feature_name": feature_name,
                "ks_stat": float(statistic),
                "ks_pvalue": float(p_value),
                "cohen_d": abs(_cohen_d(real_features[:, index], sim_features[:, index])),
            }
        )
    worst_by_ks = max(ks_rows, key=lambda row: row["ks_stat"])
    worst_by_effect = max(ks_rows, key=lambda row: row["cohen_d"])
    auc, auc_ci_low, auc_ci_high = _train_holdout_classifier(real_features, sim_features)
    mmd_stat, mmd_pvalue = _mmd_permutation_test(real_features, sim_features)
    indistinguishable = bool(
        auc_ci_low <= 0.5 <= auc_ci_high
        and auc_ci_high < 0.55
        and min(row["ks_pvalue"] for row in ks_rows) >= 0.01
        and worst_by_effect["cohen_d"] < 0.1
    )
    return {
        "max_ks_stat": worst_by_ks["ks_stat"],
        "min_ks_pvalue": min(row["ks_pvalue"] for row in ks_rows),
        "max_abs_cohen_d": worst_by_effect["cohen_d"],
        "worst_ks_feature": worst_by_ks["feature_name"],
        "worst_effect_feature": worst_by_effect["feature_name"],
        "auc": auc,
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "mmd_stat": mmd_stat,
        "mmd_pvalue": mmd_pvalue,
        "indistinguishable": indistinguishable,
    }


def run_distribution(
    *,
    track_names: list[str],
    verifier_counts: list[int],
    samples_per_class: int,
    formal_only: bool,
    mode: str = "optimized-hybrid",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_tracks = filter_requested_tracks(track_names, formal_only=formal_only)
    for track_name in selected_tracks:
        pp = setup(get_parameter_track(track_name))
        for verifier_count in verifier_counts:
            authorizer = keygen(pp, actor_label=f"{track_name}-authorizer-dist")
            proxy = keygen(pp, actor_label=f"{track_name}-proxy-dist")
            verifiers = [keygen(pp, actor_label=f"{track_name}-verifier-dist-{i}") for i in range(1, verifier_count + 1)]
            roster = _build_roster(pp, verifiers)
            shared_keys = symmetric_keygen(roster)
            verifier_entry = roster.entries[0]
            real_signatures = []
            simulated_signatures = []
            for sample_index in range(samples_per_class):
                context = _build_context(roster, sample_index, action_type="distribution")
                policy = _build_policy(sample_index)
                certificate = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
                derived = proxy_keygen(pp, proxy, certificate, roster)
                real_signatures.append(
                    proxy_sign(
                        pp=pp,
                        message=f"distribution-real-{track_name}-{sample_index}",
                        certificate=certificate,
                        context=context,
                        roster=roster,
                        proxy_derived_key=derived,
                        shared_keys=shared_keys,
                        mode=mode,
                    )
                )
                simulated_signatures.append(
                    simulate(
                        pp=pp,
                        message=f"distribution-sim-{track_name}-{sample_index}",
                        certificate=certificate,
                        context=context,
                        proxy_public_key=proxy.public_key,
                        roster=roster,
                        verifier_entry=verifier_entry,
                        verifier_keypair=verifiers[0],
                        shared_keys=shared_keys,
                    )
                )
            real_features, feature_names = _feature_matrix(real_signatures)
            sim_features, _ = _feature_matrix(simulated_signatures)
            summary = _distribution_summary(real_features, sim_features, feature_names)
            rows.append(
                {
                    "track": track_name,
                    "verifier_count": verifier_count,
                    "samples_per_class": samples_per_class,
                    **summary,
                }
            )
    frame = pd.DataFrame(rows)
    if formal_only:
        frame = finalize_results(frame)
    return frame.sort_values(["track", "verifier_count"]).reset_index(drop=True)


def summarize_bench(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    grouped = (
        frame.groupby(["track", "mode", "verifier_count", "algorithm"], as_index=False)
        .agg(
            repetitions=("repetitions", "sum"),
            mean_ms=("elapsed_ms", "mean"),
            std_ms=("elapsed_ms", "std"),
            success=("success", "sum"),
            failure=("failure", "sum"),
            error=("error", "sum"),
            signature_bytes=("signature_bytes", "mean"),
            m_rcpt_bytes=("m_rcpt_bytes", "mean"),
            receipt_bytes=("receipt_bytes", "mean"),
            open_bytes=("open_bytes", "mean"),
        )
        .fillna({"std_ms": 0.0})
        .sort_values(["track", "verifier_count", "algorithm"])
        .reset_index(drop=True)
    )
    return grouped


def finalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    return filter_formal_tracks(frame)
