from __future__ import annotations

import copy
import base64
import hashlib
import hmac
import math
import os
import secrets
from datetime import datetime, timezone
from typing import Any

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from dvgrpsig.receipt_audit import build_receipt_proof, build_receipt_proof_capsule, verify_receipt_proof
from dvgrpsig.serialization import canonical_decode, canonical_encode
from dvgrpsig.types import (
    AuditorKeypair,
    AuditorPublicKey,
    AuditorSecretKey,
    Context,
    GPVTrapdoor,
    JudgeResult,
    OpenPackage,
    ParameterTrack,
    PartyKeypair,
    ProxyCertificate,
    ProxyDerivedKey,
    ProxyPolicy,
    ProxySignature,
    PublicParams,
    Receipt,
    ReceiptProof,
    Roster,
    RosterEntry,
    SignedPermutation,
    StructuredPublicKey,
    StructuredSecretKey,
    VerifyResult,
)
from dvgrpsig.utils import (
    apply_signed_permutation_to_columns,
    blocks_norm_ok,
    build_sparse_challenge,
    gadget_basis_matrix,
    mod_q,
    sample_centered_gaussian,
    secret_challenge_response,
    secret_transpose_apply,
    signed_permutation_descriptor,
    vector_norm_ok,
)


def setup(track: ParameterTrack) -> PublicParams:
    gadget_width = track.m // (2 * track.n)
    return PublicParams(
        track=track,
        q=track.q,
        n=track.n,
        m=track.m,
        sigma1=track.sigma1,
        sigma2=track.sigma2,
        sigma_enc=track.sigma_enc,
        bz=track.bz,
        beta_s=track.beta_s,
        delta=track.q // 2,
        challenge_weight=track.challenge_weight,
        stern_rounds=track.stern_rounds,
        trapdoor_offsets=track.trapdoor_offsets,
        gadget_width=gadget_width,
    )


def _rng_for(*items: object) -> np.random.Generator:
    digest = hashlib.sha256()
    for item in items:
        digest.update(canonical_encode(item))
    seed = np.frombuffer(digest.digest(), dtype=np.uint32)
    return np.random.default_rng(seed)


def _fresh_rng() -> np.random.Generator:
    return np.random.default_rng(secrets.randbits(128))


def _mac_tag(shared_key: bytes, bind_value: bytes) -> bytes:
    return hmac.new(shared_key, bind_value, hashlib.sha256).digest()


def _lyubashevsky_acceptance_probability(
    z_vector: np.ndarray,
    shift_vector: np.ndarray,
    *,
    sigma: float,
    m_rej: float,
) -> float:
    z = z_vector.astype(np.float64, copy=False)
    shifted = z - shift_vector.astype(np.float64, copy=False)
    exponent = math.pi * float(np.dot(shifted, shifted) - np.dot(z, z)) / float(sigma * sigma) - math.log(m_rej)
    if exponent >= 0.0:
        return 1.0
    if exponent < -745.0:
        return 0.0
    return float(math.exp(exponent))


def _lyubashevsky_rejection_sample(
    rng: np.random.Generator,
    z_blocks: list[np.ndarray],
    shift_blocks: list[np.ndarray],
    *,
    sigma: float,
    m_rej: float,
) -> tuple[bool, float]:
    probability = _lyubashevsky_acceptance_probability(
        np.concatenate(z_blocks),
        np.concatenate(shift_blocks),
        sigma=sigma,
        m_rej=m_rej,
    )
    return bool(rng.random() <= probability), probability


def _sample_short_preimage_top(pp: PublicParams, rng: np.random.Generator) -> np.ndarray:
    nk = pp.n * pp.gadget_width
    matrix = np.zeros((nk, pp.n), dtype=np.int64)
    max_nonzero_per_column = max(0, int(pp.beta_s * pp.beta_s) - 1)
    for column in range(pp.n):
        weight = min(nk, max_nonzero_per_column)
        if weight == 0:
            continue
        rows = rng.choice(nk, size=weight, replace=False)
        matrix[rows, column] = rng.choice(np.array([-1, 1], dtype=np.int64), size=weight)
    return matrix


def _samplepre_identity(pp: PublicParams, top_preimages: np.ndarray) -> np.ndarray:
    gadget_preimages = gadget_basis_matrix(pp.n, pp.gadget_width)
    preimage_matrix = np.concatenate([top_preimages, gadget_preimages], axis=0).astype(np.int64)
    column_norms = np.linalg.norm(preimage_matrix.astype(np.float64), axis=0)
    if float(column_norms.max(initial=0.0)) > pp.beta_s:
        raise ValueError("SamplePre produced a preimage outside beta_s")
    return preimage_matrix


def _trapgen_samplepre(pp: PublicParams, actor_label: str, rng: np.random.Generator) -> PartyKeypair:
    nk = pp.n * pp.gadget_width
    if pp.m != 2 * nk:
        raise ValueError("GPV SamplePre backend expects m = 2*n*gadget_width")
    top_preimages = _sample_short_preimage_top(pp, rng)
    preimage_matrix = _samplepre_identity(pp, top_preimages)
    a_left = rng.integers(0, pp.q, size=(pp.n, nk), dtype=np.int64)
    a_right = rng.integers(0, pp.q, size=(pp.n, nk), dtype=np.int64)
    basis_indices = pp.gadget_width * np.arange(pp.n, dtype=np.int64)
    a_right[:, basis_indices] = mod_q(np.eye(pp.n, dtype=np.int64) - a_left @ top_preimages, pp.q)
    public_key = StructuredPublicKey(
        owner_label=actor_label,
        matrix=mod_q(np.concatenate([a_left, a_right], axis=1), pp.q),
        q=pp.q,
    )
    trapdoor = GPVTrapdoor(
        scheme="gpv08-samplepre",
        gaussian_parameter=float(pp.beta_s),
        preimage_bound=float(pp.beta_s),
        gadget_width=pp.gadget_width,
    )
    secret_key = StructuredSecretKey(
        owner_label=actor_label,
        q=pp.q,
        n=pp.n,
        m=pp.m,
        gadget_width=pp.gadget_width,
        trapdoor=trapdoor,
        preimage_matrix=preimage_matrix,
    )
    return PartyKeypair(actor_label=actor_label, public_key=public_key, secret_key=secret_key)


def keygen(pp: PublicParams, actor_label: str) -> PartyKeypair:
    rng = _fresh_rng()
    return _trapgen_samplepre(pp, actor_label, rng)


_FLOAT64_EXACT_INT_LIMIT = (1 << 53) - 1


def _max_abs_int64(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    return max(abs(int(np.min(matrix))), abs(int(np.max(matrix))))


def _float64_matmul_is_exact_safe(left: np.ndarray, right: np.ndarray) -> bool:
    terms = int(left.shape[1])
    bound = _max_abs_int64(left) * _max_abs_int64(right) * terms
    return bound <= _FLOAT64_EXACT_INT_LIMIT


def _float64_public_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    product = left.astype(np.float64, copy=False) @ right.astype(np.float64, copy=False)
    return np.rint(product).astype(np.int64)


def _cupy_public_product(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        left_gpu = cp.asarray(left, dtype=cp.float64)
        right_gpu = cp.asarray(right, dtype=cp.float64)
        product_gpu = cp.rint(left_gpu @ right_gpu)
        return cp.asnumpy(product_gpu).astype(np.int64)
    except Exception:
        return None


def _audit_public_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(f"Incompatible matrix shapes for audit public product: {left.shape!r} and {right.shape!r}")

    backend = os.environ.get("DVGRPSIG_AUDIT_MATMUL_BACKEND", "cpu").strip().lower()
    if backend == "int64" or not _float64_matmul_is_exact_safe(left, right):
        return left @ right
    if backend in {"gpu", "cupy"}:
        gpu_result = _cupy_public_product(left, right)
        if gpu_result is not None:
            return gpu_result
    return _float64_public_product(left, right)


def audit_keygen(pp: PublicParams, receipt_plaintext_bytes: int) -> AuditorKeypair:
    max_bits = max(256, receipt_plaintext_bytes * 8)
    rng = _fresh_rng()
    matrix_b = rng.integers(0, pp.q, size=(pp.n, pp.m), dtype=np.int64)
    matrix_s = sample_centered_gaussian(rng, sigma=1, shape=(pp.n, max_bits))
    matrix_e = (
        np.zeros((max_bits, pp.m), dtype=np.int64)
        if pp.track.name == "toy"
        else sample_centered_gaussian(rng, sigma=1, shape=(max_bits, pp.m))
    )
    matrix_p = mod_q(_audit_public_product(matrix_s.T, matrix_b) + matrix_e, pp.q)
    return AuditorKeypair(
        public_key=AuditorPublicKey(matrix_b=matrix_b, matrix_p=matrix_p, q=pp.q),
        secret_key=AuditorSecretKey(matrix_s=matrix_s),
        max_plaintext_bits=max_bits,
    )


def symmetric_keygen(roster: Roster) -> dict[str, bytes]:
    return {entry.member_id: secrets.token_bytes(32) for entry in roster.entries}


def _bind_public_identity(public_key: StructuredPublicKey) -> dict[str, Any]:
    return public_key.to_canonical()


def _certificate_payload(
    pp: PublicParams,
    authorizer: PartyKeypair,
    proxy_public_key: StructuredPublicKey,
    roster: Roster,
    policy: ProxyPolicy,
) -> dict[str, Any]:
    return {
        "policy": policy.to_canonical(),
        "authorizer": _bind_public_identity(authorizer.public_key),
        "proxy": _bind_public_identity(proxy_public_key),
        "roster": roster.to_canonical(),
        "epoch": roster.epoch,
        "roster_root": roster.roster_root,
    }


def _parse_context_time(value: str) -> datetime | None:
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _policy_context_error(policy_payload: dict[str, Any], context: Context) -> str | None:
    allowed_actions = tuple(policy_payload.get("allowed_action_types", ()))
    if allowed_actions and context.action_type not in allowed_actions:
        return "policy-invalid"

    valid_from = _parse_context_time(str(policy_payload.get("valid_from", "")))
    valid_to = _parse_context_time(str(policy_payload.get("valid_to", "")))
    tau = _parse_context_time(context.tau)
    if valid_from is None or valid_to is None or tau is None:
        return "policy-invalid"
    if not valid_from <= tau <= valid_to:
        return "policy-invalid"
    return None


def _hash_h1(pp: PublicParams, left: np.ndarray, w_payload: dict[str, Any]) -> np.ndarray:
    return build_sparse_challenge(pp, canonical_encode({"left": left.tolist(), "w_payload": w_payload}), b"H1")


def _hash_hc(pp: PublicParams, kappa: bytes, t: np.ndarray, bind_value: bytes, u_blocks: list[np.ndarray]) -> np.ndarray:
    payload = canonical_encode(
        {
            "kappa": kappa.hex(),
            "t": t.tolist(),
            "bind": bind_value.hex(),
            "u_blocks": [block.tolist() for block in u_blocks],
        }
    )
    return build_sparse_challenge(pp, payload, b"Hc")


def _hash_hb(pp: PublicParams, message: str, certificate: ProxyCertificate, context: Context, roster: Roster) -> bytes:
    return pp.hash_bytes(
        canonical_encode(
            {
                "message": message,
                "certificate": certificate.to_canonical(),
                "context": context.to_canonical(),
                "roster": roster.to_canonical(),
            }
        ),
        32,
        domain=b"Hb",
    )


def _hash_hk(pp: PublicParams, rho: np.ndarray, u_blocks: list[np.ndarray], bind_value: bytes) -> bytes:
    return pp.hash_bytes(
        canonical_encode({"rho": rho.tolist(), "u_blocks": [block.tolist() for block in u_blocks], "bind": bind_value.hex()}),
        32,
        domain=b"Hk",
    )


def _hash_hbind(pp: PublicParams, message: str, signature: ProxySignature, context: Context, roster: Roster) -> bytes:
    return pp.hash_bytes(
        canonical_encode(
            {
                "message": message,
                "signature": signature.to_canonical(),
                "context": context.to_canonical(),
                "roster": roster.to_canonical(),
            }
        ),
        32,
        domain=b"Hbind",
    )


def _hash_hrcpt(
    pp: PublicParams,
    message: str,
    signature: ProxySignature,
    context: Context,
    member_id: str,
    bind_value: bytes,
    receipt_proof: ReceiptProof,
    rho: np.ndarray,
) -> str:
    return pp.hash_bytes(
        canonical_encode(
            {
                "message": message,
                "signature": signature.to_canonical(),
                "context": context.to_canonical(),
                "member_id": member_id,
                "bind": bind_value.hex(),
                "receipt_proof": receipt_proof.to_canonical(),
                "rho": rho.tolist(),
            }
        ),
        32,
        domain=b"Hrcpt",
    ).hex()


def estimate_receipt_plaintext_bytes(
    pp: PublicParams,
    verifier_entry: RosterEntry,
) -> int:
    nk = pp.n * pp.gadget_width
    extended_len = 3 * nk * pp.n

    def _encoded_stub(dtype: str, length: int, bytes_per_entry: int) -> dict[str, Any]:
        return {
            "dtype": dtype,
            "shape": [length],
            "data_b64": base64.b64encode(b"\x00" * (length * bytes_per_entry)).decode("ascii"),
        }

    worst_round_template = {
        "round_index": 999,
        "commitments": {
            "c1": {"digest_hex": "00" * 32},
            "c2": {"digest_hex": "00" * 32},
            "c3": {"digest_hex": "00" * 32},
        },
        "challenge": 2,
        "response": {
            "branch": 2,
            "permutation": _encoded_stub("int32", extended_len, 4),
            "z_vector": _encoded_stub("int32", extended_len, 4),
        },
        "openings": {
            "c1": {"salt_hex": "00" * 32},
            "c3": {"salt_hex": "00" * 32},
        },
    }
    proof_common = {
        "scheme": "fs-sternext-compact",
        "bind_hex": "00" * 32,
        "statement_digest_hex": "00" * 32,
        "witness_shape": [nk, pp.n],
        "response_bound": 1.0,
        "compact_secret": None,
    }

    payload_one_round = canonical_encode(
        {
            "member_id": verifier_entry.member_id,
            "bind_hex": "00" * 32,
            "receipt_proof": {**proof_common, "rounds": [worst_round_template]},
            "eta": "00" * 32,
            "rho": np.zeros(pp.n, dtype=np.int64).tolist(),
        }
    )
    payload_two_rounds = canonical_encode(
        {
            "member_id": verifier_entry.member_id,
            "bind_hex": "00" * 32,
            "receipt_proof": {**proof_common, "rounds": [worst_round_template, worst_round_template]},
            "eta": "00" * 32,
            "rho": np.zeros(pp.n, dtype=np.int64).tolist(),
        }
    )
    per_round_increment = len(payload_two_rounds) - len(payload_one_round)
    return len(payload_one_round) + (pp.stern_rounds - 1) * per_round_increment


def _signed_permutation_for_certificate(pp: PublicParams, certificate: ProxyCertificate) -> SignedPermutation:
    return signed_permutation_descriptor(pp, canonical_encode(certificate.to_canonical()))


def proxy_authorize(
    pp: PublicParams,
    authorizer: PartyKeypair,
    proxy_public_key: StructuredPublicKey,
    roster: Roster,
    policy: ProxyPolicy,
) -> ProxyCertificate:
    w_payload = _certificate_payload(pp, authorizer, proxy_public_key, roster, policy)
    rng = _fresh_rng()
    while True:
        x1 = sample_centered_gaussian(rng, pp.sigma1, (pp.m,))
        v1 = _hash_h1(pp, mod_q(authorizer.public_key.matrix @ x1, pp.q), w_payload)
        y1 = x1 + secret_challenge_response(authorizer.secret_key, v1)
        if vector_norm_ok(y1, pp.bz, pp.q):
            return ProxyCertificate(
                w_payload=w_payload,
                challenge_v1=v1,
                response_y1=y1,
                authorizer_public_key=authorizer.public_key,
                proxy_public_key=proxy_public_key,
            )


def _certificate_failure_reason(
    pp: PublicParams,
    certificate: ProxyCertificate,
    *,
    roster: Roster | None = None,
    proxy_public_key: StructuredPublicKey | None = None,
    context: Context | None = None,
) -> str | None:
    required_top_level = {"policy", "authorizer", "proxy", "roster", "epoch", "roster_root"}
    if not required_top_level.issubset(certificate.w_payload):
        return "certificate-invalid"
    authorizer_payload = certificate.w_payload["authorizer"]
    proxy_payload = certificate.w_payload["proxy"]
    roster_payload = certificate.w_payload["roster"]
    if authorizer_payload.get("fingerprint") is None or proxy_payload.get("fingerprint") is None:
        return "certificate-invalid"
    if roster_payload.get("roster_root") != certificate.w_payload["roster_root"]:
        return "certificate-invalid"
    if certificate.authorizer_public_key.fingerprint != authorizer_payload.get("fingerprint"):
        return "certificate-invalid"
    if certificate.proxy_public_key.fingerprint != proxy_payload.get("fingerprint"):
        return "certificate-invalid"
    if proxy_public_key is not None and proxy_public_key.fingerprint != proxy_payload.get("fingerprint"):
        return "proxy-fingerprint-mismatch"
    if roster is not None:
        if roster.to_canonical() != roster_payload:
            return "roster-binding-mismatch"
        if roster.epoch != certificate.w_payload["epoch"] or roster.roster_root != certificate.w_payload["roster_root"]:
            return "roster-binding-mismatch"
    if context is not None:
        if context.epoch != certificate.w_payload["epoch"] or context.roster_root != certificate.w_payload["roster_root"]:
            return "context-invalid"
        policy_error = _policy_context_error(certificate.w_payload["policy"], context)
        if policy_error is not None:
            return policy_error
    response = certificate.response_y1
    challenge = certificate.challenge_v1
    if response.shape != (pp.m,) or challenge.shape != (pp.n,):
        return "certificate-invalid"
    if not vector_norm_ok(response, pp.bz, pp.q):
        return "certificate-invalid"
    left = mod_q(certificate.authorizer_public_key.matrix @ response - challenge, pp.q)
    expected_challenge = _hash_h1(pp, left, certificate.w_payload)
    if not np.array_equal(expected_challenge, challenge):
        return "certificate-invalid"
    return None


def _verify_certificate(
    pp: PublicParams,
    certificate: ProxyCertificate,
    *,
    roster: Roster | None = None,
    proxy_public_key: StructuredPublicKey | None = None,
    context: Context | None = None,
) -> bool:
    return (
        _certificate_failure_reason(
            pp,
            certificate,
            roster=roster,
            proxy_public_key=proxy_public_key,
            context=context,
        )
        is None
    )


def proxy_keygen(pp: PublicParams, proxy: PartyKeypair, certificate: ProxyCertificate, roster: Roster | None = None) -> ProxyDerivedKey:
    if certificate.w_payload["proxy"]["fingerprint"] != proxy.public_key.fingerprint:
        raise ValueError("Certificate proxy binding mismatch")
    if certificate.w_payload["roster"]["epoch"] != certificate.w_payload["epoch"]:
        raise ValueError("Certificate roster epoch mismatch")
    if not _verify_certificate(pp, certificate, roster=roster, proxy_public_key=proxy.public_key):
        raise ValueError("Certificate verification failed")
    descriptor = _signed_permutation_for_certificate(pp, certificate)
    derived_public_matrix = apply_signed_permutation_to_columns(proxy.public_key.matrix, descriptor)
    derived_public = StructuredPublicKey(
        owner_label=f"{proxy.actor_label}::derived",
        matrix=mod_q(derived_public_matrix, pp.q),
        q=pp.q,
    )
    derived_secret = StructuredSecretKey(
        owner_label=f"{proxy.actor_label}::derived",
        q=proxy.secret_key.q,
        n=proxy.secret_key.n,
        m=proxy.secret_key.m,
        gadget_width=proxy.secret_key.gadget_width,
        trapdoor=proxy.secret_key.trapdoor,
        preimage_matrix=proxy.secret_key.preimage_matrix,
        row_permutation=descriptor,
    )
    return ProxyDerivedKey(public_key=derived_public, secret_key=derived_secret, signed_permutation=descriptor)


def _find_roster_entry(roster: Roster, member_id: str) -> RosterEntry:
    for entry in roster.entries:
        if entry.member_id == member_id:
            return entry
    raise KeyError(member_id)


def proxy_sign(
    *,
    pp: PublicParams,
    message: str,
    certificate: ProxyCertificate,
    context: Context,
    roster: Roster,
    proxy_derived_key: ProxyDerivedKey,
    shared_keys: dict[str, bytes],
    mode: str,
) -> ProxySignature:
    cert_error = _certificate_failure_reason(
        pp,
        certificate,
        roster=roster,
        proxy_public_key=certificate.proxy_public_key,
        context=context,
    )
    if cert_error is not None:
        raise ValueError(f"Certificate verification failed: {cert_error}")
    bind_value = _hash_hb(pp, message, certificate, context, roster)
    rng = _fresh_rng()
    rho = rng.integers(0, 2, size=(pp.n,), dtype=np.int64)
    u_blocks: list[np.ndarray] = []
    for entry in roster.entries:
        noise = sample_centered_gaussian(rng, pp.sigma_enc, (pp.m,))
        u_i = mod_q(entry.public_key.matrix.T @ (pp.delta * rho) + noise, pp.q)
        u_blocks.append(u_i)
    theta_map = {entry.member_id: _mac_tag(shared_keys[entry.member_id], bind_value) for entry in roster.entries}
    kappa = _hash_hk(pp, rho, u_blocks, bind_value)
    while True:
        x_blocks = [sample_centered_gaussian(rng, pp.sigma2, (pp.m,)) for _ in range(len(roster.entries) + 1)]
        t = mod_q(proxy_derived_key.public_key.matrix @ x_blocks[0], pp.q)
        for entry, block in zip(roster.entries, x_blocks[1:]):
            t = mod_q(t + entry.public_key.matrix @ block, pp.q)
        challenge_c = _hash_hc(pp, kappa, t, bind_value, u_blocks)
        z_blocks = [x_blocks[0] + secret_challenge_response(proxy_derived_key.secret_key, challenge_c)]
        z_blocks.extend(x_blocks[1:])
        if blocks_norm_ok(z_blocks, pp.bz, pp.q):
            return ProxySignature(
                u_blocks=u_blocks,
                challenge_c=challenge_c,
                z_blocks=z_blocks,
                certificate=certificate,
                context=copy.deepcopy(context),
                theta_map=theta_map,
                metadata={"mode": mode},
            )


def _decode_rho(pp: PublicParams, vector: np.ndarray) -> np.ndarray | None:
    decoded = np.zeros(pp.n, dtype=np.int64)
    for index, value in enumerate(vector):
        centered = int(value) % pp.q
        if abs(centered - pp.delta) <= pp.delta // 3:
            decoded[index] = 1
        elif centered <= pp.delta // 3 or centered >= pp.q - pp.delta // 3:
            decoded[index] = 0
        else:
            return None
    return decoded


def verify(
    *,
    pp: PublicParams,
    message: str,
    signature: ProxySignature,
    proxy_public_key: StructuredPublicKey,
    roster: Roster,
    verifier_entry: RosterEntry,
    verifier_keypair: PartyKeypair,
    shared_key: bytes,
) -> VerifyResult:
    certificate = signature.certificate
    cert_error = _certificate_failure_reason(
        pp,
        certificate,
        roster=roster,
        proxy_public_key=proxy_public_key,
        context=signature.context,
    )
    if cert_error is not None:
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": cert_error})
    if signature.context.epoch != roster.epoch or signature.context.roster_root != roster.roster_root:
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": "context-invalid"})
    if signature.context.epoch != certificate.w_payload["epoch"]:
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": "certificate-epoch-mismatch"})
    if len(signature.u_blocks) != len(roster.entries) or len(signature.z_blocks) != len(roster.entries) + 1:
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": "signature-shape-invalid"})
    if any(block.shape != (pp.m,) for block in signature.u_blocks + signature.z_blocks):
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": "signature-shape-invalid"})
    bind_value = _hash_hb(pp, message, certificate, signature.context, roster)
    try:
        entry_index = next(index for index, entry in enumerate(roster.entries) if entry.member_id == verifier_entry.member_id)
    except StopIteration:
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": "missing-verifier"})
    d_i = mod_q(secret_transpose_apply(verifier_keypair.secret_key, signature.u_blocks[entry_index]), pp.q)
    rho = _decode_rho(pp, d_i)
    if rho is None:
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": "decode-failed"})
    expected_theta = _mac_tag(shared_key, bind_value)
    if signature.theta_map.get(verifier_entry.member_id) != expected_theta:
        return VerifyResult(False, None, None, verifier_entry.member_id, {"reason": "theta-mismatch"})
    kappa = _hash_hk(pp, rho, signature.u_blocks, bind_value)
    descriptor = _signed_permutation_for_certificate(pp, certificate)
    derived_public_matrix = apply_signed_permutation_to_columns(proxy_public_key.matrix, descriptor)
    t_prime = mod_q(derived_public_matrix @ signature.z_blocks[0], pp.q)
    for roster_item, block in zip(roster.entries, signature.z_blocks[1:]):
        t_prime = mod_q(t_prime + roster_item.public_key.matrix @ block, pp.q)
    t_prime = mod_q(t_prime - signature.challenge_c, pp.q)
    expected_c = _hash_hc(pp, kappa, t_prime, bind_value, signature.u_blocks)
    accepted = np.array_equal(expected_c, signature.challenge_c) and blocks_norm_ok(signature.z_blocks, pp.bz, pp.q)
    return VerifyResult(
        accepted=bool(accepted),
        rho=rho if accepted else None,
        bind_value=bind_value if accepted else None,
        verifier_member_id=verifier_entry.member_id,
        debug={"reason": "ok" if accepted else "challenge-mismatch"},
    )

def _bytes_to_bits(payload: bytes, width_bits: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    padded = np.zeros(width_bits, dtype=np.int64)
    padded[: bits.shape[0]] = bits.astype(np.int64)
    return padded


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    trimmed = bits.astype(np.uint8)
    packed = np.packbits(trimmed)
    return packed.tobytes()


def _lwe_encrypt_bits(pp: PublicParams, auditor_public_key: AuditorPublicKey, bits: np.ndarray, seed_material: bytes) -> tuple[np.ndarray, np.ndarray]:
    rng = _fresh_rng()
    r = sample_centered_gaussian(rng, sigma=1, shape=(pp.m,))
    c1 = mod_q(auditor_public_key.matrix_b @ r, pp.q)
    c2 = mod_q(auditor_public_key.matrix_p[: bits.shape[0], :] @ r + pp.delta * bits, pp.q)
    return c1, c2


def _lwe_decrypt_bits(pp: PublicParams, auditor_keypair: AuditorKeypair, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    bits = c2.shape[0]
    recovered = mod_q(c2 - auditor_keypair.secret_key.matrix_s[:, :bits].T @ c1, pp.q)
    decoded = np.zeros(bits, dtype=np.int64)
    for index, value in enumerate(recovered):
        centered = int(value) % pp.q
        decoded[index] = 1 if abs(centered - pp.delta) <= pp.delta // 3 else 0
    return decoded


def _hybrid_encrypt_payload(pp: PublicParams, auditor_public_key: AuditorPublicKey, payload: bytes, seed_material: bytes) -> tuple[np.ndarray, np.ndarray, bytes, bytes]:
    session_key = secrets.token_bytes(32)
    session_bits = _bytes_to_bits(session_key, 256)
    c1, c2 = _lwe_encrypt_bits(pp, auditor_public_key, session_bits, seed_material + b"-session")
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(session_key).encrypt(nonce, payload, None)
    return c1, c2, nonce, ciphertext


def _hybrid_decrypt_payload(pp: PublicParams, auditor_keypair: AuditorKeypair, c1: np.ndarray, c2: np.ndarray, nonce: bytes, ciphertext: bytes) -> bytes:
    session_bits = _lwe_decrypt_bits(pp, auditor_keypair, c1, c2)[:256]
    session_key = _bits_to_bytes(session_bits)[:32]
    return AESGCM(session_key).decrypt(nonce, ciphertext, None)


def receipt_gen(
    *,
    pp: PublicParams,
    message: str,
    signature: ProxySignature,
    context: Context,
    roster: Roster,
    verifier_entry: RosterEntry,
    verifier_keypair: PartyKeypair,
    verify_result: VerifyResult,
    auditor_public_key: AuditorPublicKey,
    mode: str,
) -> Receipt:
    if not verify_result.accepted or verify_result.rho is None or verify_result.bind_value is None:
        raise ValueError("Receipt generation requires an accepted verification result")
    if verifier_entry.public_key.fingerprint != verifier_keypair.public_key.fingerprint:
        raise ValueError("Verifier keypair does not match roster entry")
    bind_value = _hash_hbind(pp, message, signature, context, roster)
    proof_builder = build_receipt_proof if mode == "reference-full" else build_receipt_proof_capsule
    receipt_proof = proof_builder(
        pp=pp,
        verifier_public_key=verifier_keypair.public_key,
        verifier_secret_key=verifier_keypair.secret_key,
        bind_value=bind_value,
    )
    eta = _hash_hrcpt(
        pp,
        message,
        signature,
        context,
        verifier_entry.member_id,
        bind_value,
        receipt_proof,
        verify_result.rho,
    )
    payload = canonical_encode(
        {
            "member_id": verifier_entry.member_id,
            "bind_hex": bind_value.hex(),
            "receipt_proof": receipt_proof.to_canonical(),
            "eta": eta,
            "rho": verify_result.rho.tolist(),
        }
    )
    seed_material = canonical_encode({"eta": eta, "member_id": verifier_entry.member_id, "mode": mode})
    if mode == "reference-full":
        bits = _bytes_to_bits(payload, auditor_public_key.matrix_p.shape[0])
        c1, c2 = _lwe_encrypt_bits(pp, auditor_public_key, bits, seed_material)
        return Receipt(mode=mode, ciphertext_1=c1, ciphertext_2=c2, public_tag=eta)
    c1, c2, nonce, payload_ciphertext = _hybrid_encrypt_payload(pp, auditor_public_key, payload, seed_material)
    return Receipt(
        mode=mode,
        ciphertext_1=c1,
        ciphertext_2=c2,
        public_tag=eta,
        payload_ciphertext=payload_ciphertext,
        stream_nonce=nonce,
    )


def open_receipt(
    *,
    pp: PublicParams,
    auditor_keypair: AuditorKeypair,
    receipt: Receipt,
    message: str,
    signature: ProxySignature,
    context: Context,
    roster: Roster,
) -> OpenPackage:
    if receipt.mode == "reference-full":
        bits = _lwe_decrypt_bits(pp, auditor_keypair, receipt.ciphertext_1, receipt.ciphertext_2)
        payload = canonical_decode(_bits_to_bytes(bits).rstrip(b"\x00"))
    else:
        if receipt.stream_nonce is None or receipt.payload_ciphertext is None:
            raise ValueError("Hybrid receipt missing payload components")
        payload = canonical_decode(
            _hybrid_decrypt_payload(
                pp,
                auditor_keypair,
                receipt.ciphertext_1,
                receipt.ciphertext_2,
                receipt.stream_nonce,
                receipt.payload_ciphertext,
            )
        )
    rho = np.asarray(payload["rho"], dtype=np.int64)
    proof = ReceiptProof(**payload["receipt_proof"])
    expected_eta = _hash_hrcpt(
        pp,
        message,
        signature,
        context,
        payload["member_id"],
        bytes.fromhex(payload["bind_hex"]),
        proof,
        rho,
    )
    if payload["eta"] != receipt.public_tag or payload["eta"] != expected_eta:
        raise ValueError("Receipt tag mismatch")
    return OpenPackage(
        receipt=receipt,
        member_id=payload["member_id"],
        bind_hex=payload["bind_hex"],
        receipt_proof=proof,
        eta=payload["eta"],
        rho=rho,
        payload=payload,
    )

def _judge_receipt_proof(pp: PublicParams, entry: RosterEntry, bind_hex: str, receipt_proof: ReceiptProof) -> bool:
    return verify_receipt_proof(
        pp=pp,
        verifier_public_key=entry.public_key,
        bind_hex=bind_hex,
        proof=receipt_proof,
    )


def judge(
    *,
    pp: PublicParams,
    opened: OpenPackage,
    message: str,
    signature: ProxySignature,
    proxy_public_key: StructuredPublicKey,
    roster: Roster,
) -> JudgeResult:
    try:
        entry = _find_roster_entry(roster, opened.member_id)
    except KeyError:
        return JudgeResult(False, "member-not-in-roster", {})
    bind_expected = _hash_hbind(pp, message, signature, signature.context, roster).hex()
    if bind_expected != opened.bind_hex:
        return JudgeResult(False, "bind-mismatch", {"expected": bind_expected, "actual": opened.bind_hex})
    eta_expected = _hash_hrcpt(
        pp,
        message,
        signature,
        signature.context,
        opened.member_id,
        bytes.fromhex(opened.bind_hex),
        opened.receipt_proof,
        opened.rho,
    )
    if eta_expected != opened.eta or eta_expected != opened.receipt.public_tag:
        return JudgeResult(False, "eta-mismatch", {"expected": eta_expected, "actual": opened.eta})
    if not _judge_receipt_proof(pp, entry, opened.bind_hex, opened.receipt_proof):
        return JudgeResult(False, "receipt-proof-invalid", {})
    certificate = signature.certificate
    cert_error = _certificate_failure_reason(
        pp,
        certificate,
        roster=roster,
        proxy_public_key=proxy_public_key,
        context=signature.context,
    )
    if cert_error is not None:
        return JudgeResult(False, cert_error, {})
    if len(signature.u_blocks) != len(roster.entries) or len(signature.z_blocks) != len(roster.entries) + 1:
        return JudgeResult(False, "signature-shape-invalid", {})
    if any(block.shape != (pp.m,) for block in signature.u_blocks + signature.z_blocks):
        return JudgeResult(False, "signature-shape-invalid", {})
    descriptor = _signed_permutation_for_certificate(pp, certificate)
    derived_public_matrix = apply_signed_permutation_to_columns(proxy_public_key.matrix, descriptor)
    bind_hb = _hash_hb(pp, message, certificate, signature.context, roster)
    kappa = _hash_hk(pp, opened.rho, signature.u_blocks, bind_hb)
    t_value = mod_q(derived_public_matrix @ signature.z_blocks[0], pp.q)
    for roster_item, block in zip(roster.entries, signature.z_blocks[1:]):
        t_value = mod_q(t_value + roster_item.public_key.matrix @ block, pp.q)
    t_value = mod_q(t_value - signature.challenge_c, pp.q)
    expected_c = _hash_hc(pp, kappa, t_value, bind_hb, signature.u_blocks)
    if not np.array_equal(expected_c, signature.challenge_c):
        return JudgeResult(False, "signature-core-invalid", {"reason": "challenge-mismatch"})
    if not blocks_norm_ok(signature.z_blocks, pp.bz, pp.q):
        return JudgeResult(False, "signature-core-invalid", {"reason": "norm-failed"})
    return JudgeResult(True, "ok", {"member_id": opened.member_id})


def simulate(
    *,
    pp: PublicParams,
    message: str,
    certificate: ProxyCertificate,
    context: Context,
    proxy_public_key: StructuredPublicKey,
    roster: Roster,
    verifier_entry: RosterEntry,
    verifier_keypair: PartyKeypair,
    shared_keys: dict[str, bytes],
) -> ProxySignature:
    cert_error = _certificate_failure_reason(
        pp,
        certificate,
        roster=roster,
        proxy_public_key=proxy_public_key,
        context=context,
    )
    if cert_error is not None:
        raise ValueError(f"Certificate verification failed: {cert_error}")
    bind_value = _hash_hb(pp, message, certificate, context, roster)
    rng = _fresh_rng()
    rho = rng.integers(0, 2, size=(pp.n,), dtype=np.int64)
    descriptor = _signed_permutation_for_certificate(pp, certificate)
    derived_public_matrix = apply_signed_permutation_to_columns(proxy_public_key.matrix, descriptor)
    u_blocks: list[np.ndarray] = []
    for entry in roster.entries:
        noise = sample_centered_gaussian(rng, pp.sigma_enc, (pp.m,))
        u_blocks.append(mod_q(entry.public_key.matrix.T @ (pp.delta * rho) + noise, pp.q))
    theta_map = {}
    for entry in roster.entries:
        if entry.member_id == verifier_entry.member_id:
            theta_map[entry.member_id] = _mac_tag(shared_keys[entry.member_id], bind_value)
        else:
            theta_map[entry.member_id] = pp.hash_bytes(
                canonical_encode({"bind": bind_value.hex(), "member_id": entry.member_id, "mode": "simulate"}),
                32,
                domain=b"sim-theta",
            )
    kappa = _hash_hk(pp, rho, u_blocks, bind_value)
    rejection_trials = 0
    m_rej = math.e
    while True:
        rejection_trials += 1
        x_blocks = [sample_centered_gaussian(rng, pp.sigma2, (pp.m,)) for _ in range(len(roster.entries) + 1)]
        t = np.zeros(pp.n, dtype=np.int64)
        t = mod_q(t + derived_public_matrix @ x_blocks[0], pp.q)
        for entry, block in zip(roster.entries, x_blocks[1:]):
            t = mod_q(t + entry.public_key.matrix @ block, pp.q)
        challenge_c = _hash_hc(pp, kappa, t, bind_value, u_blocks)
        z_blocks = [x_blocks[0]]
        shift_blocks = [np.zeros(pp.m, dtype=np.int64)]
        for entry, block in zip(roster.entries, x_blocks[1:]):
            if entry.member_id == verifier_entry.member_id:
                shift = secret_challenge_response(verifier_keypair.secret_key, challenge_c)
                z_blocks.append(block + shift)
                shift_blocks.append(shift)
            else:
                z_blocks.append(block)
                shift_blocks.append(np.zeros(pp.m, dtype=np.int64))
        if blocks_norm_ok(z_blocks, pp.bz, pp.q):
            accepted, acceptance_probability = _lyubashevsky_rejection_sample(
                rng,
                z_blocks,
                shift_blocks,
                sigma=float(pp.sigma2),
                m_rej=m_rej,
            )
            if not accepted:
                continue
            return ProxySignature(
                u_blocks=u_blocks,
                challenge_c=challenge_c,
                z_blocks=z_blocks,
                certificate=certificate,
                context=copy.deepcopy(context),
                theta_map=theta_map,
                metadata={
                    "mode": "simulate",
                    "rejection_sampling": {
                        "scheme": "lyubashevsky",
                        "M_rej": m_rej,
                        "acceptance_probability": acceptance_probability,
                        "trials": rejection_trials,
                    },
                },
            )
