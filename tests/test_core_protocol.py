from __future__ import annotations

import copy
import math

import numpy as np

from dvgrpsig.config import get_parameter_track
from dvgrpsig.protocol import (
    audit_keygen,
    estimate_receipt_plaintext_bytes,
    judge,
    keygen,
    open_receipt,
    proxy_authorize,
    proxy_keygen,
    proxy_sign,
    setup,
    simulate,
    symmetric_keygen,
    verify,
    receipt_gen,
)
from dvgrpsig.serialization import canonical_encode
from dvgrpsig.types import Context, GPVTrapdoor, ProxyPolicy, Roster, RosterEntry
from dvgrpsig.utils import mod_matmul, secret_challenge_response, secret_materialize_dense, signed_permutation_matrix


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
    return Roster(
        entries=entries,
        epoch="epoch-001",
        roster_root=pp.hash_bytes(canonical_encode([e.member_id for e in entries]), 32).hex(),
    )


def _demo_context(roster):
    return Context(
        epoch=roster.epoch,
        roster_root=roster.roster_root,
        session_id="session-001",
        tx_id="tx-001",
        action_type="release",
        tau="2026-04-27T12:00:00Z",
    )


def _demo_policy():
    return ProxyPolicy(
        policy_id="policy-001",
        description="toy release policy",
        valid_from="2026-04-27T00:00:00Z",
        valid_to="2026-12-31T23:59:59Z",
    )


def test_signed_permutation_matrix_is_orthogonal_mod_q():
    pp = setup(get_parameter_track("toy"))
    matrix = signed_permutation_matrix(pp, b"h2-test")
    product = mod_matmul(matrix.T, matrix, pp.q)
    assert np.array_equal(product % pp.q, np.eye(pp.m, dtype=np.int64) % pp.q)


def test_symmetric_keygen_returns_one_256_bit_key_per_roster_member():
    pp = setup(get_parameter_track("toy"))
    verifiers = [keygen(pp, actor_label=f"verifier-{i}") for i in range(1, 3)]
    roster = _build_roster(pp, verifiers)

    shared_keys = symmetric_keygen(roster)

    assert set(shared_keys) == {entry.member_id for entry in roster.entries}
    assert all(len(key) == 32 for key in shared_keys.values())
    assert shared_keys[roster.entries[0].member_id] != shared_keys[roster.entries[1].member_id]


def test_keygen_outputs_short_right_inverse():
    pp = setup(get_parameter_track("toy"))
    pair = keygen(pp, actor_label="alice")
    product = mod_matmul(pair.public_key, pair.secret_key, pp.q)
    assert np.array_equal(product % pp.q, np.eye(pp.n, dtype=np.int64) % pp.q)


def test_keygen_uses_gpv_trapgen_and_samplepre_secret():
    pp = setup(get_parameter_track("toy"))
    pair = keygen(pp, actor_label="alice")

    assert isinstance(pair.secret_key.trapdoor, GPVTrapdoor)
    assert pair.secret_key.trapdoor.scheme == "gpv08-samplepre"
    assert not hasattr(pair.secret_key.trapdoor, "offsets")
    sampled_preimage = secret_materialize_dense(pair.secret_key)

    assert np.array_equal(sampled_preimage, pair.secret_key.preimage_matrix)
    assert int(np.max(np.abs(sampled_preimage))) <= pp.beta_s
    assert np.array_equal(mod_matmul(pair.public_key, sampled_preimage, pp.q), np.eye(pp.n, dtype=np.int64) % pp.q)


def test_keygen_samples_fresh_keypairs_for_same_label():
    pp = setup(get_parameter_track("toy"))
    first = keygen(pp, actor_label="alice")
    second = keygen(pp, actor_label="alice")

    assert first.public_key.fingerprint != second.public_key.fingerprint


def test_proxy_keygen_rejects_tampered_authorization_challenge():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label="verifier-1")]
    roster = _build_roster(pp, verifiers)
    policy = _demo_policy()
    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)

    tampered = copy.deepcopy(cert)
    tampered.challenge_v1 = cert.challenge_v1.copy()
    tampered.challenge_v1[0] = -tampered.challenge_v1[0] if tampered.challenge_v1[0] != 0 else 1

    try:
        proxy_keygen(pp, proxy, tampered)
    except ValueError as exc:
        assert "Certificate verification failed" in str(exc)
    else:
        raise AssertionError("tampered certificate was accepted")


def test_reference_full_toy_flow_accepts_and_binds():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label=f"verifier-{i}") for i in range(1, 3)]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    auditor = audit_keygen(
        pp,
        receipt_plaintext_bytes=estimate_receipt_plaintext_bytes(pp, roster.entries[0]),
    )
    shared_keys = {entry.member_id: bytes([entry.member_index]) * 32 for entry in roster.entries}

    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    derived = proxy_keygen(pp, proxy, cert, roster)
    signature = proxy_sign(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        roster=roster,
        proxy_derived_key=derived,
        shared_keys=shared_keys,
        mode="reference-full",
    )

    verify_result = verify(
        pp=pp,
        message="hello world",
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_key=shared_keys[roster.entries[0].member_id],
    )
    assert verify_result.accepted is True

    receipt = receipt_gen(
        pp=pp,
        message="hello world",
        signature=signature,
        context=ctx,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        verify_result=verify_result,
        auditor_public_key=auditor.public_key,
        mode="reference-full",
    )
    opened = open_receipt(
        pp=pp,
        auditor_keypair=auditor,
        receipt=receipt,
        message="hello world",
        signature=signature,
        context=ctx,
        roster=roster,
    )
    verdict = judge(
        pp=pp,
        opened=opened,
        message="hello world",
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
    )
    assert verdict.accepted is True
    assert opened.member_id == roster.entries[0].member_id
    proof_payload = opened.receipt_proof.to_canonical()
    assert "surrogate" not in opened.receipt_proof.scheme
    assert opened.receipt_proof.compact_secret is None
    assert len(opened.receipt_proof.rounds) == pp.stern_rounds
    assert "commitments" in opened.receipt_proof.rounds[0]
    assert "response_digest_hex" not in opened.receipt_proof.rounds[0]
    assert "coefficients" not in str(proof_payload)
    assert estimate_receipt_plaintext_bytes(pp, roster.entries[0]) >= len(canonical_encode(opened.payload))


def test_audit_public_product_matches_integer_matmul(monkeypatch):
    from dvgrpsig import protocol

    monkeypatch.setenv("DVGRPSIG_AUDIT_MATMUL_BACKEND", "cpu")
    left = np.array([[2, -1, 0], [0, 3, -2]], dtype=np.int64)
    right = np.array([[5, 7, 11], [13, 17, 19], [23, 29, 31]], dtype=np.int64)

    assert np.array_equal(protocol._audit_public_product(left, right), left @ right)


def test_audit_public_product_uses_integer_fallback_when_float_exactness_is_not_safe(monkeypatch):
    from dvgrpsig import protocol

    monkeypatch.setenv("DVGRPSIG_AUDIT_MATMUL_BACKEND", "cpu")
    large = 1 << 30
    left = np.array([[large, large]], dtype=np.int64)
    right = np.array([[large], [large]], dtype=np.int64)

    assert np.array_equal(protocol._audit_public_product(left, right), left @ right)


def test_optimized_hybrid_receipt_uses_symmetric_capsule_without_full_proof_materialization(monkeypatch):
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label=f"verifier-{i}") for i in range(1, 3)]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    auditor = audit_keygen(pp, receipt_plaintext_bytes=64)
    shared_keys = symmetric_keygen(roster)

    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    derived = proxy_keygen(pp, proxy, cert, roster)
    signature = proxy_sign(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        roster=roster,
        proxy_derived_key=derived,
        shared_keys=shared_keys,
        mode="optimized-hybrid",
    )
    verify_result = verify(
        pp=pp,
        message="hello world",
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_key=shared_keys[roster.entries[0].member_id],
    )
    assert verify_result.accepted is True

    import dvgrpsig.protocol as protocol

    def fail_full_proof(**kwargs):
        raise AssertionError("optimized-hybrid must not materialize the full M_rcpt proof payload")

    monkeypatch.setattr(protocol, "build_receipt_proof", fail_full_proof)
    receipt = receipt_gen(
        pp=pp,
        message="hello world",
        signature=signature,
        context=ctx,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        verify_result=verify_result,
        auditor_public_key=auditor.public_key,
        mode="optimized-hybrid",
    )

    assert receipt.payload_ciphertext is not None
    assert receipt.stream_nonce is not None
    opened = open_receipt(
        pp=pp,
        auditor_keypair=auditor,
        receipt=receipt,
        message="hello world",
        signature=signature,
        context=ctx,
        roster=roster,
    )
    assert opened.receipt_proof.scheme == "fs-sternext-symmetric-capsule"
    verdict = judge(
        pp=pp,
        opened=opened,
        message="hello world",
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
    )
    assert verdict.accepted is True


def test_verify_rejects_tampered_context_binding():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label="verifier-1")]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    shared_key = bytes([1]) * 32
    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    derived = proxy_keygen(pp, proxy, cert, roster)
    signature = proxy_sign(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        roster=roster,
        proxy_derived_key=derived,
        shared_keys={roster.entries[0].member_id: shared_key},
        mode="reference-full",
    )

    tampered = copy.deepcopy(signature)
    tampered.context.tx_id = "tx-evil"
    result = verify(
        pp=pp,
        message="hello world",
        signature=tampered,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_key=shared_key,
    )
    assert result.accepted is False


def test_verify_rejects_expired_policy_window():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label="verifier-1")]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    shared_key = bytes([1]) * 32
    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    derived = proxy_keygen(pp, proxy, cert, roster)
    signature = proxy_sign(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        roster=roster,
        proxy_derived_key=derived,
        shared_keys={roster.entries[0].member_id: shared_key},
        mode="reference-full",
    )
    expired_transcript = copy.deepcopy(signature)
    expired_transcript.context.tau = "2020-12-31T23:59:59Z"

    result = verify(
        pp=pp,
        message="hello world",
        signature=expired_transcript,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_key=shared_key,
    )
    assert result.accepted is False
    assert result.debug["reason"] == "policy-invalid"


def test_simulate_produces_verifier_acceptable_transcript():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label=f"verifier-{i}") for i in range(1, 3)]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    shared_keys = {entry.member_id: bytes([entry.member_index]) * 32 for entry in roster.entries}

    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    transcript = simulate(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_keys=shared_keys,
    )

    result = verify(
        pp=pp,
        message="hello world",
        signature=transcript,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_key=shared_keys[roster.entries[0].member_id],
    )
    assert result.accepted is True
    assert "rho_hint" not in transcript.metadata


def test_simulate_records_lyubashevsky_rejection_sampling_probability():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label=f"verifier-{i}") for i in range(1, 3)]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    shared_keys = {entry.member_id: bytes([entry.member_index]) * 32 for entry in roster.entries}
    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)

    transcript = simulate(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_keys=shared_keys,
    )

    rejection = transcript.metadata["rejection_sampling"]
    z_flat = np.concatenate(transcript.z_blocks).astype(np.float64)
    shift_blocks = [np.zeros(pp.m, dtype=np.int64) for _ in transcript.z_blocks]
    shift_blocks[1] = secret_challenge_response(verifiers[0].secret_key, transcript.challenge_c)
    shift_flat = np.concatenate(shift_blocks).astype(np.float64)
    numerator_exp = float(np.dot(z_flat - shift_flat, z_flat - shift_flat) - np.dot(z_flat, z_flat))
    expected = min(math.exp(math.pi * numerator_exp / float(pp.sigma2 * pp.sigma2)) / rejection["M_rej"], 1.0)

    assert rejection["scheme"] == "lyubashevsky"
    assert rejection["trials"] >= 1
    assert 0.0 < rejection["acceptance_probability"] <= 1.0
    assert math.isclose(rejection["acceptance_probability"], expected, rel_tol=1e-12)


def test_judge_rejects_tampered_real_receipt_proof_round():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label=f"verifier-{i}") for i in range(1, 3)]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    auditor = audit_keygen(
        pp,
        receipt_plaintext_bytes=estimate_receipt_plaintext_bytes(pp, roster.entries[0]),
    )
    shared_keys = {entry.member_id: bytes([entry.member_index]) * 32 for entry in roster.entries}

    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    derived = proxy_keygen(pp, proxy, cert, roster)
    signature = proxy_sign(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        roster=roster,
        proxy_derived_key=derived,
        shared_keys=shared_keys,
        mode="reference-full",
    )
    verify_result = verify(
        pp=pp,
        message="hello world",
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_key=shared_keys[roster.entries[0].member_id],
    )
    receipt = receipt_gen(
        pp=pp,
        message="hello world",
        signature=signature,
        context=ctx,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        verify_result=verify_result,
        auditor_public_key=auditor.public_key,
        mode="reference-full",
    )
    opened = open_receipt(
        pp=pp,
        auditor_keypair=auditor,
        receipt=receipt,
        message="hello world",
        signature=signature,
        context=ctx,
        roster=roster,
    )
    tampered = copy.deepcopy(opened)
    tampered.receipt_proof.rounds[0]["commitments"]["c1"]["digest_hex"] = "00" * 32
    tampered.eta = pp.hash_bytes(
        canonical_encode(
            {
                "message": "hello world",
                "signature": signature.to_canonical(),
                "context": ctx.to_canonical(),
                "member_id": tampered.member_id,
                "bind": tampered.bind_hex,
                "receipt_proof": tampered.receipt_proof.to_canonical(),
                "rho": tampered.rho.tolist(),
            }
        ),
        32,
        domain=b"Hrcpt",
    ).hex()
    tampered.receipt.public_tag = tampered.eta
    verdict = judge(
        pp=pp,
        opened=tampered,
        message="hello world",
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
    )
    assert verdict.accepted is False
    assert verdict.reason == "receipt-proof-invalid"


def test_open_receipt_rejects_tampered_public_tag():
    pp = setup(get_parameter_track("toy"))
    authorizer = keygen(pp, actor_label="authorizer")
    proxy = keygen(pp, actor_label="proxy")
    verifiers = [keygen(pp, actor_label="verifier-1")]
    roster = _build_roster(pp, verifiers)
    ctx = _demo_context(roster)
    policy = _demo_policy()
    auditor = audit_keygen(
        pp,
        receipt_plaintext_bytes=estimate_receipt_plaintext_bytes(pp, roster.entries[0]),
    )
    shared_keys = {entry.member_id: bytes([entry.member_index]) * 32 for entry in roster.entries}
    cert = proxy_authorize(pp, authorizer, proxy.public_key, roster, policy)
    derived = proxy_keygen(pp, proxy, cert, roster)
    signature = proxy_sign(
        pp=pp,
        message="hello world",
        certificate=cert,
        context=ctx,
        roster=roster,
        proxy_derived_key=derived,
        shared_keys=shared_keys,
        mode="reference-full",
    )
    verify_result = verify(
        pp=pp,
        message="hello world",
        signature=signature,
        proxy_public_key=proxy.public_key,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        shared_key=shared_keys[roster.entries[0].member_id],
    )
    receipt = receipt_gen(
        pp=pp,
        message="hello world",
        signature=signature,
        context=ctx,
        roster=roster,
        verifier_entry=roster.entries[0],
        verifier_keypair=verifiers[0],
        verify_result=verify_result,
        auditor_public_key=auditor.public_key,
        mode="reference-full",
    )
    receipt.public_tag = "00" * 32

    try:
        open_receipt(
            pp=pp,
            auditor_keypair=auditor,
            receipt=receipt,
            message="hello world",
            signature=signature,
            context=ctx,
            roster=roster,
        )
    except ValueError as exc:
        assert "Receipt tag mismatch" in str(exc)
    else:
        raise AssertionError("tampered receipt tag was opened")
