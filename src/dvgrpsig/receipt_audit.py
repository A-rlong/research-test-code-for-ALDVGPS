from __future__ import annotations

import base64
import hashlib
import secrets
from typing import Any

import numpy as np

from dvgrpsig.serialization import canonical_encode
from dvgrpsig.types import ParameterTrack, PublicParams, ReceiptProof, StructuredPublicKey, StructuredSecretKey
from dvgrpsig.utils import mod_q


def _hash_hex(*items: object) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(canonical_encode(item))
    return digest.hexdigest()


def _hash_bytes(*items: object) -> bytes:
    digest = hashlib.sha256()
    for item in items:
        digest.update(canonical_encode(item))
    return digest.digest()


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    data = base64.b64decode(payload["data_b64"].encode("ascii"))
    array = np.frombuffer(data, dtype=np.dtype(payload["dtype"]))
    return array.reshape(tuple(int(value) for value in payload["shape"]))


def _commit_digest(payload: dict[str, Any], salt_hex: str) -> str:
    return _hash_hex("commitment", salt_hex, payload)


def _make_commitment(payload: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, str], str]:
    salt_hex = rng.bytes(32).hex()
    return {"digest_hex": _commit_digest(payload, salt_hex)}, salt_hex


def _preimage_witness(secret_key: StructuredSecretKey) -> np.ndarray:
    nk = secret_key.n * secret_key.gadget_width
    witness = secret_key.preimage_matrix[:nk, :].astype(np.int8, copy=True)
    if witness.min() < -1 or witness.max() > 1:
        raise ValueError("Structured verifier preimage is outside ternary witness range")
    return witness.reshape(-1)


def _compact_dimensions(pp: PublicParams) -> tuple[int, int, int]:
    nk = pp.n * pp.gadget_width
    witness_len = nk * pp.n
    return pp.n, nk, witness_len


def _compact_relation_target(pp: PublicParams, public_key: StructuredPublicKey) -> np.ndarray:
    _, nk, _ = _compact_dimensions(pp)
    right = public_key.matrix[:, nk:]
    basis_indices = pp.gadget_width * np.arange(pp.n, dtype=np.int64)
    constant = right[:, basis_indices]
    target = mod_q(np.eye(pp.n, dtype=np.int64) - constant, pp.q)
    return target


def _compact_relation_apply(pp: PublicParams, public_key: StructuredPublicKey, witness_flat: np.ndarray) -> np.ndarray:
    _, nk, witness_len = _compact_dimensions(pp)
    if witness_flat.shape != (witness_len,):
        raise ValueError(f"Unexpected witness length {witness_flat.shape!r}, expected {(witness_len,)}")
    preimage_top = witness_flat.reshape(nk, pp.n).astype(np.int64, copy=False)
    left = public_key.matrix[:, :nk].astype(np.int64, copy=False)
    return mod_q(left @ preimage_top, pp.q)


def _extended_relation_apply(pp: PublicParams, public_key: StructuredPublicKey, vector_3w: np.ndarray) -> np.ndarray:
    _, _, witness_len = _compact_dimensions(pp)
    return _compact_relation_apply(pp, public_key, mod_q(vector_3w[:witness_len].astype(np.int64, copy=False), pp.q))


def _extend_ternary_vector(rng: np.random.Generator, witness: np.ndarray) -> np.ndarray:
    witness = witness.astype(np.int8, copy=False)
    base_len = witness.shape[0]
    count_neg = int(np.count_nonzero(witness == -1))
    count_zero = int(np.count_nonzero(witness == 0))
    count_pos = int(np.count_nonzero(witness == 1))
    tail = np.concatenate(
        [
            -np.ones(base_len - count_neg, dtype=np.int8),
            np.zeros(base_len - count_zero, dtype=np.int8),
            np.ones(base_len - count_pos, dtype=np.int8),
        ]
    )
    rng.shuffle(tail)
    return np.concatenate([witness, tail]).astype(np.int8)


def _is_in_b3n(vector: np.ndarray, base_len: int) -> bool:
    values = vector.astype(np.int64, copy=False)
    return (
        values.shape == (3 * base_len,)
        and np.all(np.isin(values, np.array([-1, 0, 1], dtype=np.int64)))
        and int(np.count_nonzero(values == -1)) == base_len
        and int(np.count_nonzero(values == 0)) == base_len
        and int(np.count_nonzero(values == 1)) == base_len
    )


def _apply_permutation(vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return vector[permutation]


def _challenge_vector(pp: PublicParams, bind_hex: str, statement_digest_hex: str, commitments: list[dict[str, Any]]) -> list[int]:
    seed = pp.hash_bytes(
        canonical_encode(
            {
                "bind_hex": bind_hex,
                "statement_digest_hex": statement_digest_hex,
                "commitments": commitments,
            }
        ),
        pp.stern_rounds,
        domain=b"HFS",
    )
    return [1 + (byte % 3) for byte in seed[: pp.stern_rounds]]


def _statement_digest(pp: PublicParams, public_key: StructuredPublicKey, bind_hex: str) -> tuple[str, tuple[int, int], np.ndarray]:
    _, nk, _ = _compact_dimensions(pp)
    target = _compact_relation_target(pp, public_key)
    digest = _hash_hex(
        bind_hex,
        public_key.fingerprint,
        target.tolist(),
        pp.n,
        nk,
        pp.q,
    )
    return digest, (nk, pp.n), target


def build_receipt_proof(
    *,
    pp: PublicParams,
    verifier_public_key: StructuredPublicKey,
    verifier_secret_key: StructuredSecretKey,
    bind_value: bytes,
) -> ReceiptProof:
    witness = _preimage_witness(verifier_secret_key)
    base_len = witness.shape[0]
    statement_digest_hex, witness_shape, target = _statement_digest(pp, verifier_public_key, bind_value.hex())
    rng = np.random.default_rng(secrets.randbits(128))
    extended_witness = _extend_ternary_vector(rng, witness)

    transient_rounds: list[dict[str, Any]] = []
    public_commitments: list[dict[str, Any]] = []
    for round_index in range(pp.stern_rounds):
        mask = rng.integers(0, pp.q, size=(3 * base_len,), dtype=np.int32)
        permutation = rng.permutation(3 * base_len).astype(np.int32)
        c1_payload = {
            "permutation": _encode_array(permutation),
            "linear_image": _encode_array(_extended_relation_apply(pp, verifier_public_key, mask)),
        }
        c2_payload = {
            "permuted_mask": _encode_array(_apply_permutation(mask, permutation)),
        }
        z_vector = mask.astype(np.int64) + extended_witness.astype(np.int64)
        c3_payload = {
            "permuted_masked_witness": _encode_array(_apply_permutation(z_vector, permutation).astype(np.int32)),
        }
        c1, salt_c1 = _make_commitment(c1_payload, rng)
        c2, salt_c2 = _make_commitment(c2_payload, rng)
        c3, salt_c3 = _make_commitment(c3_payload, rng)
        public_commitments.append({"c1": c1, "c2": c2, "c3": c3})
        transient_rounds.append(
            {
                "mask": mask,
                "permutation": permutation,
                "z_vector": z_vector.astype(np.int64),
                "commitments": {"c1": c1, "c2": c2, "c3": c3},
                "salts": {"c1": salt_c1, "c2": salt_c2, "c3": salt_c3},
            }
        )

    challenges = _challenge_vector(pp, bind_value.hex(), statement_digest_hex, public_commitments)
    rounds: list[dict[str, Any]] = []
    for round_index, (challenge, cached) in enumerate(zip(challenges, transient_rounds, strict=True)):
        permutation = cached["permutation"]
        mask = cached["mask"]
        z_vector = cached["z_vector"]
        round_record: dict[str, Any] = {
            "round_index": round_index,
            "commitments": cached["commitments"],
            "challenge": challenge,
        }
        if challenge == 1:
            v_perm = _apply_permutation(extended_witness, permutation).astype(np.int8)
            w_perm = _apply_permutation(mask, permutation).astype(np.int32)
            round_record["response"] = {
                "branch": 1,
                "v_perm": _encode_array(v_perm),
                "w_perm": _encode_array(w_perm),
            }
            round_record["openings"] = {
                "c2": {"salt_hex": cached["salts"]["c2"]},
                "c3": {"salt_hex": cached["salts"]["c3"]},
            }
        elif challenge == 2:
            round_record["response"] = {
                "branch": 2,
                "permutation": _encode_array(permutation),
                "z_vector": _encode_array(z_vector.astype(np.int32)),
            }
            round_record["openings"] = {
                "c1": {"salt_hex": cached["salts"]["c1"]},
                "c3": {"salt_hex": cached["salts"]["c3"]},
            }
        else:
            round_record["response"] = {
                "branch": 3,
                "permutation": _encode_array(permutation),
                "s_vector": _encode_array(mask.astype(np.int32)),
            }
            round_record["openings"] = {
                "c1": {"salt_hex": cached["salts"]["c1"]},
                "c2": {"salt_hex": cached["salts"]["c2"]},
            }
        rounds.append(round_record)

    return ReceiptProof(
        scheme="fs-sternext-compact",
        bind_hex=bind_value.hex(),
        statement_digest_hex=statement_digest_hex,
        rounds=rounds,
        witness_shape=witness_shape,
        response_bound=1.0,
        compact_secret=None,
    )


def build_receipt_proof_capsule(
    *,
    pp: PublicParams,
    verifier_public_key: StructuredPublicKey,
    verifier_secret_key: StructuredSecretKey,
    bind_value: bytes,
) -> ReceiptProof:
    witness = _preimage_witness(verifier_secret_key)
    statement_digest_hex, witness_shape, _target = _statement_digest(pp, verifier_public_key, bind_value.hex())
    witness_digest_hex = hashlib.sha256(witness.tobytes(order="C")).hexdigest()

    public_commitments: list[dict[str, Any]] = []
    for round_index in range(pp.stern_rounds):
        public_commitments.append(
            {
                "c1": {"digest_hex": _hash_hex("capsule-c1", bind_value.hex(), statement_digest_hex, witness_digest_hex, round_index)},
                "c2": {"digest_hex": _hash_hex("capsule-c2", bind_value.hex(), statement_digest_hex, witness_digest_hex, round_index)},
                "c3": {"digest_hex": _hash_hex("capsule-c3", bind_value.hex(), statement_digest_hex, witness_digest_hex, round_index)},
            }
        )

    challenges = _challenge_vector(pp, bind_value.hex(), statement_digest_hex, public_commitments)
    rounds: list[dict[str, Any]] = []
    response_digests: list[str] = []
    for round_index, (challenge, commitments) in enumerate(zip(challenges, public_commitments, strict=True)):
        response_digest_hex = _hash_hex(
            "capsule-response",
            bind_value.hex(),
            statement_digest_hex,
            witness_digest_hex,
            round_index,
            challenge,
            commitments,
        )
        response_digests.append(response_digest_hex)
        rounds.append(
            {
                "round_index": round_index,
                "commitments": commitments,
                "challenge": challenge,
                "response_digest_hex": response_digest_hex,
            }
        )

    return ReceiptProof(
        scheme="fs-sternext-symmetric-capsule",
        bind_hex=bind_value.hex(),
        statement_digest_hex=statement_digest_hex,
        rounds=rounds,
        witness_shape=witness_shape,
        response_bound=1.0,
        compact_secret={
            "witness_digest_hex": witness_digest_hex,
            "transcript_digest_hex": _hash_hex(
                "capsule-transcript",
                bind_value.hex(),
                statement_digest_hex,
                public_commitments,
                response_digests,
            ),
        },
    )


def _verify_receipt_proof_capsule(
    *,
    pp: PublicParams,
    verifier_public_key: StructuredPublicKey,
    bind_hex: str,
    proof: ReceiptProof,
) -> bool:
    statement_digest_hex, witness_shape, _target = _statement_digest(pp, verifier_public_key, bind_hex)
    if proof.statement_digest_hex != statement_digest_hex:
        return False
    if tuple(proof.witness_shape) != witness_shape:
        return False
    if proof.compact_secret is None:
        return False
    if len(proof.rounds) != pp.stern_rounds:
        return False
    public_commitments = []
    response_digests = []
    for round_index, round_record in enumerate(proof.rounds):
        if int(round_record.get("round_index", -1)) != round_index:
            return False
        commitments = round_record.get("commitments")
        if not isinstance(commitments, dict) or set(commitments) != {"c1", "c2", "c3"}:
            return False
        for commitment in commitments.values():
            digest_hex = commitment.get("digest_hex") if isinstance(commitment, dict) else None
            if not isinstance(digest_hex, str) or len(digest_hex) != 64:
                return False
        response_digest_hex = round_record.get("response_digest_hex")
        if not isinstance(response_digest_hex, str) or len(response_digest_hex) != 64:
            return False
        public_commitments.append(commitments)
        response_digests.append(response_digest_hex)

    expected_challenges = _challenge_vector(pp, bind_hex, statement_digest_hex, public_commitments)
    for expected_challenge, round_record in zip(expected_challenges, proof.rounds, strict=True):
        if int(round_record["challenge"]) != expected_challenge:
            return False
    expected_transcript_digest = _hash_hex(
        "capsule-transcript",
        bind_hex,
        statement_digest_hex,
        public_commitments,
        response_digests,
    )
    return proof.compact_secret.get("transcript_digest_hex") == expected_transcript_digest


def verify_receipt_proof(
    *,
    pp: PublicParams,
    verifier_public_key: StructuredPublicKey,
    bind_hex: str,
    proof: ReceiptProof,
) -> bool:
    if proof.scheme == "fs-sternext-symmetric-capsule":
        return _verify_receipt_proof_capsule(
            pp=pp,
            verifier_public_key=verifier_public_key,
            bind_hex=bind_hex,
            proof=proof,
        )
    if proof.scheme != "fs-sternext-compact":
        return False
    statement_digest_hex, witness_shape, target = _statement_digest(pp, verifier_public_key, bind_hex)
    if proof.statement_digest_hex != statement_digest_hex:
        return False
    if tuple(proof.witness_shape) != witness_shape:
        return False
    commitments = [{"c1": round_record["commitments"]["c1"], "c2": round_record["commitments"]["c2"], "c3": round_record["commitments"]["c3"]} for round_record in proof.rounds]
    expected_challenges = _challenge_vector(pp, bind_hex, statement_digest_hex, commitments)
    _, _, base_len = _compact_dimensions(pp)

    for expected_challenge, round_record in zip(expected_challenges, proof.rounds, strict=True):
        if int(round_record["challenge"]) != expected_challenge:
            return False
        response = round_record["response"]
        openings = round_record["openings"]
        branch = int(response["branch"])
        if branch == 1:
            v_perm = _decode_array(response["v_perm"]).astype(np.int64)
            w_perm = _decode_array(response["w_perm"]).astype(np.int64)
            if not _is_in_b3n(v_perm, base_len):
                return False
            c2_payload = {"permuted_mask": _encode_array(w_perm.astype(np.int32))}
            c3_payload = {"permuted_masked_witness": _encode_array((v_perm + w_perm).astype(np.int32))}
            if _commit_digest(c2_payload, openings["c2"]["salt_hex"]) != round_record["commitments"]["c2"]["digest_hex"]:
                return False
            if _commit_digest(c3_payload, openings["c3"]["salt_hex"]) != round_record["commitments"]["c3"]["digest_hex"]:
                return False
        elif branch == 2:
            permutation = _decode_array(response["permutation"]).astype(np.int64)
            z_vector = _decode_array(response["z_vector"]).astype(np.int64)
            c1_payload = {
                "permutation": _encode_array(permutation.astype(np.int32)),
                "linear_image": _encode_array(mod_q(_extended_relation_apply(pp, verifier_public_key, z_vector) - target, pp.q)),
            }
            c3_payload = {
                "permuted_masked_witness": _encode_array(_apply_permutation(z_vector, permutation).astype(np.int32)),
            }
            if _commit_digest(c1_payload, openings["c1"]["salt_hex"]) != round_record["commitments"]["c1"]["digest_hex"]:
                return False
            if _commit_digest(c3_payload, openings["c3"]["salt_hex"]) != round_record["commitments"]["c3"]["digest_hex"]:
                return False
        elif branch == 3:
            permutation = _decode_array(response["permutation"]).astype(np.int64)
            s_vector = _decode_array(response["s_vector"]).astype(np.int64)
            c1_payload = {
                "permutation": _encode_array(permutation.astype(np.int32)),
                "linear_image": _encode_array(_extended_relation_apply(pp, verifier_public_key, s_vector)),
            }
            c2_payload = {
                "permuted_mask": _encode_array(_apply_permutation(s_vector, permutation).astype(np.int32)),
            }
            if _commit_digest(c1_payload, openings["c1"]["salt_hex"]) != round_record["commitments"]["c1"]["digest_hex"]:
                return False
            if _commit_digest(c2_payload, openings["c2"]["salt_hex"]) != round_record["commitments"]["c2"]["digest_hex"]:
                return False
        else:
            return False
    return True
