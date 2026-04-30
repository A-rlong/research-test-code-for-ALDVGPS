from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from dvgrpsig.serialization import canonical_encode
from dvgrpsig.types import PublicParams, SignedPermutation, StructuredPublicKey, StructuredSecretKey


def mod_q(value: np.ndarray, q: int) -> np.ndarray:
    return np.mod(value, q).astype(np.int64, copy=False)


def mod_matmul(left, right, q: int) -> np.ndarray:
    left_matrix = materialize_matrix(left)
    right_matrix = materialize_matrix(right)
    return mod_q(left_matrix @ right_matrix, q)


def materialize_matrix(value) -> np.ndarray:
    if isinstance(value, StructuredPublicKey):
        return value.matrix.astype(np.int64, copy=False)
    if isinstance(value, StructuredSecretKey):
        return secret_materialize_dense(value)
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    raise TypeError(f"Unsupported matrix type: {type(value)!r}")


def gadget_basis_matrix(n: int, gadget_width: int) -> np.ndarray:
    basis = np.zeros((n * gadget_width, n), dtype=np.int64)
    for row in range(n):
        basis[row * gadget_width, row] = 1
    return basis


def sample_centered_gaussian(rng: np.random.Generator, sigma: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.rint(rng.normal(loc=0.0, scale=float(max(sigma, 1)), size=shape)).astype(np.int64)


def secret_challenge_response(secret_key: StructuredSecretKey, challenge: np.ndarray) -> np.ndarray:
    response = (secret_key.preimage_matrix.astype(np.int64, copy=False) @ challenge.astype(np.int64, copy=False)).astype(
        np.int64
    )
    if secret_key.row_permutation is None:
        return response
    return apply_signed_permutation_to_vector(response, secret_key.row_permutation)


def secret_transpose_apply(secret_key: StructuredSecretKey, vector: np.ndarray) -> np.ndarray:
    if secret_key.row_permutation is not None:
        vector = apply_inverse_signed_permutation_to_vector(vector, secret_key.row_permutation)
    return (secret_key.preimage_matrix.astype(np.int64, copy=False).T @ vector.astype(np.int64, copy=False)).astype(np.int64)


def secret_materialize_dense(secret_key: StructuredSecretKey) -> np.ndarray:
    if secret_key.row_permutation is None:
        return secret_key.preimage_matrix.astype(np.int64, copy=True)
    basis = np.eye(secret_key.n, dtype=np.int64)
    dense_columns = [secret_challenge_response(secret_key, basis[:, col]) for col in range(secret_key.n)]
    return np.column_stack(dense_columns).astype(np.int64)


def split_signature_vector(vector: np.ndarray, block_width: int, blocks: int) -> list[np.ndarray]:
    return [vector[index * block_width : (index + 1) * block_width].astype(np.int64) for index in range(blocks)]


def canonical_hash_hex(*items: object) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(canonical_encode(item))
    return digest.hexdigest()


def signed_permutation_descriptor(pp: PublicParams, seed_material: bytes) -> SignedPermutation:
    raw = pp.hash_bytes(seed_material, pp.m * 8, domain=b"H2")
    rng = np.random.default_rng(np.frombuffer(raw[:32], dtype=np.uint64))
    permutation = rng.permutation(pp.m).astype(np.int64)
    signs = rng.choice(np.array([-1, 1], dtype=np.int64), size=pp.m).astype(np.int64)
    return SignedPermutation(permutation=permutation, signs=signs)


def signed_permutation_matrix(pp: PublicParams, seed_material: bytes) -> np.ndarray:
    descriptor = signed_permutation_descriptor(pp, seed_material)
    matrix = np.zeros((pp.m, pp.m), dtype=np.int64)
    matrix[np.arange(pp.m), descriptor.permutation] = descriptor.signs
    return matrix


def apply_signed_permutation_to_vector(vector: np.ndarray, descriptor: SignedPermutation) -> np.ndarray:
    return (descriptor.signs * vector[descriptor.permutation]).astype(np.int64)


def apply_inverse_signed_permutation_to_vector(vector: np.ndarray, descriptor: SignedPermutation) -> np.ndarray:
    result = np.zeros_like(vector, dtype=np.int64)
    result[descriptor.permutation] = descriptor.signs * vector
    return result


def apply_signed_permutation_to_columns(matrix: np.ndarray, descriptor: SignedPermutation) -> np.ndarray:
    return (matrix[:, descriptor.permutation] * descriptor.signs).astype(np.int64)


def build_sparse_challenge(pp: PublicParams, seed_material: bytes, domain: bytes) -> np.ndarray:
    challenge = np.zeros(pp.n, dtype=np.int64)
    material = pp.hash_bytes(seed_material, max(64, pp.challenge_weight * 8), domain=domain)
    cursor = 0
    used: set[int] = set()
    while len(used) < pp.challenge_weight:
        if cursor + 4 > len(material):
            material += pp.hash_bytes(material, len(material), domain=domain + b"-expand")
        index = int.from_bytes(material[cursor : cursor + 2], "little") % pp.n
        sign = 1 if material[cursor + 2] % 2 == 0 else -1
        cursor += 3
        if index in used:
            continue
        used.add(index)
        challenge[index] = sign
    return challenge


def vector_norm_ok(vector: np.ndarray, bz: float, q: int) -> bool:
    return float(np.linalg.norm(vector.astype(np.float64))) <= bz and int(np.max(np.abs(vector))) <= q // 4


def blocks_norm_ok(blocks: Iterable[np.ndarray], bz: float, q: int) -> bool:
    flat = np.concatenate([np.asarray(block, dtype=np.int64) for block in blocks])
    return vector_norm_ok(flat, bz, q)
