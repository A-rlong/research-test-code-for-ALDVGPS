from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ParameterTrack:
    name: str
    n: int
    m: int
    q: int
    sigma1: int
    sigma2: int
    sigma_enc: int
    bz: float
    beta_s: int
    challenge_weight: int
    stern_rounds: int
    trapdoor_offsets: tuple[int, ...]


@dataclass(slots=True)
class SignedPermutation:
    permutation: np.ndarray
    signs: np.ndarray


@dataclass(slots=True)
class PublicParams:
    track: ParameterTrack
    q: int
    n: int
    m: int
    sigma1: int
    sigma2: int
    sigma_enc: int
    bz: float
    beta_s: int
    delta: int
    challenge_weight: int
    stern_rounds: int
    trapdoor_offsets: tuple[int, ...]
    gadget_width: int

    def hash_bytes(self, payload: bytes, out_len: int, domain: bytes = b"") -> bytes:
        import hashlib

        shake = hashlib.shake_256()
        shake.update(domain)
        shake.update(payload)
        return shake.digest(out_len)


@dataclass(slots=True)
class GPVTrapdoor:
    scheme: str
    gaussian_parameter: float
    preimage_bound: float
    gadget_width: int

    def to_canonical(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "gaussian_parameter": self.gaussian_parameter,
            "preimage_bound": self.preimage_bound,
            "gadget_width": self.gadget_width,
        }


@dataclass(slots=True)
class StructuredPublicKey:
    owner_label: str
    matrix: np.ndarray
    q: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.matrix.tobytes())
        digest.update(self.owner_label.encode("utf-8"))
        self.fingerprint = digest.hexdigest()

    def to_canonical(self) -> dict[str, Any]:
        return {
            "owner_label": self.owner_label,
            "shape": list(self.matrix.shape),
            "q": self.q,
            "fingerprint": self.fingerprint,
        }


@dataclass(slots=True)
class StructuredSecretKey:
    owner_label: str
    q: int
    n: int
    m: int
    gadget_width: int
    trapdoor: GPVTrapdoor
    preimage_matrix: np.ndarray
    row_permutation: SignedPermutation | None = None

    def to_canonical(self) -> dict[str, Any]:
        return {
            "owner_label": self.owner_label,
            "q": self.q,
            "n": self.n,
            "m": self.m,
            "gadget_width": self.gadget_width,
            "trapdoor": self.trapdoor.to_canonical(),
            "preimage_matrix": self.preimage_matrix.tolist(),
            "row_permutation": None
            if self.row_permutation is None
            else {
                "permutation": self.row_permutation.permutation.tolist(),
                "signs": self.row_permutation.signs.tolist(),
            },
        }


@dataclass(slots=True)
class PartyKeypair:
    actor_label: str
    public_key: StructuredPublicKey
    secret_key: StructuredSecretKey


@dataclass(slots=True)
class AuditorPublicKey:
    matrix_b: np.ndarray
    matrix_p: np.ndarray
    q: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.matrix_b.tobytes())
        digest.update(self.matrix_p.tobytes())
        self.fingerprint = digest.hexdigest()

    def to_canonical(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "b_shape": list(self.matrix_b.shape),
            "p_shape": list(self.matrix_p.shape),
            "fingerprint": self.fingerprint,
        }


@dataclass(slots=True)
class AuditorSecretKey:
    matrix_s: np.ndarray


@dataclass(slots=True)
class AuditorKeypair:
    public_key: AuditorPublicKey
    secret_key: AuditorSecretKey
    max_plaintext_bits: int


@dataclass(slots=True)
class Context:
    epoch: str
    roster_root: str
    session_id: str
    tx_id: str
    action_type: str
    tau: str

    def to_canonical(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "roster_root": self.roster_root,
            "session_id": self.session_id,
            "tx_id": self.tx_id,
            "action_type": self.action_type,
            "tau": self.tau,
        }


@dataclass(slots=True)
class ProxyPolicy:
    policy_id: str
    description: str
    valid_from: str
    valid_to: str
    allowed_action_types: tuple[str, ...] = ("release",)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "description": self.description,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "allowed_action_types": list(self.allowed_action_types),
        }


@dataclass(slots=True)
class RosterEntry:
    member_index: int
    member_id: str
    public_key: StructuredPublicKey

    def to_canonical(self) -> dict[str, Any]:
        return {
            "member_index": self.member_index,
            "member_id": self.member_id,
            "public_key": self.public_key.to_canonical(),
        }


@dataclass(slots=True)
class Roster:
    entries: list[RosterEntry]
    epoch: str
    roster_root: str

    def to_canonical(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_canonical() for entry in self.entries],
            "epoch": self.epoch,
            "roster_root": self.roster_root,
        }


@dataclass(slots=True)
class ProxyCertificate:
    w_payload: dict[str, Any]
    challenge_v1: np.ndarray
    response_y1: np.ndarray
    authorizer_public_key: StructuredPublicKey
    proxy_public_key: StructuredPublicKey

    def to_canonical(self) -> dict[str, Any]:
        return {
            "w_payload": self.w_payload,
            "challenge_v1": self.challenge_v1.tolist(),
            "response_y1": self.response_y1.tolist(),
        }


@dataclass(slots=True)
class ProxyDerivedKey:
    public_key: StructuredPublicKey
    secret_key: StructuredSecretKey
    signed_permutation: SignedPermutation


@dataclass(slots=True)
class ProxySignature:
    u_blocks: list[np.ndarray]
    challenge_c: np.ndarray
    z_blocks: list[np.ndarray]
    certificate: ProxyCertificate
    context: Context
    theta_map: dict[str, bytes]
    metadata: dict[str, Any]

    def to_canonical(self) -> dict[str, Any]:
        return {
            "u_blocks": [block.tolist() for block in self.u_blocks],
            "challenge_c": self.challenge_c.tolist(),
            "z_blocks": [block.tolist() for block in self.z_blocks],
            "certificate": self.certificate.to_canonical(),
            "context": self.context.to_canonical(),
            "theta_map": {key: value.hex() for key, value in self.theta_map.items()},
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class VerifyResult:
    accepted: bool
    rho: np.ndarray | None
    bind_value: bytes | None
    verifier_member_id: str
    debug: dict[str, Any]


@dataclass(slots=True)
class ReceiptProof:
    scheme: str
    bind_hex: str
    statement_digest_hex: str
    rounds: list[dict[str, Any]]
    witness_shape: tuple[int, int]
    response_bound: float
    compact_secret: dict[str, Any] | None = None

    def to_canonical(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "bind_hex": self.bind_hex,
            "statement_digest_hex": self.statement_digest_hex,
            "rounds": self.rounds,
            "witness_shape": list(self.witness_shape),
            "response_bound": self.response_bound,
            "compact_secret": self.compact_secret,
        }


@dataclass(slots=True)
class Receipt:
    mode: str
    ciphertext_1: np.ndarray
    ciphertext_2: np.ndarray
    public_tag: str
    payload_ciphertext: bytes | None = None
    stream_nonce: bytes | None = None

    def to_canonical(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ciphertext_1": self.ciphertext_1.tolist(),
            "ciphertext_2": self.ciphertext_2.tolist(),
            "public_tag": self.public_tag,
            "payload_ciphertext": self.payload_ciphertext,
            "stream_nonce": self.stream_nonce,
        }


@dataclass(slots=True)
class OpenPackage:
    receipt: Receipt
    member_id: str
    bind_hex: str
    receipt_proof: ReceiptProof
    eta: str
    rho: np.ndarray
    payload: dict[str, Any]


@dataclass(slots=True)
class JudgeResult:
    accepted: bool
    reason: str
    debug: dict[str, Any]
