from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives.asymmetric import ed25519


@dataclass(frozen=True, slots=True)
class VerificationPayload:
    """Canonical payload used for execution verification."""

    task_id: str
    step_order: int
    success: bool
    output: Any = None
    error: str | None = None

    def canonical_bytes(self) -> bytes:
        """Return a deterministic byte representation of the payload."""

        payload = {
            "error": self.error,
            "output": self.output,
            "step_order": self.step_order,
            "success": self.success,
            "task_id": self.task_id,
        }

        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical payload."""

        return hashlib.sha256(
            self.canonical_bytes()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class SignedExecutionRecord:
    """Cryptographically signed execution record."""

    payload: VerificationPayload
    signer: str
    signature: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.payload,
            VerificationPayload,
        ):
            raise TypeError(
                "payload must be a VerificationPayload"
            )

        if not isinstance(self.signer, str):
            raise TypeError("signer must be a string")

        if not self.signer.strip():
            raise ValueError("signer must not be empty")

        if not isinstance(self.signature, str):
            raise TypeError("signature must be a string")

        if not self.signature.strip():
            raise ValueError(
                "signature must not be empty"
            )


def sign_execution_record(
    payload: VerificationPayload,
    private_key: ed25519.Ed25519PrivateKey,
    signer: str,
) -> SignedExecutionRecord:
    """Sign an execution payload with an Ed25519 private key."""

    if not isinstance(
        payload,
        VerificationPayload,
    ):
        raise TypeError(
            "payload must be a VerificationPayload"
        )

    if not isinstance(
        private_key,
        ed25519.Ed25519PrivateKey,
    ):
        raise TypeError(
            "private_key must be an Ed25519 private key"
        )

    if not isinstance(signer, str):
        raise TypeError("signer must be a string")

    if not signer.strip():
        raise ValueError("signer must not be empty")

    signature = private_key.sign(
        payload.canonical_bytes()
    )

    encoded_signature = (
        base64.urlsafe_b64encode(signature)
        .decode("ascii")
        .rstrip("=")
    )

    return SignedExecutionRecord(
        payload=payload,
        signer=signer,
        signature=encoded_signature,
    )


def verify_execution_record(
    record: SignedExecutionRecord,
    public_key: ed25519.Ed25519PublicKey,
) -> bool:
    """Verify the signature of an execution record."""

    if not isinstance(
        record,
        SignedExecutionRecord,
    ):
        raise TypeError(
            "record must be a SignedExecutionRecord"
        )

    if not isinstance(
        public_key,
        ed25519.Ed25519PublicKey,
    ):
        raise TypeError(
            "public_key must be an Ed25519 public key"
        )

    padding = "=" * (
        (-len(record.signature)) % 4
    )

    try:
        signature = base64.urlsafe_b64decode(
            record.signature + padding
        )
    except Exception:
        return False

    try:
        public_key.verify(
            signature,
            record.payload.canonical_bytes(),
        )
    except Exception:
        return False

    return True