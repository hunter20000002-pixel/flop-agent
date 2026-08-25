"""Ed25519 DID:key identity management."""

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58(data: bytes) -> str:
    """Encode bytes using Bitcoin Base58."""

    n = int.from_bytes(data, "big")
    result = []

    while n > 0:
        n, remainder = divmod(n, 58)
        result.append(B58[remainder])

    prefix = "1" * (len(data) - len(data.lstrip(b"\x00")))

    return prefix + "".join(reversed(result))


def load_or_create_identity(
    path: Path,
) -> tuple[ed25519.Ed25519PrivateKey, str]:
    """
    Load an existing identity or create a new one.

    IMPORTANT:
    The generated identity file contains the private key and must
    never be uploaded to GitHub.
    """

    if path.exists():

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        private_key = (
            ed25519.Ed25519PrivateKey.from_private_bytes(
                bytes.fromhex(data["private_key_hex"])
            )
        )

        return private_key, data["did"]

    print("[*] No identity file found.")
    print("[*] Generating a new Ed25519 DID...")

    private_key = ed25519.Ed25519PrivateKey.generate()

    raw_private = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )

    raw_public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    did = "did:key:z" + b58(
        b"\xed\x01" + raw_public
    )

    path.write_text(
        json.dumps(
            {
                "did": did,
                "private_key_hex": raw_private.hex(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return private_key, did


def sign_message(
    private_key: ed25519.Ed25519PrivateKey,
    room: str,
    nonce: str,
    text: str,
) -> str:
    """
    Sign:

        room|nonce|text

    using Ed25519 and return URL-safe Base64 without '=' padding.
    """

    payload = (
        f"{room}|{nonce}|{text}"
    ).encode("utf-8")

    signature = private_key.sign(payload)

    return (
        base64.urlsafe_b64encode(signature)
        .decode("ascii")
        .rstrip("=")
    )