"""Ed25519 DID:key identity management."""

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class IdentityError(RuntimeError):
    """Raised when the local identity is invalid or corrupted."""


def b58(data: bytes) -> str:
    """Encode bytes using Bitcoin Base58."""

    n = int.from_bytes(data, "big")
    result = []

    while n > 0:
        n, remainder = divmod(n, 58)
        result.append(B58[remainder])

    prefix = "1" * (
        len(data) - len(data.lstrip(b"\x00"))
    )

    return prefix + "".join(reversed(result))


def did_from_private_key(
    private_key: ed25519.Ed25519PrivateKey,
) -> str:
    """Derive the did:key identifier from an Ed25519 private key."""

    raw_public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    return "did:key:z" + b58(
        b"\xed\x01" + raw_public
    )


def load_or_create_identity(
    path: Path,
) -> tuple[ed25519.Ed25519PrivateKey, str]:
    """
    Load an existing identity or create a new one.

    The identity file contains the private key and must never
    be uploaded to GitHub.
    """

    if path.exists():

        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )

            stored_did = data["did"]
            private_key_hex = data["private_key_hex"]

            if not isinstance(stored_did, str):
                raise IdentityError(
                    "Invalid identity file: 'did' must be a string."
                )

            if not isinstance(private_key_hex, str):
                raise IdentityError(
                    "Invalid identity file: "
                    "'private_key_hex' must be a string."
                )

            raw_private = bytes.fromhex(
                private_key_hex
            )

            if len(raw_private) != 32:
                raise IdentityError(
                    "Invalid identity file: "
                    "Ed25519 private keys must contain 32 bytes."
                )

            private_key = (
                ed25519.Ed25519PrivateKey.from_private_bytes(
                    raw_private
                )
            )

        except IdentityError:
            raise

        except (OSError, json.JSONDecodeError) as exc:

            raise IdentityError(
                f"Could not read identity file: {exc}"
            ) from exc

        except (KeyError, ValueError, TypeError) as exc:

            raise IdentityError(
                "Invalid identity file format."
            ) from exc

        derived_did = did_from_private_key(
            private_key
        )

        if stored_did != derived_did:

            raise IdentityError(
                "Identity file integrity check failed: "
                "the stored DID does not match the "
                "stored private key."
            )

        return private_key, stored_did

    print("[*] No identity file found.")
    print("[*] Generating a new Ed25519 DID...")

    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    raw_private = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )

    did = did_from_private_key(
        private_key
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

    signature = private_key.sign(
        payload
    )

    return (
        base64.urlsafe_b64encode(
            signature
        )
        .decode("ascii")
        .rstrip("=")
    )