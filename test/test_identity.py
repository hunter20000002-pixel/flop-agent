import base64

import pytest
from cryptography.exceptions import InvalidSignature

from src.identity import (
    load_or_create_identity,
    sign_message,
)


def test_identity_persists(tmp_path):
    identity_file = tmp_path / "identity.json"

    key1, did1 = load_or_create_identity(identity_file)
    key2, did2 = load_or_create_identity(identity_file)

    assert did1 == did2
    assert identity_file.exists()

    signature = sign_message(
        key2,
        "lobby",
        "123456",
        "hello",
    )

    assert signature


def test_signature_is_cryptographically_valid(tmp_path):
    identity_file = tmp_path / "identity.json"

    private_key, _ = load_or_create_identity(
        identity_file
    )

    room = "lobby"
    nonce = "123456"
    text = "hello"

    signature = sign_message(
        private_key,
        room,
        nonce,
        text,
    )

    signature_bytes = base64.urlsafe_b64decode(
        signature + "=" * (-len(signature) % 4)
    )

    payload = f"{room}|{nonce}|{text}".encode("utf-8")

    public_key = private_key.public_key()

    # This should not raise InvalidSignature.
    public_key.verify(
        signature_bytes,
        payload,
    )


def test_tampered_message_fails_verification(tmp_path):
    identity_file = tmp_path / "identity.json"

    private_key, _ = load_or_create_identity(
        identity_file
    )

    room = "lobby"
    nonce = "123456"
    original_text = "hello"

    signature = sign_message(
        private_key,
        room,
        nonce,
        original_text,
    )

    signature_bytes = base64.urlsafe_b64decode(
        signature + "=" * (-len(signature) % 4)
    )

    tampered_payload = (
        f"{room}|{nonce}|THIS MESSAGE WAS CHANGED"
    ).encode("utf-8")

    public_key = private_key.public_key()

    with pytest.raises(InvalidSignature):
        public_key.verify(
            signature_bytes,
            tampered_payload,
        )