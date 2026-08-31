import pytest

from cryptography.hazmat.primitives.asymmetric import (
    ed25519,
)

from src.agent.verification import (
    SignedExecutionRecord,
    VerificationPayload,
    sign_execution_record,
    verify_execution_record,
)


def make_payload() -> VerificationPayload:
    return VerificationPayload(
        task_id="task-123",
        step_order=1,
        success=True,
        output="execution complete",
    )


def test_verification_payload_is_deterministic():
    first = make_payload()
    second = make_payload()

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.digest() == second.digest()


def test_verification_payload_digest_changes_when_content_changes():
    first = make_payload()

    second = VerificationPayload(
        task_id="task-123",
        step_order=1,
        success=True,
        output="different output",
    )

    assert first.digest() != second.digest()


def test_sign_execution_record():
    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    payload = make_payload()

    record = sign_execution_record(
        payload,
        private_key,
        "did:key:test",
    )

    assert isinstance(
        record,
        SignedExecutionRecord,
    )

    assert record.payload == payload
    assert record.signer == "did:key:test"
    assert record.signature


def test_signed_execution_record_verifies():
    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    public_key = private_key.public_key()

    record = sign_execution_record(
        make_payload(),
        private_key,
        "did:key:test",
    )

    assert verify_execution_record(
        record,
        public_key,
    )


def test_modified_payload_fails_verification():
    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    public_key = private_key.public_key()

    record = sign_execution_record(
        make_payload(),
        private_key,
        "did:key:test",
    )

    modified_payload = VerificationPayload(
        task_id="task-123",
        step_order=1,
        success=True,
        output="tampered output",
    )

    modified_record = SignedExecutionRecord(
        payload=modified_payload,
        signer=record.signer,
        signature=record.signature,
    )

    assert not verify_execution_record(
        modified_record,
        public_key,
    )


def test_wrong_public_key_fails_verification():
    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    wrong_private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    record = sign_execution_record(
        make_payload(),
        private_key,
        "did:key:test",
    )

    assert not verify_execution_record(
        record,
        wrong_private_key.public_key(),
    )


def test_invalid_payload_is_rejected():
    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    with pytest.raises(TypeError):
        sign_execution_record(
            "invalid",  # type: ignore[arg-type]
            private_key,
            "did:key:test",
        )


def test_invalid_private_key_is_rejected():
    with pytest.raises(TypeError):
        sign_execution_record(
            make_payload(),
            "invalid",  # type: ignore[arg-type]
            "did:key:test",
        )


def test_invalid_record_is_rejected():
    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    with pytest.raises(TypeError):
        verify_execution_record(
            "invalid",  # type: ignore[arg-type]
            private_key.public_key(),
        )


def test_invalid_signature_returns_false():
    private_key = (
        ed25519.Ed25519PrivateKey.generate()
    )

    record = SignedExecutionRecord(
        payload=make_payload(),
        signer="did:key:test",
        signature="invalid-signature",
    )

    assert not verify_execution_record(
        record,
        private_key.public_key(),
    )