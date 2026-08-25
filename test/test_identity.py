from src.identity import (
    load_or_create_identity,
    sign_message,
)


def test_identity_persists(tmp_path):

    identity_file = (
        tmp_path / "identity.json"
    )

    key1, did1 = load_or_create_identity(
        identity_file
    )

    key2, did2 = load_or_create_identity(
        identity_file
    )

    assert did1 == did2

    signature = sign_message(
        key2,
        "lobby",
        "123456",
        "hello",
    )

    assert signature