"""Interactive Technocore agent."""

from datetime import datetime, timezone
import hashlib
import time
import urllib.parse
import urllib.request

from .client import (
    TechnocoreError,
    read_since,
    send_signed,
)
from .config import DEFAULT_CONFIG
from .identity import (
    load_or_create_identity,
    sign_message,
)
from .parser import (
    find_matching_message,
    parse_messages,
)


def publish_did(
    base_url: str,
    did: str,
    user_agent: str,
) -> None:
    """
    Publish the DID using Technocore's DID presence endpoint.

    Failure here does not stop the agent because the signed message
    itself is the important authenticated operation.
    """

    fingerprint = hashlib.sha256(
        did.encode("utf-8")
    ).hexdigest()[:16]

    url = (
        f"{base_url.rstrip('/')}"
        f"/kv/did/{fingerprint}/set/"
        f"{urllib.parse.quote(did, safe='')}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=20,
        ):
            pass

    except Exception:
        pass


def identify_sent_message(
    response_body: str,
    text: str,
    did: str,
):
    """Find our message in the original server response."""

    did_suffix = did[-4:]

    for message in parse_messages(response_body):

        if (
            message.text == text
            and message.short_did.endswith(did_suffix)
        ):
            return message

    return None


def main() -> None:

    config = DEFAULT_CONFIG

    # Load your existing identity if the identity file exists.
    private_key, did = load_or_create_identity(
        config.key_file
    )

    publish_did(
        config.base_url,
        did,
        config.user_agent,
    )

    print()
    print("=" * 60)
    print("              TECHNOC0RE AGENT")
    print("=" * 60)

    print(f"DID: {did}")

    text = input(
        "\nEnter your message: "
    )

    if not text.strip():

        raise SystemExit(
            "[-] Message cannot be empty."
        )

    room = config.room

    # Millisecond nonce, matching the original working script.
    nonce = str(
        int(time.time() * 1000)
    )

    signature = sign_message(
        private_key,
        room,
        nonce,
        text,
    )

    print(
        "\n[*] Sending signed message..."
    )

    try:

        response_body = send_signed(
            config.base_url,
            room,
            did,
            signature,
            nonce,
            text,
            config.user_agent,
        )

    except TechnocoreError as exc:

        raise SystemExit(
            f"[-] Send failed: {exc}"
        ) from exc

    # -----------------------------------------------------
    # Identify our message in the original response.
    # -----------------------------------------------------

    message = identify_sent_message(
        response_body,
        text,
        did,
    )

    if message is None:

        print()
        print(
            "[-] Server request returned successfully,"
        )
        print(
            "    but the new message could not be identified."
        )
        print()
        print(
            "Raw server response:"
        )
        print(
            response_body
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Initial confirmation.
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("[+] MESSAGE ACCEPTED BY SERVER")
    print("=" * 60)

    print(
        f"Message ID: {message.seq}"
    )

    print(
        f"Timestamp:  {message.timestamp}"
    )

    print(
        f"DID:        {did}"
    )

    print(
        f"Short DID:  {message.short_did}"
    )

    print(
        f"Room URL:   "
        f"{config.base_url}/r/{room}"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Independent verification.
    # -----------------------------------------------------

    print()
    print(
        "[*] Performing independent verification..."
    )

    try:

        verification_body = read_since(
            config.base_url,
            room,
            max(0, message.seq - 1),
            config.user_agent,
        )

    except TechnocoreError as exc:

        raise SystemExit(
            f"[-] Verification request failed: {exc}"
        ) from exc

    verified = find_matching_message(
        verification_body,
        message.seq,
        text,
        did[-4:],
    )

    print()
    print("=" * 60)

    if verified is not None:

        print(
            "[+] VERIFIED: MESSAGE EXISTS ON TECHNCORE"
        )

        print("=" * 60)

        print(
            f"Message ID: {verified.seq}"
        )

        print(
            f"Timestamp:  {verified.timestamp}"
        )

        print(
            f"DID:        {did}"
        )

        print(
            f"Short DID:  {verified.short_did}"
        )

        print(
            f"Room URL:   "
            f"{config.base_url}/r/{room}"
        )

        print()
        print(
            "[+] Exact server line:"
        )

        print(
            "-" * 40
        )

        # This is the actual line returned by the fresh
        # verification request.
        print(
            verified.raw_line
        )

        print(
            "-" * 40
        )

        print()
        print(
            "[+] The message was found again in a"
        )

        print(
            "    fresh request after the original send."
        )

    else:

        print(
            "[-] VERIFICATION FAILED"
        )

        print("=" * 60)

        print(
            "The original request succeeded, but the"
        )

        print(
            "message was not found in the fresh request."
        )

    print("=" * 60)

    print()
    print(
        "Verification completed at:"
    )

    print(
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    print()


if __name__ == "__main__":
    main()