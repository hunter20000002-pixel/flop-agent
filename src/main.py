"""Interactive Technocore agent."""

from datetime import datetime, timezone
import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request

from .client import (
    TechnocoreError,
    publish_did,
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

    print()
    print("=" * 60)
    print("              TECHNOC0RE AGENT")
    print("=" * 60)

    print(f"DID: {did}")

    print()
    print("[*] Publishing DID presence...")

    try:
        publish_did(
            config.base_url,
            did,
            config.user_agent,
        )

        print(
            "[+] DID presence published successfully."
        )

    except TechnocoreError as exc:

        print(
            f"[!] DID publication failed: {exc}"
        )

        print(
            "[*] Continuing because message sending "
            "does not depend on DID publication."
        )

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