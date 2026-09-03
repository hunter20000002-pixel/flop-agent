"""Technocore response parsing."""
import re

from .client import Message


MESSAGE_RE = re.compile(
    r"^\[?(\d+)\]?\s+"
    r"(\S+)\s+"
    r"<?([^>\s]+)>?\s+"
    r"(.*)$"
)


def parse_messages(
    body: str,
) -> list[Message]:
    """Extract message records from a Technocore response."""

    messages = []

    for line in body.splitlines():
        match = MESSAGE_RE.match(line)

        if not match:
            continue

        messages.append(
            Message(
                seq=int(match.group(1)),
                timestamp=match.group(2),
                short_did=match.group(3),
                text=match.group(4),
                raw_line=line,
            )
        )

    return messages


def find_matching_message(
    body: str,
    message_id: int,
    text: str,
    did_suffix: str,
) -> Message | None:
    """
    Find a specific message using:

    - server message ID
    - exact message text
    - DID suffix
    """

    for message in parse_messages(body):
        if (
            message.seq == message_id
            and message.text == text
            and message.short_did.endswith(did_suffix)
        ):
            return message

    return None