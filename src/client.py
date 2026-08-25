"""HTTP client for Technocore."""

from dataclasses import dataclass
import urllib.error
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class Message:
    """A parsed Technocore message."""

    seq: int
    timestamp: str
    short_did: str
    text: str
    raw_line: str


class TechnocoreError(RuntimeError):
    """Raised when a Technocore request fails."""


def _request(
    url: str,
    user_agent: str,
) -> str:
    """Perform a GET request and return the response body."""

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
        ) as response:

            return response.read().decode(
                "utf-8",
                errors="replace",
            )

    except urllib.error.HTTPError as exc:

        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        raise TechnocoreError(
            f"HTTP {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:

        raise TechnocoreError(
            f"Network error: {exc.reason}"
        ) from exc

    except TimeoutError as exc:

        raise TechnocoreError(
            "Request timed out."
        ) from exc


def publish_did(
    base_url: str,
    did: str,
    user_agent: str,
) -> None:
    """
    Publish DID presence through Technocore's DID endpoint.

    Raises TechnocoreError when the request fails.
    """

    import hashlib

    fingerprint = hashlib.sha256(
        did.encode("utf-8")
    ).hexdigest()[:16]

    url = (
        f"{base_url.rstrip('/')}"
        f"/kv/did/{fingerprint}/set/"
        f"{urllib.parse.quote(did, safe='')}"
    )

    _request(
        url,
        user_agent,
    )


def send_signed(
    base_url: str,
    room: str,
    did: str,
    signature: str,
    nonce: str,
    text: str,
    user_agent: str,
) -> str:
    """Send a signed message to a Technocore room."""

    url = (
        f"{base_url.rstrip('/')}"
        f"/r/{room}"
        f"/say-signed/"
        f"{urllib.parse.quote(did, safe='')}/"
        f"{urllib.parse.quote(signature, safe='')}/"
        f"{urllib.parse.quote(nonce, safe='')}/"
        f"{urllib.parse.quote(text, safe='')}"
    )

    return _request(
        url,
        user_agent,
    )


def read_since(
    base_url: str,
    room: str,
    since: int,
    user_agent: str,
) -> str:
    """Read room messages after a given sequence number."""

    url = (
        f"{base_url.rstrip('/')}"
        f"/r/{room}"
        f"?since={since}"
    )

    return _request(
        url,
        user_agent,
    )