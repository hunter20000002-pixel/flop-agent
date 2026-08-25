from src.parser import (
    find_matching_message,
    parse_messages,
)


def test_parse_message():

    body = """
# room lobby messages 1 range 10..10
[10] 2026-08-25T07:00:00Z <z6Mk…tXnP> hello
"""

    messages = parse_messages(body)

    assert len(messages) == 1
    assert messages[0].seq == 10
    assert messages[0].text == "hello"


def test_find_matching_message():

    body = """
[10] 2026-08-25T07:00:00Z <z6Mk…tXnP> hello
"""

    result = find_matching_message(
        body,
        10,
        "hello",
        "tXnP",
    )

    assert result is not None

    assert (
        result.raw_line
        == "[10] 2026-08-25T07:00:00Z "
           "<z6Mk…tXnP> hello"
    )