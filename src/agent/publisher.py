from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

from src.agent.loop import AgentLoopResult
from src.agent.task_source import ObservedTask
from src.client import (
    TechnocoreError,
    publish_did,
    send_signed,
)
from src.config import Config, DEFAULT_CONFIG
from src.identity import (
    did_from_private_key,
    load_or_create_identity,
    sign_message,
)


@dataclass(frozen=True, slots=True)
class PublishedMessage:
    """Record describing a successfully published Technocore result."""

    response_text: str
    nonce: str
    did: str
    server_response: str


class TechnocoreResultPublisher:
    """Publish autonomous execution results to Technocore."""

    def __init__(
        self,
        *,
        config: Config = DEFAULT_CONFIG,
        key_file: str | Path | None = None,
        publish_presence: bool = True,
    ) -> None:
        if not isinstance(config, Config):
            raise TypeError(
                "config must be a Config"
            )

        if key_file is not None:
            key_file = Path(key_file)

        self.config = config
        self.key_file = (
            key_file
            if key_file is not None
            else config.key_file
        )
        self.publish_presence = publish_presence

        self._private_key = None
        self._did: str | None = None
        self._presence_published = False

    @property
    def did(self) -> str | None:
        """Return the loaded DID, if the identity has been loaded."""

        return self._did

    def publish(
        self,
        observed: ObservedTask,
        result: AgentLoopResult,
    ) -> PublishedMessage:
        """
        Publish the result of an autonomous task execution.

        The message is signed with the persistent local Ed25519 identity.
        """

        if not isinstance(observed, ObservedTask):
            raise TypeError(
                "observed must be an ObservedTask"
            )

        if not isinstance(result, AgentLoopResult):
            raise TypeError(
                "result must be an AgentLoopResult"
            )

        self._ensure_identity()

        assert self._did is not None
        assert self._private_key is not None

        if (
            self.publish_presence
            and not self._presence_published
        ):
            publish_did(
                self.config.base_url,
                self._did,
                self.config.user_agent,
            )

            self._presence_published = True

        response_text = self._format_result(
            observed,
            result,
        )

        nonce = str(
            time.time_ns()
        )

        signature = sign_message(
            self._private_key,
            self.config.room,
            nonce,
            response_text,
        )

        server_response = send_signed(
            self.config.base_url,
            self.config.room,
            self._did,
            signature,
            nonce,
            response_text,
            self.config.user_agent,
        )

        return PublishedMessage(
            response_text=response_text,
            nonce=nonce,
            did=self._did,
            server_response=server_response,
        )

    def _ensure_identity(self) -> None:
        """Load or create the persistent agent identity once."""

        if (
            self._private_key is not None
            and self._did is not None
        ):
            return

        private_key, did = load_or_create_identity(
            self.key_file
        )

        derived_did = did_from_private_key(
            private_key
        )

        if did != derived_did:
            raise TechnocoreError(
                "Loaded identity DID does not match "
                "the derived DID."
            )

        self._private_key = private_key
        self._did = did

    @staticmethod
    def _format_result(
        observed: ObservedTask,
        result: AgentLoopResult,
    ) -> str:
        """Create a deterministic, human-readable result message."""

        execution = result.result

        lines = [
            "FLOP Agent autonomous execution result",
            f"source_message: {observed.message_id}",
            f"task_id: {result.task_id}",
            f"status: {execution.status.value}",
            f"iterations: {result.iterations}",
            f"action: {result.action.value}",
        ]

        if execution.succeeded:
            lines.append("success: true")
        else:
            lines.append("success: false")

        if execution.output is not None:
            lines.append(
                f"output: {TechnocoreResultPublisher._stringify(
                    execution.output
                )}"
            )

        if execution.error:
            lines.append(
                f"error: {execution.error}"
            )

        return "\n".join(lines)

    @staticmethod
    def _stringify(value: Any) -> str:
        """Convert arbitrary execution output into safe single-line text."""

        text = str(value)

        return " ".join(
            text.split()
        )