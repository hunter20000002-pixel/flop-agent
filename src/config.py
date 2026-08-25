"""Configuration for the Technocore agent."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    base_url: str = "https://technocore.chat"
    room: str = "lobby"

    # This will be copied into the new project from your existing
    # working agent. It is deliberately excluded from Git.
    key_file: Path = Path("flop_agent_identity.json")

    user_agent: str = "technocore-agent/0.1.0"


DEFAULT_CONFIG = Config()