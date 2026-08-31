"""Backward-compatible imports for the autonomy policy API.

The canonical autonomy implementation lives in ``decision.py``.
This module remains as a compatibility shim for callers that import
the ``src.agent.autonomy`` path.
"""

from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)

__all__ = [
    "AutonomyAction",
    "AutonomyDecision",
    "AutonomyPolicy",
]