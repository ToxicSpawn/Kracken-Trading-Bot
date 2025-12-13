from __future__ import annotations

"""Global singleton state holder (Python 3.9 compatible).

This project may be run on Python 3.9 where PEP604 union syntax (X | None) is unsupported.
"""

from typing import Optional

from core.state import GlobalState

_state: Optional[GlobalState] = None


def get_global_state() -> GlobalState:
    global _state
    if _state is None:
        _state = GlobalState()
    return _state


def reset_global_state() -> None:
    global _state
    _state = None
