from __future__ import annotations

"""Global singleton state holder.

Compatibility note:
- This file avoids using PEP604 union syntax (X | None) so it works on Python 3.9.
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
    """Test helper to clear the singleton."""
    global _state
    _state = None
