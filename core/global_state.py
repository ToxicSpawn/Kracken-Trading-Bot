from core.state import GlobalState

_state: GlobalState | None = None

def get_global_state() -> GlobalState:
    global _state
    if _state is None:
        _state = GlobalState()
    return _state
