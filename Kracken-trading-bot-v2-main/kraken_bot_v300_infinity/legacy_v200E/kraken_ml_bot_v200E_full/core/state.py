from __future__ import annotations

"""Global state with intent bus and health monitoring.

Compatibility note:
- Uses Optional[X] instead of X | None for Python 3.9 compatibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PositionSnapshot:
    symbol: str
    size: float = 0.0
    entry_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountState:
    name: str
    equity: float = 0.0
    balance: float = 0.0
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairSnapshot:
    symbol: str
    last_price: Optional[float] = None
    last_update_ts: Optional[float] = None
    # Optional legacy fields may still exist in older code; do not rely on them for execution.
    last_signal_short: Optional[str] = None
    last_signal_mid: Optional[str] = None
    last_signal_long: Optional[str] = None
    trend_score: float = 0.0
    vol_score: float = 0.0
    slippage_bps: float = 0.0
    stress_multiplier: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalState:
    accounts: Dict[str, AccountState] = field(default_factory=dict)
    pairs: Dict[str, PairSnapshot] = field(default_factory=dict)

    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0

    # Safety / ops
    trading_enabled: bool = True
    mode: str = "paper"  # shadow/backtest/sim/paper/live
    system_mode: str = "NORMAL"  # NORMAL | DEGRADED | MANAGE_ONLY | PAUSED

    # Unified metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    # Level-14: health + intents
    agent_health: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _intent_q: deque = field(default_factory=deque, repr=False)

    # Namespaces (research vs live)
    live: Dict[str, Any] = field(default_factory=dict)
    research: Dict[str, Any] = field(default_factory=dict)

    def set_system_mode(self, mode: str, reason: Optional[str] = None) -> None:
        self.system_mode = mode
        if reason:
            self.meta.setdefault("system_mode_reason", reason)
        self.meta["system_mode_updated_at"] = utcnow().isoformat()

    def heartbeat(self, agent_name: str, status: str = "OK", last_error: Optional[str] = None) -> None:
        self.agent_health.setdefault(agent_name, {})
        self.agent_health[agent_name]["last_heartbeat"] = utcnow().isoformat()
        self.agent_health[agent_name]["status"] = status
        if last_error is not None:
            self.agent_health[agent_name]["last_error"] = last_error

    # ---- Intent bus (single source of execution requests) ----
    def submit_intent(self, intent: Dict[str, Any]) -> None:
        self._intent_q.append(intent)

    def pop_next_intent(self) -> Optional[Dict[str, Any]]:
        if not self._intent_q:
            return None
        return self._intent_q.popleft()

    def drain_intents(self, max_n: int = 1000) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(min(max_n, len(self._intent_q))):
            out.append(self._intent_q.popleft())
        return out
